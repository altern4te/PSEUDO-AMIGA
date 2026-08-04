"""
Train a U-Net on GF-NIRNDVI composite -> crop/weed/soil masks.

Usage:
    python train.py --config config.yaml
"""
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from augment import (
    JointCompose,
    RandomBrightnessContrast,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotate90,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from dataset import (
    SegmentationDataset,
    build_file_pairs,
    build_file_pairs_recursive,
    load_color_map,
    split_pairs,
)
from losses import CEDiceLoss, compute_class_weights
from metrics import ConfusionMatrixTracker
from model import UNet
from utils import CSVLogger, EarlyStopper, load_checkpoint, plot_training_curves, save_checkpoint, set_seed


def build_transforms(cfg, train: bool):
    size = cfg["train"]["image_size"]
    if train:
        tlist = [RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), RandomRotate90()]
        if cfg["train"].get("random_resized_crop", True):
            tlist.append(RandomResizedCrop(size, scale=(0.7, 1.0)))
        else:
            tlist.append(Resize(size))
        if cfg["train"].get("photometric_augment", False):
            tlist.append(RandomBrightnessContrast())
        tlist.append(ToTensor())
    else:
        tlist = [Resize(size), ToTensor()]
    return JointCompose(tlist)


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp, amp_dtype, scaler, num_classes):
    model.train()
    total_loss, n = 0.0, 0
    tracker = ConfusionMatrixTracker(num_classes)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)
            if scaler is not None:  # only used for float16; bfloat16 needs no loss scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        tracker.update(logits.argmax(dim=1).detach(), masks.detach())

    return total_loss / max(n, 1), tracker


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, n = 0.0, 0
    tracker = ConfusionMatrixTracker(num_classes)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        tracker.update(logits.argmax(dim=1), masks)

    return total_loss / max(n, 1), tracker


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"].get("seed", 42))

    color_map = load_color_map(cfg["data"]["color_map"])
    num_classes = len(color_map)
    class_names = [c["name"] for c in color_map]

    if "root_dir" in cfg["data"]:
        # Nested multi-session/multi-capture layout, e.g.
        # plant data/<session>/<capture_id>/GF-NIRNDVI/*.jpg
        pairs = build_file_pairs_recursive(
            cfg["data"]["root_dir"],
            composite_dirname=cfg["data"].get("composite_dirname", "GF-NIRNDVI"),
            mask_dirname=cfg["data"].get("mask_dirname", "masks"),
        )
    else:
        # Old flat layout: one composites folder, one masks folder.
        pairs = build_file_pairs(cfg["data"]["composites_dir"], cfg["data"]["masks_dir"])

    if len(pairs) == 0:
        raise RuntimeError(
            "No matching composite/mask pairs found -- check data.root_dir (or "
            "data.composites_dir/data.masks_dir) in config.yaml, and confirm filenames "
            "and folder names line up (see README)."
        )

    train_pairs, val_pairs = split_pairs(
        pairs,
        val_fraction=cfg["data"].get("val_fraction", 0.15),
        seed=cfg["data"].get("split_seed", 42),
        group_by_sequence=cfg["data"].get("group_by_sequence", True),
        sequence_regex=cfg["data"].get("sequence_regex"),
    )
    print(f"Dataset: {len(pairs)} pairs total -> {len(train_pairs)} train / {len(val_pairs)} val")
    if len(val_pairs) == 0:
        raise RuntimeError("Validation split is empty -- check val_fraction / group_by_sequence in config.yaml.")

    train_tf = build_transforms(cfg, train=True)
    val_tf = build_transforms(cfg, train=False)

    train_ds = SegmentationDataset(train_pairs, color_map, transform=train_tf)
    val_ds = SegmentationDataset(val_pairs, color_map, transform=val_tf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_workers = cfg["train"].get("num_workers", 4)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_ds) > cfg["train"]["batch_size"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    model = UNet(
        in_channels=cfg["model"].get("in_channels", 3),
        num_classes=num_classes,
        base_channels=cfg["model"].get("base_channels", 64),
        bilinear=cfg["model"].get("bilinear", True),
    ).to(device)

    class_weights = None
    if cfg["train"].get("use_class_weights", True):
        class_weights = compute_class_weights(train_ds, num_classes).to(device)
        print(f"Class weights {dict(zip(class_names, [round(w, 3) for w in class_weights.tolist()]))}")

    criterion = CEDiceLoss(class_weights=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4)
    )
    epochs = cfg["train"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg["train"].get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if (use_amp and amp_dtype == torch.float16) else None

    out_dir = Path(cfg["checkpoint"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_fieldnames = [f"train_iou_{name}" for name in class_names] + [
        f"val_iou_{name}" for name in class_names
    ]
    logger = CSVLogger(
        out_dir / "training_log.csv",
        fieldnames=["epoch", "train_loss", "val_loss", "train_miou", "val_miou", "val_pixel_acc", "lr"]
        + per_class_fieldnames,
    )

    start_epoch, best_miou = 0, -1.0
    resume = cfg["checkpoint"].get("resume")
    if resume:
        start_epoch, best_miou = load_checkpoint(resume, model, optimizer, scheduler)
        print(f"Resumed from {resume} at epoch {start_epoch}, best_miou={best_miou:.4f}")

    stopper = EarlyStopper(patience=cfg["train"].get("early_stopping_patience", 20))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_tracker = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp, amp_dtype, scaler, num_classes
        )
        val_loss, val_tracker = validate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        train_miou = train_tracker.mean_iou().item()
        val_miou = val_tracker.mean_iou().item()
        val_acc = val_tracker.pixel_accuracy().item()
        train_class_ious = train_tracker.iou_per_class().tolist()
        val_class_ious = val_tracker.iou_per_class().tolist()
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"[{epoch + 1}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_mIoU={train_miou:.4f} val_mIoU={val_miou:.4f} val_acc={val_acc:.4f} "
            f"lr={lr_now:.2e} ({dt:.1f}s)"
        )
        val_class_iou_str = "  ".join(
            f"{i}:{name}={iou:.4f}" for i, (name, iou) in enumerate(zip(class_names, val_class_ious))
        )
        print(f"    val IoU by class: {val_class_iou_str}")

        logger.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_miou": train_miou,
                "val_miou": val_miou,
                "val_pixel_acc": val_acc,
                "lr": lr_now,
                **{f"train_iou_{name}": iou for name, iou in zip(class_names, train_class_ious)},
                **{f"val_iou_{name}": iou for name, iou in zip(class_names, val_class_ious)},
            }
        )

        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch + 1, best_miou, color_map)

        if (epoch + 1) % cfg["checkpoint"].get("save_every", 10) == 0:
            save_checkpoint(out_dir / f"epoch_{epoch + 1}.pt", model, optimizer, scheduler, epoch + 1, best_miou, color_map)

        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch + 1, best_miou, color_map)

        if stopper.step(val_miou):
            print(f"Early stopping at epoch {epoch + 1} (no val mIoU improvement for {stopper.patience} epochs).")
            break

    print(f"Training complete. Best val mIoU: {best_miou:.4f}")
    try:
        plot_training_curves(out_dir / "training_log.csv", out_dir / "training_curves.png")
        print(f"Saved training curves to {out_dir / 'training_curves.png'}")
    except Exception as e:
        print(f"Could not generate training curve plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)