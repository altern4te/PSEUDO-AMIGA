import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_miou, color_map):

    torch.save(
        {
            "epoch": epoch,
            "best_miou": best_miou,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "color_map": color_map,
            "model_config": {
                "in_channels": model.in_channels,
                "num_classes": model.num_classes,
                "base_channels": model.base_channels,
                "bilinear": model.bilinear,
            },
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    # weights_only=False: newer torch defaults to True, which would reject the
    # non-tensor color_map/model_config metadata we intentionally store above.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", -1.0)


class EarlyStopper:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0

    def step(self, value):
        """Returns True if training should stop."""
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def plot_training_curves(csv_path, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, train_loss, val_loss = [], [], []
    train_class_iou, val_class_iou = {}, {}  # class_name -> list of per-epoch values

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        train_iou_cols = [c for c in reader.fieldnames if c.startswith("train_iou_")]
        val_iou_cols = [c for c in reader.fieldnames if c.startswith("val_iou_")]
        for c in train_iou_cols:
            train_class_iou[c[len("train_iou_"):]] = []
        for c in val_iou_cols:
            val_class_iou[c[len("val_iou_"):]] = []

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            for c in train_iou_cols:
                train_class_iou[c[len("train_iou_"):]].append(float(row[c]))
            for c in val_iou_cols:
                val_class_iou[c[len("val_iou_"):]].append(float(row[c]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    class_names = sorted(set(train_class_iou) | set(val_class_iou))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, name in enumerate(class_names):
        color = colors[i % len(colors)]
        if name in train_class_iou:
            axes[1].plot(epochs, train_class_iou[name], color=color, linestyle="-", label=f"{name} (train)")
        if name in val_class_iou:
            axes[1].plot(epochs, val_class_iou[name], color=color, linestyle="--", label=f"{name} (val)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU by Class")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
