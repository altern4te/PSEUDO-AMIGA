"""
Run inference with a trained U-Net checkpoint on new GF-NIRNDVI composite images.

Usage:
    python predict.py --checkpoint checkpoints/best.pt --input path/to/image_or_dir --out_dir predictions
    python predict.py --checkpoint checkpoints/best.pt --input frame.png --out_dir predictions --overlay
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import UNet

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}

CLASS_COLORS = {
    1: (0, 255, 0),   # crop -> green
    2: (255, 0, 0),   # weed -> red
}

def colorize_prediction(pred_rgb):
    out = pred_rgb.copy()
    for class_id, color in CLASS_COLORS.items():
        out[(pred_rgb == class_id).all(axis=-1)] = color
    return out

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = UNet(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        base_channels=cfg.get("base_channels", 64),
        bilinear=cfg["bilinear"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["color_map"]


def index_to_rgb(index_mask, color_map):
    h, w = index_mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, entry in enumerate(color_map):
        out[index_mask == idx] = entry["color"]
    return out


@torch.no_grad()
def predict_image(model, color_map, image_path, device, image_size):
    image = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = image.shape[:2]

    resized = np.array(Image.fromarray(image).resize((image_size, image_size), Image.BILINEAR))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    logits = model(tensor)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    pred_rgb = index_to_rgb(pred, color_map)
    # nearest-neighbor back to original resolution -- keep hard class boundaries, no blending
    pred_rgb_full = np.array(
        Image.fromarray(pred_rgb).resize((orig_w, orig_h), Image.NEAREST)
    )
    return pred_rgb_full


def make_overlay(image, mask_rgb, alpha=0.5):
    return (image.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="A single image file or a directory of images")
    parser.add_argument("--out_dir", default="predictions")
    parser.add_argument("--image_size", type=int, default=512, help="Must match the size the model was trained at")
    parser.add_argument("--overlay", action="store_true", help="Also save a blended overlay for visual QA")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, color_map = load_model(args.checkpoint, device)

    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    else:
        image_paths = [input_path]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in image_paths:
        pred_rgb = predict_image(model, color_map, p, device, args.image_size)
        Image.fromarray(pred_rgb).save(out_dir / f"{p.stem}_pred.png")
        if args.overlay:
            pred_rgb = colorize_prediction(pred_rgb)
            orig = np.array(Image.open(p).convert("RGB"))
            overlay = make_overlay(orig, pred_rgb)
            Image.fromarray(overlay).save(out_dir / f"{p.stem}_overlay.png")
        print(f"Saved prediction for {p.name}")


if __name__ == "__main__":
    main()
