# U-Net training pipeline: GF-NIRNDVI composite -> crop/weed segmentation

Trains a U-Net that takes the 3-channel GF-NIRNDVI composite (from
`tiff_processing.py`) as input and predicts a per-pixel class map,
using the masks produced by SAM2 labeling tool as ground truth.

## Setup

```bash
pip install -r requirements.txt
```

## Data layout this expects

Two directories of same-sized images, matched by filename:

```
data/composites/frame_0001.png   <- 3-channel GF-NIRNDVI composite
data/masks/frame_0001.png        <- RGB colour-coded mask, same frame
```

## Usage

1. **Point `config.yaml` at your data**: set `data.composites_dir` and
   `data.masks_dir`. (`num_classes` is derived automatically from
   `color_map.json` there's nothing to keep in sync manually.)
2. **Train**:
   ```bash
   python train.py --config config.yaml
   ```
   Writes `checkpoints/best.pt` (highest val mIoU), `checkpoints/last.pt`,
   periodic `epoch_N.pt`, a `training_log.csv`, and a `training_curves.png`
   plot of loss/mIoU. Ctrl-C-safe: rerun with `checkpoint.resume:
   checkpoints/last.pt` in the config to continue.
3. **Predict** on new frames:
   ```bash
   python predict.py --checkpoint checkpoints/best.pt --input path/to/frame_or_dir --out_dir predictions --overlay
   ```
   `--overlay` also saves a blended visualization over the original composite
   for quick visual QA.

## Design notes

**Train/val split is grouped by sequence, not per-frame random.** Since
masks came from SAM2-labeled video, adjacent frames are near-duplicates. A
naive random split would leak near-identical frames into both train and val,
making val mIoU look better than it really is. `split_pairs` groups frames by
a filename-prefix heuristic (strips a trailing `_<number>`) and keeps whole
groups on one side of the split.

**Loss is CE + Dice**, with automatic inverse-frequency class weights
(`use_class_weights: true`). Weed pixels are often a small
minority of any frame; Dice pushes on region overlap directly rather than
per-pixel likelihood alone, and class weighting keeps rare-class gradients
from being drowned out by background.

**Checkpoints are self-describing.** Each `.pt` file embeds the model's
shape config (`in_channels`, `num_classes`, `base_channels`, `bilinear`) and
the `color_map` used at train time, so `predict.py` reconstructs the exact
right architecture and color mapping from the checkpoint alone so don't
need to keep `config.yaml` or `color_map.json` in sync with old runs.

## File overview

| File | Purpose |
|---|---|
| `model.py` | U-Net (4 down/up stages, skip connections, configurable width/upsampling mode) |
| `dataset.py` | File pairing, sequence-aware train/val split, RGB-mask-to-class-index conversion |
| `augment.py` | Joint image+mask transforms (flip/rotate/crop, optional photometric jitter) |
| `losses.py` | CE+Dice loss, inverse-frequency class weight computation |
| `metrics.py` | IoU/Dice/pixel-accuracy via an accumulated confusion matrix |
| `utils.py` | Seeding, checkpoint save/load, early stopping, CSV logging, curve plotting |
| `train.py` | Main training loop |
| `predict.py` | Inference on new images, RGB mask + overlay output |

## Troubleshooting

- **"No matching composite/mask pairs found"**: your filenames don't line up
  after suffix stripping. Check `data.composites_dir`/`data.masks_dir`, or
  add your own suffix patterns to the `build_file_pairs` call in `train.py`.
- **Out of memory**: lower `train.image_size` and/or `train.batch_size`
  first, then `model.base_channels` (64 -> 32) if still tight.
- **Windows + `num_workers > 0`**: `train.py` already guards its entry point
  with `if __name__ == "__main__":`, which multiprocessing DataLoader workers
  require on Windows. If you still see worker spawn errors, set
  `train.num_workers: 0` to rule out an environment issue.
