"""
Dataset for GF-NIRNDVI composite -> segmentation mask training pairs.
Two ways to build the pair list:

1. Flat layout (single pair of folders):
     composites_dir: 3-channel GF-NIRNDVI composite images (PNG/JPG)
     masks_dir:      RGB color-coded masks (as produced by the SAM2 labeling tool)
   -> use build_file_pairs(composites_dir, masks_dir)

2. Nested multi-session layout, e.g.:
     plant data/JUL29 7-8PM/004/GF-NIRNDVI/IMG_0801_GF-NIRNDVI.jpg
     plant data/JUL29 7-8PM/004/masks/IMG_0801_mask.png
     plant data/JUL30 6-7AM/012/GF-NIRNDVI/IMG_0142_GF-NIRNDVI.jpg
     ...
   -> use build_file_pairs_recursive(root_dir), which walks the whole tree,
      finds every "GF-NIRNDVI" folder, and pairs it against a sibling mask
      folder in the same parent directory.
"""
import json
import random
import re
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class FramePair(NamedTuple):
    composite: Path
    mask: Path
    group: Optional[str] = None


def load_color_map(path):
    with open(path) as f:
        data = json.load(f)
    return data["classes"]


def _strip_known_suffixes(stem, suffixes):
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def build_file_pairs(
    composites_dir,
    masks_dir,
    composite_suffixes=("_GF-NIRNDVI",),
    mask_suffixes=("_mask",),
):
    composites_dir = Path(composites_dir)
    masks_dir = Path(masks_dir)

    composite_files = [p for p in composites_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    mask_files = [p for p in masks_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

    def key_for_composite(p):
        return _strip_known_suffixes(p.stem, composite_suffixes)

    def key_for_mask(p):
        return _strip_known_suffixes(p.stem, mask_suffixes)

    mask_lookup = {}
    for p in mask_files:
        mask_lookup.setdefault(key_for_mask(p), []).append(p)

    pairs = []
    unmatched_composites = []
    for p in composite_files:
        k = key_for_composite(p)
        candidates = mask_lookup.get(k)
        if candidates:
            if len(candidates) > 1:
                print(
                    f"Warning: multiple masks match '{p.name}' (key='{k}') in {masks_dir}: "
                    f"{[c.name for c in candidates]}. Using the first."
                )
            pairs.append((p, candidates[0]))
        else:
            unmatched_composites.append(p)

    matched_keys = {key_for_composite(p) for p, _ in pairs}
    unmatched_masks = [p for p in mask_files if key_for_mask(p) not in matched_keys]

    if unmatched_composites:
        print(
            f"Warning: in {composites_dir}, {len(unmatched_composites)} composite(s) had no "
            f"matching mask, e.g. {[p.name for p in unmatched_composites[:5]]}"
        )
    if unmatched_masks:
        print(
            f"Warning: in {masks_dir}, {len(unmatched_masks)} mask(s) had no matching composite, "
            f"e.g. {[p.name for p in unmatched_masks[:5]]}"
        )

    pairs.sort(key=lambda pc: pc[0].name)
    return pairs


def find_composite_dirs(root_dir, composite_dirname="RGB"):
    root_dir = Path(root_dir)
    return sorted(p for p in root_dir.rglob(composite_dirname) if p.is_dir())


def build_file_pairs_recursive(
    root_dir,
    composite_dirname="GF-NIRNDVI",
    mask_dirname="masks",
    composite_suffixes=("_GF-NIRNDVI",),
    mask_suffixes=("_mask",),
):
    root_dir = Path(root_dir)
    all_pairs = []
    composite_dirs = find_composite_dirs(root_dir, composite_dirname)

    if not composite_dirs:
        print(f"Warning: no '{composite_dirname}' folders found anywhere under {root_dir}")

    for comp_dir in composite_dirs:
        mask_dir = comp_dir.parent / mask_dirname
        if not mask_dir.is_dir():
            print(
                f"Warning: skipping {comp_dir} -- no sibling mask folder found at {mask_dir} "
                f"(pass a different mask_dirname if your masks live elsewhere)"
            )
            continue

        pairs = build_file_pairs(comp_dir, mask_dir, composite_suffixes, mask_suffixes)
        group = str(comp_dir.parent.relative_to(root_dir))
        all_pairs.extend(FramePair(c, m, group) for c, m in pairs)

    all_pairs.sort(key=lambda fp: (fp.group, fp.composite.name))
    return all_pairs


def split_pairs(pairs, val_fraction=0.15, seed=42, group_by_sequence=True, sequence_regex=None):
    rng = random.Random(seed)
    pairs = list(pairs)

    if not group_by_sequence:
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(val_fraction * len(shuffled))))
        return shuffled[n_val:], shuffled[:n_val]

    has_explicit_group = bool(pairs) and isinstance(pairs[0], FramePair) and pairs[0].group is not None

    if has_explicit_group:
        def seq_id(pc):
            return pc.group
    else:
        pattern = re.compile(sequence_regex) if sequence_regex else re.compile(r"^(.*?)_?\d+$")

        def seq_id(pc):
            m = pattern.match(pc[0].stem)
            if m and m.group(1):
                return m.group(1)
            return pc[0].stem

    groups = {}
    for pc in pairs:
        groups.setdefault(seq_id(pc), []).append(pc)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n_val_target = int(round(val_fraction * len(pairs)))
    val_pairs, train_pairs = [], []
    val_count = 0
    for k in group_keys:
        if val_count < n_val_target:
            val_pairs.extend(groups[k])
            val_count += len(groups[k])
        else:
            train_pairs.extend(groups[k])

    if not val_pairs and len(pairs) > 1:
        print("Warning: sequence grouping produced an empty val set; falling back to a random split.")
        return split_pairs(pairs, val_fraction, seed, group_by_sequence=False)

    return train_pairs, val_pairs


def rgb_mask_to_index(mask_rgb, color_map):
    """(H, W, 3) RGB array -> (H, W) int64 class-index array, via exact color match."""
    h, w, _ = mask_rgb.shape
    index_mask = np.zeros((h, w), dtype=np.int64)
    for class_idx, entry in enumerate(color_map):
        color = np.array(entry["color"], dtype=np.uint8)
        index_mask[np.all(mask_rgb == color, axis=-1)] = class_idx
    return index_mask


class SegmentationDataset(Dataset):
    def __init__(self, pairs, color_map, transform=None):
        self.pairs = pairs
        self.color_map = color_map
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        composite_path, mask_path = self.pairs[idx][0], self.pairs[idx][1]
        image = np.array(Image.open(composite_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        if image.shape[:2] != mask_rgb.shape[:2]:
            raise ValueError(
                f"Size mismatch: composite {composite_path.name} {image.shape[:2]} "
                f"vs mask {mask_path.name} {mask_rgb.shape[:2]}"
            )

        mask_idx = rgb_mask_to_index(mask_rgb, self.color_map)

        if self.transform:
            image, mask_idx = self.transform(image, mask_idx)

        return image, mask_idx


if __name__ == "__main__":
    # Example for a nested multi-session tree like:
    #   plant data/JUL29 7-8PM/004/GF-NIRNDVI/IMG_0801_GF-NIRNDVI.jpg
    #   plant data/JUL29 7-8PM/004/masks/IMG_0801_mask.png
    root = r"C:\Users\nate_\Desktop\plant data"

    pairs = build_file_pairs_recursive(root)
    print(f"Found {len(pairs)} composite/mask pairs across all sessions/captures")

    color_map = load_color_map("color_map.json")
    train_pairs, val_pairs = split_pairs(pairs)
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    train_ds = SegmentationDataset(train_pairs, color_map)
    val_ds = SegmentationDataset(val_pairs, color_map)