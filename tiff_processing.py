#!/usr/bin/env python3
"""
Combine 4-band single-band TIFFs (480/550/670/850 nm) into:
  1. An RGB composite (Red=670nm, Green=550nm, Blue=480nm)
  2. A "GF-NIRNDVI" composite: Green (550nm) + bilateral-filtered NIR (850nm) + NDVI

Usage:
  Fill in INPUT_DIR / OUTPUT_DIR / IMAGE_ID below, then run:
  Highly recommend using REFERENCE_IMAGE_ID to pick a single image to compute the alignment from
  python tiff_processing.py
"""

import os
import re
import sys
from collections import defaultdict
 
import cv2
import numpy as np
import tifffile as tiff
from PIL import Image

INPUT_DIR = "C:\\Users\\nate_\\Downloads\\sam2-main\\sam2-main\\micasense_images"
OUTPUT_DIR = "C:\\Users\\nate_\\Downloads\\sam2-main\\sam2-main\\processed_images"
IMAGE_ID = None 
REFERENCE_IMAGE_ID = "IMG_0000"
REFINE_WITH_OPTICAL_FLOW = True

BAND_WAVELENGTHS = {1: 475, 2: 560, 3: 668, 4: 842, 5: 717, 6: 634.5}
BLUE, GREEN, RED, NIR, REDEDGE, PANCHROMA = 1, 2, 3, 4, 5, 6 # REDEDGE and PANCHROMA are unused here, but included for completeness
 
REFERENCE_BAND = GREEN
ALIGNMENT_MOTION_MODEL = cv2.MOTION_HOMOGRAPHY  # falls back to AFFINE -> EUCLIDEAN -> TRANSLATION if this doesn't converge
JPEG_QUALITY = 95
 
ENCODER_DEPTH = 4
DIM_MULTIPLE = 2 ** ENCODER_DEPTH
 
# Standardized output size for every RGB/GF-NIRNDVI image, across every run
TARGET_WIDTH = 1280
TARGET_HEIGHT = 1024
assert TARGET_HEIGHT % DIM_MULTIPLE == 0 and TARGET_WIDTH % DIM_MULTIPLE == 0, (
    f"TARGET_HEIGHT/TARGET_WIDTH must both be divisible by {DIM_MULTIPLE} for a "
    f"{ENCODER_DEPTH}-level U-Net encoder"
)
 
 
def find_image_groups(input_dir):
    """Group files like IMG_0455_1.tif..IMG_0455_4.tif by their IMAGE_ID prefix."""
    pattern = re.compile(r"^(?P<id>.+)_(?P<band>[1-6])\.tif{1,2}$", re.IGNORECASE)
    groups = defaultdict(dict)
    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if m:
            image_id = m.group("id")
            band = int(m.group("band"))
            groups[image_id][band] = os.path.join(input_dir, fname)
    return groups
 
 
def load_bands(band_paths):
    bands = {}
    for band_num, path in band_paths.items():
        arr = tiff.imread(path).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected single-band image, got shape {arr.shape}")
        bands[band_num] = arr
    missing = set(BAND_WAVELENGTHS) - set(bands)
    if missing:
        raise ValueError(f"Missing bands: {sorted(missing)}")
    return bands
 
 
def gradient_magnitude(img01):
    gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = mag.max()
    if peak > 0:
        mag = mag / peak
    return mag.astype(np.float32)
 
 
def _to_3x3(warp_matrix, motion_model):
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        return warp_matrix.astype(np.float32)
    return np.vstack([warp_matrix, [0.0, 0.0, 1.0]]).astype(np.float32)
 
 
def _from_3x3(warp_3x3, motion_model):
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        return warp_3x3.astype(np.float32)
    return warp_3x3[:2, :].astype(np.float32)
 
 
def _rescale_warp(warp_matrix, motion_model, scale):
    A = _to_3x3(warp_matrix, motion_model)
    D = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    D_inv = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]], dtype=np.float32)
    A_full = D_inv @ A @ D
    return _from_3x3(A_full, motion_model)
 
 
def estimate_translation_seed(band01, reference01, small_dim=80, min_response=0.1):
    h, w = reference01.shape
    scale = small_dim / max(h, w)
    small_size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
 
    ksize = max(9, (min(h, w) // 20) | 1)  # odd kernel, big enough to blur out row periodicity
    ref_blur = cv2.GaussianBlur(reference01, (ksize, ksize), 0)
    band_blur = cv2.GaussianBlur(band01, (ksize, ksize), 0)
 
    ref_small = cv2.resize(ref_blur, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    band_small = cv2.resize(band_blur, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
 
    window = cv2.createHanningWindow(small_size, cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(ref_small, band_small, window)
    except cv2.error:
        return 0.0, 0.0
 
    if response < min_response:
        return 0.0, 0.0
 
    scale_x = w / small_size[0]
    scale_y = h / small_size[1]
    return dx * scale_x, dy * scale_y
 
 
def estimate_warp(band01, reference01, motion_model, max_dim=400, coarse_iters=200, refine_iters=50):
    """
    NOTE ON SPEED: ECC cost scales with pixel count x iteration count, and
    homography (8 free parameters) is the slowest motion model. max_dim/
    coarse_iters/refine_iters are deliberately conservative for
    speed -- raise them if you have a handful of images and want tighter
    sub-pixel alignment; lower them (or drop to MOTION_AFFINE) if
    batch-processing many images and alignment is taking too long.
    """
    h, w = reference01.shape
    seed_tx, seed_ty = estimate_translation_seed(band01, reference01)
 
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        small_size = (int(round(w * scale)), int(round(h * scale)))
        ref_small = cv2.resize(reference01, small_size, interpolation=cv2.INTER_AREA)
        band_small = cv2.resize(band01, small_size, interpolation=cv2.INTER_AREA)
    else:
        ref_small, band_small = reference01, band01
 
    warp_matrix = np.eye(3, 3, dtype=np.float32) if motion_model == cv2.MOTION_HOMOGRAPHY \
        else np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = seed_tx * scale
    warp_matrix[1, 2] = seed_ty * scale
    coarse_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, coarse_iters, 1e-6)
 
    try:
        _, warp_matrix = cv2.findTransformECC(
            gradient_magnitude(ref_small), gradient_magnitude(band_small),
            warp_matrix, motion_model, coarse_criteria, None, 5)
    except cv2.error:
        return None, motion_model
 
    if scale < 1.0:
        warp_matrix = _rescale_warp(warp_matrix, motion_model, scale)
 
    # short refinement at full resolution, starting from the coarse estimate
    refine_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, refine_iters, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(
            gradient_magnitude(reference01), gradient_magnitude(band01),
            warp_matrix, motion_model, refine_criteria, None, 5)
    except cv2.error:
        pass  # keep the coarse (downsampled-estimate) warp if refinement fails
 
    return warp_matrix, motion_model
 
 
def warp_band(band, warp_matrix, motion_model, shape_hw):
    h, w = shape_hw
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(band, warp_matrix, (w, h), flags=flags,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.warpAffine(band, warp_matrix, (w, h), flags=flags,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
 
 
def refine_with_optical_flow(band, reference, winsize=25, iterations=5):
    ref01 = normalize01(reference)
    band01 = normalize01(band)
    ref_u8 = to_uint8(gradient_magnitude(ref01))
    band_u8 = to_uint8(gradient_magnitude(band01))
 
    flow = cv2.calcOpticalFlowFarneback(
        ref_u8, band_u8, None,
        pyr_scale=0.5, levels=3, winsize=winsize,
        iterations=iterations, poly_n=5, poly_sigma=1.2, flags=0)
 
    h, w = reference.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(band, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
 
 
def fit_to_target(arr, target_h, target_w):
    h, w = arr.shape[:2]
 
    if h > target_h:
        top = (h - target_h) // 2
        arr = arr[top:top + target_h]
    elif h < target_h:
        pad = target_h - h
        top, bottom = pad // 2, pad - pad // 2
        pad_width = ((top, bottom),) + ((0, 0),) * (arr.ndim - 1)
        arr = np.pad(arr, pad_width, mode="edge")
 
    h2, w2 = arr.shape[:2]
    if w2 > target_w:
        left = (w2 - target_w) // 2
        arr = arr[:, left:left + target_w]
    elif w2 < target_w:
        pad = target_w - w2
        left, right = pad // 2, pad - pad // 2
        pad_width = ((0, 0), (left, right)) + ((0, 0),) * (arr.ndim - 2)
        arr = np.pad(arr, pad_width, mode="edge")
 
    return arr
 
 
def compute_batch_alignment(reference_bands, reference_band=REFERENCE_BAND, motion_model=ALIGNMENT_MOTION_MODEL):
    ref = reference_bands[reference_band]
    h, w = ref.shape
    ref01 = normalize01(ref)
 
    fallback_models, seen = [], set()
    for m in (motion_model, cv2.MOTION_AFFINE, cv2.MOTION_EUCLIDEAN, cv2.MOTION_TRANSLATION):
        if m not in seen:
            fallback_models.append(m)
            seen.add(m)
 
    warp_matrices = {}
    valid_mask = np.ones((h, w), dtype=np.uint8)
 
    for band_num, band in reference_bands.items():
        if band_num == reference_band:
            continue
        band01 = normalize01(band)
 
        warp_matrix = used_model = None
        for model in fallback_models:
            warp_matrix, used_model = estimate_warp(band01, ref01, model)
            if warp_matrix is not None:
                break
 
        if warp_matrix is None:
            print(f"[warn] reference image: band {band_num} alignment did not converge; "
                  f"band {band_num} will be left unaligned for the entire batch")
            warp_matrices[band_num] = None
            continue
 
        warp_matrices[band_num] = (warp_matrix, used_model)
        ones_warped = warp_band(np.ones((h, w), dtype=np.float32), warp_matrix, used_model, (h, w))
        valid_mask &= (ones_warped > 0.999).astype(np.uint8)
 
    rows, cols = np.any(valid_mask, axis=1), np.any(valid_mask, axis=0)
    if not rows.any() or not cols.any():
        print("[warn] no common valid region found after alignment; skipping crop")
        crop_box = (0, h, 0, w)
    else:
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        crop_box = (int(r0), int(r1) + 1, int(c0), int(c1) + 1)
 
    out_h, out_w = crop_box[1] - crop_box[0], crop_box[3] - crop_box[2]
    if out_h >= TARGET_HEIGHT and out_w >= TARGET_WIDTH:
        print(f"[info] valid overlap region is {out_h}x{out_w}; will be center-cropped "
              f"down to the standard {TARGET_HEIGHT}x{TARGET_WIDTH}")
    else:
        print(f"[warn] valid overlap region is {out_h}x{out_w}, smaller than the standard "
              f"{TARGET_HEIGHT}x{TARGET_WIDTH} in at least one dimension -- the shortfall will be "
              f"edge-padded rather than cropped. If this happens often, consider a REFERENCE_IMAGE_ID "
              f"with less lens-to-lens offset, since that's what shrinks the mutual overlap region.")
 
    return warp_matrices, crop_box
 
 
def apply_batch_alignment(bands, warp_matrices, crop_box, reference_band=REFERENCE_BAND, refine_with_flow=REFINE_WITH_OPTICAL_FLOW):
    ref_shape = bands[reference_band].shape
    h, w = ref_shape
 
    aligned = {reference_band: bands[reference_band]}
    for band_num, band in bands.items():
        if band_num == reference_band:
            continue
        if band.shape != ref_shape:
            band = cv2.resize(band, (w, h), interpolation=cv2.INTER_LINEAR)
 
        entry = warp_matrices.get(band_num)
        if entry is None:
            aligned[band_num] = band  # unaligned fallback; already warned about at computation time
            continue
        warp_matrix, used_model = entry
        aligned[band_num] = warp_band(band, warp_matrix, used_model, (h, w))
 
    r0, r1, c0, c1 = crop_box
    cropped = {b: arr[r0:r1, c0:c1] for b, arr in aligned.items()}
 
    if refine_with_flow:
        ref_cropped = cropped[reference_band]
        for band_num in list(cropped):
            if band_num == reference_band:
                continue
            cropped[band_num] = refine_with_optical_flow(cropped[band_num], ref_cropped)
 
    return cropped
 
 
def normalize01(band, low_pct=1.0, high_pct=99.0):
    lo, hi = np.percentile(band, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)
 
 
def to_uint8(band01):
    return np.clip(band01 * 255.0, 0, 255).astype(np.uint8)
 
 
def build_rgb(bands):
    r = to_uint8(normalize01(bands[RED]))
    g = to_uint8(normalize01(bands[GREEN]))
    b = to_uint8(normalize01(bands[BLUE]))
    return np.dstack([r, g, b])  # HxWx3, standard R,G,B channel order
 
 
def compute_ndvi(nir, red): # Measuring reflected light to determine plant stress / health
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    denom = np.where(denom == 0, 1e-6, denom)
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)

def compute_ndre(nir, rededge): # Measuring chlorophyll levels, plant stress, and nitrogen status in crops
    nir = nir.astype(np.float32)
    rededge = rededge.astype(np.float32)
    denom = nir + rededge
    denom = np.where(denom == 0, 1e-6, denom)
    ndre = (nir - rededge) / denom
    return np.clip(ndre, -1.0, 1.0)

def compute_gndvi(nir, green): # Measuring plant health, chlorophyll content, and water / nitrogen uptake
    nir = nir.astype(np.float32)
    green = green.astype(np.float32)
    denom = nir + green
    denom = np.where(denom == 0, 1e-6, denom)
    gndvi = (nir - green) / denom
    return np.clip(gndvi, -1.0, 1.0) 

def compute_osavi(nir, red): #Measuring canopy density and plant health
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red + 0.16
    denom = np.where(denom == 0, 1e-6, denom)
    osavi = (nir - red) / denom
    return np.clip(osavi, -1.0, 1.0)
 
def bilateral_filter01(band01, d=9, sigma_color=0.1, sigma_space=9):
    """Edge-preserving smoothing on a float32 [0,1] single-channel band."""
    return cv2.bilateralFilter(np.ascontiguousarray(band01), d, sigma_color, sigma_space)
 
 
def build_gf_nirndvi(bands):
    """
    NOTE: JPEG is 8-bit and lossy, so this quantizes/compresses the true NDVI
    values. If you need exact NDVI (e.g. for downstream analysis rather than
    viewing), compute compute_ndvi() directly and save that array as a
    float32 TIFF/NumPy file instead of going through this JPEG path.

    I just use JPEG here for SAM V2 Compatibility, it only works on JPEGs (sobbing)
    Feel free to adjust.
    """
    green01 = normalize01(bands[GREEN])
    nir01 = normalize01(bands[NIR])
    nir_filtered01 = bilateral_filter01(nir01)
    ndvi = compute_ndvi(bands[NIR], bands[RED])
 
    ndvi_disp = to_uint8((ndvi + 1.0) / 2.0)  # remap [-1,1] -> [0,255] for viewing
    return np.dstack([to_uint8(green01), to_uint8(nir_filtered01), ndvi_disp])
 
 
def process_image(image_id, band_paths, rgb_output_dir, gfn_output_dir, warp_matrices, crop_box):
    bands = load_bands(band_paths)
    cropped = apply_batch_alignment(bands, warp_matrices, crop_box)
 
    rgb = fit_to_target(build_rgb(cropped), TARGET_HEIGHT, TARGET_WIDTH)
    rgb_path = os.path.join(rgb_output_dir, f"{image_id}_RGB.jpg")
    Image.fromarray(rgb).save(rgb_path, quality=JPEG_QUALITY)
 
    gfn = fit_to_target(build_gf_nirndvi(cropped), TARGET_HEIGHT, TARGET_WIDTH)
    gfn_path = os.path.join(gfn_output_dir, f"{image_id}_GF-NIRNDVI.jpg")
    Image.fromarray(gfn).save(gfn_path, quality=JPEG_QUALITY)
 
    return rgb_path, gfn_path
 
 
def main():
    if not INPUT_DIR or not OUTPUT_DIR:
        sys.exit("Set INPUT_DIR and OUTPUT_DIR at the top of this script before running.")
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    RGB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "RGB")
    GFN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "GF-NIRNDVI")
 
    os.makedirs(RGB_OUTPUT_DIR, exist_ok=True)
    os.makedirs(GFN_OUTPUT_DIR, exist_ok=True)
    groups = find_image_groups(INPUT_DIR)
 
    if IMAGE_ID:
        if IMAGE_ID not in groups:
            sys.exit(f"No band files found for image ID '{IMAGE_ID}' in {INPUT_DIR}")
        groups = {IMAGE_ID: groups[IMAGE_ID]}
 
    if not groups:
        sys.exit(f"No <ID>_1..4.tif band groups found in {INPUT_DIR}")
 
    ref_id = REFERENCE_IMAGE_ID or IMAGE_ID or sorted(groups)[0]
    if ref_id not in groups:
        sys.exit(f"REFERENCE_IMAGE_ID '{ref_id}' not found among the image groups in {INPUT_DIR}")
    if not REFERENCE_IMAGE_ID:
        print(f"[info] REFERENCE_IMAGE_ID not set; defaulting to '{ref_id}'.")
 
    print(f"[info] computing alignment once from reference image '{ref_id}' ...")
    reference_bands = load_bands(groups[ref_id])
    warp_matrices, crop_box = compute_batch_alignment(reference_bands)
    print(f"[info] alignment computed; reusing it for all {len(groups)} image(s) in this batch")
 
    for image_id, band_paths in sorted(groups.items()):
        if len(band_paths) < 4:
            print(f"[skip] {image_id}: only found bands {sorted(band_paths)}, need 1-4")
            continue
        rgb_path, gfn_path = process_image(image_id, band_paths, RGB_OUTPUT_DIR, GFN_OUTPUT_DIR, warp_matrices, crop_box)
        print(f"[ok] {image_id}")
        print(f"      RGB          -> {rgb_path}")
        print(f"      GF-NIRNDVI   -> {gfn_path}")
 
 
if __name__ == "__main__":
    main()
 
