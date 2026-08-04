"""
Joint image+mask transforms. Each transform is a callable(image, mask) -> (image, mask)
so geometric changes are always applied identically to both.

Interpolation is matched to content type: BILINEAR for the continuous-valued
composite image, NEAREST for the categorical mask (never interpolate class
labels -- that invents fractional "classes" at edges).

Photometric jitter (brightness/contrast) is provided but OFF by default. The
GF-NIRNDVI composite's channels encode derived index values, not natural RGB
light -- invented brightness/contrast shifts don't correspond to any real
physical variation and can teach the model implausible input statistics. Turn
it on (train.photometric_augment: true) only if you've checked it still makes
sense for your specific composite encoding.
"""
import random

import numpy as np
import torch
from PIL import Image


class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.ascontiguousarray(image[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        return image, mask


class RandomRotate90:
    """Rotate by 0/90/180/270 degrees. No interpolation needed at all, so this
    is "free" augmentation with zero risk of inventing pixel values.
    """

    def __call__(self, image, mask):
        k = random.randint(0, 3)
        image = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
        mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
        return image, mask


class RandomResizedCrop:
    """Random-area crop, then resize to a fixed square size."""

    def __init__(self, size, scale=(0.7, 1.0), aspect_jitter=0.1):
        self.size = size
        self.scale = scale
        self.aspect_jitter = aspect_jitter

    def __call__(self, image, mask):
        h, w = image.shape[:2]
        area = h * w
        target_area = random.uniform(*self.scale) * area
        aspect = random.uniform(1 - self.aspect_jitter, 1 + self.aspect_jitter)

        new_h = int(round((target_area / aspect) ** 0.5))
        new_w = int(round((target_area * aspect) ** 0.5))
        new_h = max(1, min(new_h, h))
        new_w = max(1, min(new_w, w))

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        image_c = image[top : top + new_h, left : left + new_w]
        mask_c = mask[top : top + new_h, left : left + new_w]

        image_r = np.array(Image.fromarray(image_c).resize((self.size, self.size), Image.BILINEAR))
        mask_r = np.array(
            Image.fromarray(mask_c.astype(np.int32), mode="I").resize((self.size, self.size), Image.NEAREST)
        ).astype(np.int64)
        return image_r, mask_r


class Resize:
    """Deterministic resize, no cropping. Used for val/inference."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image_r = np.array(Image.fromarray(image).resize((self.size, self.size), Image.BILINEAR))
        mask_r = np.array(
            Image.fromarray(mask.astype(np.int32), mode="I").resize((self.size, self.size), Image.NEAREST)
        ).astype(np.int64)
        return image_r, mask_r


class RandomBrightnessContrast:
    """Mild photometric jitter, image only. See module docstring -- off by default."""

    def __init__(self, p=0.3, brightness=0.1, contrast=0.1):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, mask):
        if random.random() < self.p:
            img = image.astype(np.float32)
            b = 1 + random.uniform(-self.brightness, self.brightness)
            c = 1 + random.uniform(-self.contrast, self.contrast)
            mean = img.mean()
            img = (img - mean) * c + mean
            img = img * b
            image = np.clip(img, 0, 255).astype(np.uint8)
        return image, mask


class ToTensor:
    """(H,W,3) uint8 image + (H,W) int64 mask -> (3,H,W) float32 in [0,1] + (H,W) long tensor."""

    def __call__(self, image, mask):
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        mask_t = torch.from_numpy(mask.copy()).long()
        return image_t, mask_t
