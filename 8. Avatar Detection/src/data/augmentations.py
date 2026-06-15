"""Forensic-aware augmentation pipeline for avatar detection.

Provides train and val transforms matching AGENTS.md Section 7 Trial 2 specs,
plus FrequencyAugmentation for Trial 2 config 2f (FreqBlender-style).
"""

import numpy as np
from scipy.fft import dctn, idctn

import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrequencyAugmentation(ImageOnlyTransform):
    """Zero out a random frequency band in the DCT domain.

    Inspired by FreqBlender (arXiv 2304.07193). Forces the model
    to not rely on any single frequency band for detection.

    Args:
        p: Probability of applying the transform. Default 0.3 per AGENTS.md.
        bands: Tuple of band names to sample from. Default ('low', 'mid', 'high').
    """

    def __init__(self, p: float = 0.3, bands: tuple = ("low", "mid", "high")):
        super().__init__(p=p)
        self.bands = bands

    def _get_band_mask(self, h: int, w: int, band: str) -> np.ndarray:
        """Create a boolean mask that selects a frequency band in 2D DCT space.

        Low band: upper-left triangle (DC + low frequencies)
        Mid band: middle frequencies
        High band: lower-right triangle (high frequencies)
        """
        mask = np.zeros((h, w), dtype=bool)

        rows, cols = np.indices((h, w))
        freq_radius = np.sqrt(rows**2 + cols**2).astype(np.float64)

        max_radius = np.sqrt(float(h**2 + w**2))

        if band == "low":
            mask = freq_radius < max_radius * 0.15
        elif band == "mid":
            mask = (freq_radius >= max_radius * 0.15) & (freq_radius < max_radius * 0.4)
        elif band == "high":
            mask = freq_radius >= max_radius * 0.4
        else:
            raise ValueError(f"Unknown band '{band}'. Must be 'low', 'mid', or 'high'.")

        return mask

    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """Apply frequency band zeroing to the image.

        Args:
            img: (H, W, C) uint8 image array.

        Returns:
            (H, W, C) uint8 image with one frequency band zeroed.
        """
        if img.ndim != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {img.shape}")

        h, w, c = img.shape
        band = np.random.default_rng().choice(self.bands)
        mask = self._get_band_mask(h, w, band)

        result = img.astype(np.float64)
        for ch in range(c):
            dct_coeffs = dctn(result[:, :, ch], type=2, norm="ortho")
            dct_coeffs[mask] = 0.0
            result[:, :, ch] = idctn(dct_coeffs, type=2, norm="ortho")

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def get_transform_init_args_names(self):
        return ("bands",)


def get_train_transforms(img_size: int = 256) -> A.Compose:
    """Training augmentation pipeline per AGENTS.md Section 7 Trial 2.

    Includes JPEG compression (quality 40-90) which is critical for
    deployment realism — real interview frames go through H.264 then JPEG.

    Args:
        img_size: Target image size (height and width). Default 256.

    Returns:
        albumentations Compose pipeline producing (C, H, W) float tensor.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3
            ),
            A.ImageCompression(quality_range=(40, 90), compression_type="jpeg", p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """Validation/test augmentation pipeline — resize and normalize only.

    Args:
        img_size: Target image size (height and width). Default 256.

    Returns:
        albumentations Compose pipeline producing (C, H, W) float tensor.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
