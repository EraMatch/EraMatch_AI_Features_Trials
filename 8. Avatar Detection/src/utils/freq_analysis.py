"""Frequency analysis utilities for avatar detection.

Provides DCT, FFT, SRM residual computation, and frequency comparison tools.
Per AGENTS.md Section 7: compute 2D DCT of grayscale image → log1p(abs(DCT)) → normalize.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dctn


def _to_grayscale(image_tensor: torch.Tensor) -> torch.Tensor:
    """Convert image tensor to grayscale using ITU-R BT.601 luminance weights.

    Args:
        image_tensor: (B, C, H, W) or (C, H, W) or (H, W) tensor.
            If C=3, converts RGB→grayscale. If C=1, passes through.

    Returns:
        Grayscale tensor with shape matching input minus channel dim:
        - (B, 3, H, W) → (B, 1, H, W)
        - (B, 1, H, W) → (B, 1, H, W)
        - (H, W) → (1, 1, H, W) via _ensure_4d
    """
    tensor = _ensure_4d(image_tensor)

    if tensor.shape[1] == 1:
        return tensor
    if tensor.shape[1] == 3:
        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=tensor.device, dtype=tensor.dtype
        )
        weights = weights.view(1, 3, 1, 1)
        gray = (tensor * weights).sum(dim=1, keepdim=True)
        return gray

    raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[1]}")


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has 4 dimensions: (B, C, H, W).

    Handles: (H, W) → (1, 1, H, W), (C, H, W) → (1, C, H, W).
    """
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected 2-4 dimensions, got {tensor.ndim}")


def compute_dct(image_tensor: torch.Tensor) -> torch.Tensor:
    """Compute 2D DCT magnitude of an image, converted to grayscale.

    Uses scipy.fft.dctn with type=2, norm='ortho' (standard image processing DCT).
    Returns log1p(abs(DCT)) for visualization stability.

    Args:
        image_tensor: (B, C, H, W), (C, H, W), or (H, W) tensor.
            RGB images are converted to grayscale first.

    Returns:
        (B, 1, H, W) tensor of log-scaled DCT magnitudes.
    """
    gray = _to_grayscale(image_tensor)
    B, C, H, W = gray.shape

    magnitudes = []
    for b in range(B):
        img_np = gray[b, 0].cpu().numpy()
        dct_result = dctn(img_np, type=2, norm="ortho")
        magnitudes.append(torch.from_numpy(dct_result))

    stacked = torch.stack(magnitudes, dim=0).float()
    stacked = stacked.unsqueeze(1)
    log_magnitude = torch.log1p(stacked.abs())

    return log_magnitude.to(image_tensor.device)


def compute_fft_magnitude(image_tensor: torch.Tensor) -> torch.Tensor:
    """Compute FFT magnitude spectrum of an image, converted to grayscale.

    Zero-frequency is shifted to center. Returns log1p(abs(FFT)).

    Args:
        image_tensor: (B, C, H, W), (C, H, W), or (H, W) tensor.

    Returns:
        (B, 1, H, W) tensor of log-scaled FFT magnitudes with DC centered.
    """
    gray = _to_grayscale(image_tensor)
    fft_result = torch.fft.fft2(gray)
    fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
    magnitude = torch.log1p(fft_shifted.abs())

    return magnitude.to(image_tensor.device)


def compute_srm_residual(
    image_tensor: torch.Tensor,
    srm_kernels: torch.Tensor,
) -> torch.Tensor:
    """Apply SRM high-pass filters to a grayscale image tensor.

    Args:
        image_tensor: (B, 1, H, W) grayscale tensor.
        srm_kernels: (3, 1, 5, 5) filter kernel tensor (non-trainable).

    Returns:
        (B, 3, H, W) residual maps clamped to [-3, 3] after normalization.

    Raises:
        ValueError: If input is not single-channel.
    """
    if image_tensor.shape[1] != 1:
        raise ValueError(
            f"SRM residual requires single-channel input, got {image_tensor.shape[1]} channels"
        )

    residual = F.conv2d(image_tensor, srm_kernels, bias=None, padding=2)

    std = residual.std(dim=[2, 3], keepdim=True).clamp(min=1e-8)
    normalized = residual / std

    return normalized.clamp(-3.0, 3.0)


def compute_mean_frequency_spectrum(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute mean |DCT| spectrum for real vs fake images.

    Args:
        images: (N, C, H, W) image tensor.
        labels: (N,) long tensor where 0=real, 1=fake.

    Returns:
        Dict with keys:
            real_mean: (1, H, W) mean DCT magnitude of real images
            fake_mean: (1, H, W) mean DCT magnitude of fake images
            ratio: (1, H, W) fake_mean / (real_mean + epsilon)
    """
    dct_magnitude = compute_dct(images)
    H, W = dct_magnitude.shape[2], dct_magnitude.shape[3]

    real_mask = labels == 0
    fake_mask = labels == 1

    if real_mask.sum() > 0:
        real_mean = dct_magnitude[real_mask].mean(dim=0)
    else:
        real_mean = torch.zeros(1, H, W, device=images.device)

    if fake_mask.sum() > 0:
        fake_mean = dct_magnitude[fake_mask].mean(dim=0)
    else:
        fake_mean = torch.zeros(1, H, W, device=images.device)

    eps = 1e-8
    ratio = fake_mean / (real_mean + eps)

    return {
        "real_mean": real_mean,
        "fake_mean": fake_mean,
        "ratio": ratio,
    }


def visualize_frequency_comparison(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    save_path: Union[str, Path],
) -> None:
    """Generate 3-panel frequency heatmap comparison: real spectrum, fake spectrum, ratio.

    Args:
        real_images: (N, 3, H, W) real face images.
        fake_images: (M, 3, H, W) fake face images.
        save_path: Path to save the comparison plot PNG.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_real = real_images.shape[0]
    n_fake = fake_images.shape[0]
    real_labels = torch.zeros(n_real, dtype=torch.long)
    fake_labels = torch.ones(n_fake, dtype=torch.long)

    all_images = torch.cat([real_images, fake_images], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)

    spectra = compute_mean_frequency_spectrum(all_images, all_labels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(spectra["real_mean"][0].cpu().numpy(), cmap="inferno")
    axes[0].set_title("Real |DCT| Mean")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(spectra["fake_mean"][0].cpu().numpy(), cmap="inferno")
    axes[1].set_title("Fake |DCT| Mean")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        spectra["ratio"][0].cpu().numpy(), cmap="RdBu_r", vmin=0.5, vmax=2.0
    )
    axes[2].set_title("Fake/Real Ratio")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
