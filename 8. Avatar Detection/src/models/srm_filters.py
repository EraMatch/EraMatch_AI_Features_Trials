"""
SRM (Steganalysis Rich Model) filter layers and residual encoder.

References:
    Fridrich & Kodovsky (2012). Rich Models for Steganalysis of Digital Images. IEEE TIFS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SRM_KERNELS


class SRMConv2d(nn.Module):
    """Fixed (non-trainable) SRM high-pass filter layer.

    Applies 3 SRM 5×5 kernels to a grayscale image, producing 3 residual
    maps clamped to [-3, 3].

    Args:
        None — kernel weights are loaded from ``src.config.SRM_KERNELS``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        # Load fixed SRM kernels and freeze them
        self.conv.weight.data.copy_(SRM_KERNELS)
        self.conv.weight.requires_grad_(False)

    @staticmethod
    def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
        """Convert (B, 3, H, W) RGB to (B, 1, H, W) grayscale."""
        return (
            0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SRM filters.

        Args:
            x: Input tensor — either ``(B, 1, H, W)`` grayscale
               or ``(B, 3, H, W)`` RGB.

        Returns:
            ``(B, 3, H, W)`` residual maps clamped to [-3, 3].
        """
        if x.size(1) == 3:
            x = self._rgb_to_gray(x)
        residuals = self.conv(x)
        # Normalize by per-channel std then clamp
        std = residuals.std(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        residuals = residuals / std
        residuals = residuals.clamp(-3.0, 3.0)
        return residuals


class SRMResidualEncoder(nn.Module):
    """Small CNN encoder that maps SRM residual maps to a 256-d feature vector.

    Architecture:
        Conv(3→32, 3×3, BN, ReLU) →
        Conv(32→64, 3×3, stride=2, BN, ReLU) →
        Conv(64→128, 3×3, stride=2, BN, ReLU) →
        AdaptiveAvgPool2d(1) → Linear(128, 256)

    Args:
        in_channels: Number of input channels (default 3, matching SRMConv2d output).
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode SRM residual maps into a feature vector.

        Args:
            x: ``(B, 3, H, W)`` residual maps from SRMConv2d.

        Returns:
            ``(B, 256)`` feature vector.
        """
        feat = self.encoder(x)
        feat = feat.flatten(1)
        return self.fc(feat)
