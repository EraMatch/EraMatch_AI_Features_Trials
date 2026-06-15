"""
DCT 4-channel EfficientNet-B1 model.

Adds a DCT magnitude channel to RGB input, then passes through
EfficientNet-B1 with a modified 4-channel stem.
"""

import numpy as np
import torch
import torch.nn as nn
import timm
from scipy.fft import dctn


class DCT4ChannelModel(nn.Module):
    """EfficientNet-B1 with 4-channel input (RGB + DCT magnitude).

    The DCT channel is computed from the grayscale version of the input.
    Weight surgery: the 4th stem channel is initialized as the mean of
    the 3 RGB channels.

    Args:
        pretrained: Whether to load ImageNet pretrained weights (default False
            for local tests; set True in Modal training).
        num_classes: Number of output classes (default 2).
    """

    def __init__(self, pretrained: bool = False, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b1",
            pretrained=pretrained,
            in_chans=4,
            num_classes=num_classes,
        )
        if not pretrained:
            self._init_4th_channel()

    def _init_4th_channel(self) -> None:
        """Initialize the 4th input channel as the mean of the first 3."""
        stem = self.backbone.conv_stem
        with torch.no_grad():
            stem.weight.data[:, 3, :, :] = stem.weight.data[:, :3, :, :].mean(dim=1)

    @staticmethod
    def compute_dct(x: torch.Tensor) -> torch.Tensor:
        """Compute DCT magnitude map from an RGB image.

        Args:
            x: ``(B, 3, H, W)`` RGB image tensor in [0, 1] range.

        Returns:
            ``(B, 1, H, W)`` normalized log-DCT magnitude map.
        """
        gray = (
            0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        )
        gray_np = np.array(gray.squeeze(1).cpu().tolist(), dtype=np.float32)
        B, H, W = gray_np.shape
        dct_maps = []
        for i in range(B):
            d = dctn(gray_np[i], type=2, norm="ortho")
            d = torch.as_tensor(d, dtype=torch.float32)
            d = torch.log1p(d.abs())
            d = d / (d.max() + 1e-8)
            dct_maps.append(d)
        result = torch.stack(dct_maps, dim=0).unsqueeze(1)
        return result.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: concatenate DCT channel to RGB, pass through model.

        Args:
            x: ``(B, 3, H, W)`` RGB image tensor.

        Returns:
            ``(B, num_classes)`` logits.
        """
        dct_channel = self.compute_dct(x)
        x_4ch = torch.cat([x, dct_channel], dim=1)
        return self.backbone(x_4ch)
