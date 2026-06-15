"""
Combined BCE + Supervised Contrastive loss for avatar detection.

L_total = L_BCE + lambda_supcon * L_SupCon
"""

import torch
import torch.nn as nn
from pytorch_metric_learning.losses import SupConLoss


class AvatarDetectionLoss(nn.Module):
    """Combined BCE + Supervised Contrastive loss.

    Args:
        lambda_supcon: Weight for SupCon loss (default 0.3 per AGENTS.md).
        supcon_temperature: Temperature for SupCon loss (default 0.07 per AGENTS.md).
        label_smoothing: Label smoothing for BCE (default 0.05 per AGENTS.md).
    """

    def __init__(
        self,
        lambda_supcon: float = 0.3,
        supcon_temperature: float = 0.07,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.supcon = SupConLoss(temperature=supcon_temperature)
        self.lambda_supcon = lambda_supcon

    def forward(
        self, logits: torch.Tensor, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: ``(B, C)`` classifier output.
            features: ``(B, D)`` embedding vectors for SupCon.
            labels: ``(B,)`` ground-truth class indices.

        Returns:
            Scalar loss tensor.
        """
        loss_bce = self.cross_entropy(logits, labels)
        loss_supcon = self.supcon(features, labels)
        return loss_bce + self.lambda_supcon * loss_supcon
