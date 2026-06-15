from typing import Optional

import torch
import torch.nn as nn
import timm

from src.models.srm_filters import SRMConv2d, SRMResidualEncoder


class _SRMBranch(nn.Module):
    """Combined SRM filter + residual encoder branch.

    Outputs (B, 256) feature vector from fixed SRM high-pass residuals.
    The Conv2d kernels inside SRMConv2d are frozen (requires_grad=False).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = SRMConv2d().conv
        self.encoder = SRMResidualEncoder(in_channels=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3:
            gray = SRMConv2d._rgb_to_gray(x)
        else:
            gray = x
        residuals = self.conv(gray)
        std = residuals.std(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        residuals = residuals / std
        residuals = residuals.clamp(-3.0, 3.0)
        return self.encoder(residuals)


class DualBranchSRM(nn.Module):
    """Dual-branch SRM + RGB model with cross-modal attention and SupCon head.

    Architecture matches AGENTS.md Section 7 Trial 2:
      - SRM branch: fixed 5x5 kernels → 3→32→64→128→256 CNN encoder
      - RGB branch: ConvNeXt-tiny feature extractor (768-d, spatial 8×8 at 256px)
      - Cross-modal attention: SRM_feat(256) → FC(256,64) → sigmoid → 8×8 mask
      - Fusion: concat(rgb_attended=768, srm_feat=256) = 1024 → classifier
      - SupCon projection: 1024 → 256 → 128 (only if use_supcon=True)

    All components are optional via constructor flags for ablation study.

    Args:
        use_srm: Enable SRM residual branch (default True).
        use_attention: Enable cross-modal attention gate (default True).
        use_supcon: Enable SupCon projection head (default True).
        backbone: timm model name for RGB branch (default 'convnext_tiny.fb_in1k').
        num_classes: Output classes (default 2).
        feature_dim: SRM encoder output dim (default 256).
        dropout: Classifier dropout rate (default 0.4).
        pretrained: Load ImageNet pretrained weights for backbone (default False).
    """

    def __init__(
        self,
        use_srm: bool = True,
        use_attention: bool = True,
        use_supcon: bool = True,
        backbone: str = "convnext_tiny.fb_in1k",
        num_classes: int = 2,
        feature_dim: int = 256,
        dropout: float = 0.4,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.use_srm = use_srm
        self.use_attention = use_attention
        self.use_supcon = use_supcon
        self.feature_dim = feature_dim

        self.rgb_backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        rgb_feat_dim = self.rgb_backbone.num_features

        self.srm_branch = _SRMBranch() if use_srm else None

        if use_srm and use_attention:
            self.attention_fc = nn.Linear(feature_dim, 64)
        elif use_attention and not use_srm:
            self.learned_attention = nn.Parameter(torch.zeros(1, 1, 8, 8))
            nn.init.constant_(self.learned_attention, 0.0)

        fusion_dim = rgb_feat_dim + feature_dim if use_srm else rgb_feat_dim
        classifier_in = fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        if use_supcon:
            proj_in = fusion_dim
            self.projection_head = nn.Sequential(
                nn.Linear(proj_in, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
        else:
            self.projection_head = None

    def _compute_attention(
        self,
        srm_feat: Optional[torch.Tensor],
        spatial_h: int,
        spatial_w: int,
        device: torch.device,
        batch_size: int,
    ):
        if self.use_srm and self.use_attention:
            raw = self.attention_fc(srm_feat)
            attn = torch.sigmoid(raw).view(batch_size, 1, 8, 8)
            if spatial_h != 8 or spatial_w != 8:
                attn = torch.nn.functional.interpolate(
                    attn,
                    size=(spatial_h, spatial_w),
                    mode="bilinear",
                    align_corners=False,
                )
            return attn, True
        if self.use_attention and not self.use_srm:
            attn = torch.sigmoid(self.learned_attention.expand(batch_size, -1, -1, -1))
            if spatial_h != 8 or spatial_w != 8:
                attn = torch.nn.functional.interpolate(
                    attn,
                    size=(spatial_h, spatial_w),
                    mode="bilinear",
                    align_corners=False,
                )
            return attn, True
        return torch.zeros(batch_size, 1, spatial_h, spatial_w, device=device), False

    def _extract_rgb_features(self, x: torch.Tensor):
        spatial = self.rgb_backbone.forward_features(x)
        if spatial.dim() != 4:
            spatial = spatial.unsqueeze(-1).unsqueeze(-1)
        return spatial

    def _apply_attention_pool(
        self,
        spatial: torch.Tensor,
        attention: torch.Tensor,
        attention_computed: bool,
    ) -> torch.Tensor:
        if not attention_computed:
            rgb_attended = spatial
        else:
            if (
                spatial.shape[2] != attention.shape[2]
                or spatial.shape[3] != attention.shape[3]
            ):
                attention = torch.nn.functional.interpolate(
                    attention,
                    size=(spatial.shape[2], spatial.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            rgb_attended = spatial * attention
        pooled = self.rgb_backbone.head.global_pool(rgb_attended)
        pooled = self.rgb_backbone.head.norm(pooled)
        pooled = self.rgb_backbone.head.flatten(pooled)
        pooled = self.rgb_backbone.head.drop(pooled)
        return pooled

    def forward(self, x: torch.Tensor) -> dict:
        B = x.size(0)
        device = x.device

        srm_feat = (
            self.srm_branch(x)
            if self.use_srm
            else torch.zeros(B, self.feature_dim, device=device)
        )

        spatial = self._extract_rgb_features(x)

        s_h, s_w = spatial.shape[2], spatial.shape[3]

        attention, attention_computed = self._compute_attention(
            srm_feat, s_h, s_w, device, B
        )

        rgb_attended = self._apply_attention_pool(
            spatial, attention, attention_computed
        )

        if self.use_srm:
            fused = torch.cat([rgb_attended, srm_feat], dim=1)
        else:
            fused = rgb_attended

        logits = self.classifier(fused)

        if self.use_supcon and self.projection_head is not None:
            embedding = self.projection_head(fused)
        else:
            embedding = torch.zeros(B, 128, device=device)

        attn_out = (
            attention
            if attention_computed
            else torch.zeros(B, 1, s_h, s_w, device=device)
        )

        return {
            "logits": logits,
            "embedding": embedding,
            "srm_feat": srm_feat,
            "attention": attn_out,
        }
