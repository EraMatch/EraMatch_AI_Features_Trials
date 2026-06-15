# mypy: ignore-errors
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LayoutLMv3Model

from eraparse.sge_losses import binary_positive_weight, token_class_weights


class SchemaGuidedLayoutLMv3(nn.Module):
    """LayoutLMv3 with schema queries, grounded field logits, and record grouping."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        num_fields: int,
        query_layers: int = 2,
        field_token_weight: float = 1.0,
        presence_weight: float = 0.25,
        grouping_weight: float = 0.5,
        evidence_weight: float = 0.5,
    ) -> None:
        super().__init__()
        for name, value in {
            "field_token_weight": field_token_weight,
            "presence_weight": presence_weight,
            "grouping_weight": grouping_weight,
            "evidence_weight": evidence_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if field_token_weight == 0:
            raise ValueError("field_token_weight must be positive")
        if query_layers < 1:
            raise ValueError("query_layers must be at least one")
        self.encoder = LayoutLMv3Model.from_pretrained(model_id, revision=revision)
        self.loss_weights = {
            "field_token": field_token_weight,
            "presence": presence_weight,
            "grouping": grouping_weight,
            "evidence": evidence_weight,
        }
        hidden = self.encoder.config.hidden_size
        heads = self.encoder.config.num_attention_heads
        self.schema_queries = nn.Parameter(torch.empty(num_fields, hidden))
        nn.init.normal_(self.schema_queries, std=0.02)
        self.query_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden, heads, batch_first=True) for _ in range(query_layers)]
        )
        self.query_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(query_layers)])
        self.token_projection = nn.Linear(hidden, hidden)
        self.query_projection = nn.Linear(hidden, hidden)
        self.outside_head = nn.Linear(hidden, 1)
        self.presence_head = nn.Linear(hidden, 1)
        self.group_left = nn.Linear(hidden, hidden)
        self.group_right = nn.Linear(hidden, hidden)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_final_encoder_layers(self, count: int = 4) -> None:
        if count < 1 or count > len(self.encoder.encoder.layer):
            raise ValueError(
                f"count must be between 1 and {len(self.encoder.encoder.layer)}"
            )
        for layer in self.encoder.encoder.layer[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        *,
        input_ids,
        bbox,
        attention_mask,
        pixel_values,
        labels=None,
        presence_labels=None,
        record_ids=None,
    ):
        output = self.encoder(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        hidden = output.last_hidden_state
        batch = hidden.shape[0]
        queries = self.schema_queries.unsqueeze(0).expand(batch, -1, -1)
        padding_mask = ~attention_mask.bool()
        if padding_mask.shape[1] < hidden.shape[1]:
            visual_padding = torch.zeros(
                batch,
                hidden.shape[1] - padding_mask.shape[1],
                dtype=torch.bool,
                device=hidden.device,
            )
            padding_mask = torch.cat([padding_mask, visual_padding], dim=1)
        for attention, norm in zip(self.query_layers, self.query_norms, strict=True):
            attended, _ = attention(queries, hidden, hidden, key_padding_mask=padding_mask)
            queries = norm(queries + attended)
        token_features = self.token_projection(hidden)
        query_features = self.query_projection(queries)
        field_logits = torch.einsum("bnh,bqh->bnq", token_features, query_features)
        field_logits = field_logits / math.sqrt(token_features.shape[-1])
        logits = torch.cat([self.outside_head(hidden), field_logits], dim=-1)
        presence_logits = self.presence_head(queries).squeeze(-1)
        grouping_logits = torch.einsum(
            "bnh,bmh->bnm",
            self.group_left(hidden),
            self.group_right(hidden),
        ) / math.sqrt(hidden.shape[-1])
        result = {
            "logits": logits,
            "presence_logits": presence_logits,
            "grouping_logits": grouping_logits,
            "evidence_logits": field_logits.amax(dim=-1),
            "loss": None,
        }
        losses = []
        if labels is not None:
            text_length = labels.shape[1]
            losses.append(
                self.loss_weights["field_token"]
                * F.cross_entropy(
                    logits[:, :text_length].reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    weight=logits.new_tensor(token_class_weights(logits.shape[-1] - 1)),
                )
            )
            evidence_targets = (labels > 0).float()
            evidence_mask = labels != -100
            losses.append(
                self.loss_weights["evidence"]
                * F.binary_cross_entropy_with_logits(
                    result["evidence_logits"][:, :text_length][evidence_mask],
                    evidence_targets[evidence_mask],
                )
            )
        if presence_labels is not None and self.loss_weights["presence"] > 0:
            losses.append(
                self.loss_weights["presence"]
                * F.binary_cross_entropy_with_logits(presence_logits, presence_labels.float())
            )
        if record_ids is not None and self.loss_weights["grouping"] > 0:
            text_length = record_ids.shape[1]
            valid = record_ids >= 0
            pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
            if pair_mask.any():
                same_record = (record_ids.unsqueeze(2) == record_ids.unsqueeze(1)).float()
                pair_targets = same_record[pair_mask]
                positive_count = int(pair_targets.sum().item())
                negative_count = int(pair_targets.numel() - positive_count)
                positive_weight = binary_positive_weight(
                    positive_count=positive_count,
                    negative_count=negative_count,
                )
                losses.append(
                    self.loss_weights["grouping"]
                    * F.binary_cross_entropy_with_logits(
                        grouping_logits[:, :text_length, :text_length][pair_mask],
                        pair_targets,
                        pos_weight=grouping_logits.new_tensor(positive_weight),
                    )
                )
        if losses:
            result["loss"] = sum(losses)
        return result
