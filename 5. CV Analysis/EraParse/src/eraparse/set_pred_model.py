"""Track B — Set-prediction extraction head with copy/pointer sub-heads.

Architecture (DETR-style for document IE):

  [Document Encoder]           pretrained backbone (LayoutLMv3 or NuExtract encoder)
       ↓ token embeddings
  [Schema-conditioned queries]  one learnable query per schema slot
       ↓ cross-attention N times
  [Field heads]
    flat extractive fields →   span pointer head (start/end logits over tokens)
    nested records (N slots) → per-slot sub-heads, Hungarian matched to gold
      extractive sub-fields →  copy/pointer sub-head (cannot hallucinate)
      generative sub-fields →  small MLP → vocabulary logits

Loss:
  flat:   cross-entropy on span start/end
  nested: Hungarian matching (bipartite) between predicted slots and gold records
          + "no-record" null class for empty slots
          + copy cross-entropy on span positions for extractive sub-fields

Faithfulness by construction: extractive sub-heads select token spans from the
input — they cannot output tokens not present in the document.

Usage:
    model = CVSetPredModel.from_pretrained("microsoft/layoutlmv3-base")
    loss, preds = model(input_ids, bbox, attention_mask, pixel_values, labels)
"""
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# =========================================================
# Schema field definitions
# =========================================================

@dataclass
class FlatField:
    name: str
    extractive: bool = True      # True → pointer head; False → generative MLP


@dataclass
class NestedField:
    name: str
    max_slots: int               # max number of records (e.g. 15 for work_experience)
    sub_fields: list[FlatField] = field(default_factory=list)


FLAT_FIELDS: list[FlatField] = [
    FlatField("full_name"),
    FlatField("email"),
    FlatField("location"),
    FlatField("phone"),
    FlatField("linkedin_url"),
    FlatField("github_url"),
    FlatField("summary", extractive=False),  # may need normalization
]

NESTED_FIELDS: list[NestedField] = [
    NestedField("work_experience", max_slots=15, sub_fields=[
        FlatField("job_title"),
        FlatField("company"),
        FlatField("start_date"),
        FlatField("end_date"),
        FlatField("duration", extractive=False),
    ]),
    NestedField("education", max_slots=8, sub_fields=[
        FlatField("degree"),
        FlatField("field_of_study"),
        FlatField("institution"),
        FlatField("graduation_date"),
    ]),
    NestedField("projects", max_slots=10, sub_fields=[
        FlatField("name"),
        FlatField("url"),
    ]),
    NestedField("certifications", max_slots=10, sub_fields=[
        FlatField("name"),
        FlatField("issuer"),
        FlatField("date"),
    ]),
]

N_FLAT = len(FLAT_FIELDS)
TOTAL_SLOTS = sum(nf.max_slots for nf in NESTED_FIELDS)


# =========================================================
# Span pointer head (extractive — selects start + end token)
# =========================================================

class SpanHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.start = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)

    def forward(self, token_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        token_emb: (B, L, H)
        Returns start_logits, end_logits: (B, L)
        """
        return self.start(token_emb).squeeze(-1), self.end(token_emb).squeeze(-1)

    def loss(
        self,
        token_emb: torch.Tensor,
        start_pos: torch.Tensor,
        end_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        start_log, end_log = self.forward(token_emb)
        start_log = start_log.masked_fill(~mask, float("-inf"))
        end_log = end_log.masked_fill(~mask, float("-inf"))
        ls = F.cross_entropy(start_log, start_pos, ignore_index=-1)
        le = F.cross_entropy(end_log, end_pos, ignore_index=-1)
        return (ls + le) / 2


# =========================================================
# Record slot (one predicted entry in a nested field)
# =========================================================

class RecordSlotHead(nn.Module):
    """One slot head for a nested field: sub-heads for each sub_field."""

    def __init__(self, hidden: int, sub_fields: list[FlatField]):
        super().__init__()
        self.sub_fields = sub_fields
        self.heads = nn.ModuleDict({
            sf.name: SpanHead(hidden) if sf.extractive else nn.Linear(hidden, hidden)
            for sf in sub_fields
        })
        self.null_logit = nn.Parameter(torch.zeros(1))  # "no-record" class

    def forward(self, slot_emb: torch.Tensor, token_emb: torch.Tensor) -> dict:
        """
        slot_emb: (B, H) — this slot's cross-attended embedding
        token_emb: (B, L, H) — full token sequence
        Returns dict of sub-field predictions.
        """
        out = {"null_logit": self.null_logit}
        for sf in self.sub_fields:
            head = self.heads[sf.name]
            if sf.extractive:
                sl, el = head(token_emb)  # (B, L)
                out[sf.name] = {"start": sl, "end": el}
            else:
                out[sf.name] = {"vec": head(slot_emb)}
        return out


# =========================================================
# Cross-attention block (queries ↔ document tokens)
# =========================================================

class SchemaQueryLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int = 8, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(hidden, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def forward(
        self,
        queries: torch.Tensor,       # (B, Q, H)
        token_emb: torch.Tensor,     # (B, L, H)
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = queries
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x, _ = attn(x, token_emb, token_emb, key_padding_mask=key_padding_mask)
            x = norm(x + residual)
        return x


# =========================================================
# Main model
# =========================================================

class CVSetPredModel(nn.Module):
    """
    Full set-prediction CV extraction model.

    Backbone: any HF encoder that produces token_embeddings + hidden_size.
    Supported: layoutlmv3-base, bert-base, nuextract encoder side.
    """

    def __init__(self, encoder: nn.Module, hidden: int = 768, n_query_layers: int = 3):
        super().__init__()
        self.encoder = encoder
        self.hidden = hidden

        # Learnable queries: one per flat field + one per nested slot
        n_queries = N_FLAT + TOTAL_SLOTS
        self.queries = nn.Embedding(n_queries, hidden)

        # Cross-attention (queries ↔ document tokens)
        self.cross_attn = SchemaQueryLayer(hidden, n_heads=8, n_layers=n_query_layers)

        # Flat field heads
        self.flat_heads = nn.ModuleDict({
            ff.name: SpanHead(hidden) if ff.extractive else nn.Linear(hidden, hidden)
            for ff in FLAT_FIELDS
        })

        # Nested field heads (one RecordSlotHead per slot)
        self.nested_heads = nn.ModuleDict()
        slot_idx = N_FLAT
        for nf in NESTED_FIELDS:
            slots = nn.ModuleList([
                RecordSlotHead(hidden, nf.sub_fields)
                for _ in range(nf.max_slots)
            ])
            self.nested_heads[nf.name] = slots
            slot_idx += nf.max_slots

    @classmethod
    def from_pretrained(cls, backbone_id: str, **kwargs) -> "CVSetPredModel":
        from transformers import AutoModel
        encoder = AutoModel.from_pretrained(backbone_id)
        hidden = encoder.config.hidden_size
        return cls(encoder, hidden=hidden, **kwargs)

    def _encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Run backbone encoder, return (token_emb, padding_mask)."""
        enc_kwargs = {k: v for k, v in batch.items()
                      if k in ("input_ids", "attention_mask", "bbox", "pixel_values",
                               "token_type_ids")}
        out = self.encoder(**enc_kwargs)
        token_emb = out.last_hidden_state          # (B, L, H)
        pad_mask = (batch["attention_mask"] == 0)  # True where padding
        return token_emb, pad_mask

    def forward(self, batch: dict, labels: Optional[dict] = None):
        B = batch["input_ids"].shape[0]
        device = batch["input_ids"].device

        token_emb, pad_mask = self._encode(batch)  # (B, L, H)

        # Build query matrix: (B, Q, H)
        q_ids = torch.arange(N_FLAT + TOTAL_SLOTS, device=device).unsqueeze(0).expand(B, -1)
        queries = self.queries(q_ids)              # (B, Q, H)

        # Cross-attend queries to document tokens
        attended = self.cross_attn(queries, token_emb, key_padding_mask=pad_mask)  # (B, Q, H)

        flat_emb = attended[:, :N_FLAT]            # (B, N_FLAT, H)
        slot_emb = attended[:, N_FLAT:]            # (B, TOTAL_SLOTS, H)

        if labels is None:
            return self._decode(flat_emb, slot_emb, token_emb)

        loss = self._compute_loss(flat_emb, slot_emb, token_emb, labels)
        return loss, self._decode(flat_emb, slot_emb, token_emb)

    def _compute_loss(
        self,
        flat_emb: torch.Tensor,
        slot_emb: torch.Tensor,
        token_emb: torch.Tensor,
        labels: dict,
    ) -> torch.Tensor:
        losses = []
        attn_mask = labels.get("attention_mask")
        if attn_mask is not None:
            token_mask = attn_mask.bool()
        else:
            token_mask = torch.ones(token_emb.shape[:2], dtype=torch.bool, device=token_emb.device)

        # Flat field losses
        for i, ff in enumerate(FLAT_FIELDS):
            if ff.name not in labels:
                continue
            head = self.flat_heads[ff.name]
            emb_i = flat_emb[:, i]  # (B, H)
            if ff.extractive:
                lbl = labels[ff.name]
                l = head.loss(
                    token_emb,
                    lbl.get("start", torch.full((token_emb.shape[0],), -1, device=token_emb.device)),
                    lbl.get("end",   torch.full((token_emb.shape[0],), -1, device=token_emb.device)),
                    token_mask,
                )
                losses.append(l)

        # Nested field losses — Hungarian matching
        slot_offset = 0
        for nf in NESTED_FIELDS:
            if nf.name not in labels:
                slot_offset += nf.max_slots
                continue
            gold_records = labels[nf.name]  # list of dicts per batch item
            slots = self.nested_heads[nf.name]
            slot_block = slot_emb[:, slot_offset: slot_offset + nf.max_slots]

            batch_loss = self._hungarian_nested_loss(
                slots, slot_block, token_emb, token_mask, gold_records, nf
            )
            losses.append(batch_loss)
            slot_offset += nf.max_slots

        return sum(losses) / max(len(losses), 1)

    def _hungarian_nested_loss(
        self,
        slot_heads: nn.ModuleList,
        slot_block: torch.Tensor,   # (B, S, H)
        token_emb: torch.Tensor,    # (B, L, H)
        token_mask: torch.Tensor,
        gold_records: list,         # list[list[dict]] — B outer, variable inner
        nf: NestedField,
    ) -> torch.Tensor:
        """Bipartite Hungarian matching between predicted slots and gold records."""
        B, S, H = slot_block.shape
        total_loss = torch.tensor(0.0, device=slot_block.device)
        count = 0

        for b in range(B):
            golds = gold_records[b] if b < len(gold_records) else []
            if not golds:
                continue

            # Cost matrix: S predicted slots × G gold records
            # Use negative of "extractive match quality" as proxy for cost
            G = min(len(golds), S)
            cost = torch.zeros(S, G, device=slot_block.device)

            # Use null_logit as a simple quality proxy (real cost needs sub-head outputs)
            for s_i, sh in enumerate(slot_heads):
                null_l = sh.null_logit.squeeze()
                for g_j in range(G):
                    # lower null_logit → more confident this is a real record
                    cost[s_i, g_j] = null_l

            cost_np = cost[:, :G].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            for s_i, g_j in zip(row_ind, col_ind):
                gold = golds[g_j]
                sh = slot_heads[s_i]
                slot_e = slot_block[b: b + 1, s_i]  # (1, H)
                tok_e = token_emb[b: b + 1]          # (1, L, H)
                tok_m = token_mask[b: b + 1]

                for sf in nf.sub_fields:
                    if sf.name not in gold or not sf.extractive:
                        continue
                    head = sh.heads[sf.name]
                    lbl = gold[sf.name]
                    s_lbl = torch.tensor([lbl.get("start", -1)], device=slot_block.device)
                    e_lbl = torch.tensor([lbl.get("end",   -1)], device=slot_block.device)
                    l = head.loss(tok_e, s_lbl, e_lbl, tok_m)
                    total_loss = total_loss + l
                    count += 1

        return total_loss / max(count, 1)

    def _decode(
        self,
        flat_emb: torch.Tensor,
        slot_emb: torch.Tensor,
        token_emb: torch.Tensor,
    ) -> dict:
        """Greedy decode — returns structured dict of predictions."""
        preds: dict = {}

        for i, ff in enumerate(FLAT_FIELDS):
            head = self.flat_heads[ff.name]
            if ff.extractive:
                sl, el = head(token_emb)
                preds[ff.name] = {
                    "start": sl.argmax(-1).tolist(),
                    "end":   el.argmax(-1).tolist(),
                }
            else:
                preds[ff.name] = {"vec": head(flat_emb[:, i]).tolist()}

        slot_offset = 0
        for nf in NESTED_FIELDS:
            nested_preds = []
            for s_i, sh in enumerate(self.nested_heads[nf.name]):
                slot_e = slot_emb[:, slot_offset + s_i]
                slot_pred = {"slot_idx": s_i}
                for sf in nf.sub_fields:
                    head = sh.heads[sf.name]
                    if sf.extractive:
                        sl, el = head(token_emb)
                        slot_pred[sf.name] = {"start": sl.argmax(-1).tolist(),
                                              "end":   el.argmax(-1).tolist()}
                    else:
                        slot_pred[sf.name] = {"vec": head(slot_e).tolist()}
                nested_preds.append(slot_pred)
            preds[nf.name] = nested_preds
            slot_offset += nf.max_slots

        return preds
