"""Track B — Train the set-prediction extraction head on Modal.

Attaches the CVSetPredModel head to a pretrained backbone (LayoutLMv3-base by
default) and trains with Hungarian matching loss on the full train split.

Architecture summary (see src/eraparse/set_pred_model.py):
  backbone encoder → schema-conditioned cross-attention queries →
  SpanHead (copy/pointer) for extractive fields →
  RecordSlotHead + Hungarian loss for nested records

The backbone is frozen for the first half of training (head warm-up), then
unfrozen with a 10× lower LR for joint fine-tuning.

Prerequisite:
    modal volume put eraparse-adapters artifacts/manifests/train.jsonl /manifests/train.jsonl
    modal volume put eraparse-adapters artifacts/representations/pymupdf4llm_markdown/ /reps/
    modal volume put eraparse-adapters ../eramatch_benchmark_v4/schema_targets/reduced/ /schema_targets/reduced/

Run:
    modal run modal_apps/set_pred_train.py
"""
import json
from pathlib import Path

import modal

GPU = "T4"
TIMEOUT = 6 * 60 * 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.50",
        "scipy",
        "safetensors",
        "datasets>=3.0",
        "accelerate>=1.0",
        "sentencepiece",
        "Pillow",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("set-pred-train", image=image)
vol = modal.Volume.from_name("eraparse-adapters", create_if_missing=True)

BACKBONE = "microsoft/layoutlmv3-base"


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/adapters": vol})
def train(
    manifest_path: str = "/adapters/manifests/train.jsonl",
    reps_dir: str = "/adapters/reps",
    schema_targets_dir: str = "/adapters/schema_targets/reduced",
    out_name: str = "set-pred-layoutlmv3",
    epochs: int = 20,
    lr_head: float = 5e-4,
    lr_backbone: float = 5e-5,
    batch_size: int = 8,
    warmup_epochs: int = 5,
) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    import sys
    sys.path.insert(0, "/adapters/src")

    # Inline the model code since the src/ package isn't installed in Modal
    # (upload src/ to volume or pip install if packaging is set up)
    from eraparse.set_pred_model import CVSetPredModel, FLAT_FIELDS, NESTED_FIELDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # =========================================================
    # Span alignment helper
    # =========================================================

    def find_token_span(doc_input_ids, value_str, tokenizer):
        """Return (start, end) inclusive token indices in the document, or (-1, -1).

        Tokenizes `value_str` via the backend BPE tokenizer (filtering out any
        pad tokens that LayoutLMv3TokenizerFast injects), then slides a window
        over `doc_input_ids` looking for a contiguous match.

        RoBERTa-based tokenizers (LayoutLMv3 included) prepend a leading-space
        marker Ġ to the first subword when tokenizing a standalone string, but in
        the document sequence the very first subword of a word appearing after a
        special token or punctuation often lacks that Ġ.  We search with both the
        original id sequence AND a variant whose first id has the Ġ stripped, and
        return the first match found (original preferred, no-space as fallback).
        """
        if not value_str or not isinstance(value_str, str):
            return (-1, -1)
        val_str = value_str.strip()
        if not val_str:
            return (-1, -1)

        # Encode without special tokens via the backend tokenizer.
        # LayoutLMv3TokenizerFast pads the output to max_length, so we filter
        # out pad ids (id == tokenizer.pad_token_id, typically 1).
        pad_id = tokenizer.pad_token_id
        enc = tokenizer.backend_tokenizer.encode(val_str, add_special_tokens=False)
        tokens = [t for t, i in zip(enc.tokens, enc.ids) if i != pad_id]
        val_ids = [i for i in enc.ids if i != pad_id]
        if not val_ids:
            return (-1, -1)

        # Build no-leading-space variant for the first token (handles position-
        # dependent Ġ differences).
        if tokens and tokens[0].startswith("Ġ"):
            no_space_id = tokenizer.convert_tokens_to_ids(tokens[0][1:])
            val_ids_ns = [no_space_id] + val_ids[1:]
        else:
            val_ids_ns = val_ids

        doc_ids = list(doc_input_ids)
        n = len(val_ids)
        for i in range(len(doc_ids) - n + 1):
            window = doc_ids[i: i + n]
            if window == val_ids or window == val_ids_ns:
                return (i, i + n - 1)
        return (-1, -1)

    # =========================================================
    # Dataset
    # =========================================================

    class CVTokenDataset(Dataset):
        def __init__(
            self,
            manifest_path: str,
            reps_dir: str,
            schema_targets_dir: str,
            tokenizer,
            max_len: int = 512,
        ):
            self.rows = [
                json.loads(x)
                for x in Path(manifest_path).read_text().splitlines()
                if x.strip()
            ]
            self.reps = Path(reps_dir)
            self.schema_dir = Path(schema_targets_dir)
            self.tok = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            cv_id = row["cv_id"]

            # ---- document text ----
            md_path = self.reps / f"{cv_id}.md"
            text = md_path.read_text(errors="ignore") if md_path.exists() else ""

            # LayoutLMv3Tokenizer requires pre-tokenized words + bounding boxes.
            # We have markdown text (no layout coords), so split into words and
            # supply dummy normalized boxes — keeps architecture intact while
            # training the set-pred head on text-only input.
            words = text.split()
            boxes = [[0, 0, 1000, 1000]] * len(words)

            enc = self.tok(
                words,
                boxes=boxes,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}

            # ---- ground-truth span labels ----
            schema_path = self.schema_dir / f"{cv_id}.json"
            if not schema_path.exists():
                return enc, {}

            gold = json.loads(schema_path.read_text())
            doc_ids = enc["input_ids"].tolist()
            tok = self.tok

            labels: dict = {"attention_mask": enc["attention_mask"]}

            # -- flat fields --
            for ff in FLAT_FIELDS:
                value = gold.get(ff.name)
                if not value or not isinstance(value, str):
                    continue
                if ff.extractive:
                    s, e = find_token_span(doc_ids, value, tok)
                    if s >= 0:
                        labels[ff.name] = {
                            "start": torch.tensor(s, dtype=torch.long),
                            "end":   torch.tensor(e, dtype=torch.long),
                        }
                    # if span not found we omit the field — loss uses ignore_index=-1

            # -- nested fields --
            for nf in NESTED_FIELDS:
                gold_records_raw = gold.get(nf.name)
                if not gold_records_raw or not isinstance(gold_records_raw, list):
                    continue

                slot_labels = []  # list of sub-field dicts (one per gold record)
                for record in gold_records_raw:
                    if not isinstance(record, dict):
                        continue
                    slot_dict: dict = {}
                    for sf in nf.sub_fields:
                        value = record.get(sf.name)
                        if not value or not isinstance(value, str):
                            continue
                        if sf.extractive:
                            s, e = find_token_span(doc_ids, value, tok)
                            if s >= 0:
                                slot_dict[sf.name] = {
                                    "start": s,
                                    "end":   e,
                                }
                    slot_labels.append(slot_dict)

                if slot_labels:
                    labels[nf.name] = slot_labels

            return enc, labels

    # =========================================================
    # Label collation
    # =========================================================

    def collate_labels(label_list: list[dict]) -> dict:
        """Merge a list of per-sample label dicts into one batched label dict.

        Flat fields: stack individual start/end tensors along dim-0.  Samples
        that are missing a field contribute ignore_index=-1 so SpanHead.loss
        skips them via cross_entropy(ignore_index=-1).

        Nested fields: keep as a list[list[dict]] — the Hungarian loss in
        _hungarian_nested_loss iterates over the batch dimension explicitly.

        attention_mask: stack as (B, L) tensor.
        """
        B = len(label_list)
        out: dict = {}

        # attention_mask — stack if present in any sample
        masks = [lbl.get("attention_mask") for lbl in label_list]
        if any(m is not None for m in masks):
            # fill missing with ones (no padding suppression)
            ref = next(m for m in masks if m is not None)
            L = ref.shape[0]
            stacked = torch.stack([
                m if m is not None else torch.ones(L, dtype=torch.long)
                for m in masks
            ])
            out["attention_mask"] = stacked

        # flat fields
        for ff in FLAT_FIELDS:
            starts, ends = [], []
            has_any = False
            for lbl in label_list:
                field_lbl = lbl.get(ff.name)
                if field_lbl is not None:
                    starts.append(field_lbl["start"])
                    ends.append(field_lbl["end"])
                    has_any = True
                else:
                    starts.append(torch.tensor(-1, dtype=torch.long))
                    ends.append(torch.tensor(-1, dtype=torch.long))
            if has_any:
                out[ff.name] = {
                    "start": torch.stack(starts),
                    "end":   torch.stack(ends),
                }

        # nested fields: list[list[dict]] — one inner list per batch item
        for nf in NESTED_FIELDS:
            records_per_sample = [lbl.get(nf.name, []) for lbl in label_list]
            if any(records_per_sample):
                out[nf.name] = records_per_sample

        return out

    def collate_fn(batch):
        encs, label_list = zip(*batch)
        # stack encoder outputs (all tensors)
        enc_batch = {k: torch.stack([e[k] for e in encs]) for k in encs[0]}
        labels_batch = collate_labels(list(label_list))
        return enc_batch, labels_batch

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    ds = CVTokenDataset(manifest_path, reps_dir, schema_targets_dir, tokenizer)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=2,
        collate_fn=collate_fn,
    )
    print(f"dataset: {len(ds)} examples, {len(loader)} batches/epoch")

    # --- Model ---
    model = CVSetPredModel.from_pretrained(BACKBONE).to(device)

    # Phase 1: freeze backbone, train head only
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=1e-2)
    total_steps = len(loader) * epochs
    warmup_steps = len(loader) * warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    out_dir = Path(f"/adapters/{out_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []

    for epoch in range(epochs):
        # Phase 2: unfreeze backbone after warmup
        if epoch == warmup_epochs:
            print(f"epoch {epoch}: unfreezing backbone with lr={lr_backbone}")
            for p in model.encoder.parameters():
                p.requires_grad_(True)
            optimizer.add_param_group(
                {"params": list(model.encoder.parameters()), "lr": lr_backbone}
            )

        model.train()
        epoch_loss = 0.0
        for batch_idx, (enc, labels) in enumerate(loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            # Move batched label tensors to device (flat fields only;
            # nested labels are lists-of-dicts and moved inside the model)
            labels_dev: dict = {}
            for k, v in labels.items():
                if k == "attention_mask":
                    labels_dev[k] = v.to(device)
                elif isinstance(v, dict):
                    # flat field: {"start": Tensor, "end": Tensor}
                    labels_dev[k] = {sk: sv.to(device) for sk, sv in v.items()}
                else:
                    # nested field: list[list[dict]] — stays on CPU, iterated per item
                    labels_dev[k] = v

            if labels_dev:
                loss, _ = model(enc, labels_dev)
            else:
                # no labels available for this batch (schema file missing)
                _ = model(enc)
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  epoch {epoch} step {batch_idx}/{len(loader)} loss {loss.item():.4f}")

        mean_loss = epoch_loss / len(loader)
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch {epoch} mean_loss={mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  saved best at epoch {epoch}")

    # Save final
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    vol.commit()

    result = {
        "out_dir": str(out_dir),
        "best_loss": best_loss,
        "epochs": epochs,
        "backbone": BACKBONE,
    }
    print("DONE:", json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    out_name: str = "set-pred-layoutlmv3",
    epochs: int = 20,
    backbone: str = BACKBONE,
):
    result = train.remote(out_name=out_name, epochs=epochs)
    print("Training complete:", result)
