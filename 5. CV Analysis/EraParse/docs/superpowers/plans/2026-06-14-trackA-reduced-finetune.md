# Track A Foundation — Reduced-Schema Small-Model Fine-tune + Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Gemma-3-1B on the eraparse **reduced** schema and evaluate it through the *existing* eraparse evaluator, producing a base-vs-fine-tuned comparison (accuracy / validity / faithfulness / latency) directly comparable to lanes A/B/C.

**Architecture:** Three small components on Modal mirroring existing apps: (1) a local data-gen script that builds SFT pairs from `pymupdf4llm` markdown → cleaned reduced-schema JSON; (2) a Modal LoRA fine-tune app (mirrors `modal_apps/sge_train.py`); (3) a Modal inference app that loads base + LoRA adapter (mirrors `modal_apps/mapper_trial.py` + `PeftModel`). Evaluation reuses `eraparse trials ingest-mapper` unchanged.

**Tech Stack:** Python, Modal (L4 GPU), Unsloth/PEFT LoRA, Transformers, the existing eraparse CLI + evaluator.

**Plan series context:** This is **Plan 1 of the thesis** (see `docs/THESIS_DESIGN.md`). Follow-on plans: Plan 2 = 4B + rich-schema; Plan 3 = constrained decoding (RQ3); Plan 4 = set-prediction head (Track B / RQ4); Plan 5 = frozen eval + Pareto.

---

## File structure

| File | Responsibility |
|---|---|
| `scripts/build_sft_reduced.py` (create) | Build cleaned reduced-schema SFT JSONL from a manifest |
| `modal_apps/gemma_finetune.py` (create) | Modal LoRA fine-tune of Gemma-3 on SFT JSONL → adapter on a Modal Volume |
| `modal_apps/gemma_adapter_infer.py` (create) | Modal inference: base + LoRA adapter → mapper responses (ingest-mapper format) |
| `tests/test_build_sft_reduced.py` (create) | Unit tests for the data-gen cleaning logic |
| `src/eraparse/constants.py` (read) | `REDUCED_SCHEMA_TEMPLATE` for the prompt |
| `src/eraparse/models.py` (read) | `ReducedCVTarget` schema |

**Reused unchanged:** `eraparse trials ingest-mapper`, `src/eraparse/evaluate.py`, `scripts/analyze_labels_and_router.py` (clean-metric logic).

---

## Task 1: SFT data-gen — cleaned reduced-schema pairs

**Files:**
- Create: `scripts/build_sft_reduced.py`
- Test: `tests/test_build_sft_reduced.py`

The training target must be the reduced ground truth **with evidence-absent cells removed** (phantom github/linkedin URLs + the 4 non-rendering project templates) so the model learns faithfulness, not to hallucinate. This reuses the exclusion logic already in `scripts/analyze_labels_and_router.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_sft_reduced.py
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_sft_reduced import clean_target, build_example

def test_clean_target_drops_phantom_github():
    truth = {"github_url": "https://github.com/x", "work_experience": []}
    # text has no 'github' token -> evidence-absent -> drop the url
    cleaned = clean_target(truth, doc_text="some cv text", template="T1_classic")
    assert "github_url" not in cleaned or cleaned["github_url"] in (None, "")

def test_clean_target_drops_nonrendering_projects():
    truth = {"projects": [{"name": "Vector Search API"}]}
    cleaned = clean_target(truth, doc_text="no project section here", template="T3_table")
    assert not cleaned.get("projects")

def test_build_example_shape():
    ex = build_example("cv_1", "MARKDOWN TEXT", {"full_name": "A"})
    roles = [m["role"] for m in ex["conversations"]]
    assert roles == ["system", "user", "assistant"]
    assert json.loads(ex["conversations"][2]["content"])["full_name"] == "A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/anasahmed/AI_Main_Repo/CV_parsing_main/eraparse && uv run pytest tests/test_build_sft_reduced.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_sft_reduced'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/build_sft_reduced.py
"""Build cleaned reduced-schema SFT pairs (ShareGPT format) from a manifest.

Input  per CV: pymupdf4llm markdown representation.
Target per CV: reduced ground-truth JSON with evidence-absent cells removed
              (phantom URLs + non-rendering project templates) so the model
              learns faithful extraction.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DS = ROOT.parent / "eramatch_benchmark_v4"
REPS = ROOT / "artifacts" / "representations" / "pymupdf4llm_markdown"
import sys
sys.path.insert(0, str(ROOT / "src"))
from eraparse.constants import REDUCED_SCHEMA_TEMPLATE

PROJECTS_ABSENT_TEMPLATES = {"T1_functional", "T1_executive", "T3_table", "T5_minimal"}
SYSTEM = (
    "You are a precise CV information extraction system. Extract the CV into "
    "exactly this JSON schema. Return one valid JSON object. Use empty strings "
    "or empty arrays for missing values. Do not invent values absent from the CV.\n\n"
    "Schema:\n" + json.dumps(REDUCED_SCHEMA_TEMPLATE, indent=2)
)


def clean_target(truth: dict, doc_text: str, template: str) -> dict:
    out = dict(truth)
    low = (doc_text or "").lower()
    for field, needle in (("github_url", "github"), ("linkedin_url", "linkedin")):
        if out.get(field) and needle not in low:
            out[field] = None
    if template in PROJECTS_ABSENT_TEMPLATES and out.get("projects"):
        if not any((p.get("name") or "").lower() in low for p in out["projects"] if p.get("name")):
            out["projects"] = []
    return out


def build_example(cv_id: str, markdown: str, target: dict) -> dict:
    return {
        "id": cv_id,
        "conversations": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": markdown},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
    }


def reduced_truth(row: dict) -> dict:
    ref = row["artifacts"]["schema_reduced"]
    return json.loads((DS / ref["path"]).read_text())


def doc_text(row: dict) -> str:
    return (DS / row["artifacts"]["pymupdf_text"]["path"]).read_text(errors="ignore")


def build(manifest_path: Path, out_path: Path) -> int:
    rows = [json.loads(x) for x in manifest_path.read_text().splitlines()]
    written = 0
    with out_path.open("w") as fh:
        for row in rows:
            cv_id = row["cv_id"]
            md = REPS / f"{cv_id}.md"
            if not md.exists():
                continue
            target = clean_target(reduced_truth(row), doc_text(row), row.get("template", ""))
            fh.write(json.dumps(build_example(cv_id, md.read_text(errors="ignore"), target)) + "\n")
            written += 1
    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "artifacts/manifests/train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/sft/train.reduced.sft.jsonl"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = build(Path(a.manifest), Path(a.out))
    print(f"wrote {n} SFT examples -> {a.out}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_build_sft_reduced.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Generate train + validation SFT data**

```bash
cd /Users/anasahmed/AI_Main_Repo/CV_parsing_main/eraparse
uv run python scripts/build_sft_reduced.py --manifest artifacts/manifests/train.jsonl --out artifacts/sft/train.reduced.sft.jsonl
uv run python scripts/build_sft_reduced.py --manifest artifacts/manifests/validation.jsonl --out artifacts/sft/validation.reduced.sft.jsonl
```
Expected: `wrote 1445 ...` and `wrote 310 ...`

- [ ] **Step 6: Commit**

```bash
git add scripts/build_sft_reduced.py tests/test_build_sft_reduced.py
git commit -m "feat: reduced-schema SFT data-gen with faithfulness cleaning"
```

---

## Task 2: Modal LoRA fine-tune app

**Files:**
- Create: `modal_apps/gemma_finetune.py`
- Reference (mirror): `modal_apps/sge_train.py` (Modal app/volume/GPU patterns), Kaggle notebook config (r=16, alpha=16, lr=2e-4, 2 epochs)

- [ ] **Step 1: Write the fine-tune app**

```python
# modal_apps/gemma_finetune.py
"""LoRA fine-tune Gemma-3 on reduced-schema SFT data; save adapter to a Volume."""
import json
from pathlib import Path
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("unsloth", "unsloth_zoo", "trl", "peft", "accelerate",
                 "bitsandbytes", "datasets", "torch")
)
app = modal.App("gemma-finetune-reduced", image=image)
vol = modal.Volume.from_name("eraparse-adapters", create_if_missing=True)


@app.function(gpu="L4", timeout=2 * 60 * 60, volumes={"/adapters": vol})
def train(sft_jsonl: str, model_name: str, out_name: str,
          max_seq_length: int = 8192, epochs: int = 2, lr: float = 2e-4) -> dict:
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    rows = [json.loads(x) for x in Path(sft_jsonl).read_text().splitlines()]
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_length, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    def fmt(ex):
        return {"text": tok.apply_chat_template(ex["conversations"], tokenize=False)}

    ds = Dataset.from_list(rows).map(fmt)
    out_dir = f"/adapters/{out_name}"
    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        dataset_text_field="text", max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=out_dir, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, warmup_ratio=0.03,
            num_train_epochs=epochs, learning_rate=lr, bf16=True,
            logging_steps=10, save_strategy="epoch", seed=42,
        ),
    )
    stats = trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    vol.commit()
    return {"out_dir": out_dir, "train_loss": float(stats.training_loss), "examples": len(rows)}


@app.local_entrypoint()
def main(sft_jsonl: str, model_name: str = "unsloth/gemma-3-1b-it",
         out_name: str = "gemma3-1b-reduced", max_seq_length: int = 8192):
    # Upload SFT data into the container via a temporary mount is avoided:
    # read it locally and pass the path after `modal volume put` (Step 2).
    print(train.remote(sft_jsonl=f"/adapters/sft/{Path(sft_jsonl).name}",
                       model_name=model_name, out_name=out_name,
                       max_seq_length=max_seq_length))
```

- [ ] **Step 2: Upload SFT data to the adapter volume, then train**

```bash
cd /Users/anasahmed/AI_Main_Repo/CV_parsing_main/eraparse
uv run modal volume put eraparse-adapters artifacts/sft/train.reduced.sft.jsonl /sft/train.reduced.sft.jsonl
uv run modal run modal_apps/gemma_finetune.py --sft-jsonl artifacts/sft/train.reduced.sft.jsonl --model-name unsloth/gemma-3-1b-it --out-name gemma3-1b-reduced
```
Expected: prints `{'out_dir': '/adapters/gemma3-1b-reduced', 'train_loss': <~0.x>, 'examples': 1445}`

- [ ] **Step 3: Verify adapter saved**

```bash
uv run modal volume ls eraparse-adapters gemma3-1b-reduced
```
Expected: lists `adapter_model.safetensors`, `adapter_config.json`

- [ ] **Step 4: Commit**

```bash
git add modal_apps/gemma_finetune.py
git commit -m "feat: modal LoRA fine-tune app for reduced-schema gemma3"
```

---

## Task 3: Modal inference app (base + adapter → mapper responses)

**Files:**
- Create: `modal_apps/gemma_adapter_infer.py`
- Reference (mirror): `modal_apps/mapper_trial.py:160-234` (worker, prompt, response shape)

Output must match the `ingest-mapper` response format: one JSON line per CV with `cv_id`, `raw_output`, `latency_seconds`, `model_id`, `revision`, `parser_id`.

- [ ] **Step 1: Write the inference app**

```python
# modal_apps/gemma_adapter_infer.py
"""Run a fine-tuned Gemma-3 LoRA adapter as a mapper; emit ingest-mapper rows."""
import json, os, time
from pathlib import Path
import modal

image = modal.Image.debian_slim().pip_install(
    "transformers", "peft", "accelerate", "bitsandbytes", "torch"
)
app = modal.App("gemma-adapter-infer", image=image)
vol = modal.Volume.from_name("eraparse-adapters")


@app.cls(gpu="L4", volumes={"/adapters": vol}, timeout=60 * 60)
class Worker:
    base_model: str = modal.parameter()
    adapter_name: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, dtype=torch.bfloat16).to("cuda")
        self.model = PeftModel.from_pretrained(base, f"/adapters/{self.adapter_name}").eval()

    @modal.method()
    def predict(self, requests: list[dict], max_new_tokens: int = 1200) -> list[dict]:
        out = []
        for item in requests:
            msgs = [{"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["text"]}]
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
            n_in = int(enc.input_ids.shape[-1])
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, do_sample=False,
                                          max_new_tokens=max_new_tokens,
                                          pad_token_id=self.tok.pad_token_id)
            raw = self.tok.batch_decode(gen[:, n_in:], skip_special_tokens=True)[0].strip()
            out.append({"cv_id": item["cv_id"], "raw_output": raw,
                        "latency_seconds": time.perf_counter() - t0,
                        "model_id": f"eraparse/{self.adapter_name}", "revision": "ft-v1",
                        "parser_id": "pymupdf4llm_markdown"})
        return out


@app.local_entrypoint()
def main(requests_path: str, output_path: str,
         base_model: str = "unsloth/gemma-3-1b-it",
         adapter_name: str = "gemma3-1b-reduced", chunk_size: int = 25):
    reqs = [json.loads(x) for x in Path(requests_path).read_text().splitlines()]
    done = set()
    if Path(output_path).exists():
        done = {json.loads(x)["cv_id"] for x in Path(output_path).read_text().splitlines()}
    reqs = [r for r in reqs if r["cv_id"] not in done]
    w = Worker(base_model=base_model, adapter_name=adapter_name)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(reqs), chunk_size):
        chunk = reqs[i:i + chunk_size]
        print(f"chunk {i//chunk_size+1} ({len(chunk)})")
        res = w.predict.remote(chunk)
        with open(output_path, "a") as fh:
            for r in res:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush(); os.fsync(fh.fileno())
```

- [ ] **Step 2: Build adapter-inference requests for validation**

The requests reuse the validation Qwen mapper requests (already built) but add the `system` field. Quick local script:

```bash
cd /Users/anasahmed/AI_Main_Repo/CV_parsing_main/eraparse
uv run python -c "
import json, sys
sys.path.insert(0, 'src')
from eraparse.constants import REDUCED_SCHEMA_TEMPLATE
SYSTEM = 'You are a precise CV information extraction system. Extract the CV into exactly this JSON schema. Return one valid JSON object. Use empty strings or empty arrays for missing values. Do not invent values absent from the CV.\n\nSchema:\n' + json.dumps(REDUCED_SCHEMA_TEMPLATE, indent=2)
src = 'artifacts/trials/router/validation.qwen-mapper-full.requests.jsonl'
out = 'artifacts/trials/router/validation.gemma-ft.requests.jsonl'
with open(out,'w') as o:
    for l in open(src):
        r = json.loads(l); r['system'] = SYSTEM
        o.write(json.dumps(r)+chr(10))
print('wrote', out)
"
```

- [ ] **Step 3: Run inference on validation (retry-loop for Modal heartbeat)**

```bash
RESP=artifacts/trials/router/validation.gemma-ft.responses.jsonl
while [ "$(wc -l < $RESP 2>/dev/null || echo 0)" -lt 310 ]; do
  uv run modal run modal_apps/gemma_adapter_infer.py \
    --requests-path artifacts/trials/router/validation.gemma-ft.requests.jsonl \
    --output-path $RESP --base-model unsloth/gemma-3-1b-it --adapter-name gemma3-1b-reduced
  sleep 5
done
echo "done: $(wc -l < $RESP)/310"
```
Expected: ends at `done: 310/310`

- [ ] **Step 4: Commit**

```bash
git add modal_apps/gemma_adapter_infer.py
git commit -m "feat: modal inference app for fine-tuned gemma adapter (mapper format)"
```

---

## Task 4: Evaluate via existing ingest-mapper + clean metric

**Files:**
- Reuse: `eraparse trials ingest-mapper`, `scripts/analyze_labels_and_router.py`

- [ ] **Step 1: Ingest fine-tuned responses (reduced schema, existing evaluator)**

```bash
cd /Users/anasahmed/AI_Main_Repo/CV_parsing_main/eraparse
uv run eraparse trials ingest-mapper \
  --model-id eraparse/gemma3-1b-reduced --revision ft-v1 \
  --representation pymupdf4llm_markdown \
  --requests artifacts/trials/router/validation.gemma-ft.requests.jsonl \
  --responses artifacts/trials/router/validation.gemma-ft.responses.jsonl \
  --output-dir artifacts/trials/router/validation-gemma-ft-ingested \
  --allow-partial --json
```
Expected: JSON with `aggregate.macro_score`, `schema_valid_rate`, `unsupported_evidence_rate`, `mean_latency_seconds`.

- [ ] **Step 2: Compute the fully-clean metric (reuse the analysis helpers)**

```bash
uv run python -c "
import json, statistics, sys
from pathlib import Path
sys.path.insert(0,'scripts')
import analyze_labels_and_router as A
man = A.load_manifest('validation')
res = next(Path('artifacts/trials/router/validation-gemma-ft-ingested').rglob('results.jsonl'))
A.clean_macro(res, man, 'Gemma3-1B fine-tuned (validation)')
"
```
Expected: prints raw / url-clean / fully-clean macro + nested + per-field.

- [ ] **Step 3: Record the base-vs-FT comparison**

Append to `docs/RESULTS_AND_PARETO.md` a row: base 1B (31% valid, from `PRIOR_KAGGLE_TRIALS.md`) vs fine-tuned 1B (this run) — validity %, fully-clean macro, nested macro, unsupported-evidence, mean latency. This is the **RQ2 answer for 1B/reduced**.

- [ ] **Step 4: Commit**

```bash
git add docs/RESULTS_AND_PARETO.md
git commit -m "feat: fine-tuned gemma3-1b reduced-schema eval vs base (RQ2 1B)"
```

---

## Acceptance / exit

- Fine-tuned 1B evaluated on the SAME validation split + clean metric as A/B/C.
- A concrete base(31% valid) → fine-tuned comparison on validity / accuracy / faithfulness / latency.
- All three components (`build_sft_reduced.py`, `gemma_finetune.py`, `gemma_adapter_infer.py`) committed and reusable for Plan 2 (4B + rich schema).
- **Decision gate:** if fine-tuned 1B reaches usable validity (>90%) and nested accuracy approaching lanes A/B, Track A is validated → scale to 4B + rich schema (Plan 2). If not, diagnose (truncation? data? size?) before scaling.

## Self-review notes
- Schema source: `REDUCED_SCHEMA_TEMPLATE` (constants.py) used identically in data-gen, inference requests, and the evaluator's reduced target — consistent.
- `parser_id`/`representation` = `pymupdf4llm_markdown` everywhere (matches the reps used to build SFT input).
- Training targets are cleaned (phantom URLs/projects removed) so faithfulness is trained, and evaluation uses the same clean metric — no train/eval metric mismatch.
- Modal heartbeat disconnect (seen with background runs) handled by the resume-loop in Task 3 Step 3, mirroring the validated NuExtract3/Qwen retry pattern.
