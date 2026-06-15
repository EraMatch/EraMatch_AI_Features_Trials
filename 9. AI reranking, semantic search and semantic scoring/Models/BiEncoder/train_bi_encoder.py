"""
train_bi_encoder.py
===================
Fine-tunes nomic-ai/nomic-embed-text-v1.5 as a bi-encoder on the CV-query triplet dataset.

Pipeline:
  1. Load all triplets/*.jsonl files
  2. Serialize CV JSON → structured text
  3. Apply nomic task prefixes  (search_query: / search_document:)
  4. Oversample hard negatives HARD_NEG_REPEAT × relative to easy negatives
  5. Train with MultipleNegativesRankingLoss — explicit hard negative per pair
     plus all other in-batch pairs as easy negatives automatically
  6. Save checkpoint every SAVE_STEPS steps; auto-resume from latest on restart

Note on Unsloth: not applicable.
  Unsloth accelerates decoder-only causal LLMs (Llama, Qwen, Mistral …).
  nomic-embed-text-v1.5 is a BERT-style bidirectional encoder — Unsloth cannot
  be used here. sentence-transformers handles training directly.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator


# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
MODEL_NAME      = "nomic-ai/nomic-embed-text-v1.5"

_ROOT           = Path(__file__).parent.parent.parent   # …/Candidate Reranking and Semantic search/
TRIPLETS_DIR    = _ROOT / "Data generation" / "triplets"
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
OUTPUT_DIR      = Path(__file__).parent / "output"

BATCH_SIZE      = 16
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-5
WARMUP_RATIO    = 0.1
SAVE_STEPS      = 200
EVAL_STEPS      = 200
LOGGING_STEPS   = 50
SAVE_TOTAL_LIMIT = 3        # keep only the 3 latest checkpoints
VAL_SPLIT       = 0.02      # 2 % of rows reserved for evaluation
EVAL_SAMPLE_CAP = 512       # max rows fed to TripletEvaluator per eval pass

HARD_NEG_REPEAT = 1         # hard-negative rows duplicated this many extra times

QUERY_PREFIX    = "search_query: "
DOC_PREFIX      = "search_document: "


# ═════════════════════════════════════════════════════════════════════════════
# CV JSON → structured text
# ═════════════════════════════════════════════════════════════════════════════
def cv_to_text(cv: Dict[str, Any]) -> str:
    parts: List[str] = []

    name      = cv.get("full_name", "")
    location  = cv.get("location", "")
    seniority = cv.get("seniority_level", "")
    yoe       = cv.get("years_of_experience", "")
    summary   = cv.get("summary", "")

    if name:
        parts.append(f"Name: {name}")
    if location:
        parts.append(f"Location: {location}")
    if seniority or yoe:
        parts.append(f"Seniority: {seniority} | Experience: {yoe} years")
    if summary:
        parts.append(f"Summary: {summary}")

    skills = cv.get("skills", [])
    if skills:
        skill_str = ", ".join(
            f"{s['skill_name']} ({s.get('proficiency', '')})"
            for s in skills[:12]
        )
        parts.append(f"Skills: {skill_str}")

    for job in cv.get("work_experience", [])[:3]:
        title   = job.get("job_title", "")
        company = job.get("company", "")
        desc    = job.get("description", "")[:120]
        techs   = ", ".join(job.get("technologies", [])[:6])
        line    = f"  {title} @ {company}"
        if techs:
            line += f" [{techs}]"
        if desc:
            line += f": {desc}"
        parts.append(line)

    for edu in cv.get("education", [])[:1]:
        degree = edu.get("degree", "")
        field  = edu.get("field_of_study", "")
        inst   = edu.get("institution", "")
        if degree and inst:
            parts.append(f"Education: {degree} in {field} — {inst}")

    certs = cv.get("certifications", [])
    if certs:
        parts.append(f"Certifications: {', '.join(certs[:4])}")

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════
def load_triplets(triplets_dir: Path) -> List[Dict[str, str]]:
    jsonl_files = sorted(triplets_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {triplets_dir}")

    easy: List[Dict[str, str]] = []
    hard: List[Dict[str, str]] = []

    for path in jsonl_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                row = {
                    "anchor":   QUERY_PREFIX + rec["anchor"],
                    "positive": DOC_PREFIX   + cv_to_text(rec["positive"]),
                    "negative": DOC_PREFIX   + cv_to_text(rec["negative"]),
                }
                if rec.get("negative_type") == "hard":
                    hard.append(row)
                else:
                    easy.append(row)

    print(f"Loaded  {len(easy):>7,} easy  +  {len(hard):>6,} hard negatives")

    combined = easy + hard * HARD_NEG_REPEAT
    random.shuffle(combined)
    print(f"Dataset {len(combined):>7,} rows  (hard negatives ×{HARD_NEG_REPEAT} oversampled)")
    return combined


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint detection
# ═════════════════════════════════════════════════════════════════════════════
def _checkpoint_step(p: Path) -> int:
    try:
        return int(p.name.split("-")[-1])
    except ValueError:
        return -1


def find_last_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=_checkpoint_step)
    return str(checkpoints[-1]) if checkpoints else None


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Model ────────────────────────────────────────────────────────────────
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    rows    = load_triplets(TRIPLETS_DIR)
    dataset = Dataset.from_list(rows)
    split   = dataset.train_test_split(test_size=VAL_SPLIT, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"Train   {len(train_ds):>7,} rows  |  Eval {len(eval_ds):,} rows")

    # ── Loss ─────────────────────────────────────────────────────────────────
    # Columns: anchor, positive, negative
    # MNRL uses the explicit `negative` as a hard negative AND treats every
    # other positive in the batch as an in-batch easy negative simultaneously.
    loss = MultipleNegativesRankingLoss(model)

    # ── Evaluator ────────────────────────────────────────────────────────────
    eval_sample = eval_ds.select(range(min(EVAL_SAMPLE_CAP, len(eval_ds))))
    evaluator = TripletEvaluator(
        anchors   = eval_sample["anchor"],
        positives = eval_sample["positive"],
        negatives = eval_sample["negative"],
        name      = "cv-query",
    )

    # ── Checkpoint detection ─────────────────────────────────────────────────
    last_ckpt = find_last_checkpoint(CHECKPOINT_DIR)
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
    else:
        print("No checkpoint found — starting from scratch")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Training arguments ───────────────────────────────────────────────────
    args = SentenceTransformerTrainingArguments(
        output_dir                  = str(CHECKPOINT_DIR),
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        fp16                        = True,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        logging_steps               = LOGGING_STEPS,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        dataloader_num_workers      = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        loss          = loss,
        evaluator     = evaluator,
    )

    trainer.train(resume_from_checkpoint=last_ckpt)

    # ── Save final model ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    print(f"\nFinal model saved → {OUTPUT_DIR}")
    print(f"Checkpoints kept  → {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
