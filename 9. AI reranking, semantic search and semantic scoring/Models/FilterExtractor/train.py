"""
Fine-tune Qwen2.5-Coder-1.5B-Instruct with Unsloth to extract structured filters
from HR natural language job queries.

Run on a Linux machine with an NVIDIA GPU (VRAM >= 8GB recommended).
Install dependencies with:  pip install -r requirements.txt
Then install Unsloth per README / requirements.txt comments.
"""
from unsloth import FastLanguageModel

import json
import os
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
DATASET_PATH = "../../Data generation/deepseek_tech_filters_5k.jsonl"
OUTPUT_DIR = f"./qwen_filter_extractor_{MODEL_NAME.split('/')[-1]}"  # e.g. qwen_filter_extractor-1.5B

SYSTEM_PROMPT = (
    "You are an expert HR filter extractor. "
    "Given a natural language job requirement query, extract structured filters as a valid JSON object.\n\n"
    "Output ONLY valid JSON. Include only fields that are clearly present in the query. Common fields:\n"
    "  role, skills, preferred_skills, min_experience_years, remote_ok, timezone, location,\n"
    "  employment_type, seniority_level, industry, start_date, visa_sponsorship,\n"
    "  certifications, education_level, no_c2c, citizenship_required, contract_duration,\n"
    "  direct_hire, experience_domain, security_clearance, startup_experience"
)

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 42
EVAL_SPLIT = 0.05


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed line {line_no}: {e}")
    return records


def format_example(example: dict, tokenizer) -> str:
    """Format a single dataset example into a ChatML string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["query"]},
        {
            "role": "assistant",
            "content": json.dumps(example["target_filters"], ensure_ascii=False),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def build_hf_dataset(records: list[dict], tokenizer) -> Dataset:
    formatted = [{"text": format_example(r, tokenizer)} for r in records]
    return Dataset.from_list(formatted)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=f"Fine-tune {MODEL_NAME} for filter extraction")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to .jsonl dataset")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[1/5] Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq_len,
        dtype=None,           # auto-detect float16 / bfloat16
        load_in_4bit=True,
    )

    print(f"[2/5] Attaching LoRA adapters (r={args.lora_r})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    print(f"[3/5] Loading dataset from: {args.dataset}")
    records = load_jsonl(args.dataset)
    print(f"      {len(records)} examples loaded")

    dataset = build_hf_dataset(records, tokenizer)
    split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=SEED)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"      Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    print("[4/5] Setting up trainer")
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=args.lr,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    print("[5/5] Training ...")
    trainer_stats = trainer.train()
    print(f"      Done. Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Model saved to {out.resolve()}")


if __name__ == "__main__":
    main()
