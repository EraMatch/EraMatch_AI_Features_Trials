"""LoRA fine-tune NuExtract-1.5 (3.8B, Phi-3.5-mini base) on reduced-schema SFT data.

RQ2b mid-size point: extraction-pretrained 3.8B vs general 1B (Gemma) and
extraction-pretrained 0.5B (NuExtract-tiny) on the same faithfulness-cleaned data.

Run:
    modal run modal_apps/nuextract_15_finetune.py
"""
import json
from pathlib import Path

import modal

GPU = "A10G"  # 3.8B needs more VRAM than L4 can comfortably hold at 8192 ctx
TIMEOUT = 3 * 60 * 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("unsloth", "unsloth_zoo", "datasets>=3.0", "sentencepiece", "protobuf")
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("nuextract-15-finetune-reduced", image=image)
vol = modal.Volume.from_name("eraparse-adapters", create_if_missing=True)


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/adapters": vol})
def train(
    sft_jsonl: str = "/adapters/sft/train.nuextract.sft.jsonl",
    model_name: str = "numind/NuExtract-1.5",
    out_name: str = "nuextract-15-reduced",
    max_seq_length: int = 8192,
    epochs: int = 2,
    lr: float = 2e-4,
) -> dict:
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    rows = [json.loads(x) for x in Path(sft_jsonl).read_text().splitlines() if x.strip()]
    print(f"loaded {len(rows)} SFT examples from {sft_jsonl}")

    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    ds = Dataset.from_list(rows)
    out_dir = f"/adapters/{out_name}"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            num_train_epochs=epochs,
            learning_rate=lr,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=25,
            save_strategy="no",
            seed=42,
            report_to="none",
        ),
    )
    stats = trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    vol.commit()

    result = {
        "out_dir": out_dir,
        "train_loss": float(stats.training_loss),
        "examples": len(rows),
        "model_name": model_name,
        "epochs": epochs,
    }
    print("DONE:", json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    model_name: str = "numind/NuExtract-1.5",
    out_name: str = "nuextract-15-reduced",
    sft_jsonl: str = "/adapters/sft/train.nuextract.sft.jsonl",
    max_seq_length: int = 8192,
    epochs: int = 2,
):
    result = train.remote(
        sft_jsonl=sft_jsonl,
        model_name=model_name,
        out_name=out_name,
        max_seq_length=max_seq_length,
        epochs=epochs,
    )
    print("Training complete:", result)
