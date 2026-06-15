"""Track C — Fine-tune SmolVLM2-2.2B-Instruct on (CV image, schema, JSON) triples.

Replicates the NuExtract recipe on a fresher/faster base specialized to CVs:
  general VLM base + schema-conditioned SFT on (image, schema-template, JSON)
  with schema augmentation so the model follows arbitrary target schemas.

RQ6: Can a domain-specialized 2.2B VLM match NuExtract3 (12B) accuracy at
lower size/latency with better faithfulness?

Prerequisite:
    uv run python scripts/build_sft_vision.py
    modal volume put eraparse-adapters artifacts/sft/train.vision.sft.jsonl /sft/train.vision.sft.jsonl
    # upload page_images directory:
    modal volume put eraparse-adapters <path-to-benchmark>/page_images /page_images

Run:
    modal run modal_apps/smolvlm2_finetune.py
"""
import json
from pathlib import Path

import modal

GPU = "A100"  # VLM SFT; 2.2B + image encoder needs ~40GB for batch training
TIMEOUT = 6 * 60 * 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "libgl1")
    .pip_install(
        "unsloth[huggingface]",
        "unsloth_zoo",
        "datasets>=3.0",
        "Pillow",
        "sentencepiece",
        "protobuf",
        "torchvision",
        "num2words",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("smolvlm2-finetune", image=image)
vol = modal.Volume.from_name("eraparse-adapters", create_if_missing=True)

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/adapters": vol})
def train(
    sft_jsonl: str = "/adapters/sft/train.vision.sft.jsonl",
    images_dir: str = "/adapters/page_images",
    out_name: str = "smolvlm2-cv-reduced",
    max_seq_length: int = 4096,
    epochs: int = 2,
    lr: float = 1e-4,
) -> dict:
    import torch
    from PIL import Image
    from transformers import TrainingArguments, AutoProcessor, Trainer
    from datasets import Dataset
    from unsloth import FastVisionModel

    rows = [json.loads(x) for x in Path(sft_jsonl).read_text().splitlines() if x.strip()]
    print(f"loaded {len(rows)} vision SFT examples")

    # FastVisionModel returns (model, tokenizer) — tokenizer only, not a multimodal processor
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    # Load the full multimodal processor (tokenizer + image processor) separately
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # TRL SFTTrainer calls processing_class.eos_token — proxy it from the inner tokenizer
    if not hasattr(processor, "eos_token"):
        processor.eos_token = processor.tokenizer.eos_token
    print(f"model loaded: {MODEL_ID}")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=42,
    )
    model.print_trainable_parameters()

    images_path = Path(images_dir)

    def load_images(image_paths: list[str]) -> list:
        imgs = []
        for p in image_paths:
            full = images_path / Path(p).name
            if full.exists():
                imgs.append(Image.open(full).convert("RGB"))
        return imgs or None

    def collate(batch):
        texts, all_images = [], []
        for ex in batch:
            imgs = load_images(ex["image_paths"])
            convs = ex["conversations"]
            # SmolVLM2 expects structured content with {"type": "image"} dict,
            # not a raw "<image>" string token.
            if imgs:
                # one {"type": "image"} placeholder per page image
                user_content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": convs[1]["content"]}]
            else:
                user_content = [{"type": "text", "text": convs[1]["content"]}]
            messages = [
                {"role": "system", "content": [{"type": "text", "text": convs[0]["content"]}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": convs[2]["content"]}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            all_images.append(imgs if imgs else [])

        inputs = processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    ds = Dataset.from_list(rows)
    out_dir = f"/adapters/{out_name}"

    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=collate,
        args=TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.03,
            num_train_epochs=epochs,
            learning_rate=lr,
            bf16=True,
            logging_steps=25,
            save_strategy="no",
            seed=42,
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        ),
    )
    stats = trainer.train()
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)  # saves full multimodal processor (tokenizer + image proc)
    vol.commit()

    result = {
        "out_dir": out_dir,
        "train_loss": float(stats.training_loss),
        "examples": len(rows),
        "model": MODEL_ID,
        "epochs": epochs,
    }
    print("DONE:", json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    out_name: str = "smolvlm2-cv-reduced",
    sft_jsonl: str = "/adapters/sft/train.vision.sft.jsonl",
    max_seq_length: int = 4096,
    epochs: int = 2,
):
    result = train.remote(
        sft_jsonl=sft_jsonl,
        out_name=out_name,
        max_seq_length=max_seq_length,
        epochs=epochs,
    )
    print("Training complete:", result)
