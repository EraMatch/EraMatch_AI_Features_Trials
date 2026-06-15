import json
import random
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "naver-clova-ix/donut-base"
MODEL_REVISION = "a959cf33c20e09215873e338299c900f57047c61"
TASK_PROMPT = "<s_eraparse>"
DATA_ROOT = Path("/data/eraparse")
MODEL_CACHE = "/models"
CHECKPOINT_ROOT = Path("/checkpoints")

app = modal.App("eraparse-donut-train")
dataset_volume = modal.Volume.from_name("eraparse-dataset", create_if_missing=True)
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("eraparse-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "safetensors==0.6.2",
        "pillow==11.3.0",
        "sentencepiece==0.2.1",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("eraparse")
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={
        "/data": dataset_volume,
        MODEL_CACHE: model_volume,
        str(CHECKPOINT_ROOT): checkpoint_volume,
    },
)
def train(
    *,
    train_records_path: str,
    validation_records_path: str,
    run_name: str,
    epochs: int,
    max_steps: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    decoder_max_length: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    from eraparse.training_utils import gradient_accumulation_step

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_records = _read_jsonl(Path(train_records_path))
    validation_records = _read_jsonl(Path(validation_records_path))
    special_tokens = sorted(
        {
            str(token)
            for record in train_records
            for token in record.get("special_tokens", [TASK_PROMPT])
        }
        | {TASK_PROMPT}
    )
    processor = DonutProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    task_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_PROMPT)
    model.config.decoder_start_token_id = task_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.max_length = decoder_max_length
    model.decoder.config.max_position_embeddings = decoder_max_length
    model.decoder.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to("cuda")

    class DonutDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            record = self.records[index]
            pages = [Image.open(DATA_ROOT / path).convert("RGB") for path in record["page_images"]]
            width = max(page.width for page in pages)
            resized = [
                page
                if page.width == width
                else page.resize(
                    (width, round(page.height * width / page.width)),
                    Image.Resampling.LANCZOS,
                )
                for page in pages
            ]
            canvas = Image.new(
                "RGB",
                (width, sum(page.height for page in resized) + 16 * (len(resized) - 1)),
                "white",
            )
            top = 0
            for page in resized:
                canvas.paste(page, (0, top))
                top += page.height + 16
            pixel_values = processor(
                canvas,
                return_tensors="pt",
            ).pixel_values.squeeze(0)
            target = record["target"] + processor.tokenizer.eos_token
            labels = processor.tokenizer(
                target,
                add_special_tokens=False,
                max_length=decoder_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            labels[labels == processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    token_lengths = [
        len(processor.tokenizer(record["target"], add_special_tokens=False).input_ids)
        for record in train_records
    ]
    truncated_count = sum(length >= decoder_max_length for length in token_lengths)
    train_loader = DataLoader(
        DonutDataset(train_records),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        DonutDataset(validation_records),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    training_losses: list[float] = []
    epoch_summaries = []
    run_dir = CHECKPOINT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    for epoch in range(epochs):
        model.train()
        epoch_training_losses = []
        accumulated_loss = 0.0
        accumulated_batches = 0
        scheduled_epoch_micro_steps = len(train_loader)
        if max_steps:
            remaining_steps = max_steps - completed_steps
            scheduled_epoch_micro_steps = min(
                scheduled_epoch_micro_steps,
                remaining_steps * gradient_accumulation_steps,
            )
        for epoch_micro_steps, batch in enumerate(train_loader, start=1):
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                unscaled_loss = model(**batch).loss
                loss = unscaled_loss / gradient_accumulation_steps
            loss.backward()
            accumulated_batches += 1
            accumulated_loss += float(unscaled_loss.item())
            should_step, remainder_scale = gradient_accumulation_step(
                epoch_micro_steps,
                total_micro_steps=scheduled_epoch_micro_steps,
                accumulation_steps=gradient_accumulation_steps,
            )
            if should_step:
                if remainder_scale != 1.0:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(remainder_scale)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                step_loss = accumulated_loss / accumulated_batches
                training_losses.append(step_loss)
                epoch_training_losses.append(step_loss)
                accumulated_loss = 0.0
                accumulated_batches = 0
                print(f"epoch={epoch + 1} step={completed_steps} loss={training_losses[-1]:.6f}")
            if epoch_micro_steps >= scheduled_epoch_micro_steps:
                break

        model.eval()
        epoch_validation_losses = []
        with torch.inference_mode():
            for batch in validation_loader:
                batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    epoch_validation_losses.append(float(model(**batch).loss.item()))
        checkpoint_dir = run_dir / f"epoch_{epoch + 1:02d}"
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        processor.save_pretrained(checkpoint_dir)
        epoch_summary = {
            "epoch": epoch + 1,
            "optimizer_steps_total": completed_steps,
            "mean_training_loss": sum(epoch_training_losses) / len(epoch_training_losses),
            "mean_validation_loss": sum(epoch_validation_losses) / len(epoch_validation_losses),
            "checkpoint_path": str(checkpoint_dir),
        }
        epoch_summaries.append(epoch_summary)
        (run_dir / "training_summary.partial.json").write_text(
            json.dumps({"run_name": run_name, "epochs": epoch_summaries}, indent=2) + "\n",
            encoding="utf-8",
        )
        checkpoint_volume.commit()
        if max_steps and completed_steps >= max_steps:
            break

    summary = {
        "run_name": run_name,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "epochs_requested": epochs,
        "optimizer_steps": completed_steps,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "decoder_max_length": decoder_max_length,
        "special_token_count": len(special_tokens),
        "target_format": train_records[0].get("target_format", "raw_json"),
        "max_target_tokens": max(token_lengths),
        "truncated_target_count": truncated_count,
        "mean_training_loss": sum(training_losses) / len(training_losses),
        "epochs": epoch_summaries,
        "checkpoint_path": str(run_dir),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_volume.commit()
    model_volume.commit()
    return summary


def _upload_records(records_path: Path, dataset_root: Path) -> str:
    records = _read_jsonl(records_path)
    remote_records_path = f"/eraparse/records/{records_path.stem}.jsonl"
    with dataset_volume.batch_upload(force=True) as batch:
        batch.put_file(records_path, remote_records_path)
        for record in records:
            for relative_path in record["page_images"]:
                batch.put_file(dataset_root / relative_path, f"/eraparse/{relative_path}")
    return f"/data{remote_records_path}"


@app.local_entrypoint()
def main(
    train_records: str,
    validation_records: str,
    dataset_root: str,
    run_name: str,
    epochs: int = 1,
    max_steps: int = 2,
    learning_rate: float = 3e-5,
    gradient_accumulation_steps: int = 8,
    decoder_max_length: int = 1536,
    seed: int = 20260609,
) -> None:
    root = Path(dataset_root)
    remote_train = _upload_records(Path(train_records), root)
    remote_validation = _upload_records(Path(validation_records), root)
    summary = train.remote(
        train_records_path=remote_train,
        validation_records_path=remote_validation,
        run_name=run_name,
        epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        decoder_max_length=decoder_max_length,
        seed=seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
