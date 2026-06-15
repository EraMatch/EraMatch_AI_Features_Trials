import json
import time
from pathlib import Path

import modal

MODEL_ID = "microsoft/layoutlmv3-base"
MODEL_REVISION = "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4"
MODEL_CACHE = "/models"
DATA_ROOT = Path("/data/eraparse")
CHECKPOINT_ROOT = Path("/checkpoints")

app = modal.App("eraparse-sge-smoke")
dataset_volume = modal.Volume.from_name("eraparse-dataset", create_if_missing=True)
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("eraparse-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers==4.57.3",
        "pillow==11.3.0",
        "safetensors==0.6.2",
    )
    .env({"HF_HUB_CACHE": MODEL_CACHE, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("eraparse")
)


@app.function(image=image, cpu=2, memory=4096, timeout=2 * 60)
def validate_remote_imports() -> dict:
    import torch
    import transformers

    from eraparse.sge_losses import binary_positive_weight, token_class_weights
    from eraparse.sge_model import SchemaGuidedLayoutLMv3
    from eraparse.sge_training import training_schedule

    return {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "model_class": SchemaGuidedLayoutLMv3.__name__,
        "schedule_callable": callable(training_schedule),
        "loss_helpers": [
            token_class_weights(2),
            binary_positive_weight(positive_count=1, negative_count=2),
        ],
    }


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=15 * 60,
    volumes={
        "/data": dataset_volume,
        MODEL_CACHE: model_volume,
        str(CHECKPOINT_ROOT): checkpoint_volume,
    },
)
def train_smoke(
    *,
    records_path: str,
    run_name: str,
    mode: str,
    max_steps: int,
    seed: int,
    unfreeze_final_layers: int,
) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

    from eraparse.constants import SGE_FIELD_PATHS
    from eraparse.sge_model import SchemaGuidedLayoutLMv3
    from eraparse.sge_training import (
        _weighted_token_loss,
        training_schedule,
        unfreeze_final_layoutlmv3_layers,
    )

    torch.manual_seed(seed)
    records = [
        json.loads(line)
        for line in Path(records_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        apply_ocr=False,
    )
    def new_model():
        if mode == "baseline":
            baseline = LayoutLMv3ForTokenClassification.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                num_labels=len(SGE_FIELD_PATHS) + 1,
            )
            for parameter in baseline.layoutlmv3.parameters():
                parameter.requires_grad = False
            if unfreeze_final_layers:
                unfreeze_final_layoutlmv3_layers(baseline, unfreeze_final_layers)
            return baseline
        if mode == "sge":
            sge = SchemaGuidedLayoutLMv3(
                MODEL_ID,
                revision=MODEL_REVISION,
                num_fields=len(SGE_FIELD_PATHS),
            )
            sge.freeze_encoder()
            if unfreeze_final_layers:
                sge.unfreeze_final_encoder_layers(unfreeze_final_layers)
            return sge
        raise ValueError("mode must be baseline or sge")

    model = new_model()
    model.to("cuda").train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-5,
    )
    started = time.perf_counter()
    losses = []
    for step, record in enumerate(training_schedule(records, max_steps=max_steps), start=1):
        image_path = DATA_ROOT / record["page_image"]
        page_image = Image.open(image_path).convert("RGB")
        encoded = processor(
            page_image,
            record["words"],
            boxes=record["boxes"],
            word_labels=record["field_labels"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        group_labels = record.get("record_group_labels", record["record_indices"])
        record_ids = torch.tensor(
            [-1 if word_id is None else int(group_labels[word_id]) for word_id in word_ids],
            dtype=torch.long,
        ).unsqueeze(0)
        batch = {key: value.to("cuda") for key, value in encoded.items()}
        record_ids = record_ids.to("cuda")
        if mode == "sge":
            present = torch.zeros((1, len(SGE_FIELD_PATHS)), device="cuda")
            for label in record["field_labels"]:
                if label > 0:
                    present[0, label - 1] = 1
            output = model(**batch, presence_labels=present, record_ids=record_ids)
            loss = output["loss"]
        else:
            labels = batch.pop("labels")
            output = model(**batch)
            loss = _weighted_token_loss(torch, output.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
        print(f"step={step} loss={losses[-1]:.6f}")
    run_dir = CHECKPOINT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "training_state.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "mode": mode,
            "seed": seed,
            "steps": len(losses),
            "unfreeze_final_layers": unfreeze_final_layers,
        },
        checkpoint_path,
    )
    processor.save_pretrained(run_dir)
    reloaded_model = new_model().to("cuda").train()
    reloaded_optimizer = torch.optim.AdamW(
        [parameter for parameter in reloaded_model.parameters() if parameter.requires_grad],
        lr=3e-5,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    reloaded_model.load_state_dict(checkpoint["model_state"], strict=True)
    reloaded_optimizer.load_state_dict(checkpoint["optimizer_state"])
    summary = {
        "run_name": run_name,
        "mode": mode,
        "steps": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "runtime_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_path": str(run_dir),
        "checkpoint_reload_verified": True,
        "seed": seed,
        "unfreeze_final_layers": unfreeze_final_layers,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checkpoint_volume.commit()
    model_volume.commit()
    return summary


def _upload_records(records_path: Path, dataset_root: Path) -> str:
    records = [
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line
    ]
    remote_path = f"/eraparse/records/{records_path.stem}.jsonl"
    with dataset_volume.batch_upload(force=True) as batch:
        batch.put_file(records_path, remote_path)
        for record in records:
            batch.put_file(dataset_root / record["page_image"], f"/eraparse/{record['page_image']}")
    return f"/data{remote_path}"


@app.local_entrypoint()
def main(
    records_path: str,
    dataset_root: str,
    run_name: str,
    mode: str = "sge",
    max_steps: int = 1,
    stage_spent: float = 0.0,
    stage_budget: float = 18.0,
    remaining_credit: float = 20.0,
    seed: int = 20260609,
    unfreeze_final_layers: int = 4,
) -> None:
    from eraparse.budget import projected_t4_cost, require_budget

    projected = projected_t4_cost(15 * 60)
    require_budget(
        projected,
        spent=stage_spent,
        budget=stage_budget,
        remaining_credit=remaining_credit,
    )
    import_smoke = validate_remote_imports.remote()
    remote_records = _upload_records(Path(records_path), Path(dataset_root))
    summary = train_smoke.remote(
        records_path=remote_records,
        run_name=run_name,
        mode=mode,
        max_steps=max_steps,
        seed=seed,
        unfreeze_final_layers=unfreeze_final_layers,
    )
    summary["remote_import_smoke"] = import_smoke
    summary["maximum_projected_gpu_cost"] = projected
    print(json.dumps(summary, indent=2, sort_keys=True))
