from __future__ import annotations

import os
import platform
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from eraparse.constants import LAYOUTLMV3_MODEL_ID, LAYOUTLMV3_REVISION, SEED, SGE_FIELD_PATHS
from eraparse.io import atomic_write_json, atomic_write_jsonl, read_jsonl

TrainingMode = Literal["baseline", "sge"]
DecoderStrategy = Literal["auto", "learned", "sequence"]


def resolve_primary_decoder(mode: TrainingMode, strategy: DecoderStrategy) -> str:
    if mode == "baseline":
        if strategy == "learned":
            raise ValueError("baseline mode does not support the learned decoder")
        return "sequence"
    return "learned" if strategy == "learned" else "sequence"


def validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    mode: TrainingMode,
    seed: int,
    unfreeze_final_layers: int,
    query_layers: int,
    presence_weight: float,
    grouping_weight: float,
    evidence_weight: float,
) -> None:
    expected = {
        "mode": mode,
        "seed": seed,
        "unfreeze_final_layers": unfreeze_final_layers,
        "query_layers": query_layers,
        "loss_weights": {
            "field_token": 1.0,
            "presence": presence_weight,
            "grouping": grouping_weight,
            "evidence": evidence_weight,
        },
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume checkpoint {key}={checkpoint.get(key)!r}, expected {value!r}")


def training_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    max_steps: int,
) -> list[Mapping[str, Any]]:
    if not records:
        raise ValueError("training requires at least one record")
    if max_steps < 0:
        raise ValueError("max_steps must not be negative")
    return [records[index % len(records)] for index in range(max_steps)]


def first_subword_positions(word_ids: Sequence[int | None]) -> list[int]:
    positions = []
    previous = None
    for position, word_id in enumerate(word_ids):
        if word_id is not None and word_id != previous:
            positions.append(position)
        previous = word_id
    return positions


def select_device(torch_module: Any, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _weighted_token_loss(torch_module: Any, logits: Any, labels: Any) -> Any:
    from eraparse.sge_losses import token_class_weights

    text_length = labels.shape[1]
    weights = logits.new_tensor(token_class_weights(logits.shape[-1] - 1))
    return torch_module.nn.functional.cross_entropy(
        logits[:, :text_length].reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        weight=weights,
    )


def unfreeze_final_layoutlmv3_layers(model: Any, count: int) -> None:
    layers = model.layoutlmv3.encoder.layer
    if count < 1 or count > len(layers):
        raise ValueError(f"count must be between 1 and {len(layers)}")
    for layer in layers[-count:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


def _load_model(
    mode: TrainingMode,
    *,
    unfreeze_final_layers: int = 0,
    presence_weight: float = 0.25,
    grouping_weight: float = 0.5,
    evidence_weight: float = 0.5,
    query_layers: int = 2,
) -> Any:
    if unfreeze_final_layers < 0:
        raise ValueError("unfreeze_final_layers cannot be negative")
    for name, value in {
        "presence_weight": presence_weight,
        "grouping_weight": grouping_weight,
        "evidence_weight": evidence_weight,
    }.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    if query_layers < 1:
        raise ValueError("query_layers must be at least one")
    from transformers import LayoutLMv3ForTokenClassification

    if mode == "baseline":
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            LAYOUTLMV3_MODEL_ID,
            revision=LAYOUTLMV3_REVISION,
            num_labels=len(SGE_FIELD_PATHS) + 1,
        )
        for parameter in model.layoutlmv3.parameters():
            parameter.requires_grad = False
        if unfreeze_final_layers:
            unfreeze_final_layoutlmv3_layers(model, unfreeze_final_layers)
        return model
    if mode == "sge":
        from eraparse.sge_model import SchemaGuidedLayoutLMv3

        model = SchemaGuidedLayoutLMv3(
            LAYOUTLMV3_MODEL_ID,
            revision=LAYOUTLMV3_REVISION,
            num_fields=len(SGE_FIELD_PATHS),
            presence_weight=presence_weight,
            grouping_weight=grouping_weight,
            evidence_weight=evidence_weight,
            query_layers=query_layers,
        )
        model.freeze_encoder()
        if unfreeze_final_layers:
            model.unfreeze_final_encoder_layers(unfreeze_final_layers)
        return model
    raise ValueError("mode must be baseline or sge")


def _encode_record(
    processor: Any,
    record: Mapping[str, Any],
    dataset_root: Path,
    torch_module: Any,
    device: str,
) -> tuple[dict[str, Any], Any, Any, list[int | None]]:
    from PIL import Image

    image = Image.open(dataset_root / str(record["page_image"])).convert("RGB")
    encoded = processor(
        image,
        record["words"],
        boxes=record["boxes"],
        word_labels=record["field_labels"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    group_labels = record.get("record_group_labels", record["record_indices"])
    record_ids = torch_module.tensor(
        [-1 if word_id is None else int(group_labels[word_id]) for word_id in word_ids],
        dtype=torch_module.long,
    ).unsqueeze(0)
    presence = torch_module.zeros((1, len(SGE_FIELD_PATHS)))
    for label in record["field_labels"]:
        if label > 0:
            presence[0, int(label) - 1] = 1
    batch = {key: value.to(device) for key, value in encoded.items()}
    return batch, presence.to(device), record_ids.to(device), word_ids


def _evaluate_model(
    model: Any,
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    torch_module: Any,
    device: str,
    mode: TrainingMode,
    output_dir: Path,
    *,
    file_prefix: str = "",
    primary_decoder: str = "sequence",
) -> dict[str, Any]:
    from eraparse.assembly import assemble_prediction
    from eraparse.evaluate import aggregate_evaluations, evaluate_document
    from eraparse.models import EvidenceBundle
    from eraparse.sge import decode_word_candidates

    decoder_names = ("learned", "sequence") if mode == "sge" else ("sequence",)
    by_decoder_candidates = {
        decoder_name: defaultdict(list) for decoder_name in decoder_names
    }
    by_cv_truth = {}
    by_cv_words = defaultdict(list)
    model.eval()
    with torch_module.no_grad():
        for record in records:
            batch, _, _, word_ids = _encode_record(
                processor,
                record,
                dataset_root,
                torch_module,
                device,
            )
            batch.pop("labels", None)
            output = model(**batch)
            logits = output["logits"] if isinstance(output, dict) else output.logits
            positions = first_subword_positions(word_ids)
            word_count = min(len(positions), len(record["words"]))
            positions = positions[:word_count]
            probabilities = torch_module.softmax(logits[0, positions], dim=-1)
            word_labels = probabilities.argmax(dim=-1).detach().cpu().tolist()
            confidences = probabilities.amax(dim=-1).detach().cpu().tolist()
            grouping_probabilities = None
            if mode == "sge":
                grouping = torch_module.sigmoid(output["grouping_logits"][0])
                grouping_probabilities = grouping[positions][:, positions].detach().cpu().tolist()
            record_view = dict(record)
            record_view["words"] = list(record["words"])[:word_count]
            record_view["evidence_ids"] = list(record["evidence_ids"])[:word_count]
            for decoder_name in decoder_names:
                by_decoder_candidates[decoder_name][str(record["cv_id"])].extend(
                    decode_word_candidates(
                        record_view,
                        word_labels=word_labels,
                        confidences=confidences,
                        grouping_probabilities=(
                            grouping_probabilities if decoder_name == "learned" else None
                        ),
                    )
                )
            by_cv_truth[str(record["cv_id"])] = record["truth"]
            by_cv_words[str(record["cv_id"])].extend(record_view["words"])
    decoder_evaluations = {}
    for decoder_name in decoder_names:
        predictions = [
            assemble_prediction(cv_id, by_decoder_candidates[decoder_name][cv_id])
            for cv_id in sorted(by_cv_truth)
        ]
        evaluations = [
            evaluate_document(
                by_cv_truth[prediction.cv_id],
                prediction.prediction.model_dump(mode="json"),
                EvidenceBundle(parser_text=" ".join(by_cv_words[prediction.cv_id])),
            )
            for prediction in predictions
        ]
        aggregate = aggregate_evaluations(evaluations).model_dump(mode="json")
        decoder_evaluations[decoder_name] = aggregate
        atomic_write_jsonl(
            output_dir / f"{file_prefix}predictions_{decoder_name}.jsonl",
            [prediction.model_dump(mode="json") for prediction in predictions],
        )
        atomic_write_json(output_dir / f"{file_prefix}evaluation_{decoder_name}.json", aggregate)
    if primary_decoder not in decoder_evaluations:
        raise ValueError(f"decoder {primary_decoder!r} is unavailable for mode {mode!r}")
    atomic_write_jsonl(
        output_dir / f"{file_prefix}predictions.jsonl",
        read_jsonl(output_dir / f"{file_prefix}predictions_{primary_decoder}.jsonl"),
    )
    atomic_write_json(
        output_dir / f"{file_prefix}evaluation.json",
        decoder_evaluations[primary_decoder],
    )
    return decoder_evaluations


def run_local_smoke(
    records_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    mode: TrainingMode = "sge",
    max_steps: int = 1,
    max_records: int = 1,
    requested_device: str = "auto",
    seed: int = SEED,
    unfreeze_final_layers: int = 0,
    presence_weight: float = 0.25,
    grouping_weight: float = 0.5,
    evidence_weight: float = 0.5,
    query_layers: int = 2,
    decoder_strategy: DecoderStrategy = "auto",
    evaluate_training: bool = True,
    evaluation_records_path: Path | None = None,
    max_evaluation_records: int | None = None,
    checkpoint_path: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from transformers import AutoProcessor

    records = list(read_jsonl(records_path))[:max_records]
    evaluation_records = (
        list(read_jsonl(evaluation_records_path))[:max_evaluation_records]
        if evaluation_records_path is not None
        else None
    )
    if max_steps == 0 and resume_checkpoint is None:
        raise ValueError("zero-step evaluation requires a resume checkpoint")
    schedule = training_schedule(records, max_steps=max_steps)
    device = select_device(torch, requested_device)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    processor = AutoProcessor.from_pretrained(
        LAYOUTLMV3_MODEL_ID,
        revision=LAYOUTLMV3_REVISION,
        apply_ocr=False,
    )
    model = _load_model(
        mode,
        unfreeze_final_layers=unfreeze_final_layers,
        presence_weight=presence_weight,
        grouping_weight=grouping_weight,
        evidence_weight=evidence_weight,
        query_layers=query_layers,
    ).to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-5,
    )
    resumed_steps = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        validate_resume_checkpoint(
            checkpoint,
            mode=mode,
            seed=seed,
            unfreeze_final_layers=unfreeze_final_layers,
            query_layers=query_layers,
            presence_weight=presence_weight,
            grouping_weight=grouping_weight,
            evidence_weight=evidence_weight,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        resumed_steps = int(checkpoint["completed_steps"])
    losses = []
    started = time.perf_counter()
    training_started = started
    for step, record in enumerate(schedule, start=1):
        batch, presence, record_ids, _ = _encode_record(
            processor,
            record,
            dataset_root,
            torch,
            device,
        )
        if mode == "sge":
            output = model(**batch, presence_labels=presence, record_ids=record_ids)
            loss = output["loss"]
        else:
            labels = batch.pop("labels")
            output = model(**batch)
            loss = _weighted_token_loss(torch, output.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu().item()))
        print(f"step={step} loss={losses[-1]:.6f}")
    training_seconds = time.perf_counter() - training_started
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "completed_steps": resumed_steps + len(losses),
                "mode": mode,
                "seed": seed,
                "unfreeze_final_layers": unfreeze_final_layers,
                "loss_weights": {
                    "field_token": 1.0,
                    "presence": presence_weight,
                    "grouping": grouping_weight,
                    "evidence": evidence_weight,
                },
                "query_layers": query_layers,
            },
            temporary_checkpoint,
        )
        os.replace(temporary_checkpoint, checkpoint_path)
    primary_decoder = resolve_primary_decoder(mode, decoder_strategy)
    evaluation_started = time.perf_counter()
    decoder_evaluations = (
        _evaluate_model(
            model,
            processor,
            records,
            dataset_root,
            torch,
            device,
            mode,
            output_dir,
            file_prefix="training_" if evaluation_records is not None else "",
            primary_decoder=primary_decoder,
        )
        if evaluate_training
        else None
    )
    validation_decoder_evaluations = (
        _evaluate_model(
            model,
            processor,
            evaluation_records,
            dataset_root,
            torch,
            device,
            mode,
            output_dir,
            file_prefix="validation_",
            primary_decoder=primary_decoder,
        )
        if evaluation_records is not None
        else None
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    summary = {
        "mode": mode,
        "device": device,
        "machine": platform.machine(),
        "records_path": str(records_path),
        "evaluation_records_path": (
            str(evaluation_records_path) if evaluation_records_path is not None else None
        ),
        "max_records": max_records,
        "max_evaluation_records": max_evaluation_records,
        "steps": len(losses),
        "resumed_steps": resumed_steps,
        "completed_steps": resumed_steps + len(losses),
        "losses": losses,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "runtime_seconds": time.perf_counter() - started,
        "training_seconds": training_seconds,
        "training_steps_per_second": len(losses) / training_seconds if losses else 0.0,
        "evaluation_seconds": evaluation_seconds,
        "evaluated_records": (len(records) if evaluate_training else 0)
        + (len(evaluation_records) if evaluation_records is not None else 0),
        "model_id": LAYOUTLMV3_MODEL_ID,
        "model_revision": LAYOUTLMV3_REVISION,
        "torch_version": torch.__version__,
        "seed": seed,
        "unfreeze_final_layers": unfreeze_final_layers,
        "loss_weights": {
            "field_token": 1.0,
            "presence": presence_weight,
            "grouping": grouping_weight,
            "evidence": evidence_weight,
        },
        "query_layers": query_layers,
        "decoder_strategy": decoder_strategy,
        "evaluate_training": evaluate_training,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "primary_decoder": primary_decoder,
        "training_evaluation": (
            decoder_evaluations[primary_decoder] if decoder_evaluations is not None else None
        ),
        "decoder_evaluations": decoder_evaluations,
        "validation_evaluation": (
            validation_decoder_evaluations[primary_decoder]
            if validation_decoder_evaluations is not None
            else None
        ),
        "validation_decoder_evaluations": validation_decoder_evaluations,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
