import json
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console

from eraparse.constants import (
    DEFAULT_ATS_ROOT,
    DEFAULT_AUDIT_REPORT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_MANIFEST_ROOT,
    DEFAULT_REPRESENTATION_ROOT,
    DEFAULT_RUN_DB,
    DEFAULT_SGE_ROOT,
    DEFAULT_TRIAL_ROOT,
    NUEXTRACT3_MODEL_ID,
    NUEXTRACT3_REVISION,
    PADDLEOCR_VL_MODEL_ID,
    PADDLEOCR_VL_REVISION,
)

app = typer.Typer(help="EraParse research foundation")
data_app = typer.Typer(help="Audit data and build deterministic manifests")
eval_app = typer.Typer(help="Evaluate structured CV predictions")
runs_app = typer.Typer(help="Manage the local DuckDB run store")
trials_app = typer.Typer(help="Prepare and ingest staged research trials")
parsers_app = typer.Typer(help="Generate document representations")
ats_app = typer.Typer(help="Run persisted ATS compatibility and screening baselines")
evidence_app = typer.Typer(help="Build and validate grounded document evidence")
sge_app = typer.Typer(help="Prepare and evaluate schema-guided extraction trials")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(runs_app, name="runs")
app.add_typer(trials_app, name="trials")
app.add_typer(parsers_app, name="parsers")
app.add_typer(ats_app, name="ats")
app.add_typer(evidence_app, name="evidence")
app.add_typer(sge_app, name="sge")
console = Console(stderr=True)


def _emit(value: Any, json_output: bool) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if json_output:
        typer.echo(json.dumps(value, indent=2, sort_keys=True))
    else:
        console.print(value)


@data_app.command("audit")
def audit_command(
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output: Path = typer.Option(DEFAULT_AUDIT_REPORT),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.data import audit_dataset

    report = audit_dataset(dataset_root, report_path=output, hash_artifacts=True)
    _emit(report, json_output)
    if not report.passed:
        raise typer.Exit(1)


@data_app.command("build-manifests")
def build_manifests_command(
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output_dir: Path = typer.Option(DEFAULT_MANIFEST_ROOT),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.data import build_manifests

    summary = build_manifests(dataset_root, output_dir)
    _emit(summary, json_output)


@data_app.command("validate-manifests")
def validate_manifests_command(
    manifest_dir: Path = typer.Option(DEFAULT_MANIFEST_ROOT, exists=True, file_okay=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.data import validate_manifests

    result = validate_manifests(manifest_dir)
    _emit(result, json_output)
    if not result["passed"]:
        raise typer.Exit(1)


@eval_app.command("score")
def score_command(
    truth: Path = typer.Option(..., exists=True, dir_okay=False),
    prediction: Path = typer.Option(..., exists=True, dir_okay=False),
    evidence: Path | None = typer.Option(None, exists=True, dir_okay=False),
    output: Path | None = typer.Option(None),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.evaluate import evaluate_files

    result = evaluate_files(truth, prediction, evidence)
    if output is not None:
        from eraparse.io import atomic_write_json

        atomic_write_json(output, result.model_dump(mode="json"))
    _emit(result, json_output)
    if not result.json_valid or not result.schema_valid:
        raise typer.Exit(1)


@runs_app.command("init")
def runs_init_command(
    database: Path = typer.Option(DEFAULT_RUN_DB),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.run_store import initialize_database

    initialize_database(database)
    _emit({"database": str(database), "initialized": True}, json_output)


@trials_app.command("prepare-nuextract")
def prepare_nuextract_command(
    representation: str = typer.Option(...),
    manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "debug_50.jsonl", exists=True),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output: Path | None = typer.Option(None),
    representation_root: Path = typer.Option(DEFAULT_REPRESENTATION_ROOT),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.representations import (
        DERIVED_REPRESENTATIONS,
        GENERATED_REPRESENTATIONS,
        REPRESENTATION_ARTIFACTS,
        RepresentationName,
        build_trial_requests,
        write_trial_requests,
    )

    supported = set(REPRESENTATION_ARTIFACTS) | GENERATED_REPRESENTATIONS | DERIVED_REPRESENTATIONS
    if representation not in supported:
        raise typer.BadParameter(f"representation must be one of: {', '.join(sorted(supported))}")
    output_path = output or DEFAULT_TRIAL_ROOT / "nuextract" / representation / "requests.jsonl"
    requests = build_trial_requests(
        manifest,
        dataset_root,
        cast(RepresentationName, representation),
        representation_root,
    )
    write_trial_requests(output_path, requests)
    _emit(
        {
            "representation": representation,
            "manifest": str(manifest),
            "output": str(output_path),
            "request_count": len(requests),
        },
        json_output,
    )


@trials_app.command("ingest-nuextract")
def ingest_nuextract_command(
    representation: str = typer.Option(...),
    requests: Path = typer.Option(..., exists=True, dir_okay=False),
    responses: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path | None = typer.Option(None),
    run_db: Path = typer.Option(DEFAULT_RUN_DB),
    manifest_hash: str | None = typer.Option(None),
    repair_work_records: bool = typer.Option(
        False, "--repair-work-records/--no-repair-work-records"
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.trials import ingest_nuextract_results, read_rows

    target_dir = output_dir or DEFAULT_TRIAL_ROOT / "nuextract" / representation
    summary = ingest_nuextract_results(
        read_rows(requests),
        read_rows(responses),
        representation=representation,
        output_dir=target_dir,
        run_db=run_db,
        manifest_hash=manifest_hash,
        repair_work_records=repair_work_records,
    )
    _emit(summary, json_output)


@trials_app.command("ingest-qwen3")
def ingest_qwen3_command(
    representation: str = typer.Option(...),
    requests: Path = typer.Option(..., exists=True, dir_okay=False),
    responses: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path | None = typer.Option(None),
    run_db: Path = typer.Option(DEFAULT_RUN_DB),
    manifest_hash: str | None = typer.Option(None),
    repair_work_records: bool = typer.Option(
        False, "--repair-work-records/--no-repair-work-records"
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.constants import QWEN3_MODEL_ID, QWEN3_REVISION
    from eraparse.trials import ingest_nuextract_results, read_rows

    target_dir = output_dir or DEFAULT_TRIAL_ROOT / "qwen3" / representation
    summary = ingest_nuextract_results(
        read_rows(requests),
        read_rows(responses),
        representation=representation,
        output_dir=target_dir,
        run_db=run_db,
        manifest_hash=manifest_hash,
        model_id=QWEN3_MODEL_ID,
        revision=QWEN3_REVISION,
        run_kind="mapper_comparison",
        repair_work_records=repair_work_records,
    )
    _emit(summary, json_output)


@trials_app.command("ingest-mapper")
def ingest_mapper_command(
    model_id: str = typer.Option(...),
    revision: str = typer.Option(...),
    representation: str = typer.Option(...),
    requests: Path = typer.Option(..., exists=True, dir_okay=False),
    responses: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path | None = typer.Option(None),
    run_db: Path = typer.Option(DEFAULT_RUN_DB),
    manifest_hash: str | None = typer.Option(None),
    repair_work_records: bool = typer.Option(
        True, "--repair-work-records/--no-repair-work-records"
    ),
    allow_partial: bool = typer.Option(False, "--allow-partial/--require-complete"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.trials import ingest_nuextract_results, read_rows

    model_slug = model_id.rsplit("/", maxsplit=1)[-1].lower()
    target_dir = output_dir or DEFAULT_TRIAL_ROOT / "mappers" / model_slug / representation
    summary = ingest_nuextract_results(
        read_rows(requests),
        read_rows(responses),
        representation=representation,
        output_dir=target_dir,
        run_db=run_db,
        manifest_hash=manifest_hash,
        model_id=model_id,
        revision=revision,
        run_kind="mapper_scaling_comparison",
        repair_work_records=repair_work_records,
        allow_partial=allow_partial,
    )
    _emit(summary, json_output)


@trials_app.command("prepare-nuextract3")
def prepare_nuextract3_command(
    manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "debug_50.jsonl", exists=True),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output: Path | None = typer.Option(None),
    compact_schema: bool = typer.Option(False, "--compact-schema/--full-schema"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.donut import build_visual_trial_requests, write_visual_trial_requests

    output_path = output or DEFAULT_TRIAL_ROOT / "nuextract3" / f"{manifest.stem}.jsonl"
    requests = build_visual_trial_requests(
        manifest,
        dataset_root,
        model_family="nuextract3",
        compact_schema=compact_schema,
    )
    write_visual_trial_requests(output_path, requests)
    _emit(
        {
            "manifest": str(manifest),
            "output": str(output_path),
            "request_count": len(requests),
            "model_id": NUEXTRACT3_MODEL_ID,
            "revision": NUEXTRACT3_REVISION,
            "compact_schema": compact_schema,
        },
        json_output,
    )


@trials_app.command("ingest-nuextract3")
def ingest_nuextract3_command(
    requests: Path = typer.Option(..., exists=True, dir_okay=False),
    responses: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path | None = typer.Option(None),
    run_db: Path = typer.Option(DEFAULT_RUN_DB),
    manifest_hash: str | None = typer.Option(None),
    model_id: str = typer.Option(NUEXTRACT3_MODEL_ID),
    revision: str = typer.Option(NUEXTRACT3_REVISION),
    repair_work_records: bool = typer.Option(
        False, "--repair-work-records/--no-repair-work-records"
    ),
    compact_schema: bool = typer.Option(False, "--compact-schema/--full-schema"),
    allow_partial: bool = typer.Option(False, "--allow-partial/--require-complete"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.trials import ingest_nuextract_results, read_rows

    target_dir = output_dir or DEFAULT_TRIAL_ROOT / "nuextract3"
    summary = ingest_nuextract_results(
        read_rows(requests),
        read_rows(responses),
        representation="nuextract3_visual",
        output_dir=target_dir,
        run_db=run_db,
        manifest_hash=manifest_hash,
        model_id=model_id,
        revision=revision,
        run_kind="structured_vlm_upper_bound",
        repair_work_records=repair_work_records,
        compact_schema=compact_schema,
        allow_partial=allow_partial,
    )
    _emit(summary, json_output)


@trials_app.command("prepare-paddleocr-vl")
def prepare_paddleocr_vl_command(
    manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "debug_50.jsonl", exists=True),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output: Path | None = typer.Option(None),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.donut import build_visual_trial_requests, write_visual_trial_requests

    output_path = output or DEFAULT_TRIAL_ROOT / "paddleocr_vl" / f"{manifest.stem}.jsonl"
    requests = build_visual_trial_requests(
        manifest,
        dataset_root,
        model_family="paddleocr_vl",
    )
    write_visual_trial_requests(output_path, requests)
    _emit(
        {
            "manifest": str(manifest),
            "output": str(output_path),
            "request_count": len(requests),
            "model_id": PADDLEOCR_VL_MODEL_ID,
            "revision": PADDLEOCR_VL_REVISION,
        },
        json_output,
    )


@trials_app.command("materialize-paddleocr-vl")
def materialize_paddleocr_vl_command(
    responses: Path = typer.Option(..., exists=True, dir_okay=False),
    representation_root: Path = typer.Option(DEFAULT_REPRESENTATION_ROOT),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.representations import materialize_paddleocr_vl_outputs

    _emit(materialize_paddleocr_vl_outputs(responses, representation_root), json_output)


@trials_app.command("prepare-donut")
def prepare_donut_command(
    manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "debug_50.jsonl", exists=True),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output: Path | None = typer.Option(None),
    target_format: str = typer.Option("raw_json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.donut import build_donut_records, write_donut_records

    output_path = output or DEFAULT_TRIAL_ROOT / "donut" / f"{manifest.stem}.jsonl"
    if target_format not in {"raw_json", "native_tokens"}:
        raise typer.BadParameter("target-format must be raw_json or native_tokens")
    records = build_donut_records(manifest, dataset_root, target_format=cast(Any, target_format))
    write_donut_records(output_path, records)
    _emit(
        {
            "manifest": str(manifest),
            "output": str(output_path),
            "record_count": len(records),
            "page_count": sum(len(record["page_images"]) for record in records),
            "target_format": target_format,
        },
        json_output,
    )


@trials_app.command("ingest-donut")
def ingest_donut_command(
    records: Path = typer.Option(..., exists=True, dir_okay=False),
    responses: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(DEFAULT_TRIAL_ROOT / "donut" / "validation"),
    run_db: Path = typer.Option(DEFAULT_RUN_DB),
    manifest_hash: str | None = typer.Option(None),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.constants import DONUT_MODEL_ID, DONUT_REVISION
    from eraparse.trials import ingest_nuextract_results, read_rows

    summary = ingest_nuextract_results(
        read_rows(records),
        read_rows(responses),
        representation="donut_direct_visual",
        output_dir=output_dir,
        run_db=run_db,
        manifest_hash=manifest_hash,
        model_id=DONUT_MODEL_ID,
        revision=DONUT_REVISION,
        run_kind="direct_visual_baseline",
    )
    _emit(summary, json_output)


@trials_app.command("route-fields")
def route_fields_command(
    primary_results: Path = typer.Option(..., exists=True, dir_okay=False),
    specialist_results: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    policy: str = typer.Option("disagreement"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.router import run_selective_field_router
    from eraparse.trials import read_rows

    summary = run_selective_field_router(
        read_rows(primary_results),
        read_rows(specialist_results),
        output,
        policy=policy,
    )
    _emit(summary, json_output)


@trials_app.command("router-prepare-focused")
def router_prepare_focused_command(
    primary_results: Path = typer.Option(..., exists=True, dir_okay=False),
    mapper_requests: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.router import build_focused_specialist_requests
    from eraparse.trials import read_rows

    summary = build_focused_specialist_requests(
        read_rows(primary_results),
        read_rows(mapper_requests),
        output,
    )
    _emit(summary, json_output)


@trials_app.command("router-fuse-focused")
def router_fuse_focused_command(
    primary_results: Path = typer.Option(..., exists=True, dir_okay=False),
    focused_requests: Path = typer.Option(..., exists=True, dir_okay=False),
    specialist_responses: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.router import fuse_focused_specialist_responses
    from eraparse.trials import read_rows

    requests = read_rows(focused_requests)
    summary = fuse_focused_specialist_responses(
        read_rows(primary_results),
        read_rows(specialist_responses),
        {str(row["cv_id"]): list(row["routed_fields"]) for row in requests},
        output,
    )
    _emit(summary, json_output)


@parsers_app.command("generate")
def parser_generate_command(
    representation: str = typer.Option(...),
    manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "debug_50.jsonl", exists=True),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output_root: Path = typer.Option(DEFAULT_REPRESENTATION_ROOT),
    overwrite: bool = typer.Option(False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.parsers import ParserRepresentation, generate_representations

    supported = {
        "pymupdf4llm_markdown",
        "pymupdf4llm_json",
        "docling_markdown",
        "docling_json",
    }
    if representation not in supported:
        raise typer.BadParameter(f"representation must be one of: {', '.join(sorted(supported))}")
    summary = generate_representations(
        manifest,
        dataset_root,
        cast(ParserRepresentation, representation),
        output_root,
        overwrite=overwrite,
    )
    _emit(summary, json_output)


@evidence_app.command("build")
def evidence_build_command(
    reader: str = typer.Option("pymupdf4llm"),
    manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "debug_50.jsonl", exists=True),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    representation_root: Path = typer.Option(DEFAULT_REPRESENTATION_ROOT),
    output: Path | None = typer.Option(None),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.evidence import EvidenceReader, build_evidence_graphs

    if reader not in {"pymupdf4llm", "source_oracle"}:
        raise typer.BadParameter("reader must be pymupdf4llm or source_oracle")
    output_path = output or DEFAULT_EVIDENCE_ROOT / reader / f"{manifest.stem}.jsonl"
    summary = build_evidence_graphs(
        manifest,
        dataset_root,
        output_path,
        reader=cast(EvidenceReader, reader),
        representation_root=representation_root,
    )
    _emit(summary, json_output)
    if not summary["passed"]:
        raise typer.Exit(1)


@evidence_app.command("validate")
def evidence_validate_command(
    evidence: Path = typer.Option(..., exists=True, dir_okay=False),
    allow_oracle: bool = typer.Option(False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.evidence import read_evidence_graphs, validate_evidence_graphs

    summary = validate_evidence_graphs(
        read_evidence_graphs(evidence),
        allow_oracle=allow_oracle,
    )
    _emit(summary, json_output)
    if not summary["passed"]:
        raise typer.Exit(1)


@sge_app.command("prepare")
def sge_prepare_command(
    evidence: Path = typer.Option(..., exists=True, dir_okay=False),
    manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "debug_50.jsonl", exists=True),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output: Path | None = typer.Option(None),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import prepare_sge_records

    output_path = output or DEFAULT_SGE_ROOT / "records" / f"{manifest.stem}.jsonl"
    _emit(
        prepare_sge_records(evidence, manifest, dataset_root, output_path),
        json_output,
    )


@sge_app.command("assemble")
def sge_assemble_command(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "grounded_predictions.jsonl"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import assemble_candidate_rows
    from eraparse.trials import read_rows

    _emit(assemble_candidate_rows(read_rows(predictions), output), json_output)


@sge_app.command("evaluate")
def sge_evaluate_command(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    requests: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "evaluation.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import evaluate_grounded_rows

    _emit(evaluate_grounded_rows(predictions, requests, output), json_output)


@sge_app.command("estimate-t4")
def sge_estimate_t4_command(
    seconds: float = typer.Option(..., min=0),
    stage_spent: float = typer.Option(0.0, min=0),
    stage_budget: float = typer.Option(3.0, min=0),
    remaining_credit: float | None = typer.Option(None, min=0),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.budget import projected_t4_cost, require_budget

    projected = projected_t4_cost(seconds)
    require_budget(
        projected,
        spent=stage_spent,
        budget=stage_budget,
        remaining_credit=remaining_credit,
    )
    _emit(
        {
            "seconds": seconds,
            "projected_cost_with_safety_factor": projected,
            "stage_spent": stage_spent,
            "stage_budget": stage_budget,
            "remaining_credit": remaining_credit,
            "remaining_after_projection": stage_budget - stage_spent - projected,
        },
        json_output,
    )


@sge_app.command("local-smoke")
def sge_local_smoke_command(
    records: Path = typer.Option(..., exists=True, dir_okay=False),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output_dir: Path = typer.Option(...),
    mode: str = typer.Option("sge"),
    max_steps: int = typer.Option(1, min=0),
    max_records: int = typer.Option(1, min=1),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(20260609),
    unfreeze_final_layers: int = typer.Option(0, min=0),
    presence_weight: float = typer.Option(0.25, min=0),
    grouping_weight: float = typer.Option(0.5, min=0),
    evidence_weight: float = typer.Option(0.5, min=0),
    query_layers: int = typer.Option(2, min=1),
    decoder_strategy: str = typer.Option("auto"),
    evaluate_training: bool = typer.Option(True),
    evaluation_records: Path | None = typer.Option(None, exists=True, dir_okay=False),
    max_evaluation_records: int | None = typer.Option(None, min=1),
    checkpoint: Path | None = typer.Option(None),
    resume_checkpoint: Path | None = typer.Option(None, exists=True, dir_okay=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge_training import DecoderStrategy, TrainingMode, run_local_smoke

    if mode not in {"baseline", "sge"}:
        raise typer.BadParameter("mode must be baseline or sge")
    if decoder_strategy not in {"auto", "learned", "sequence"}:
        raise typer.BadParameter("decoder-strategy must be auto, learned, or sequence")
    _emit(
        run_local_smoke(
            records,
            dataset_root,
            output_dir,
            mode=cast(TrainingMode, mode),
            max_steps=max_steps,
            max_records=max_records,
            requested_device=device,
            seed=seed,
            unfreeze_final_layers=unfreeze_final_layers,
            presence_weight=presence_weight,
            grouping_weight=grouping_weight,
            evidence_weight=evidence_weight,
            query_layers=query_layers,
            decoder_strategy=cast(DecoderStrategy, decoder_strategy),
            evaluate_training=evaluate_training,
            evaluation_records_path=evaluation_records,
            max_evaluation_records=max_evaluation_records,
            checkpoint_path=checkpoint,
            resume_checkpoint=resume_checkpoint,
        ),
        json_output,
    )


@sge_app.command("select-overfit")
def sge_select_overfit_command(
    records: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
    document_count: int = typer.Option(10, min=1),
    min_coverage: float = typer.Option(0.95, min=0, max=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.io import atomic_write_json, atomic_write_jsonl, read_jsonl
    from eraparse.sge_selection import select_overfit_records

    selected, summary = select_overfit_records(
        list(read_jsonl(records)),
        document_count=document_count,
        min_coverage=min_coverage,
    )
    atomic_write_jsonl(output, [dict(row) for row in selected])
    atomic_write_json(output.with_suffix(".summary.json"), summary)
    _emit(summary, json_output)


@sge_app.command("report-trials")
def sge_report_trials_command(
    root: Path = typer.Option(DEFAULT_SGE_ROOT / "local_smokes", exists=True, file_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "trial_comparison.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge_reporting import write_trial_report

    _emit(write_trial_report(root, output), json_output)


@sge_app.command("oracle-ceiling")
def sge_oracle_ceiling_command(
    records: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(...),
    grouping_mode: str = typer.Option("oracle"),
    max_records: int | None = typer.Option(None, min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import evaluate_record_oracle

    if grouping_mode not in {"oracle", "sequence"}:
        raise typer.BadParameter("grouping-mode must be oracle or sequence")
    _emit(
        evaluate_record_oracle(
            records,
            output_dir,
            grouping_mode=grouping_mode,
            max_records=max_records,
        ),
        json_output,
    )


@sge_app.command("prepare-work-bank")
def sge_prepare_work_bank_command(
    records: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
    max_records: int | None = typer.Option(None, min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import prepare_work_record_bank

    _emit(
        prepare_work_record_bank(records, output, max_records=max_records),
        json_output,
    )


@sge_app.command("repair-work")
def sge_repair_work_command(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "repaired_work_predictions.jsonl"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import repair_grounded_work_predictions

    _emit(repair_grounded_work_predictions(predictions, output), json_output)


@sge_app.command("efsfr-repair")
def sge_efsfr_repair_command(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "efsfr_repaired_predictions.jsonl"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import apply_efsfr_repairs

    _emit(apply_efsfr_repairs(predictions, output), json_output)


@sge_app.command("sgrse-work")
def sge_sgrse_work_command(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "sgrse_work_predictions.jsonl"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import apply_sgrse_work_decoder

    _emit(apply_sgrse_work_decoder(predictions, output), json_output)


@sge_app.command("compare")
def sge_compare_command(
    left_predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    right_predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    requests: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "comparison.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import compare_prediction_sets

    _emit(
        compare_prediction_sets(left_predictions, right_predictions, requests, output),
        json_output,
    )


@sge_app.command("select-sgrse-work")
def sge_select_sgrse_work_command(
    baseline_predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    sgrse_predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "selected_sgrse_work_predictions.jsonl"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import apply_selective_sgrse_work_decoder

    _emit(
        apply_selective_sgrse_work_decoder(
            baseline_predictions,
            sgrse_predictions,
            output,
        ),
        json_output,
    )


@sge_app.command("repair-project-tech")
def sge_repair_project_tech_command(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "project_tech_repaired_predictions.jsonl"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import apply_project_skill_tech_repairs

    _emit(apply_project_skill_tech_repairs(predictions, output), json_output)


@sge_app.command("repair-project-url")
def sge_repair_project_url_command(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    train_records: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(DEFAULT_SGE_ROOT / "project_url_repaired_predictions.jsonl"),
    threshold: float = typer.Option(0.45, min=0, max=1),
    min_full_count: int = typer.Option(8, min=1),
    min_reduced_count: int = typer.Option(8, min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.sge import apply_project_url_repairs

    _emit(
        apply_project_url_repairs(
            predictions,
            train_records,
            output,
            threshold=threshold,
            min_full_count=min_full_count,
            min_reduced_count=min_reduced_count,
        ),
        json_output,
    )


@ats_app.command("run-baseline")
def ats_run_baseline_command(
    train_manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "train.jsonl", exists=True),
    id_manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "id_test.jsonl", exists=True),
    ood_manifest: Path = typer.Option(
        DEFAULT_MANIFEST_ROOT / "template_ood_test.jsonl", exists=True
    ),
    dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, exists=True, file_okay=False),
    output_root: Path = typer.Option(DEFAULT_ATS_ROOT),
    run_db: Path = typer.Option(DEFAULT_RUN_DB),
    skills_per_profile: int = typer.Option(12, min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.ats import run_baseline

    summary = run_baseline(
        train_manifest=train_manifest,
        test_manifests=[id_manifest, ood_manifest],
        dataset_root=dataset_root,
        output_root=output_root,
        run_db=run_db,
        skills_per_profile=skills_per_profile,
    )
    _emit(summary, json_output)


@ats_app.command("compare-predictions")
def ats_compare_predictions_command(
    profiles: Path = typer.Option(
        Path("configs/ats/domain_job_profiles_v1.json"), exists=True, dir_okay=False
    ),
    id_manifest: Path = typer.Option(DEFAULT_MANIFEST_ROOT / "id_test.jsonl", exists=True),
    id_results: Path = typer.Option(..., exists=True, dir_okay=False),
    ood_manifest: Path = typer.Option(
        DEFAULT_MANIFEST_ROOT / "template_ood_test.jsonl", exists=True
    ),
    ood_results: Path = typer.Option(..., exists=True, dir_okay=False),
    lane: str = typer.Option(...),
    output_root: Path = typer.Option(DEFAULT_ATS_ROOT),
    run_db: Path = typer.Option(DEFAULT_RUN_DB),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from eraparse.ats import run_prediction_comparison

    summary = run_prediction_comparison(
        profiles_path=profiles,
        result_manifests=[(id_manifest, id_results), (ood_manifest, ood_results)],
        lane=lane,
        output_root=output_root,
        run_db=run_db,
    )
    _emit(summary, json_output)


@app.callback()
def main() -> None:
    """EraParse command line interface."""


if __name__ == "__main__":
    app()
