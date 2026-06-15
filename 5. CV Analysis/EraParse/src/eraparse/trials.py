import json
import platform
import re
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eraparse.constants import DEFAULT_RUN_DB, NUEXTRACT_MODEL_ID, NUEXTRACT_REVISION, SEED
from eraparse.evaluate import aggregate_evaluations, evaluate_document
from eraparse.io import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file
from eraparse.models import EvidenceBundle, RunProvenance, RunRecord, ValidationEvent
from eraparse.run_store import insert_artifact, insert_run, insert_sample_result
from eraparse.sge import repair_work_record


def chunk_pending_requests(
    requests: Sequence[Mapping[str, Any]],
    completed_ids: set[str],
    *,
    chunk_size: int,
) -> list[list[Mapping[str, Any]]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    pending = [request for request in requests if str(request["cv_id"]) not in completed_ids]
    return [pending[start : start + chunk_size] for start in range(0, len(pending), chunk_size)]


def parse_generated_json(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_output.strip()
    if "<|output|>" in text:
        text = text.split("<|output|>", maxsplit=1)[1].strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        repaired_value = _parse_with_trailing_field_trim(text)
        if repaired_value is not None:
            return repaired_value, None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None, str(error)
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as inner_error:
            return None, str(inner_error)
    if not isinstance(value, dict):
        return None, "generated JSON must be an object"
    return value, None


def _parse_with_trailing_field_trim(text: str) -> dict[str, Any] | None:
    required_markers = {"full_name", "email", "location", "phone", "summary"}
    markers = [match.start() for match in re.finditer(r',\s*"', text)]
    for marker in reversed(markers):
        for offset in (0, -1):
            cutoff = marker + offset
            if cutoff <= 0:
                continue
            candidate = text[:cutoff] + "}"
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and required_markers & set(value):
                return value
    return None


def current_git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def repair_prediction_work_records(
    prediction: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[ValidationEvent]]:
    if prediction is None:
        return None, []
    repaired_prediction: dict[str, Any] = {}
    repair_events: list[ValidationEvent] = []

    known_top_level = {
        "full_name",
        "email",
        "location",
        "phone",
        "linkedin_url",
        "github_url",
        "summary",
        "skills",
        "work_experience",
        "education",
        "projects",
        "certifications",
    }
    for extra_key in sorted(set(prediction) - known_top_level):
        repair_events.append(
            ValidationEvent(
                kind="extra_key_removed",
                path=extra_key,
                message=f"removed unsupported top-level key: {extra_key}",
            )
        )

    def coerce_required_string(path: str, value: Any) -> str:
        if isinstance(value, str):
            return value
        repair_events.append(
            ValidationEvent(
                kind="required_string_coerced",
                path=path,
                message="coerced required string field to schema-compatible text",
            )
        )
        return "" if value is None else str(value)

    def coerce_optional_string(path: str, value: Any) -> str | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return value
        repair_events.append(
            ValidationEvent(
                kind="optional_string_coerced",
                path=path,
                message="coerced optional string field to schema-compatible text",
            )
        )
        return str(value)

    def coerce_string_list(path: str, value: Any, *, optional: bool) -> list[str] | None:
        if value is None:
            return None if optional else []
        if not isinstance(value, list):
            repair_events.append(
                ValidationEvent(
                    kind="list_coerced",
                    path=path,
                    message="replaced non-list field with schema-compatible list",
                )
            )
            return None if optional else []
        normalized = []
        for item in value:
            cleaned = str(item).strip() if item is not None else ""
            if cleaned:
                normalized.append(cleaned)
        return normalized

    repaired_prediction["full_name"] = coerce_required_string(
        "full_name", prediction.get("full_name")
    )
    repaired_prediction["email"] = coerce_required_string("email", prediction.get("email"))
    repaired_prediction["location"] = coerce_required_string(
        "location", prediction.get("location")
    )
    repaired_prediction["phone"] = coerce_required_string("phone", prediction.get("phone"))
    repaired_prediction["linkedin_url"] = coerce_optional_string(
        "linkedin_url", prediction.get("linkedin_url")
    )
    repaired_prediction["github_url"] = coerce_optional_string(
        "github_url", prediction.get("github_url")
    )
    repaired_prediction["summary"] = coerce_required_string("summary", prediction.get("summary"))
    repaired_prediction["skills"] = coerce_string_list(
        "skills", prediction.get("skills"), optional=False
    )

    raw_work = prediction.get("work_experience")
    if not isinstance(raw_work, list):
        repair_events.append(
            ValidationEvent(
                kind="list_coerced",
                path="work_experience",
                message="replaced non-list work_experience with an empty list",
            )
        )
        raw_work = []
    repaired_work: list[dict[str, str]] = []
    for record_index, record in enumerate(raw_work):
        if not isinstance(record, Mapping):
            repaired_work.append(
                {
                    "job_title": "",
                    "company": "",
                    "start_date": "",
                    "end_date": "",
                    "duration": "",
                }
            )
            repair_events.append(
                ValidationEvent(
                    kind="work_record_repaired",
                    path=f"work_experience.{record_index}",
                    message="replaced non-object work record with empty required-string record",
                )
            )
            continue
        sanitized_record = {
            "job_title": record.get("job_title") or "",
            "company": record.get("company") or "",
            "start_date": record.get("start_date") or "",
            "end_date": record.get("end_date") or "",
            "duration": record.get("duration") or "",
        }
        repaired_record, events = repair_work_record(sanitized_record)
        repaired_work.append(repaired_record)
        for event in events:
            updated_path = event.path.replace("*", str(record_index)) if event.path else None
            repair_events.append(
                event.model_copy(update={"path": updated_path})
            )
    repaired_prediction["work_experience"] = repaired_work

    raw_education = prediction.get("education")
    if not isinstance(raw_education, list):
        repair_events.append(
            ValidationEvent(
                kind="list_coerced",
                path="education",
                message="replaced non-list education with an empty list",
            )
        )
        raw_education = []
    repaired_prediction["education"] = [
        {
            "degree": coerce_required_string(f"education.{index}.degree", record.get("degree")),
            "field_of_study": coerce_required_string(
                f"education.{index}.field_of_study", record.get("field_of_study")
            ),
            "institution": coerce_required_string(
                f"education.{index}.institution", record.get("institution")
            ),
            "graduation_date": coerce_required_string(
                f"education.{index}.graduation_date", record.get("graduation_date")
            ),
        }
        for index, record in enumerate(item for item in raw_education if isinstance(item, Mapping))
    ]

    raw_projects = prediction.get("projects")
    if raw_projects is None:
        repaired_prediction["projects"] = None
    elif not isinstance(raw_projects, list):
        repair_events.append(
            ValidationEvent(
                kind="list_coerced",
                path="projects",
                message="replaced non-list projects with null",
            )
        )
        repaired_prediction["projects"] = None
    else:
        repaired_prediction["projects"] = [
            {
                "name": coerce_required_string(f"projects.{index}.name", record.get("name")),
                "technologies": coerce_string_list(
                    f"projects.{index}.technologies",
                    record.get("technologies"),
                    optional=False,
                ),
                "url": coerce_optional_string(f"projects.{index}.url", record.get("url")),
            }
            for index, record in enumerate(
                item for item in raw_projects if isinstance(item, Mapping)
            )
        ]

    raw_certifications = prediction.get("certifications")
    if raw_certifications is None:
        repaired_prediction["certifications"] = None
    elif not isinstance(raw_certifications, list):
        repair_events.append(
            ValidationEvent(
                kind="list_coerced",
                path="certifications",
                message="replaced non-list certifications with null",
            )
        )
        repaired_prediction["certifications"] = None
    else:
        repaired_prediction["certifications"] = [
            {
                "name": coerce_required_string(f"certifications.{index}.name", record.get("name")),
                "issuer": coerce_required_string(
                    f"certifications.{index}.issuer", record.get("issuer")
                ),
                "date": coerce_required_string(f"certifications.{index}.date", record.get("date")),
            }
            for index, record in enumerate(
                item for item in raw_certifications if isinstance(item, Mapping)
            )
        ]
    return repaired_prediction, repair_events


def ingest_nuextract_results(
    requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    *,
    representation: str,
    output_dir: Path,
    run_db: Path = DEFAULT_RUN_DB,
    manifest_hash: str | None = None,
    model_id: str = NUEXTRACT_MODEL_ID,
    revision: str = NUEXTRACT_REVISION,
    run_kind: str = "nuextract_input_ablation",
    repair_work_records: bool = False,
    compact_schema: bool = False,
    allow_partial: bool = False,
) -> dict[str, Any]:
    from eraparse.compact_schema import compact_to_reduced

    responses_by_id = {str(response["cv_id"]): response for response in responses}
    if not allow_partial and len(responses_by_id) != len(requests):
        raise ValueError("response count or IDs do not match requests")
    requests = [request for request in requests if str(request["cv_id"]) in responses_by_id]
    if not requests:
        raise ValueError("no request IDs match responses")

    model_slug = model_id.rsplit("/", maxsplit=1)[-1].lower()
    run_id = f"{model_slug}-{representation}-{uuid.uuid4().hex[:10]}"
    run_output_dir = output_dir / run_id
    started_at = datetime.now(UTC)
    run = RunRecord(
        run_id=run_id,
        kind=run_kind,
        status="running",
        provenance=RunProvenance(
            code_revision=current_git_revision(),
            manifest_hash=manifest_hash,
            environment={"platform": platform.platform()},
            seed=SEED,
            resolved_config={
                "model_id": model_id,
                "revision": revision,
                "representation": representation,
                "compact_schema": compact_schema,
                "allow_partial": allow_partial,
            },
        ),
        model_id=model_id,
        parser_id=representation,
        started_at=started_at,
    )
    insert_run(run_db, run)

    output_rows: list[dict[str, Any]] = []
    evaluations = []
    for request in requests:
        cv_id = str(request["cv_id"])
        response = responses_by_id[cv_id]
        raw_output = str(response["raw_output"])
        raw_prediction, parse_error = parse_generated_json(raw_output)
        prediction = (
            compact_to_reduced(raw_prediction)
            if compact_schema and raw_prediction is not None
            else raw_prediction
        )
        repair_events: list[ValidationEvent] = []
        if repair_work_records:
            prediction, repair_events = repair_prediction_work_records(raw_prediction)
        evidence = EvidenceBundle(
            parser_text=str(request.get("text", request.get("evidence_text", "")))
        )
        evaluation = evaluate_document(
            request["truth"],
            prediction,
            evidence,
            json_valid=parse_error is None,
            parse_error=parse_error,
        )
        if repair_events:
            evaluation = evaluation.model_copy(
                update={"validation_events": evaluation.validation_events + repair_events}
            )
        evaluations.append(evaluation)
        insert_sample_result(
            run_db,
            run_id=run_id,
            cv_id=cv_id,
            split=str(request["split"]),
            result=evaluation,
        )
        output_rows.append(
            {
                "cv_id": cv_id,
                "representation": representation,
                "model_id": model_id,
                "revision": revision,
                "raw_output": raw_output,
                "prediction_raw": raw_prediction,
                "prediction": prediction,
                "parse_error": parse_error,
                "repair_events": [event.model_dump(mode="json") for event in repair_events],
                "latency_seconds": response.get("latency_seconds"),
                "input_tokens": response.get("input_tokens"),
                "output_tokens": response.get("output_tokens"),
                "visual_tokens": response.get("visual_tokens"),
                "encoder_latency_seconds": response.get("encoder_latency_seconds"),
                "decoder_latency_seconds": response.get("decoder_latency_seconds"),
                "generated_sequence": response.get("generated_sequence"),
                "native_decode_error": response.get("native_decode_error"),
                "evaluation": evaluation.model_dump(mode="json"),
            }
        )

    run_output_dir.mkdir(parents=True, exist_ok=False)
    results_path = run_output_dir / "results.jsonl"
    summary_path = run_output_dir / "summary.json"
    atomic_write_jsonl(results_path, output_rows)
    aggregate = aggregate_evaluations(evaluations)
    summary = {
        "run_id": run_id,
        "representation": representation,
        "model_id": model_id,
        "revision": revision,
        "repair_work_records": repair_work_records,
        "compact_schema": compact_schema,
        "allow_partial": allow_partial,
        "aggregate": aggregate.model_dump(mode="json"),
        "results_path": str(results_path),
        "mean_latency_seconds": sum(float(row["latency_seconds"] or 0) for row in output_rows)
        / len(output_rows),
    }
    atomic_write_json(summary_path, summary)
    insert_artifact(
        run_db,
        run_id=run_id,
        kind="results",
        artifact_path=str(results_path),
        sha256=sha256_file(results_path),
    )
    insert_artifact(
        run_db,
        run_id=run_id,
        kind="summary",
        artifact_path=str(summary_path),
        sha256=sha256_file(summary_path),
    )
    insert_run(
        run_db,
        run.model_copy(update={"status": "completed", "completed_at": datetime.now(UTC)}),
    )
    return summary


def read_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
