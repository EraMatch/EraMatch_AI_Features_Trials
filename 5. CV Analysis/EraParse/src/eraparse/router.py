import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eraparse.io import atomic_write_jsonl

ROUTED_FIELDS = ("work_experience", "certifications")
FOCUSED_SCHEMAS = {
    "work_experience": [
        {
            "job_title": "",
            "company": "",
            "start_date": "",
            "end_date": "",
            "duration": "",
        }
    ],
    "certifications": [{"name": "", "issuer": "", "date": ""}],
}


def _field_supported(row: Mapping[str, Any], field: str) -> bool:
    field_results = row.get("evaluation", {}).get("field_results", [])
    result = next((item for item in field_results if item.get("path") == field), None)
    return bool(result.get("supported", False)) if result is not None else False


def observable_field_features(
    primary: Mapping[str, Any],
    specialist: Mapping[str, Any],
    field: str,
) -> dict[str, bool | int]:
    primary_value = primary["prediction"].get(field) or []
    specialist_value = specialist["prediction"].get(field) or []
    return {
        "disagrees": primary_value != specialist_value,
        "primary_count": len(primary_value),
        "specialist_count": len(specialist_value),
        "primary_supported": _field_supported(primary, field),
    }


def primary_only_field_features(
    primary: Mapping[str, Any],
    field: str,
) -> dict[str, bool | int]:
    primary_value = primary["prediction"].get(field) or []
    return {
        "primary_count": len(primary_value),
        "primary_characters": len(json.dumps(primary_value, sort_keys=True)),
        "primary_supported": _field_supported(primary, field),
    }


def route_specialist(field: str, features: Mapping[str, bool | int]) -> bool:
    disagrees = bool(features["disagrees"])
    primary_count = int(features["primary_count"])
    specialist_count = int(features["specialist_count"])
    if field == "work_experience":
        return disagrees and primary_count == specialist_count
    if field == "certifications":
        return disagrees and primary_count > 0 and primary_count != specialist_count
    raise ValueError(f"unsupported routed field: {field}")


def route_primary_only_specialist(field: str, features: Mapping[str, bool | int]) -> bool:
    primary_count = int(features["primary_count"])
    primary_characters = int(features["primary_characters"])
    primary_supported = bool(features["primary_supported"])
    if field == "work_experience":
        return primary_supported and primary_characters <= 383
    if field == "certifications":
        return primary_supported and primary_count > 0
    raise ValueError(f"unsupported routed field: {field}")


def run_selective_field_router(
    primary_rows: Sequence[Mapping[str, Any]],
    specialist_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    policy: str = "disagreement",
) -> dict[str, Any]:
    specialist_by_id = {str(row["cv_id"]): row for row in specialist_rows}
    if set(specialist_by_id) != {str(row["cv_id"]) for row in primary_rows}:
        raise ValueError("primary and specialist result IDs must match")

    routed_counts: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for primary in primary_rows:
        cv_id = str(primary["cv_id"])
        specialist = specialist_by_id[cv_id]
        prediction = dict(primary["prediction"])
        routed_fields = []
        features_by_field = {}
        for field in ROUTED_FIELDS:
            features = (
                observable_field_features(primary, specialist, field)
                if policy == "disagreement"
                else primary_only_field_features(primary, field)
            )
            features_by_field[field] = features
            should_route = (
                route_specialist(field, features)
                if policy == "disagreement"
                else route_primary_only_specialist(field, features)
            )
            if should_route:
                prediction[field] = specialist["prediction"][field]
                routed_fields.append(field)
                routed_counts[field] += 1

        primary_latency = float(primary.get("latency_seconds") or 0.0)
        specialist_latency = (
            float(specialist.get("latency_seconds") or 0.0) if routed_fields else 0.0
        )
        output_rows.append(
            {
                "cv_id": cv_id,
                "raw_output": json.dumps(prediction, separators=(",", ":")),
                "latency_seconds": primary_latency + specialist_latency,
                "routed_fields": routed_fields,
                "router_features": features_by_field,
            }
        )

    atomic_write_jsonl(output_path, output_rows)
    escalated_documents = sum(bool(row["routed_fields"]) for row in output_rows)
    return {
        "document_count": len(output_rows),
        "policy": policy,
        "escalated_documents": escalated_documents,
        "escalation_rate": escalated_documents / len(output_rows) if output_rows else 0.0,
        "routed_field_counts": dict(sorted(routed_counts.items())),
        "mean_projected_latency_seconds": (
            sum(float(row["latency_seconds"]) for row in output_rows) / len(output_rows)
            if output_rows
            else 0.0
        ),
        "output_path": str(output_path),
    }


def build_focused_specialist_requests(
    primary_rows: Sequence[Mapping[str, Any]],
    mapper_requests: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    requests_by_id = {str(row["cv_id"]): row for row in mapper_requests}
    output_rows = []
    routed_counts: Counter[str] = Counter()
    for primary in primary_rows:
        cv_id = str(primary["cv_id"])
        request = requests_by_id[cv_id]
        routed_fields = [
            field
            for field in ROUTED_FIELDS
            if route_primary_only_specialist(field, primary_only_field_features(primary, field))
        ]
        if not routed_fields:
            continue
        routed_counts.update(routed_fields)
        output_rows.append(
            {
                "cv_id": cv_id,
                "text": request["text"],
                "parser_id": request.get("parser_id", request.get("representation")),
                "schema": {field: FOCUSED_SCHEMAS[field] for field in routed_fields},
                "routed_fields": routed_fields,
            }
        )
    atomic_write_jsonl(output_path, output_rows)
    return {
        "document_count": len(primary_rows),
        "request_count": len(output_rows),
        "request_rate": len(output_rows) / len(primary_rows) if primary_rows else 0.0,
        "routed_field_counts": dict(sorted(routed_counts.items())),
        "output_path": str(output_path),
    }


def fuse_focused_specialist_responses(
    primary_rows: Sequence[Mapping[str, Any]],
    specialist_responses: Sequence[Mapping[str, Any]],
    routed_fields_by_id: Mapping[str, Sequence[str]],
    output_path: Path,
) -> dict[str, Any]:
    from eraparse.trials import parse_generated_json

    responses_by_id = {str(row["cv_id"]): row for row in specialist_responses}
    output_rows = []
    parse_error_count = 0
    for primary in primary_rows:
        cv_id = str(primary["cv_id"])
        prediction = dict(primary["prediction"])
        routed_fields = list(routed_fields_by_id.get(cv_id, []))
        response = responses_by_id.get(cv_id)
        parse_error = None
        if response is not None:
            partial_prediction, parse_error = parse_generated_json(str(response["raw_output"]))
            if partial_prediction is not None:
                for field in routed_fields:
                    if field in partial_prediction:
                        prediction[field] = partial_prediction[field]
        if parse_error is not None:
            parse_error_count += 1
        output_rows.append(
            {
                "cv_id": cv_id,
                "raw_output": json.dumps(prediction, separators=(",", ":")),
                "latency_seconds": float(primary.get("latency_seconds") or 0.0)
                + float(response.get("latency_seconds") or 0.0)
                if response is not None
                else float(primary.get("latency_seconds") or 0.0),
                "routed_fields": routed_fields,
                "specialist_parse_error": parse_error,
            }
        )
    atomic_write_jsonl(output_path, output_rows)
    return {
        "document_count": len(output_rows),
        "specialist_response_count": len(responses_by_id),
        "parse_error_count": parse_error_count,
        "output_path": str(output_path),
    }
