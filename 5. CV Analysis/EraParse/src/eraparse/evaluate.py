import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError
from rapidfuzz.fuzz import ratio
from scipy.optimize import linear_sum_assignment

from eraparse.io import read_json
from eraparse.models import (
    AggregateEvaluation,
    DocumentEvaluation,
    EvidenceBundle,
    FieldResult,
    ReducedCVTarget,
    ValidationEvent,
)

FUZZY_THRESHOLD = 0.5
TOKEN_SUPPORT_THRESHOLD = 0.5


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_email(value: Any) -> str:
    return str(value or "").strip().casefold()


def normalize_phone(value: Any) -> str:
    text = str(value or "").casefold()
    extension = ""
    match = re.search(r"(?:ext\.?|extension|x)\s*(\d+)\s*$", text)
    if match:
        extension = f"x{match.group(1)}"
        text = text[: match.start()]
    return re.sub(r"\D", "", text) + extension


def normalize_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"^(https?):\s*/\s*/", r"\1://", raw, flags=re.IGNORECASE)
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.casefold()
    host = (parsed.hostname or "").casefold()
    if host.startswith("www."):
        host = host[4:]
    try:
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        port = ""
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    normalized = urlunsplit(("", host + port, path, parsed.query, ""))
    return normalized[2:] if normalized.startswith("//") else normalized


def normalize_date(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    if text in {"present", "current", "now"}:
        return "present"
    return re.sub(r"\s+", "", text)


def exact_score(
    truth: Any, prediction: Any, normalizer: Callable[[Any], str] = normalize_text
) -> float:
    return float(normalizer(truth) == normalizer(prediction))


def anls(truth: Any, prediction: Any) -> float:
    truth_text = normalize_text(truth)
    prediction_text = normalize_text(prediction)
    if not truth_text and not prediction_text:
        return 1.0
    if not truth_text or not prediction_text:
        return 0.0
    score = ratio(truth_text, prediction_text) / 100.0
    return score if score >= FUZZY_THRESHOLD else 0.0


def set_metrics(
    truth: Iterable[Any], prediction: Iterable[Any]
) -> tuple[float, float, float, float]:
    truth_set = {normalize_text(value) for value in truth if normalize_text(value)}
    prediction_set = {normalize_text(value) for value in prediction if normalize_text(value)}
    intersection = len(truth_set & prediction_set)
    precision = intersection / len(prediction_set) if prediction_set else float(not truth_set)
    recall = intersection / len(truth_set) if truth_set else float(not prediction_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(truth_set | prediction_set)
    jaccard = intersection / union if union else 1.0
    return precision, recall, f1, jaccard


NESTED_COMPONENTS: dict[str, tuple[str, ...]] = {
    "work_experience": ("job_title", "company", "start_date", "end_date", "duration"),
    "education": ("degree", "field_of_study", "institution", "graduation_date"),
    "projects": ("name", "technologies", "url"),
    "certifications": ("name", "issuer", "date"),
}


def _component_score(field: str, truth: Any, prediction: Any) -> float:
    if field == "technologies":
        return set_metrics(truth or [], prediction or [])[2]
    if field == "url":
        return exact_score(truth, prediction, normalize_url)
    if "date" in field or field == "duration":
        return exact_score(truth, prediction, normalize_date)
    return anls(truth, prediction)


def object_similarity(
    truth: Mapping[str, Any], prediction: Mapping[str, Any], fields: Sequence[str]
) -> float:
    return sum(
        _component_score(field, truth.get(field), prediction.get(field)) for field in fields
    ) / len(fields)


def hungarian_match(
    truth: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> tuple[float, float, float]:
    if not truth and not prediction:
        return 1.0, 1.0, 1.0
    if not truth or not prediction:
        return 0.0, 0.0, 0.0
    similarities = [
        [object_similarity(truth_item, prediction_item, fields) for prediction_item in prediction]
        for truth_item in truth
    ]
    truth_indices, prediction_indices = linear_sum_assignment(similarities, maximize=True)
    accepted = [
        similarities[truth_index][prediction_index]
        for truth_index, prediction_index in zip(truth_indices, prediction_indices, strict=True)
        if similarities[truth_index][prediction_index] >= FUZZY_THRESHOLD
    ]
    true_positives = len(accepted)
    precision = true_positives / len(prediction)
    recall = true_positives / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    quality = sum(accepted) / true_positives if true_positives else 0.0
    return precision, recall, f1 * quality


def _evidence_text(bundle: EvidenceBundle | None) -> str:
    if bundle is None:
        return ""
    return "\n".join(
        text
        for text in (bundle.parser_text, bundle.canonical_text, bundle.ocr_text)
        if text is not None
    )


def value_supported(value: Any, evidence_text: str) -> bool | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, Mapping):
        supports = [value_supported(item, evidence_text) for item in value.values()]
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        supports = [value_supported(item, evidence_text) for item in value]
    else:
        normalized_value = normalize_text(value)
        normalized_evidence = normalize_text(evidence_text)
        if not normalized_value:
            return None
        if normalized_value in normalized_evidence:
            return True
        value_tokens = set(normalized_value.split())
        evidence_tokens = set(normalized_evidence.split())
        return len(value_tokens & evidence_tokens) / len(value_tokens) >= TOKEN_SUPPORT_THRESHOLD
    determinate = [support for support in supports if support is not None]
    return all(determinate) if determinate else None


def _field_result(
    path: str,
    truth: Any,
    prediction: Any,
    evidence_text: str,
) -> FieldResult:
    if path == "email":
        score = exact_score(truth, prediction, normalize_email)
        metric = "exact_email"
    elif path == "phone":
        score = exact_score(truth, prediction, normalize_phone)
        metric = "exact_phone"
    elif path.endswith("_url"):
        score = exact_score(truth, prediction, normalize_url)
        metric = "exact_url"
    elif path == "skills":
        precision, recall, f1, jaccard = set_metrics(truth or [], prediction or [])
        return FieldResult(
            path=path,
            metric="set_f1",
            score=f1,
            precision=precision,
            recall=recall,
            f1=f1,
            jaccard=jaccard,
            supported=value_supported(prediction, evidence_text),
            truth=truth,
            prediction=prediction,
        )
    elif path in NESTED_COMPONENTS:
        truth_items = [item for item in (truth or []) if isinstance(item, Mapping)]
        prediction_items = [item for item in (prediction or []) if isinstance(item, Mapping)]
        precision, recall, score = hungarian_match(
            truth_items, prediction_items, NESTED_COMPONENTS[path]
        )
        return FieldResult(
            path=path,
            metric="hungarian_f1_quality",
            score=score,
            precision=precision,
            recall=recall,
            f1=score,
            supported=value_supported(prediction, evidence_text),
            truth=truth,
            prediction=prediction,
        )
    else:
        score = anls(truth, prediction)
        metric = "anls"
    return FieldResult(
        path=path,
        metric=metric,
        score=score,
        supported=value_supported(prediction, evidence_text),
        truth=truth,
        prediction=prediction,
    )


def evaluate_document(
    truth: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    evidence: EvidenceBundle | None = None,
    *,
    json_valid: bool = True,
    parse_error: str | None = None,
) -> DocumentEvaluation:
    truth_model = ReducedCVTarget.model_validate(truth)
    truth_data = truth_model.model_dump(mode="json")
    prediction_data = dict(prediction or {})
    events: list[ValidationEvent] = []
    if parse_error:
        events.append(ValidationEvent(kind="json_parse_error", message=parse_error))
    try:
        ReducedCVTarget.model_validate(prediction_data)
        schema_valid = True
    except ValidationError as error:
        schema_valid = False
        for item in error.errors():
            events.append(
                ValidationEvent(
                    kind=str(item["type"]),
                    path=".".join(str(part) for part in item["loc"]),
                    message=str(item["msg"]),
                )
            )

    truth_keys = set(truth_data)
    prediction_keys = set(prediction_data)
    evidence_text = _evidence_text(evidence)
    field_results = [
        _field_result(path, truth_data.get(path), prediction_data.get(path), evidence_text)
        for path in truth_data
    ]
    scores = [field.score for field in field_results]
    support_values = [field.supported for field in field_results if field.supported is not None]
    unsupported_rate = (
        sum(not support for support in support_values) / len(support_values)
        if support_values
        else 0.0
    )
    return DocumentEvaluation(
        json_valid=json_valid,
        schema_valid=schema_valid,
        missing_keys=sorted(truth_keys - prediction_keys),
        extra_keys=sorted(prediction_keys - truth_keys),
        validation_events=events,
        field_results=field_results,
        micro_score=sum(scores) / len(scores) if scores else 0.0,
        macro_score=sum(scores) / len(scores) if scores else 0.0,
        unsupported_evidence_rate=unsupported_rate,
    )


def aggregate_evaluations(results: Sequence[DocumentEvaluation]) -> AggregateEvaluation:
    if not results:
        return AggregateEvaluation(
            document_count=0,
            json_valid_rate=0.0,
            schema_valid_rate=0.0,
            micro_score=0.0,
            macro_score=0.0,
            unsupported_evidence_rate=0.0,
            field_scores={},
        )
    fields: dict[str, list[float]] = {}
    for result in results:
        for field in result.field_results:
            fields.setdefault(field.path, []).append(field.score)
    return AggregateEvaluation(
        document_count=len(results),
        json_valid_rate=sum(result.json_valid for result in results) / len(results),
        schema_valid_rate=sum(result.schema_valid for result in results) / len(results),
        micro_score=sum(field.score for result in results for field in result.field_results)
        / sum(len(result.field_results) for result in results),
        macro_score=sum(result.macro_score for result in results) / len(results),
        unsupported_evidence_rate=sum(result.unsupported_evidence_rate for result in results)
        / len(results),
        field_scores={path: sum(scores) / len(scores) for path, scores in sorted(fields.items())},
    )


def evaluate_files(
    truth_path: Path,
    prediction_path: Path,
    evidence_path: Path | None = None,
) -> DocumentEvaluation:
    truth = read_json(truth_path)
    if not isinstance(truth, dict):
        raise ValueError("truth file must contain a JSON object")
    raw_prediction = prediction_path.read_text(encoding="utf-8")
    parse_error: str | None = None
    try:
        prediction = json.loads(raw_prediction)
        json_valid = isinstance(prediction, dict)
        if not json_valid:
            parse_error = "prediction JSON must be an object"
            prediction = {}
    except json.JSONDecodeError as error:
        prediction = {}
        json_valid = False
        parse_error = str(error)
    evidence: EvidenceBundle | None = None
    if evidence_path is not None:
        if evidence_path.suffix.casefold() == ".json":
            value = read_json(evidence_path)
            evidence = EvidenceBundle.model_validate(value)
        else:
            evidence = EvidenceBundle(parser_text=evidence_path.read_text(encoding="utf-8"))
    return evaluate_document(
        truth,
        prediction,
        evidence,
        json_valid=json_valid,
        parse_error=parse_error,
    )
