import json
from pathlib import Path

import pytest

from eraparse.evaluate import (
    aggregate_evaluations,
    anls,
    evaluate_document,
    evaluate_files,
    hungarian_match,
    normalize_phone,
    normalize_url,
    set_metrics,
    value_supported,
)
from eraparse.models import EvidenceBundle


def test_normalizers_cover_phone_extensions_and_urls() -> None:
    assert normalize_phone("+1 (555) 123-4567 ext. 89") == "15551234567x89"
    assert normalize_phone("1-555-123-4567 x89") == "15551234567x89"
    assert normalize_url("https://www.Example.com/path/") == "example.com/path"
    assert normalize_url("example.com/path") == "example.com/path"


def test_url_normalizer_does_not_raise_for_malformed_model_output() -> None:
    assert normalize_url("https: //linkedin.com/in/jay-silva") == "linkedin.com/in/jay-silva"


def test_set_metrics_and_anls() -> None:
    precision, recall, f1, jaccard = set_metrics(["Python", "SQL"], ["python", "Go"])
    assert (precision, recall, f1, jaccard) == pytest.approx((0.5, 0.5, 0.5, 1 / 3))
    assert anls("Example University", "example university") == 1.0
    assert anls("unrelated", "different") == 0.0


def test_hungarian_matching_handles_duplicates() -> None:
    truth = [
        {"company": "A", "job_title": "Engineer"},
        {"company": "B", "job_title": "Engineer"},
    ]
    prediction = [
        {"company": "B", "job_title": "Engineer"},
        {"company": "A", "job_title": "Engineer"},
        {"company": "A", "job_title": "Engineer"},
    ]
    precision, recall, score = hungarian_match(truth, prediction, ("company", "job_title"))
    assert precision == pytest.approx(2 / 3)
    assert recall == 1.0
    assert score == pytest.approx(0.8)


def test_evidence_support_uses_substring_and_token_coverage() -> None:
    evidence = "Jane Doe is a machine learning engineer working with Python."
    assert value_supported("Jane Doe", evidence) is True
    assert value_supported("machine learning engineer", evidence) is True
    assert value_supported("unrelated value", evidence) is False


def test_golden_document_evaluation(reduced_target: dict[str, object]) -> None:
    prediction = json.loads(json.dumps(reduced_target))
    prediction["email"] = "JANE@EXAMPLE.COM"
    prediction["phone"] = "+20-100-123-4567 x9"
    prediction["linkedin_url"] = "linkedin.com/in/jane-doe"
    prediction["skills"] = ["python", "Go"]
    result = evaluate_document(
        reduced_target,
        prediction,
        EvidenceBundle(canonical_text=json.dumps(reduced_target)),
    )
    scores = {field.path: field.score for field in result.field_results}
    assert result.json_valid is True
    assert result.schema_valid is True
    assert scores["email"] == 1.0
    assert scores["phone"] == 1.0
    assert scores["linkedin_url"] == 1.0
    assert scores["skills"] == pytest.approx(0.5)
    assert result.macro_score == pytest.approx(sum(scores.values()) / len(scores))
    aggregate = aggregate_evaluations([result])
    assert aggregate.document_count == 1
    assert aggregate.macro_score == result.macro_score


def test_malformed_prediction_is_preserved_as_validation_event(
    tmp_path: Path, reduced_target: dict[str, object]
) -> None:
    truth = tmp_path / "truth.json"
    prediction = tmp_path / "prediction.json"
    truth.write_text(json.dumps(reduced_target), encoding="utf-8")
    prediction.write_text("{bad json", encoding="utf-8")
    result = evaluate_files(truth, prediction)
    assert result.json_valid is False
    assert result.schema_valid is False
    assert result.validation_events[0].kind == "json_parse_error"
    assert result.missing_keys
