from pathlib import Path

import pytest

from eraparse.ats import (
    _false_rejection_breakdown,
    _ranking_metrics,
    bm25_scores,
    canonical_search_text,
    evaluate_lane,
    load_prediction_documents,
    normalize_phrase,
    prediction_search_text,
)
from eraparse.io import atomic_write_jsonl
from eraparse.models import ArtifactReference, ManifestRow


def test_normalize_phrase_and_canonical_text_exclude_identity(
    reduced_target: dict[str, object],
) -> None:
    text = canonical_search_text(reduced_target)
    assert normalize_phrase("Machine-Learning / Python") == "machine learning python"
    assert "Machine Learning" in text
    assert "jane@example.com" not in text
    assert "Jane Doe" not in text
    assert prediction_search_text(None) == ""
    assert "Machine Learning" in prediction_search_text(reduced_target)


def test_prediction_documents_require_exact_unique_manifest_ids(tmp_path: Path) -> None:
    artifact = ArtifactReference(kind="ground_truth", path="truth.json", sha256="a", size_bytes=1)
    manifest_row = ManifestRow(
        cv_id="cv_00001",
        tier="T1",
        template="T1_classic",
        primary_domain="Backend Engineering",
        split="id_test",
        selection_seed=20260609,
        artifacts={"ground_truth": artifact},
        page_images=[],
    )
    manifest_path = tmp_path / "id_test.jsonl"
    results_path = tmp_path / "results.jsonl"
    atomic_write_jsonl(manifest_path, [manifest_row.model_dump(mode="json")])
    atomic_write_jsonl(results_path, [{"cv_id": "unexpected", "prediction": None}])

    with pytest.raises(ValueError, match="Prediction IDs do not match manifest"):
        load_prediction_documents(manifest_path, results_path)

    atomic_write_jsonl(
        results_path,
        [
            {"cv_id": manifest_row.cv_id, "prediction": None},
            {"cv_id": manifest_row.cv_id, "prediction": None},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate prediction result"):
        load_prediction_documents(manifest_path, results_path)


def test_bm25_and_ranking_metrics() -> None:
    documents = [
        {"cv_id": "a", "domain": "ML", "tokens": ["python", "ml"]},
        {"cv_id": "b", "domain": "Web", "tokens": ["javascript"]},
        {"cv_id": "c", "domain": "ML", "tokens": ["python"]},
    ]
    scores = bm25_scores(documents, ["python", "ml"])
    assert scores[0] > scores[2] > scores[1]
    ranked = [
        {"domain": "ML"},
        {"domain": "Web"},
        {"domain": "ML"},
    ]
    metrics = _ranking_metrics(ranked, "ML")
    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_10"] == 1.0


def test_evaluate_lane_records_full_rankings() -> None:
    documents = [
        {
            "cv_id": "a",
            "domain": "ML",
            "tier": "T1",
            "template": "classic",
            "text": "Python machine learning",
            "tokens": ["python", "machine", "learning"],
        },
        {
            "cv_id": "b",
            "domain": "Web",
            "tier": "T1",
            "template": "classic",
            "text": "JavaScript",
            "tokens": ["javascript"],
        },
    ]
    profiles = [
        {
            "profile_id": "domain:ml",
            "domain": "ML",
            "required_skills": [],
            "optional_skills": ["python", "machine learning"],
        }
    ]
    results = evaluate_lane(documents, profiles, lane="pymupdf_text", split="id_test")
    assert {result["method"] for result in results} == {"boolean", "bm25"}
    assert all(len(result["ranking"]) == 2 for result in results)
    assert results[0]["metrics"]["relevant_count"] == pytest.approx(1.0)
    breakdown = _false_rejection_breakdown(results)
    assert {row["dimension"] for row in breakdown} == {"tier", "template"}
