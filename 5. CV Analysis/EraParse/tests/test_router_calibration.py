from typing import Any

import pytest

from eraparse.router_calibration import (
    PrimaryFeatureRow,
    RouterTrainingLabel,
    extract_primary_only_features,
    extract_training_labels,
    stable_fold,
    summarize_train_oof_calibration,
)


def _result(
    cv_id: str,
    *,
    work_score: float,
    work: list[dict[str, str]] | None = None,
    latency: float = 1.0,
) -> dict[str, Any]:
    return {
        "cv_id": cv_id,
        "prediction": {"work_experience": work or []},
        "latency_seconds": latency,
        "output_tokens": 20,
        "evaluation": {
            "field_results": [
                {
                    "path": "work_experience",
                    "score": work_score,
                    "truth": [{"company": "secret"}],
                }
            ]
        },
    }


def test_stable_fold_is_deterministic_seeded_and_bounded() -> None:
    first = stable_fold("cv_00001")

    assert first == stable_fold("cv_00001")
    assert 0 <= first < 5
    assert stable_fold("cv_00001", seed=7) != stable_fold("cv_00001", seed=8)


def test_extract_labels_uses_only_persisted_field_scores() -> None:
    labels = extract_training_labels(
        [_result("cv_1", work_score=0.4), _result("cv_2", work_score=0.8)],
        [_result("cv_1", work_score=0.9), _result("cv_2", work_score=0.3)],
        fields=["work_experience"],
    )

    assert [label.outcome for label in labels] == ["win", "loss"]
    assert labels[0].score_delta == pytest.approx(0.5)


def test_primary_features_do_not_include_evaluation_truth_or_scores() -> None:
    features = extract_primary_only_features(
        [_result("cv_1", work_score=0.123, work=[{"company": "A"}], latency=1.5)],
        fields=["work_experience"],
    )

    assert set(features[0].values) == {
        "primary_count",
        "primary_characters",
        "primary_empty",
        "primary_latency_seconds",
        "primary_output_tokens",
        "primary_truncated",
    }
    assert 0.123 not in features[0].values.values()
    assert "secret" not in str(features[0].values)


def test_features_and_labels_remain_separate_types() -> None:
    feature = PrimaryFeatureRow("cv_1", "work_experience", {"primary_count": 1})
    label = RouterTrainingLabel("cv_1", "work_experience", 0.2, 0.8, "win")

    assert not hasattr(feature, "outcome")
    assert not hasattr(feature, "primary_score")
    assert not hasattr(label, "values")


def test_summary_reports_fold_outcomes_and_threshold_coverage() -> None:
    features = [
        PrimaryFeatureRow("cv_1", "work_experience", {"primary_count": 0}),
        PrimaryFeatureRow("cv_2", "work_experience", {"primary_count": 1}),
        PrimaryFeatureRow("cv_3", "work_experience", {"primary_count": 2}),
    ]
    labels = [
        RouterTrainingLabel("cv_1", "work_experience", 0.1, 0.9, "win"),
        RouterTrainingLabel("cv_2", "work_experience", 0.5, 0.5, "tie"),
        RouterTrainingLabel("cv_3", "work_experience", 0.8, 0.2, "loss"),
    ]

    summary = summarize_train_oof_calibration(features, labels, partition="train_oof")
    candidate = next(
        item
        for item in summary["candidate_thresholds"]
        if item["feature"] == "primary_count"
        and item["direction"] == "le"
        and item["threshold"] == 0
    )

    assert sum(row["wins"] for row in summary["fold_outcomes"]) == 1
    assert sum(row["losses"] for row in summary["fold_outcomes"]) == 1
    assert sum(row["ties"] for row in summary["fold_outcomes"]) == 1
    assert "labels" not in summary
    assert candidate["coverage"] == pytest.approx(1 / 3)
    assert candidate["win_precision"] == 1.0
    assert candidate["win_recall"] == 1.0


def test_summary_rejects_non_train_partition_and_mismatched_rows() -> None:
    features = [PrimaryFeatureRow("cv_1", "work_experience", {"primary_count": 0})]
    labels = [RouterTrainingLabel("cv_2", "work_experience", 0.1, 0.9, "win")]

    with pytest.raises(ValueError, match="train_oof"):
        summarize_train_oof_calibration(features, labels, partition="validation")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="keys must match"):
        summarize_train_oof_calibration(features, labels, partition="train_oof")
