from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from eraparse.io import stable_hash

DEFAULT_ROUTER_SEED = 20260609
Outcome = Literal["win", "loss", "tie"]
FeatureValue = bool | int | float


@dataclass(frozen=True, slots=True)
class PrimaryFeatureRow:
    """Observable primary-model features, intentionally excluding training labels."""

    cv_id: str
    field: str
    values: Mapping[str, FeatureValue]


@dataclass(frozen=True, slots=True)
class RouterTrainingLabel:
    """Truth-derived calibration label; never an inference-time feature."""

    cv_id: str
    field: str
    primary_score: float
    specialist_score: float
    outcome: Outcome

    @property
    def score_delta(self) -> float:
        return self.specialist_score - self.primary_score


def stable_fold(cv_id: str, *, folds: int = 5, seed: int = DEFAULT_ROUTER_SEED) -> int:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    return int(stable_hash("router_oof_fold", cv_id, seed=seed), 16) % folds


def _field_scores(row: Mapping[str, Any]) -> dict[str, float]:
    field_results = row.get("evaluation", {}).get("field_results", [])
    if not isinstance(field_results, list):
        raise ValueError("evaluation.field_results must be a list")
    scores: dict[str, float] = {}
    for result in field_results:
        if not isinstance(result, Mapping):
            raise ValueError("each field result must be an object")
        path = str(result.get("path", ""))
        if path in scores:
            raise ValueError(f"duplicate persisted field score: {path}")
        if path:
            scores[path] = float(result["score"])
    return scores


def extract_training_labels(
    primary_rows: Sequence[Mapping[str, Any]],
    specialist_rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
    tie_tolerance: float = 1e-9,
) -> list[RouterTrainingLabel]:
    """Build labels only from persisted per-field evaluation scores."""

    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    primary_by_id = {str(row["cv_id"]): row for row in primary_rows}
    specialist_by_id = {str(row["cv_id"]): row for row in specialist_rows}
    if len(primary_by_id) != len(primary_rows) or len(specialist_by_id) != len(specialist_rows):
        raise ValueError("duplicate cv_id in persisted results")
    if set(primary_by_id) != set(specialist_by_id):
        raise ValueError("primary and specialist result IDs must match")

    labels: list[RouterTrainingLabel] = []
    for cv_id in sorted(primary_by_id):
        primary_scores = _field_scores(primary_by_id[cv_id])
        specialist_scores = _field_scores(specialist_by_id[cv_id])
        for field in fields:
            if field not in primary_scores or field not in specialist_scores:
                raise ValueError(f"missing persisted score for {cv_id}:{field}")
            primary_score = primary_scores[field]
            specialist_score = specialist_scores[field]
            delta = specialist_score - primary_score
            outcome: Outcome
            if abs(delta) <= tie_tolerance:
                outcome = "tie"
            elif delta > 0:
                outcome = "win"
            else:
                outcome = "loss"
            labels.append(
                RouterTrainingLabel(
                    cv_id=cv_id,
                    field=field,
                    primary_score=primary_score,
                    specialist_score=specialist_score,
                    outcome=outcome,
                )
            )
    return labels


def extract_primary_only_features(
    primary_rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
) -> list[PrimaryFeatureRow]:
    """Extract inference-safe features without reading the evaluation object."""

    features: list[PrimaryFeatureRow] = []
    seen_ids: set[str] = set()
    for row in sorted(primary_rows, key=lambda item: str(item["cv_id"])):
        cv_id = str(row["cv_id"])
        if cv_id in seen_ids:
            raise ValueError(f"duplicate cv_id in primary results: {cv_id}")
        seen_ids.add(cv_id)
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            raise ValueError(f"prediction must be an object for {cv_id}")
        for field in fields:
            value = prediction.get(field)
            count = len(value) if isinstance(value, list) else int(value not in (None, ""))
            serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            features.append(
                PrimaryFeatureRow(
                    cv_id=cv_id,
                    field=field,
                    values={
                        "primary_count": count,
                        "primary_characters": len(serialized),
                        "primary_empty": value in (None, "", []),
                        "primary_latency_seconds": float(row.get("latency_seconds") or 0.0),
                        "primary_output_tokens": int(row.get("output_tokens") or 0),
                        "primary_truncated": bool(row.get("truncated", False)),
                    },
                )
            )
    return features


def _candidate_thresholds(values: Sequence[float], maximum: int) -> list[float]:
    unique = sorted(set(values))
    if len(unique) <= maximum:
        return unique
    indexes = {round(index * (len(unique) - 1) / (maximum - 1)) for index in range(maximum)}
    return [unique[index] for index in sorted(indexes)]


def summarize_train_oof_calibration(
    features: Sequence[PrimaryFeatureRow],
    labels: Sequence[RouterTrainingLabel],
    *,
    partition: Literal["train_oof"],
    folds: int = 5,
    seed: int = DEFAULT_ROUTER_SEED,
    maximum_thresholds_per_feature: int = 20,
) -> dict[str, Any]:
    """Summarize train-OOF outcomes and primary-only threshold candidates."""

    if partition != "train_oof":
        raise ValueError("router calibration may only summarize the train_oof partition")
    if maximum_thresholds_per_feature < 2:
        raise ValueError("maximum_thresholds_per_feature must be at least 2")

    feature_by_key = {(row.cv_id, row.field): row for row in features}
    label_by_key = {(row.cv_id, row.field): row for row in labels}
    if len(feature_by_key) != len(features) or len(label_by_key) != len(labels):
        raise ValueError("duplicate cv_id/field calibration rows")
    if set(feature_by_key) != set(label_by_key):
        raise ValueError("feature and label keys must match")

    fold_counts: dict[int, Counter[str]] = {fold: Counter() for fold in range(folds)}
    for label in labels:
        fold_counts[stable_fold(label.cv_id, folds=folds, seed=seed)][label.outcome] += 1

    candidates: list[dict[str, Any]] = []
    fields = sorted({label.field for label in labels})
    for field in fields:
        field_labels = [label for label in labels if label.field == field]
        total_wins = sum(label.outcome == "win" for label in field_labels)
        feature_names = sorted(
            {
                name
                for label in field_labels
                for name in feature_by_key[(label.cv_id, label.field)].values
            }
        )
        for feature_name in feature_names:
            numeric_rows = [
                (
                    float(feature_by_key[(label.cv_id, label.field)].values[feature_name]),
                    label,
                )
                for label in field_labels
            ]
            thresholds = _candidate_thresholds(
                [value for value, _label in numeric_rows], maximum_thresholds_per_feature
            )
            for direction in ("le", "ge"):
                for threshold in thresholds:
                    routed = [
                        label
                        for value, label in numeric_rows
                        if (value <= threshold if direction == "le" else value >= threshold)
                    ]
                    outcomes = Counter(label.outcome for label in routed)
                    routed_wins = outcomes["win"]
                    candidates.append(
                        {
                            "field": field,
                            "feature": feature_name,
                            "direction": direction,
                            "threshold": threshold,
                            "coverage": len(routed) / len(field_labels),
                            "routed": len(routed),
                            "wins": routed_wins,
                            "losses": outcomes["loss"],
                            "ties": outcomes["tie"],
                            "win_precision": routed_wins / len(routed) if routed else 0.0,
                            "win_recall": routed_wins / total_wins if total_wins else 0.0,
                        }
                    )

    return {
        "partition": partition,
        "seed": seed,
        "folds": folds,
        "row_count": len(labels),
        "fold_outcomes": [
            {
                "fold": fold,
                "wins": fold_counts[fold]["win"],
                "losses": fold_counts[fold]["loss"],
                "ties": fold_counts[fold]["tie"],
            }
            for fold in range(folds)
        ],
        "candidate_thresholds": candidates,
    }
