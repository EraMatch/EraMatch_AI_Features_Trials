from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from eraparse.evaluate import normalize_text
from eraparse.evidence import truth_field_annotations


def evidence_support_fraction(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        raise ValueError("support measurement requires at least one record")
    evidence_text = normalize_text(
        " ".join(str(word) for record in records for word in record.get("words", []))
    )
    annotations = truth_field_annotations(records[0]["truth"])
    if not annotations:
        return 1.0
    supported = sum(normalize_text(item["text"]) in evidence_text for item in annotations)
    return supported / len(annotations)


def select_overfit_records(
    records: Sequence[Mapping[str, Any]],
    *,
    document_count: int,
    min_coverage: float = 0.95,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if document_count < 1:
        raise ValueError("document_count must be at least one")
    by_cv: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_cv[str(record["cv_id"])].append(record)
    coverage = {
        cv_id: evidence_support_fraction(document_records)
        for cv_id, document_records in by_cv.items()
    }
    eligible = [
        cv_id
        for cv_id in sorted(by_cv, key=lambda item: (-coverage[item], item))
        if coverage[cv_id] >= min_coverage
    ]
    if len(eligible) < document_count:
        raise ValueError(
            f"only {len(eligible)} documents meet minimum evidence coverage {min_coverage:.2%}"
        )
    selected_ids = eligible[:document_count]
    selected = [
        record
        for cv_id in selected_ids
        for record in sorted(by_cv[cv_id], key=lambda item: int(item["page"]))
    ]
    return selected, {
        "document_count": document_count,
        "record_count": len(selected),
        "minimum_coverage": min_coverage,
        "selected_ids": selected_ids,
        "coverage": {cv_id: coverage[cv_id] for cv_id in selected_ids},
    }
