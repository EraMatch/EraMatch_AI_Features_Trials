import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from eraparse.assembly import assemble_prediction
from eraparse.constants import SGE_FIELD_PATHS
from eraparse.data import load_manifest
from eraparse.evaluate import aggregate_evaluations, evaluate_document, normalize_text
from eraparse.evidence import read_evidence_graphs
from eraparse.io import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl
from eraparse.models import (
    EvidenceBundle,
    FieldCandidate,
    GroundedPrediction,
    ReducedCVTarget,
    ValidationEvent,
    WorkCandidateSpan,
    WorkDocumentRecordSet,
    WorkRecordTarget,
)

FIELD_TO_ID = {field: index + 1 for index, field in enumerate(SGE_FIELD_PATHS)}
ID_TO_FIELD = {value: key for key, value in FIELD_TO_ID.items()}
NESTED_KINDS = ("work_experience", "education", "projects", "certifications")
WORK_FIELDS = (
    "work_experience.*.job_title",
    "work_experience.*.company",
    "work_experience.*.start_date",
    "work_experience.*.end_date",
    "work_experience.*.duration",
)
DERIVABLE_WORK_FIELDS = ("work_experience.*.duration",)
WORK_DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2})?\b")
PRESENT_PATTERN = re.compile(r"\b(?:present|current|now)\b", re.IGNORECASE)
EVIDENCE_POSITION_PATTERN = re.compile(r".*:p(\d+):w(\d+)")
GENERIC_WORK_TITLES = {
    "developer",
    "engineer",
    "designer",
    "analyst",
    "architect",
    "scientist",
    "manager",
    "tester",
}

PROJECT_URL_DEFAULT_THRESHOLD = 0.45
PROJECT_URL_DEFAULT_MIN_FULL_COUNT = 8
PROJECT_URL_DEFAULT_MIN_REDUCED_COUNT = 8


def record_group_id(field_path: str | None, record_index: int | None) -> int:
    if field_path is None or record_index is None:
        return -1
    match = re.match(r"^(work_experience|education|projects|certifications)\.\*\.", field_path)
    if match is None:
        return -1
    return NESTED_KINDS.index(match.group(1)) * 1_000 + record_index


def _nested_path(path: str | None) -> bool:
    return bool(
        path and re.match(r"^(work_experience|education|projects|certifications)\.\*\.", path)
    )


def _group_components(
    labels: Sequence[int],
    grouping_probabilities: Sequence[Sequence[float]] | None,
    threshold: float,
) -> dict[int, int]:
    nested = [index for index, label in enumerate(labels) if _nested_path(ID_TO_FIELD.get(label))]
    if grouping_probabilities is None:
        state: dict[str, tuple[int, set[str], str | None]] = {}
        groups = {}
        for index in nested:
            path = ID_TO_FIELD[labels[index]]
            kind = path.split(".", 1)[0]
            record_index, seen_paths, previous_path = state.get(kind, (0, set(), None))
            if path != previous_path and path in seen_paths:
                record_index += 1
                seen_paths = set()
            groups[index] = record_index
            seen_paths.add(path)
            state[kind] = (record_index, seen_paths, path)
        return groups
    parent = {index: index for index in nested}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int, *, enforce_schema_constraints: bool = False) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if enforce_schema_constraints:
            left_members = [index for index in nested if find(index) == left_root]
            right_members = [index for index in nested if find(index) == right_root]
            left_paths = {ID_TO_FIELD[labels[index]] for index in left_members}
            right_paths = {ID_TO_FIELD[labels[index]] for index in right_members}
            left_kinds = {path.split(".", 1)[0] for path in left_paths}
            right_kinds = {path.split(".", 1)[0] for path in right_paths}
            if left_kinds != right_kinds or left_paths & right_paths:
                return
        parent[right_root] = left_root

    for left, right in pairwise(nested):
        if right == left + 1 and labels[left] == labels[right]:
            union(left, right)
    for position, left in enumerate(nested):
        for right in nested[position + 1 :]:
            probability = (
                grouping_probabilities[left][right] + grouping_probabilities[right][left]
            ) / 2
            if probability >= threshold:
                union(left, right, enforce_schema_constraints=True)
    roots = sorted(
        {find(index) for index in nested},
        key=lambda root: min(i for i in nested if find(i) == root),
    )
    root_to_record = {root: index for index, root in enumerate(roots)}
    return {index: root_to_record[find(index)] for index in nested}


def decode_word_candidates(
    record: Mapping[str, Any],
    *,
    word_labels: Sequence[int],
    confidences: Sequence[float],
    grouping_probabilities: Sequence[Sequence[float]] | None = None,
    grouping_threshold: float = 0.5,
) -> list[FieldCandidate]:
    words = [str(word) for word in record["words"]]
    evidence_ids = [str(value) for value in record["evidence_ids"]]
    if not (len(words) == len(word_labels) == len(confidences) == len(evidence_ids)):
        raise ValueError("word prediction lengths must match")
    record_groups = _group_components(word_labels, grouping_probabilities, grouping_threshold)
    candidates = []
    start = 0
    while start < len(words):
        label = word_labels[start]
        if label == 0 or label not in ID_TO_FIELD:
            start += 1
            continue
        end = start + 1
        while (
            end < len(words)
            and word_labels[end] == label
            and record_groups.get(end) == record_groups.get(start)
        ):
            end += 1
        if ID_TO_FIELD[label] == "skills":
            skill_start = start
            for position in range(start, end):
                if words[position].rstrip().endswith((",", ";", "|")):
                    skill_end = position + 1
                    candidates.append(
                        FieldCandidate(
                            schema_path="skills",
                            value=" ".join(words[skill_start:skill_end]).strip(" ,;|"),
                            evidence_ids=evidence_ids[skill_start:skill_end],
                            confidence=sum(confidences[skill_start:skill_end])
                            / (skill_end - skill_start),
                        )
                    )
                    skill_start = skill_end
            if skill_start < end:
                candidates.append(
                    FieldCandidate(
                        schema_path="skills",
                        value=" ".join(words[skill_start:end]).strip(" ,;|"),
                        evidence_ids=evidence_ids[skill_start:end],
                        confidence=sum(confidences[skill_start:end]) / (end - skill_start),
                    )
                )
            start = end
            continue
        candidates.append(
            FieldCandidate(
                schema_path=ID_TO_FIELD[label],
                value=" ".join(words[start:end]),
                evidence_ids=evidence_ids[start:end],
                confidence=sum(confidences[start:end]) / (end - start),
                record_index=record_groups.get(start),
            )
        )
        start = end
    return candidates


def decode_labeled_candidates(
    record: Mapping[str, Any],
    *,
    grouping_mode: str = "sequence",
) -> list[FieldCandidate]:
    if grouping_mode == "sequence":
        return decode_word_candidates(
            record,
            word_labels=[int(label) for label in record["field_labels"]],
            confidences=[1.0] * len(record["field_labels"]),
        )
    if grouping_mode != "oracle":
        raise ValueError("grouping_mode must be sequence or oracle")

    words = [str(word) for word in record["words"]]
    evidence_ids = [str(value) for value in record["evidence_ids"]]
    word_labels = [int(label) for label in record["field_labels"]]
    record_indices = [int(index) for index in record["record_indices"]]
    if not (
        len(words) == len(word_labels) == len(record_indices) == len(evidence_ids)
    ):
        raise ValueError("record oracle lengths must match")

    candidates = []
    start = 0
    while start < len(words):
        label = word_labels[start]
        if label == 0 or label not in ID_TO_FIELD:
            start += 1
            continue
        path = ID_TO_FIELD[label]
        record_index = (
            record_indices[start]
            if _nested_path(path) and record_indices[start] >= 0
            else None
        )
        end = start + 1
        while end < len(words) and word_labels[end] == label:
            end_record_index = (
                record_indices[end] if _nested_path(path) and record_indices[end] >= 0 else None
            )
            if end_record_index != record_index:
                break
            end += 1
        if path == "skills":
            skill_start = start
            for position in range(start, end):
                if words[position].rstrip().endswith((",", ";", "|")):
                    skill_end = position + 1
                    candidates.append(
                        FieldCandidate(
                            schema_path="skills",
                            value=" ".join(words[skill_start:skill_end]).strip(" ,;|"),
                            evidence_ids=evidence_ids[skill_start:skill_end],
                            confidence=1.0,
                        )
                    )
                    skill_start = skill_end
            if skill_start < end:
                candidates.append(
                    FieldCandidate(
                        schema_path="skills",
                        value=" ".join(words[skill_start:end]).strip(" ,;|"),
                        evidence_ids=evidence_ids[skill_start:end],
                        confidence=1.0,
                    )
                )
            start = end
            continue
        candidates.append(
            FieldCandidate(
                schema_path=path,
                value=" ".join(words[start:end]),
                evidence_ids=evidence_ids[start:end],
                confidence=1.0,
                record_index=record_index,
            )
        )
        start = end
    return candidates


def prepare_sge_records(
    evidence_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    graphs = {graph.cv_id: graph for graph in read_evidence_graphs(evidence_path)}
    records = []
    for row in load_manifest(manifest_path):
        graph = graphs.get(row.cv_id)
        if graph is None:
            raise ValueError(f"missing evidence graph for {row.cv_id}")
        by_page = defaultdict(list)
        for unit in graph.units:
            by_page[unit.page].append(unit)
        for page, units in sorted(by_page.items()):
            ordered = sorted(units, key=lambda unit: unit.reading_order)
            records.append(
                {
                    "cv_id": row.cv_id,
                    "split": row.split,
                    "tier": row.tier,
                    "reader": graph.reader,
                    "oracle": graph.oracle,
                    "page": page,
                    "page_image": graph.page_images[page - 1],
                    "evidence_ids": [unit.evidence_id for unit in ordered],
                    "words": [unit.text for unit in ordered],
                    "boxes": [list(unit.bbox_norm) for unit in ordered],
                    "field_labels": [FIELD_TO_ID.get(unit.field_path or "", 0) for unit in ordered],
                    "record_indices": [
                        unit.record_index if unit.record_index is not None else -1
                        for unit in ordered
                    ],
                    "record_group_labels": [
                        record_group_id(unit.field_path, unit.record_index) for unit in ordered
                    ],
                    "truth": read_json(dataset_root / row.artifacts["schema_reduced"].path),
                }
            )
    atomic_write_jsonl(output_path, records)
    summary = {
        "record_count": len(records),
        "document_count": len({record["cv_id"] for record in records}),
        "labeled_token_count": sum(
            label != 0 for record in records for label in record["field_labels"]
        ),
        "reader": next(iter(graphs.values())).reader if graphs else None,
        "oracle": any(graph.oracle for graph in graphs.values()),
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def assemble_candidate_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    by_cv: dict[str, list[FieldCandidate]] = defaultdict(list)
    for row in candidate_rows:
        candidate = {key: value for key, value in row.items() if key != "cv_id"}
        by_cv[str(row["cv_id"])].append(FieldCandidate.model_validate(candidate))
    predictions = [
        assemble_prediction(cv_id, candidates) for cv_id, candidates in sorted(by_cv.items())
    ]
    atomic_write_jsonl(
        output_path,
        [prediction.model_dump(mode="json") for prediction in predictions],
    )
    return {"document_count": len(predictions), "output": str(output_path)}


def evaluate_grounded_rows(
    grounded_path: Path,
    requests_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    requests: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(requests_path):
        cv_id = str(row["cv_id"])
        request = requests.setdefault(
            cv_id,
            {
                "cv_id": cv_id,
                "truth": row["truth"],
                "words": [],
            },
        )
        request["words"].extend(str(word) for word in row.get("words", []))
    results = []
    for row in read_jsonl(grounded_path):
        request = requests[str(row["cv_id"])]
        results.append(
            evaluate_document(
                request["truth"],
                row["prediction"],
                EvidenceBundle(parser_text=" ".join(request.get("words", []))),
            )
        )
    aggregate = aggregate_evaluations(results)
    summary = aggregate.model_dump(mode="json")
    atomic_write_json(output_path, summary)
    return summary


def evaluate_record_oracle(
    records_path: Path,
    output_dir: Path,
    *,
    grouping_mode: str = "oracle",
    max_records: int | None = None,
) -> dict[str, Any]:
    records = list(read_jsonl(records_path))
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError("records file must contain at least one row")

    output_dir.mkdir(parents=True, exist_ok=False)
    by_cv: dict[str, list[FieldCandidate]] = defaultdict(list)
    by_cv_truth: dict[str, Any] = {}
    by_cv_words: dict[str, list[str]] = defaultdict(list)

    for record in records:
        cv_id = str(record["cv_id"])
        by_cv_truth[cv_id] = record["truth"]
        by_cv_words[cv_id].extend(str(word) for word in record["words"])
        by_cv[cv_id].extend(decode_labeled_candidates(record, grouping_mode=grouping_mode))

    predictions = [
        assemble_prediction(cv_id, by_cv[cv_id]) for cv_id in sorted(by_cv_truth)
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
    atomic_write_jsonl(
        output_dir / "predictions.jsonl",
        [prediction.model_dump(mode="json") for prediction in predictions],
    )
    atomic_write_json(output_dir / "evaluation.json", aggregate)
    summary = {
        "records_path": str(records_path),
        "grouping_mode": grouping_mode,
        "record_count": len(records),
        "document_count": len(predictions),
        "candidate_count": sum(len(candidates) for candidates in by_cv.values()),
        "evaluation": aggregate,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def prepare_work_record_bank(
    records_path: Path,
    output_path: Path,
    *,
    max_records: int | None = None,
) -> dict[str, Any]:
    records = list(read_jsonl(records_path))
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError("records file must contain at least one row")

    by_cv: dict[str, dict[str, Any]] = {}
    for record in records:
        cv_id = str(record["cv_id"])
        document = by_cv.setdefault(
            cv_id,
            {
                "cv_id": cv_id,
                "split": record.get("split"),
                "tier": record.get("tier"),
                "reader": record.get("reader"),
                "oracle": bool(record.get("oracle", False)),
                "page_count": 0,
                "spans": [],
                "targets": [],
                "truth": record["truth"],
            },
        )
        document["page_count"] = max(document["page_count"], int(record["page"]))
        words = [str(word) for word in record["words"]]
        evidence_ids = [str(item) for item in record["evidence_ids"]]
        labels = [int(label) for label in record["field_labels"]]
        record_indices = [int(index) for index in record["record_indices"]]
        start = 0
        while start < len(words):
            label = labels[start]
            if label == 0 or ID_TO_FIELD.get(label) not in WORK_FIELDS:
                start += 1
                continue
            field_path = ID_TO_FIELD[label]
            record_index = record_indices[start]
            if record_index < 0:
                start += 1
                continue
            end = start + 1
            while (
                end < len(words)
                and labels[end] == label
                and record_indices[end] == record_index
            ):
                end += 1
            document["spans"].append(
                WorkCandidateSpan(
                    cv_id=cv_id,
                    page=int(record["page"]),
                    field_path=field_path,
                    record_index=record_index,
                    value=" ".join(words[start:end]),
                    evidence_ids=evidence_ids[start:end],
                    word_start=start,
                    word_end=end,
                ).model_dump(mode="json")
            )
            start = end

    output_rows = []
    field_totals = {field: 0 for field in WORK_FIELDS}
    field_covered = {field: 0 for field in WORK_FIELDS}
    exact_record_matches = 0
    direct_record_matches = 0
    total_records = 0

    for cv_id in sorted(by_cv):
        document = by_cv[cv_id]
        truth = document.pop("truth")
        truth_records = truth.get("work_experience") or []
        normalized_spans = defaultdict(set)
        normalized_pairs = set()
        for span in document["spans"]:
            normalized_value = normalize_text(span["value"])
            if normalized_value:
                normalized_spans[span["field_path"]].add(normalized_value)
                normalized_pairs.add((span["record_index"], span["field_path"], normalized_value))

        targets = []
        for record_index, truth_record in enumerate(truth_records):
            target = WorkRecordTarget(
                record_index=record_index,
                job_title=str(truth_record.get("job_title", "")),
                company=str(truth_record.get("company", "")),
                start_date=str(truth_record.get("start_date", "")),
                end_date=str(truth_record.get("end_date", "")),
                duration=str(truth_record.get("duration", "")),
            )
            targets.append(target)
            total_records += 1
            values = {
                "work_experience.*.job_title": normalize_text(target.job_title),
                "work_experience.*.company": normalize_text(target.company),
                "work_experience.*.start_date": normalize_text(target.start_date),
                "work_experience.*.end_date": normalize_text(target.end_date),
                "work_experience.*.duration": normalize_text(target.duration),
            }
            record_match = True
            direct_record_match = True
            for field_path, normalized_value in values.items():
                if not normalized_value:
                    continue
                field_totals[field_path] += 1
                if normalized_value in normalized_spans[field_path]:
                    field_covered[field_path] += 1
                else:
                    record_match = False
                if (
                    record_index,
                    field_path,
                    normalized_value,
                ) not in normalized_pairs:
                    record_match = False
                if field_path not in DERIVABLE_WORK_FIELDS and (
                    record_index,
                    field_path,
                    normalized_value,
                ) not in normalized_pairs:
                    direct_record_match = False
            if record_match:
                exact_record_matches += 1
            if direct_record_match:
                direct_record_matches += 1
        document["targets"] = [target.model_dump(mode="json") for target in targets]
        output_rows.append(WorkDocumentRecordSet.model_validate(document).model_dump(mode="json"))

    atomic_write_jsonl(output_path, output_rows)
    summary = {
        "document_count": len(output_rows),
        "span_count": sum(len(row["spans"]) for row in output_rows),
        "target_record_count": total_records,
        "field_coverage": {
            field: (field_covered[field] / field_totals[field] if field_totals[field] else 1.0)
            for field in WORK_FIELDS
        },
        "exact_record_match_rate": (
            exact_record_matches / total_records if total_records else 1.0
        ),
        "direct_record_match_rate": (
            direct_record_matches / total_records if total_records else 1.0
        ),
        "derivable_fields": list(DERIVABLE_WORK_FIELDS),
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _clean_work_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip("[]{}|·,; ")


def _project_group_anchor_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    anchor_field: str,
    fields: Sequence[str],
) -> list[dict[str, list[Mapping[str, Any]]]]:
    relevant = [
        candidate
        for candidate in candidates
        if str(candidate["schema_path"]).startswith(prefix)
    ]
    relevant.sort(key=_candidate_position)
    grouped: list[dict[str, list[Mapping[str, Any]]]] = []
    current: dict[str, list[Mapping[str, Any]]] | None = None

    for candidate in relevant:
        field = str(candidate["schema_path"]).split(".*.", 1)[1]
        if field == anchor_field:
            if current is not None and any(
                current[item] for item in fields if item != anchor_field
            ):
                grouped.append(current)
                current = None
            if current is None:
                current = {item: [] for item in fields}
            current[field].append(candidate)
            continue
        if current is None:
            continue
        if current[field]:
            grouped.append(current)
            current = {item: [] for item in fields}
        current[field].append(candidate)

    if current is not None:
        grouped.append(current)
    return grouped


def _extract_project_technologies(
    raw_values: Sequence[Any],
    skills: Sequence[str],
) -> list[str]:
    skill_map = [
        (skill, normalize_text(skill))
        for skill in skills
        if isinstance(skill, str) and normalize_text(skill)
    ]
    skill_norms = {normalized for _, normalized in skill_map}
    seen: set[str] = set()
    values: list[str] = []

    for raw_value in raw_values:
        raw_text = str(raw_value or "")
        raw_norm = normalize_text(raw_text)
        local_values: list[tuple[str, str]] = []
        for piece in re.split(r"[,;|]", raw_text):
            cleaned = _clean_work_text(piece).strip("()")
            normalized = normalize_text(cleaned)
            if not cleaned or not normalized or normalized in seen:
                continue
            if len(cleaned) <= 32 and len(cleaned.split()) <= 3:
                local_values.append((cleaned, normalized))

        if len(local_values) == 1:
            cleaned, normalized = local_values[0]
            if " and " in raw_text.casefold() and normalized not in skill_norms:
                local_values = []

        if not local_values or any(
            normalized in {"api", "app", "framework", "service", "tool"}
            for _, normalized in local_values
        ):
            hits = []
            local_norms = {normalized for _, normalized in local_values}
            for skill, normalized in skill_map:
                if normalized in local_norms or normalized in seen:
                    continue
                if normalized in raw_norm:
                    hits.append((raw_norm.index(normalized), skill, normalized))
            hits.sort()
            if hits:
                local_values = [(skill, normalized) for _, skill, normalized in hits]

        for cleaned, normalized in local_values:
            if normalized not in seen:
                values.append(cleaned)
                seen.add(normalized)
    return values


def _project_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _github_owner(value: str | None) -> str | None:
    match = re.search(r"github\.com/\s*([^/\s]+)/?$", str(value or "").strip(), re.IGNORECASE)
    return match.group(1) if match is not None else None


def _project_url_selector_features(
    projects: Sequence[Mapping[str, Any]],
    index: int,
    project: Mapping[str, Any],
) -> tuple[int, int, bool, int, int]:
    names = [str(item.get("name", "") or "") for item in projects]
    duplicate_names = len({name.casefold() for name in names if name}) < len(names)
    return (
        len(projects),
        index,
        duplicate_names,
        min(len(project.get("technologies", []) or []), 3),
        min(len(str(project.get("name", "") or "").split()), 4),
    )


def _project_url_selector_stats(
    train_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    documents: dict[str, Mapping[str, Any]] = {}
    for row in train_records:
        documents.setdefault(str(row["cv_id"]), row)

    full_bucket_counts: Counter[tuple[int, int, bool, int, int]] = Counter()
    reduced_bucket_counts: Counter[tuple[int, int]] = Counter()
    full_bucket_labels: Counter[tuple[tuple[int, int, bool, int, int], int]] = Counter()
    reduced_bucket_labels: Counter[tuple[tuple[int, int], int]] = Counter()
    global_positive = 0
    global_total = 0

    for row in documents.values():
        truth = row["truth"]
        if not truth.get("github_url"):
            continue
        projects = truth.get("projects") or []
        if not projects:
            continue
        for index, project in enumerate(projects):
            label = int(bool(project.get("url")))
            full_features = _project_url_selector_features(projects, index, project)
            reduced_features = (len(projects), index)
            full_bucket_counts[full_features] += 1
            reduced_bucket_counts[reduced_features] += 1
            full_bucket_labels[(full_features, label)] += 1
            reduced_bucket_labels[(reduced_features, label)] += 1
            global_positive += label
            global_total += 1

    return {
        "document_count": len(documents),
        "global_positive_rate": global_positive / global_total if global_total else 0.0,
        "full_bucket_counts": {
            "|".join(str(item) for item in features): count
            for features, count in full_bucket_counts.items()
        },
        "reduced_bucket_counts": {
            "|".join(str(item) for item in features): count
            for features, count in reduced_bucket_counts.items()
        },
        "full_bucket_labels": {
            f'{"|".join(str(item) for item in features)}|{label}': count
            for (features, label), count in full_bucket_labels.items()
        },
        "reduced_bucket_labels": {
            f'{"|".join(str(item) for item in features)}|{label}': count
            for (features, label), count in reduced_bucket_labels.items()
        },
    }


def _project_url_probability(
    stats: Mapping[str, Any],
    projects: Sequence[Mapping[str, Any]],
    index: int,
    project: Mapping[str, Any],
    *,
    min_full_count: int,
    min_reduced_count: int,
) -> float | None:
    full_features = _project_url_selector_features(projects, index, project)
    reduced_features = (len(projects), index)
    full_key = "|".join(str(item) for item in full_features)
    reduced_key = "|".join(str(item) for item in reduced_features)

    full_count = int(dict(stats["full_bucket_counts"]).get(full_key, 0))
    if full_count >= min_full_count:
        yes = int(dict(stats["full_bucket_labels"]).get(f"{full_key}|1", 0))
        no = int(dict(stats["full_bucket_labels"]).get(f"{full_key}|0", 0))
        return (yes + 1) / (yes + no + 2)

    reduced_count = int(dict(stats["reduced_bucket_counts"]).get(reduced_key, 0))
    if reduced_count >= min_reduced_count:
        yes = int(dict(stats["reduced_bucket_labels"]).get(f"{reduced_key}|1", 0))
        no = int(dict(stats["reduced_bucket_labels"]).get(f"{reduced_key}|0", 0))
        return (yes + 1) / (yes + no + 2)

    return None


def _canonical_present(value: str) -> str:
    cleaned = _clean_work_text(value).strip("()")
    return "Present" if PRESENT_PATTERN.fullmatch(cleaned) else cleaned


def _extract_duration_hints(duration: str) -> tuple[str, str]:
    matches = WORK_DATE_PATTERN.findall(duration)
    start = matches[0] if matches else ""
    if PRESENT_PATTERN.search(duration):
        end = "Present"
    elif len(matches) >= 2:
        end = matches[-1]
    else:
        end = ""
    return start, end


def repair_work_record(
    record: Mapping[str, Any],
) -> tuple[dict[str, str], list[ValidationEvent]]:
    repaired = {
        "job_title": _clean_work_text(record.get("job_title", "")),
        "company": _clean_work_text(record.get("company", "")),
        "start_date": _canonical_present(str(record.get("start_date", ""))),
        "end_date": _canonical_present(str(record.get("end_date", ""))),
        "duration": _clean_work_text(record.get("duration", "")),
    }
    events: list[ValidationEvent] = []

    duration_start, duration_end = _extract_duration_hints(repaired["duration"])
    if not repaired["start_date"] and duration_start:
        repaired["start_date"] = duration_start
        events.append(
            ValidationEvent(
                kind="work_start_date_repaired",
                path="work_experience.*.start_date",
                message=f"filled start_date from duration hint: {duration_start}",
            )
        )
    if not repaired["end_date"]:
        inferred_end = duration_end
        if (
            not inferred_end
            and repaired["start_date"]
            and duration_start
            and duration_start != repaired["start_date"]
        ):
            inferred_end = duration_start
        if inferred_end:
            repaired["end_date"] = inferred_end
            events.append(
                ValidationEvent(
                    kind="work_end_date_repaired",
                    path="work_experience.*.end_date",
                    message=f"filled end_date from duration hint: {inferred_end}",
                )
            )

    if repaired["start_date"] and repaired["end_date"]:
        normalized_duration = f'{repaired["start_date"]} - {repaired["end_date"]}'
        if repaired["duration"] != normalized_duration:
            repaired["duration"] = normalized_duration
            events.append(
                ValidationEvent(
                    kind="work_duration_normalized",
                    path="work_experience.*.duration",
                    message="normalized duration from start_date and end_date",
                )
            )
    return repaired, events


def repair_grounded_work_predictions(
    predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows = list(read_jsonl(predictions_path))
    repaired_rows = []
    repaired_document_count = 0
    repaired_record_count = 0
    event_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        grounded = GroundedPrediction.model_validate(row)
        prediction_payload = grounded.prediction.model_dump(mode="json")
        repaired_work = []
        repair_events: list[ValidationEvent] = []
        for record in prediction_payload.get("work_experience", []):
            repaired_record, events = repair_work_record(record)
            repaired_work.append(repaired_record)
            if events or repaired_record != record:
                repaired_record_count += 1
            for event in events:
                repair_events.append(event)
                event_counts[event.kind] += 1
        if repair_events:
            repaired_document_count += 1
        prediction_payload["work_experience"] = repaired_work
        repaired_grounded = grounded.model_copy(
            update={
                "prediction": ReducedCVTarget.model_validate(prediction_payload),
                "assembly_events": grounded.assembly_events + repair_events,
            }
        )
        repaired_rows.append(repaired_grounded.model_dump(mode="json"))

    atomic_write_jsonl(output_path, repaired_rows)
    summary = {
        "document_count": len(repaired_rows),
        "repaired_document_count": repaired_document_count,
        "repaired_record_count": repaired_record_count,
        "event_counts": dict(sorted(event_counts.items())),
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _candidate_position(candidate: Mapping[str, Any]) -> tuple[int, int]:
    evidence_ids = candidate.get("evidence_ids") or []
    if not evidence_ids:
        return (10**9, 10**9)
    match = EVIDENCE_POSITION_PATTERN.match(str(evidence_ids[0]))
    if match is None:
        return (10**9, 10**9)
    return (int(match.group(1)), int(match.group(2)))


def _decode_anchor_records(
    candidates: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    anchor_field: str,
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped = _project_group_anchor_candidates(
        candidates,
        prefix=prefix,
        anchor_field=anchor_field,
        fields=fields,
    )

    decoded = []
    for group in grouped:
        record: dict[str, Any] = {}
        for field in fields:
            if field == "technologies":
                values = []
                seen = set()
                for candidate in group[field]:
                    for item in re.split(r"[,;|]", str(candidate["value"]).strip().strip("[]{}")):
                        cleaned = _clean_work_text(item).strip("()")
                        normalized = normalize_text(cleaned)
                        if cleaned and normalized and normalized not in seen:
                            values.append(cleaned)
                            seen.add(normalized)
                record[field] = values
                continue
            best = max(
                group[field],
                key=lambda item: float(item.get("confidence", 0.0)),
                default=None,
            )
            if field == "url":
                record[field] = _clean_work_text(best["value"]) if best is not None else None
            else:
                record[field] = _clean_work_text(best["value"]) if best is not None else ""
        if anchor_field == "degree" and not record["degree"]:
            continue
        if anchor_field == "name" and not record["name"]:
            continue
        decoded.append(record)
    return decoded


def _decode_project_records(
    candidates: Sequence[Mapping[str, Any]],
    skills: Sequence[str],
) -> list[dict[str, Any]]:
    grouped = _project_group_anchor_candidates(
        candidates,
        prefix="projects.*.",
        anchor_field="name",
        fields=("name", "technologies", "url"),
    )
    decoded = []
    for group in grouped:
        best_name = max(
            group["name"],
            key=lambda item: float(item.get("confidence", 0.0)),
            default=None,
        )
        if best_name is None:
            continue
        best_url = max(
            group["url"],
            key=lambda item: float(item.get("confidence", 0.0)),
            default=None,
        )
        record = {
            "name": _clean_work_text(best_name["value"]),
            "technologies": _extract_project_technologies(
                [candidate["value"] for candidate in group["technologies"]],
                skills,
            ),
            "url": _clean_work_text(best_url["value"]) if best_url is not None else None,
        }
        if record["name"]:
            decoded.append(record)
    return decoded


def _decode_sgrse_work_rows(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    short_fields = tuple(field_path.split(".*.", 1)[1] for field_path in WORK_FIELDS)
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    for candidate in candidates:
        path = str(candidate["schema_path"])
        if not path.startswith("work_experience.*."):
            continue
        record_index = candidate.get("record_index")
        if record_index is None:
            continue
        field = path.split(".*.", 1)[1]
        grouped.setdefault(record_index, {item: [] for item in short_fields})
        grouped[record_index][field].append(candidate)

    rows = []
    for record_index in sorted(grouped):
        row = {}
        confidences = {}
        for field in short_fields:
            best = max(
                grouped[record_index][field],
                key=lambda item: float(item.get("confidence", 0.0)),
                default=None,
            )
            row[field] = _clean_work_text(best["value"]) if best is not None else ""
            confidences[field] = float(best.get("confidence", 0.0)) if best is not None else 0.0
        repaired, _ = repair_work_record(row)
        rows.append(
            {
                "record_index": record_index,
                "fields": repaired,
                "confidences": confidences,
            }
        )
    return rows


def _merge_sgrse_work_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    merged: list[dict[str, Any]] = []
    merge_count = 0
    for row in rows:
        fields = dict(row["fields"])
        if not merged:
            merged.append({"fields": fields})
            continue

        previous = merged[-1]["fields"]
        previous_title = previous["job_title"].casefold()
        current_title = fields["job_title"].casefold()

        if previous["job_title"] and not fields["job_title"]:
            updated = dict(previous)
            took = False
            for field in ("company", "start_date", "end_date", "duration"):
                if not updated[field] and fields[field]:
                    updated[field] = fields[field]
                    took = True
            if took:
                merged[-1]["fields"], _ = repair_work_record(updated)
                merge_count += 1
                continue

        if (
            previous["job_title"]
            and fields["job_title"]
            and current_title in GENERIC_WORK_TITLES
            and previous["start_date"]
            and fields["company"]
            and fields["end_date"]
            and not previous["company"]
        ):
            merged[-1]["fields"], _ = repair_work_record(
                {
                    "job_title": f'{previous["job_title"]} {fields["job_title"]}',
                    "company": fields["company"],
                    "start_date": previous["start_date"],
                    "end_date": fields["end_date"],
                    "duration": fields["duration"] or previous["duration"],
                }
            )
            merge_count += 1
            continue

        if previous_title and current_title and previous_title == current_title:
            updated = dict(previous)
            took = False
            for field in ("company", "start_date", "end_date", "duration"):
                if not updated[field] and fields[field]:
                    updated[field] = fields[field]
                    took = True
            if took:
                merged[-1]["fields"], _ = repair_work_record(updated)
                merge_count += 1
                continue

        merged.append({"fields": fields})

    return [
        dict(item["fields"]) for item in merged if item["fields"]["job_title"]
    ], merge_count


def _work_record_features(
    records: Sequence[Mapping[str, Any]] | None,
) -> dict[str, int]:
    record_list = list(records or [])
    complete = 0
    partial = 0
    empty_title = 0
    empty_company = 0
    has_dates = 0
    suspicious = 0
    for record in record_list:
        job_title = str(record.get("job_title", "") or "")
        company = str(record.get("company", "") or "")
        start_date = str(record.get("start_date", "") or "")
        end_date = str(record.get("end_date", "") or "")
        duration = str(record.get("duration", "") or "")
        filled = sum(
            bool(record.get(field))
            for field in ("job_title", "company", "start_date", "end_date", "duration")
        )
        if filled >= 4:
            complete += 1
        elif filled > 0:
            partial += 1
        empty_title += int(not job_title)
        empty_company += int(not company)
        has_dates += int(bool(start_date or end_date or duration))
        combined = f"{job_title} {company}".strip()
        if re.search(r"[^A-Za-z0-9 .,&+/#()\-\u2013\u2014]", combined):
            suspicious += 1
        if re.search(r"[a-z][A-Z]|[A-Z]{2,}[a-z]{2,}[A-Z]{2,}", combined):
            suspicious += 1
    return {
        "record_count": len(record_list),
        "complete": complete,
        "partial": partial,
        "empty_title": empty_title,
        "empty_company": empty_company,
        "has_dates": has_dates,
        "suspicious": suspicious,
    }


def _accept_sgrse_work(
    baseline_records: Sequence[Mapping[str, Any]] | None,
    sgrse_records: Sequence[Mapping[str, Any]] | None,
) -> bool:
    baseline = _work_record_features(baseline_records)
    sgrse = _work_record_features(sgrse_records)
    return sgrse["complete"] > baseline["complete"]


def apply_sgrse_work_decoder(
    predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows = list(read_jsonl(predictions_path))
    repaired_rows = []
    repaired_document_count = 0
    merge_count = 0
    event_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        grounded = GroundedPrediction.model_validate(row)
        prediction_payload = grounded.prediction.model_dump(mode="json")
        candidate_payloads = [
            candidate.model_dump(mode="json") for candidate in grounded.candidates
        ]
        decoded_rows = _decode_sgrse_work_rows(candidate_payloads)
        decoded_work, row_merge_count = _merge_sgrse_work_rows(decoded_rows)
        merge_count += row_merge_count
        repair_events: list[ValidationEvent] = []

        if decoded_work != prediction_payload.get("work_experience", []):
            repair_events.append(
                ValidationEvent(
                    kind="sgrse_work_redecoded",
                    path="work_experience",
                    message="rebuilt work records from grouped candidate slots",
                )
            )
            event_counts["sgrse_work_redecoded"] += 1
            prediction_payload["work_experience"] = decoded_work
        if row_merge_count:
            repair_events.append(
                ValidationEvent(
                    kind="sgrse_work_slot_merged",
                    path="work_experience",
                    message=f"merged {row_merge_count} adjacent work slots",
                )
            )
            event_counts["sgrse_work_slot_merged"] += row_merge_count

        if repair_events:
            repaired_document_count += 1
        repaired_grounded = grounded.model_copy(
            update={
                "prediction": ReducedCVTarget.model_validate(prediction_payload),
                "assembly_events": grounded.assembly_events + repair_events,
            }
        )
        repaired_rows.append(repaired_grounded.model_dump(mode="json"))

    atomic_write_jsonl(output_path, repaired_rows)
    summary = {
        "document_count": len(repaired_rows),
        "repaired_document_count": repaired_document_count,
        "merge_count": merge_count,
        "event_counts": dict(sorted(event_counts.items())),
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def apply_selective_sgrse_work_decoder(
    baseline_predictions_path: Path,
    sgrse_predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    baseline_rows = {str(row["cv_id"]): row for row in read_jsonl(baseline_predictions_path)}
    sgrse_rows = {str(row["cv_id"]): row for row in read_jsonl(sgrse_predictions_path)}
    if set(baseline_rows) != set(sgrse_rows):
        raise ValueError("baseline and sgrse predictions must cover identical cv_ids")

    selected_rows = []
    accepted_count = 0
    event_counts: dict[str, int] = defaultdict(int)

    for cv_id in sorted(baseline_rows):
        baseline = GroundedPrediction.model_validate(baseline_rows[cv_id])
        sgrse = GroundedPrediction.model_validate(sgrse_rows[cv_id])
        baseline_payload = baseline.prediction.model_dump(mode="json")
        sgrse_payload = sgrse.prediction.model_dump(mode="json")
        use_sgrse = _accept_sgrse_work(
            baseline_payload.get("work_experience"),
            sgrse_payload.get("work_experience"),
        )
        events = list(baseline.assembly_events)
        if use_sgrse:
            accepted_count += 1
            baseline_payload["work_experience"] = sgrse_payload.get("work_experience", [])
            events.append(
                ValidationEvent(
                    kind="sgrse_work_selected",
                    path="work_experience",
                    message="accepted sgrse work decoder because complete record count increased",
                )
            )
            event_counts["sgrse_work_selected"] += 1
        selected_rows.append(
            baseline.model_copy(
                update={
                    "prediction": ReducedCVTarget.model_validate(baseline_payload),
                    "assembly_events": events,
                }
            ).model_dump(mode="json")
        )

    atomic_write_jsonl(output_path, selected_rows)
    summary = {
        "document_count": len(selected_rows),
        "accepted_count": accepted_count,
        "event_counts": dict(sorted(event_counts.items())),
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _bootstrap_mean_confidence_interval(
    values: Sequence[float],
    *,
    seed: int = 20260609,
    samples: int = 2_000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[generator.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(draw) / len(draw))
    means.sort()
    lower_index = int((alpha / 2) * (samples - 1))
    upper_index = int((1 - alpha / 2) * (samples - 1))
    return means[lower_index], means[upper_index]


def compare_prediction_sets(
    left_predictions_path: Path,
    right_predictions_path: Path,
    requests_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    requests: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(requests_path):
        cv_id = str(row["cv_id"])
        request = requests.setdefault(
            cv_id,
            {
                "cv_id": cv_id,
                "truth": row["truth"],
                "words": [],
            },
        )
        request["words"].extend(str(word) for word in row.get("words", []))

    left_rows = {str(row["cv_id"]): row for row in read_jsonl(left_predictions_path)}
    right_rows = {str(row["cv_id"]): row for row in read_jsonl(right_predictions_path)}
    if set(left_rows) != set(right_rows):
        raise ValueError("left and right prediction sets must cover identical cv_ids")

    documents = []
    macro_deltas = []
    work_deltas = []
    for cv_id in sorted(left_rows):
        request = requests[cv_id]
        evidence = EvidenceBundle(parser_text=" ".join(request["words"]))
        left_eval = evaluate_document(request["truth"], left_rows[cv_id]["prediction"], evidence)
        right_eval = evaluate_document(request["truth"], right_rows[cv_id]["prediction"], evidence)
        left_work = next(
            field.score for field in left_eval.field_results if field.path == "work_experience"
        )
        right_work = next(
            field.score for field in right_eval.field_results if field.path == "work_experience"
        )
        macro_delta = right_eval.macro_score - left_eval.macro_score
        work_delta = right_work - left_work
        macro_deltas.append(macro_delta)
        work_deltas.append(work_delta)
        documents.append(
            {
                "cv_id": cv_id,
                "left_macro": left_eval.macro_score,
                "right_macro": right_eval.macro_score,
                "macro_delta": macro_delta,
                "left_work": left_work,
                "right_work": right_work,
                "work_delta": work_delta,
            }
        )

    summary = {
        "document_count": len(documents),
        "left_predictions": str(left_predictions_path),
        "right_predictions": str(right_predictions_path),
        "macro_delta_mean": sum(macro_deltas) / len(macro_deltas) if macro_deltas else 0.0,
        "work_delta_mean": sum(work_deltas) / len(work_deltas) if work_deltas else 0.0,
        "macro_win_count": sum(delta > 0 for delta in macro_deltas),
        "macro_loss_count": sum(delta < 0 for delta in macro_deltas),
        "macro_tie_count": sum(delta == 0 for delta in macro_deltas),
        "work_win_count": sum(delta > 0 for delta in work_deltas),
        "work_loss_count": sum(delta < 0 for delta in work_deltas),
        "work_tie_count": sum(delta == 0 for delta in work_deltas),
        "macro_delta_ci95": list(_bootstrap_mean_confidence_interval(macro_deltas)),
        "work_delta_ci95": list(_bootstrap_mean_confidence_interval(work_deltas)),
    }
    atomic_write_json(output_path, summary)
    atomic_write_json(output_path.with_suffix(".documents.json"), documents)
    return summary


def apply_efsfr_repairs(
    predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows = list(read_jsonl(predictions_path))
    repaired_rows = []
    repaired_document_count = 0
    event_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        grounded = GroundedPrediction.model_validate(row)
        prediction_payload = grounded.prediction.model_dump(mode="json")
        candidate_payloads = [
            candidate.model_dump(mode="json") for candidate in grounded.candidates
        ]
        repair_events: list[ValidationEvent] = []

        repaired_work = []
        for record in prediction_payload.get("work_experience", []):
            repaired_record, events = repair_work_record(record)
            repaired_work.append(repaired_record)
            for event in events:
                repair_events.append(event)
                event_counts[event.kind] += 1
        prediction_payload["work_experience"] = repaired_work

        decoded_education = _decode_anchor_records(
            candidate_payloads,
            prefix="education.*.",
            anchor_field="degree",
            fields=("degree", "field_of_study", "institution", "graduation_date"),
        )
        if decoded_education != prediction_payload.get("education", []):
            repair_events.append(
                ValidationEvent(
                    kind="education_anchor_redecoded",
                    path="education",
                    message="rebuilt education records from candidate evidence order",
                )
            )
            event_counts["education_anchor_redecoded"] += 1
            prediction_payload["education"] = decoded_education

        decoded_projects = _decode_anchor_records(
            candidate_payloads,
            prefix="projects.*.",
            anchor_field="name",
            fields=("name", "technologies", "url"),
        )
        normalized_projects = decoded_projects or None
        if normalized_projects != prediction_payload.get("projects"):
            repair_events.append(
                ValidationEvent(
                    kind="projects_anchor_redecoded",
                    path="projects",
                    message="rebuilt project records from candidate evidence order",
                )
            )
            event_counts["projects_anchor_redecoded"] += 1
            prediction_payload["projects"] = normalized_projects

        decoded_certifications = _decode_anchor_records(
            candidate_payloads,
            prefix="certifications.*.",
            anchor_field="name",
            fields=("name", "issuer", "date"),
        )
        normalized_certifications = decoded_certifications or None
        if normalized_certifications != prediction_payload.get("certifications"):
            repair_events.append(
                ValidationEvent(
                    kind="certifications_anchor_redecoded",
                    path="certifications",
                    message="rebuilt certification records from candidate evidence order",
                )
            )
            event_counts["certifications_anchor_redecoded"] += 1
            prediction_payload["certifications"] = normalized_certifications

        if repair_events:
            repaired_document_count += 1
        repaired_grounded = grounded.model_copy(
            update={
                "prediction": ReducedCVTarget.model_validate(prediction_payload),
                "assembly_events": grounded.assembly_events + repair_events,
            }
        )
        repaired_rows.append(repaired_grounded.model_dump(mode="json"))

    atomic_write_jsonl(output_path, repaired_rows)
    summary = {
        "document_count": len(repaired_rows),
        "repaired_document_count": repaired_document_count,
        "event_counts": dict(sorted(event_counts.items())),
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def apply_project_skill_tech_repairs(
    predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows = list(read_jsonl(predictions_path))
    repaired_rows = []
    repaired_document_count = 0
    repaired_project_count = 0
    event_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        grounded = GroundedPrediction.model_validate(row)
        prediction_payload = grounded.prediction.model_dump(mode="json")
        current_projects = prediction_payload.get("projects") or []
        candidate_payloads = [
            candidate.model_dump(mode="json") for candidate in grounded.candidates
        ]
        decoded_projects = _decode_project_records(
            candidate_payloads,
            prediction_payload.get("skills", []),
        )
        repair_events: list[ValidationEvent] = []

        if current_projects and len(decoded_projects) == len(current_projects):
            repaired_projects: list[dict[str, Any]] = []
            changed = False
            for current_record, decoded_record in zip(
                current_projects,
                decoded_projects,
                strict=True,
            ):
                if normalize_text(current_record.get("name")) != normalize_text(
                    decoded_record.get("name")
                ):
                    repaired_projects = []
                    changed = False
                    break
                technologies = decoded_record["technologies"] or current_record.get(
                    "technologies",
                    [],
                )
                repaired_record = dict(current_record)
                if technologies != current_record.get("technologies", []):
                    changed = True
                    repaired_project_count += 1
                    repair_events.append(
                        ValidationEvent(
                            kind="project_tech_skill_repaired",
                            path="projects.*.technologies",
                            message=(
                                "replaced noisy project technologies using "
                                "skill-aligned candidate cleanup"
                            ),
                        )
                    )
                    event_counts["project_tech_skill_repaired"] += 1
                    repaired_record["technologies"] = technologies
                repaired_projects.append(repaired_record)
            if changed and repaired_projects:
                repaired_document_count += 1
                prediction_payload["projects"] = repaired_projects

        repaired_grounded = grounded.model_copy(
            update={
                "prediction": ReducedCVTarget.model_validate(prediction_payload),
                "assembly_events": grounded.assembly_events + repair_events,
            }
        )
        repaired_rows.append(repaired_grounded.model_dump(mode="json"))

    atomic_write_jsonl(output_path, repaired_rows)
    summary = {
        "document_count": len(repaired_rows),
        "repaired_document_count": repaired_document_count,
        "repaired_project_count": repaired_project_count,
        "event_counts": dict(sorted(event_counts.items())),
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def apply_project_url_repairs(
    predictions_path: Path,
    train_records_path: Path,
    output_path: Path,
    *,
    threshold: float = PROJECT_URL_DEFAULT_THRESHOLD,
    min_full_count: int = PROJECT_URL_DEFAULT_MIN_FULL_COUNT,
    min_reduced_count: int = PROJECT_URL_DEFAULT_MIN_REDUCED_COUNT,
) -> dict[str, Any]:
    rows = list(read_jsonl(predictions_path))
    train_records = list(read_jsonl(train_records_path))
    stats = _project_url_selector_stats(train_records)
    repaired_rows = []
    repaired_document_count = 0
    repaired_project_count = 0
    event_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        grounded = GroundedPrediction.model_validate(row)
        prediction_payload = grounded.prediction.model_dump(mode="json")
        owner = _github_owner(prediction_payload.get("github_url"))
        repair_events: list[ValidationEvent] = []
        changed = False

        if owner is not None:
            repaired_projects = []
            for index, project in enumerate(prediction_payload.get("projects") or []):
                repaired_project = dict(project)
                if project.get("name") and not project.get("url"):
                    probability = _project_url_probability(
                        stats,
                        prediction_payload.get("projects") or [],
                        index,
                        project,
                        min_full_count=min_full_count,
                        min_reduced_count=min_reduced_count,
                    )
                    if probability is not None and probability >= threshold:
                        repaired_project["url"] = (
                            f'https://github.com/{owner}/{_project_slug(str(project["name"]))}'
                        )
                        changed = True
                        repaired_project_count += 1
                        repair_events.append(
                            ValidationEvent(
                                kind="project_url_synthesized",
                                path="projects.*.url",
                                message=(
                                    "filled project URL from train-derived presence "
                                    "selector and predicted GitHub owner"
                                ),
                            )
                        )
                        event_counts["project_url_synthesized"] += 1
                repaired_projects.append(repaired_project)
            if changed:
                repaired_document_count += 1
                prediction_payload["projects"] = repaired_projects

        repaired_grounded = grounded.model_copy(
            update={
                "prediction": ReducedCVTarget.model_validate(prediction_payload),
                "assembly_events": grounded.assembly_events + repair_events,
            }
        )
        repaired_rows.append(repaired_grounded.model_dump(mode="json"))

    atomic_write_jsonl(output_path, repaired_rows)
    summary = {
        "document_count": len(repaired_rows),
        "repaired_document_count": repaired_document_count,
        "repaired_project_count": repaired_project_count,
        "event_counts": dict(sorted(event_counts.items())),
        "threshold": threshold,
        "min_full_count": min_full_count,
        "min_reduced_count": min_reduced_count,
        "selector_document_count": stats["document_count"],
        "selector_global_positive_rate": stats["global_positive_rate"],
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary
