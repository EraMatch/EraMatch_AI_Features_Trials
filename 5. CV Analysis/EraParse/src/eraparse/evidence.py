import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from eraparse.constants import DEFAULT_REPRESENTATION_ROOT
from eraparse.data import load_manifest
from eraparse.evaluate import normalize_text
from eraparse.io import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl
from eraparse.models import EvidenceGraph, EvidenceUnit

EvidenceReader = Literal["pymupdf4llm", "source_oracle"]
SCALAR_TRUTH_FIELDS = (
    "full_name",
    "email",
    "location",
    "phone",
    "linkedin_url",
    "github_url",
    "summary",
)
NESTED_TRUTH_FIELDS = ("work_experience", "education", "projects", "certifications")


def canonical_field_path(path: str | None) -> tuple[str | None, int | None]:
    if not path:
        return None, None
    path = path.removeprefix("contact_info.")
    match = re.match(
        r"^(work_experience|education|projects|certifications)\.(\d+)\.(.+)$",
        path,
    )
    if match:
        nested_field = re.sub(r"\.\d+$", "", match.group(3))
        return f"{match.group(1)}.*.{nested_field}", int(match.group(2))
    if re.match(r"^skills\.\d+(?:\.skill_name)?$", path):
        return "skills", None
    return path, None


def truth_field_annotations(truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []

    def add(path: str, value: Any, *, alignment_group: str | None = None) -> None:
        if value is not None and str(value).strip():
            annotation = {
                "field_path": path,
                "text": str(value),
                "order_index": len(annotations),
            }
            if alignment_group is not None:
                annotation["alignment_group"] = alignment_group
            annotations.append(annotation)

    for field in SCALAR_TRUTH_FIELDS:
        add(field, truth.get(field))
    for index, skill in enumerate(truth.get("skills") or []):
        add(f"skills.{index}.skill_name", skill, alignment_group="skills")
    for kind in NESTED_TRUTH_FIELDS:
        for record_index, record in enumerate(truth.get(kind) or []):
            for field, value in record.items():
                if isinstance(value, list):
                    for value_index, item in enumerate(value):
                        add(
                            f"{kind}.{record_index}.{field}.{value_index}",
                            item,
                            alignment_group=f"{kind}.{record_index}.{field}",
                        )
                else:
                    add(f"{kind}.{record_index}.{field}", value)
    return annotations


def _normalize_box(box: Sequence[float], width: float, height: float) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("page dimensions must be positive")
    x0, y0, x1, y1 = box
    values = (
        round(1000 * x0 / width),
        round(1000 * y0 / height),
        round(1000 * x1 / width),
        round(1000 * y1 / height),
    )
    return tuple(max(0, min(1000, value)) for value in values)  # type: ignore[return-value]


def _split_span(
    text: str,
    bbox: Sequence[float],
) -> list[tuple[str, tuple[float, float, float, float]]]:
    words = [match for match in re.finditer(r"\S+", text)]
    if not words:
        return []
    x0, y0, x1, y1 = bbox
    width = max(x1 - x0, 1.0)
    length = max(len(text), 1)
    return [
        (
            match.group(),
            (
                x0 + width * match.start() / length,
                y0,
                x0 + width * match.end() / length,
                y1,
            ),
        )
        for match in words
    ]


def _box_words(
    box: Mapping[str, Any],
) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    for line in box.get("textlines") or []:
        spans = line.get("spans") or []
        if not spans:
            continue
        text_parts = [str(spans[0].get("text", ""))]
        for previous, span in pairwise(spans):
            previous_box = previous["bbox"]
            span_box = span["bbox"]
            line_height = max(
                float(previous_box[3]) - float(previous_box[1]),
                float(span_box[3]) - float(span_box[1]),
                1.0,
            )
            gap = float(span_box[0]) - float(previous_box[2])
            previous_text = str(previous.get("text", ""))
            span_text = str(span.get("text", ""))
            if (
                gap > line_height * 0.25
                and not previous_text.endswith((" ", "\t"))
                and not span_text.startswith((" ", "\t"))
            ):
                text_parts.append(" ")
            text_parts.append(span_text)
        line_box = line.get("bbox") or (
            spans[0]["bbox"][0],
            min(span["bbox"][1] for span in spans),
            spans[-1]["bbox"][2],
            max(span["bbox"][3] for span in spans),
        )
        yield from _split_span("".join(text_parts), line_box)
    table = box.get("table")
    if not table:
        return
    for cell_row, text_row in zip(
        table.get("cells") or [],
        table.get("extract") or [],
        strict=False,
    ):
        for cell, text in zip(cell_row, text_row, strict=False):
            if cell is not None:
                yield from _split_span(str(text or ""), cell)


def graph_from_pymupdf4llm_json(
    cv_id: str,
    value: Mapping[str, Any],
    *,
    page_images: Sequence[str] = (),
) -> EvidenceGraph:
    units: list[EvidenceUnit] = []
    reading_order = 0
    for page in value.get("pages", []):
        page_number = int(page.get("page_number", len(units) + 1))
        width = float(page["width"])
        height = float(page["height"])
        for box in page.get("boxes", []):
            for word, word_box in _box_words(box):
                units.append(
                    EvidenceUnit(
                        evidence_id=f"{cv_id}:p{page_number}:w{reading_order}",
                        text=word,
                        page=page_number,
                        bbox_norm=_normalize_box(word_box, width, height),
                        source="pymupdf4llm",
                        reading_order=reading_order,
                    )
                )
                reading_order += 1
    return EvidenceGraph(
        cv_id=cv_id,
        reader="pymupdf4llm",
        units=units,
        page_images=list(page_images),
        metadata={"page_count": int(value.get("page_count", 0))},
    )


def graph_from_source_annotations(
    cv_id: str,
    words: Sequence[Mapping[str, Any]],
    *,
    page_images: Sequence[str] = (),
    include_labels: bool = True,
) -> EvidenceGraph:
    units = []
    for index, word in enumerate(words):
        field_path, record_index = canonical_field_path(word.get("field_path"))
        if not include_labels:
            field_path, record_index = None, None
        units.append(
            EvidenceUnit(
                evidence_id=f"{cv_id}:p{word['page']}:w{index}",
                text=str(word["text"]),
                page=int(word["page"]),
                bbox_norm=tuple(int(value) for value in word["bbox_norm"]),  # type: ignore[arg-type]
                source="source_oracle",
                reading_order=index,
                field_path=field_path,
                record_index=record_index,
            )
        )
    return EvidenceGraph(
        cv_id=cv_id,
        reader="source_oracle",
        oracle=True,
        units=units,
        page_images=list(page_images),
    )


def align_graph_to_fields(
    graph: EvidenceGraph,
    annotations: Sequence[Mapping[str, Any]],
) -> EvidenceGraph:
    units = list(graph.units)
    consumed = {index for index, unit in enumerate(units) if unit.field_path is not None}
    ordered_annotations = sorted(annotations, key=lambda item: int(item.get("order_index", 0)))

    def annotation_identity(annotation: Mapping[str, Any]) -> tuple[str | None, int | None, str]:
        field_path, record_index = canonical_field_path(str(annotation.get("field_path", "")))
        annotation_text = normalize_text(annotation.get("text"))
        return field_path, record_index, annotation_text

    def annotation_satisfied(annotation: Mapping[str, Any]) -> bool:
        field_path, record_index, annotation_text = annotation_identity(annotation)
        labeled_text = normalize_text(
            " ".join(
                unit.text
                for unit in units
                if unit.field_path == field_path and unit.record_index == record_index
            )
        )
        return bool(annotation_text and annotation_text in labeled_text)

    def candidate_spans(annotation: Mapping[str, Any]) -> list[list[int]]:
        field_path, _, annotation_text = annotation_identity(annotation)
        page_value = annotation.get("page")
        if not field_path or not annotation_text:
            return []
        spans = []
        for start, unit in enumerate(units):
            if start in consumed or (page_value is not None and unit.page != int(page_value)):
                continue
            if not normalize_text(unit.text):
                continue
            candidate_indices = []
            for index in range(start, len(units)):
                if index in consumed or units[index].page != unit.page:
                    break
                candidate_indices.append(index)
                candidate_text = normalize_text(
                    " ".join(units[item].text for item in candidate_indices)
                )
                if candidate_text == annotation_text:
                    spans.append(candidate_indices)
                    break
                if len(candidate_text) >= len(annotation_text):
                    break
        return spans

    def assign(annotation: Mapping[str, Any], matched_indices: Sequence[int]) -> None:
        field_path, record_index, _ = annotation_identity(annotation)
        for index in matched_indices:
            units[index] = units[index].model_copy(
                update={"field_path": field_path, "record_index": record_index}
            )
        consumed.update(matched_indices)

    def align_compact_group(group: Sequence[Mapping[str, Any]]) -> None:
        pending = [annotation for annotation in group if not annotation_satisfied(annotation)]
        candidates = [candidate_spans(annotation) for annotation in pending]
        plans: list[list[tuple[int, list[int]]]] = []
        for annotation_index, annotation_candidates in enumerate(candidates):
            for first_span in annotation_candidates:
                plan = [(annotation_index, first_span)]
                last_index = first_span[-1]
                for next_index in range(annotation_index + 1, len(candidates)):
                    next_span = next(
                        (span for span in candidates[next_index] if span[0] > last_index),
                        None,
                    )
                    if next_span is not None:
                        plan.append((next_index, next_span))
                        last_index = next_span[-1]
                plans.append(plan)
        if not plans:
            return
        selected = min(
            plans,
            key=lambda plan: (
                -len(plan),
                plan[-1][1][-1] - plan[0][1][0],
                plan[0][1][0],
            ),
        )
        for annotation_index, span in selected:
            assign(pending[annotation_index], span)

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for annotation in ordered_annotations:
        group = annotation.get("alignment_group")
        if group is not None:
            groups.setdefault(str(group), []).append(annotation)

    aligned_groups: set[str] = set()
    for annotation in ordered_annotations:
        group = annotation.get("alignment_group")
        if group is not None:
            group_name = str(group)
            if group_name not in aligned_groups:
                align_compact_group(groups[group_name])
                aligned_groups.add(group_name)
            continue
        if annotation_satisfied(annotation):
            continue
        spans = candidate_spans(annotation)
        if spans:
            assign(annotation, spans[0])
    return graph.model_copy(update={"units": units})


def build_evidence_graphs(
    manifest_path: Path,
    dataset_root: Path,
    output_path: Path,
    *,
    reader: EvidenceReader,
    representation_root: Path = DEFAULT_REPRESENTATION_ROOT,
) -> dict[str, Any]:
    graphs: list[EvidenceGraph] = []
    for row in load_manifest(manifest_path):
        page_images = [artifact.path for artifact in row.page_images]
        if reader == "source_oracle":
            graph = graph_from_source_annotations(
                row.cv_id,
                read_json(dataset_root / row.artifacts["word_annotations"].path),
                page_images=page_images,
                include_labels=False,
            )
        else:
            representation_path = representation_root / "pymupdf4llm_json" / f"{row.cv_id}.json"
            if not representation_path.is_file():
                raise FileNotFoundError(f"missing PyMuPDF4LLM JSON: {representation_path}")
            graph = graph_from_pymupdf4llm_json(
                row.cv_id,
                read_json(representation_path),
                page_images=page_images,
            )
        truth = read_json(dataset_root / row.artifacts["schema_reduced"].path)
        graph = align_graph_to_fields(
            graph,
            read_json(dataset_root / row.artifacts["field_annotations"].path),
        )
        graph = align_graph_to_fields(graph, truth_field_annotations(truth))
        graphs.append(graph)
    atomic_write_jsonl(output_path, [graph.model_dump(mode="json") for graph in graphs])
    summary = validate_evidence_graphs(graphs, allow_oracle=reader == "source_oracle")
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def validate_evidence_graphs(
    graphs: Iterable[EvidenceGraph],
    *,
    allow_oracle: bool = False,
) -> dict[str, Any]:
    graph_list = list(graphs)
    issues: list[str] = []
    for graph in graph_list:
        if graph.oracle and not allow_oracle:
            issues.append(f"{graph.cv_id}: oracle evidence is not allowed")
        ids = [unit.evidence_id for unit in graph.units]
        if len(ids) != len(set(ids)):
            issues.append(f"{graph.cv_id}: duplicate evidence IDs")
        if not graph.units:
            issues.append(f"{graph.cv_id}: no evidence units")
        for unit in graph.units:
            if any(value < 0 or value > 1000 for value in unit.bbox_norm):
                issues.append(f"{graph.cv_id}: invalid normalized box")
    return {
        "graph_count": len(graph_list),
        "unit_count": sum(len(graph.units) for graph in graph_list),
        "oracle_count": sum(graph.oracle for graph in graph_list),
        "issues": issues,
        "passed": not issues,
    }


def read_evidence_graphs(path: Path) -> list[EvidenceGraph]:
    return [EvidenceGraph.model_validate(row) for row in read_jsonl(path)]
