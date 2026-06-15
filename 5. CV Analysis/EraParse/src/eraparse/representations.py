import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from eraparse.constants import DEFAULT_REPRESENTATION_ROOT, REDUCED_SCHEMA_TEMPLATE
from eraparse.data import load_manifest
from eraparse.io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_json,
    read_jsonl,
)
from eraparse.models import ManifestRow

RepresentationName = Literal[
    "pymupdf_text",
    "pdfminer_text",
    "oracle_text",
    "pymupdf_tesseract_fallback",
    "pymupdf4llm_markdown",
    "pymupdf4llm_json",
    "docling_markdown",
    "docling_json",
    "paddleocr_vl_markdown",
    "paddleocr_vl_json",
]
REPRESENTATION_ARTIFACTS: dict[RepresentationName, str] = {
    "pymupdf_text": "pymupdf_text",
    "pdfminer_text": "pdfminer_text",
    "oracle_text": "text_ground_truth",
}
GENERATED_REPRESENTATIONS = {
    "pymupdf4llm_markdown",
    "pymupdf4llm_json",
    "docling_markdown",
    "docling_json",
    "paddleocr_vl_markdown",
    "paddleocr_vl_json",
}
DERIVED_REPRESENTATIONS = {"pymupdf_tesseract_fallback"}


def read_representation(
    row: ManifestRow,
    dataset_root: Path,
    representation: RepresentationName,
    representation_root: Path = DEFAULT_REPRESENTATION_ROOT,
) -> str:
    if representation in GENERATED_REPRESENTATIONS:
        suffix = ".json" if representation.endswith("_json") else ".md"
        path = representation_root / representation / f"{row.cv_id}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"generated representation is missing: {path}")
        return path.read_text(encoding="utf-8")
    if representation == "pymupdf_tesseract_fallback":
        pdf_text = (dataset_root / row.artifacts["pymupdf_text"].path).read_text(encoding="utf-8")
        if pdf_text.strip():
            return pdf_text
        ocr_artifact = row.artifacts.get("tesseract_ocr")
        if ocr_artifact is None:
            raise FileNotFoundError(f"Tesseract OCR fallback is missing for {row.cv_id}")
        return (dataset_root / ocr_artifact.path).read_text(encoding="utf-8")
    artifact_kind = REPRESENTATION_ARTIFACTS[representation]
    return (dataset_root / row.artifacts[artifact_kind].path).read_text(encoding="utf-8")


def build_nuextract_prompt(text: str, template: dict[str, Any] | None = None) -> str:
    rendered_template = json.dumps(template or REDUCED_SCHEMA_TEMPLATE, indent=2)
    return f"<|input|>\n### Template:\n{rendered_template}\n### Text:\n{text}\n\n<|output|>"


def build_trial_requests(
    manifest_path: Path,
    dataset_root: Path,
    representation: RepresentationName,
    representation_root: Path = DEFAULT_REPRESENTATION_ROOT,
) -> list[dict[str, Any]]:
    rows = load_manifest(manifest_path)
    requests: list[dict[str, Any]] = []
    for row in rows:
        text = read_representation(row, dataset_root, representation, representation_root)
        requests.append(
            {
                "cv_id": row.cv_id,
                "split": row.split,
                "tier": row.tier,
                "template": row.template,
                "primary_domain": row.primary_domain,
                "representation": representation,
                "text": text,
                "prompt": build_nuextract_prompt(text),
                "truth": read_json(dataset_root / row.artifacts["schema_reduced"].path),
            }
        )
    return requests


def write_trial_requests(path: Path, requests: Sequence[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, requests)


def materialize_paddleocr_vl_outputs(
    responses_path: Path,
    representation_root: Path = DEFAULT_REPRESENTATION_ROOT,
) -> dict[str, Any]:
    markdown_dir = representation_root / "paddleocr_vl_markdown"
    json_dir = representation_root / "paddleocr_vl_json"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    failures: list[dict[str, str]] = []
    for row in read_jsonl(responses_path):
        cv_id = str(row["cv_id"])
        error = row.get("error")
        if error:
            failures.append({"cv_id": cv_id, "error": str(error)})
            continue
        markdown = row.get("markdown")
        page_json = row.get("page_json")
        if not isinstance(markdown, str):
            failures.append({"cv_id": cv_id, "error": "missing markdown output"})
            continue
        if not isinstance(page_json, list):
            failures.append({"cv_id": cv_id, "error": "missing page_json output"})
            continue
        atomic_write_text(markdown_dir / f"{cv_id}.md", markdown)
        atomic_write_json(json_dir / f"{cv_id}.json", page_json)
        completed += 1

    summary = {
        "responses": str(responses_path),
        "representation_root": str(representation_root),
        "completed": completed,
        "failures": failures,
        "failure_count": len(failures),
    }
    atomic_write_json(markdown_dir / "generation_summary.json", summary)
    atomic_write_json(json_dir / "generation_summary.json", summary)
    return summary
