import json
import time
from pathlib import Path
from typing import Any, Literal

from eraparse.constants import DEFAULT_REPRESENTATION_ROOT
from eraparse.data import load_manifest
from eraparse.io import atomic_write_json, atomic_write_text

ParserRepresentation = Literal[
    "pymupdf4llm_markdown",
    "pymupdf4llm_json",
    "docling_markdown",
    "docling_json",
]


def _write_value(path: Path, value: Any) -> None:
    if path.suffix == ".json":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {"content": value}
        atomic_write_json(path, value)
    else:
        atomic_write_text(path, str(value))


def generate_pymupdf4llm(pdf_path: Path, representation: ParserRepresentation) -> Any:
    import pymupdf4llm

    if representation == "pymupdf4llm_markdown":
        return pymupdf4llm.to_markdown(str(pdf_path))
    if representation == "pymupdf4llm_json":
        return pymupdf4llm.to_json(str(pdf_path))
    raise ValueError(f"unsupported PyMuPDF4LLM representation: {representation}")


def generate_docling(pdf_path: Path, representation: ParserRepresentation) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    document = converter.convert(pdf_path).document
    if representation == "docling_markdown":
        return document.export_to_markdown()
    if representation == "docling_json":
        return document.export_to_dict()
    raise ValueError(f"unsupported Docling representation: {representation}")


def generate_representations(
    manifest_path: Path,
    dataset_root: Path,
    representation: ParserRepresentation,
    output_root: Path = DEFAULT_REPRESENTATION_ROOT,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    rows = load_manifest(manifest_path)
    output_dir = output_root / representation
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        suffix = ".json" if representation.endswith("_json") else ".md"
        output_path = output_dir / f"{row.cv_id}{suffix}"
        if output_path.is_file() and not overwrite:
            records.append({"cv_id": row.cv_id, "cached": True, "output": str(output_path)})
            continue
        pdf_path = dataset_root / row.artifacts["pdf"].path
        started = time.perf_counter()
        try:
            if representation.startswith("pymupdf4llm"):
                value = generate_pymupdf4llm(pdf_path, representation)
            else:
                value = generate_docling(pdf_path, representation)
            _write_value(output_path, value)
            records.append(
                {
                    "cv_id": row.cv_id,
                    "cached": False,
                    "output": str(output_path),
                    "latency_seconds": time.perf_counter() - started,
                    "size_bytes": output_path.stat().st_size,
                }
            )
        except Exception as error:  # Parser failures must be recorded, not abort the batch.
            failures.append({"cv_id": row.cv_id, "error": str(error)})
        print(f"{representation}: {index}/{len(rows)}")
    summary = {
        "representation": representation,
        "manifest": str(manifest_path),
        "completed": len(records),
        "failures": failures,
        "records": records,
    }
    atomic_write_json(output_dir / f"generation_summary_{manifest_path.stem}.json", summary)
    atomic_write_json(output_dir / "generation_summary.json", summary)
    return summary
