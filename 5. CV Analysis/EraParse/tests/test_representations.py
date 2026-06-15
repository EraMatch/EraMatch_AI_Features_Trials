import json
from pathlib import Path

from eraparse.io import atomic_write_jsonl, read_json
from eraparse.models import ArtifactReference, ManifestRow
from eraparse.representations import (
    GENERATED_REPRESENTATIONS,
    build_nuextract_prompt,
    materialize_paddleocr_vl_outputs,
    read_representation,
)


def test_prompt_contains_template_text_and_output_marker() -> None:
    prompt = build_nuextract_prompt("Jane Doe")
    assert "### Template:" in prompt
    assert "### Text:\nJane Doe" in prompt
    assert prompt.endswith("<|output|>")


def test_generated_representations_are_available_to_trials() -> None:
    assert "docling_markdown" in GENERATED_REPRESENTATIONS
    assert "pymupdf4llm_json" in GENERATED_REPRESENTATIONS
    assert "paddleocr_vl_markdown" in GENERATED_REPRESENTATIONS
    assert "paddleocr_vl_json" in GENERATED_REPRESENTATIONS


def test_reads_selected_representation(tmp_path: Path) -> None:
    text_path = tmp_path / "text.txt"
    text_path.write_text("CV text", encoding="utf-8")
    artifact = ArtifactReference(
        kind="pymupdf_text",
        path="text.txt",
        sha256="abc",
        size_bytes=7,
    )
    row = ManifestRow(
        cv_id="cv_00001",
        tier="T1",
        template="T1_classic",
        primary_domain="Backend Engineering",
        split="debug_50",
        selection_seed=20260609,
        artifacts={"pymupdf_text": artifact},
        page_images=[],
    )
    assert read_representation(row, tmp_path, "pymupdf_text") == "CV text"


def test_hybrid_representation_uses_ocr_only_when_pdf_text_is_empty(tmp_path: Path) -> None:
    pdf_text = tmp_path / "pdf.txt"
    ocr_text = tmp_path / "ocr.txt"
    pdf_text.write_text("Digital CV text", encoding="utf-8")
    ocr_text.write_text("OCR fallback text", encoding="utf-8")
    artifacts = {
        "pymupdf_text": ArtifactReference(
            kind="pymupdf_text", path="pdf.txt", sha256="a", size_bytes=15
        ),
        "tesseract_ocr": ArtifactReference(
            kind="tesseract_ocr", path="ocr.txt", sha256="b", size_bytes=17
        ),
    }
    row = ManifestRow(
        cv_id="cv_00001",
        tier="T4",
        template="T4_scan",
        primary_domain="Backend Engineering",
        split="debug_50",
        selection_seed=20260609,
        artifacts=artifacts,
        page_images=[],
    )

    assert read_representation(row, tmp_path, "pymupdf_tesseract_fallback") == "Digital CV text"
    pdf_text.write_text(" \n", encoding="utf-8")
    assert read_representation(row, tmp_path, "pymupdf_tesseract_fallback") == "OCR fallback text"


def test_materialize_paddleocr_vl_outputs_writes_generated_representations(
    tmp_path: Path,
) -> None:
    responses_path = tmp_path / "responses.jsonl"
    atomic_write_jsonl(
        responses_path,
        [
            {
                "cv_id": "cv_00001",
                "markdown": "# Jane Doe",
                "page_json": [{"page": 1, "blocks": ["Jane Doe"]}],
            },
            {
                "cv_id": "cv_00002",
                "error": "runtime failure",
            },
        ],
    )

    summary = materialize_paddleocr_vl_outputs(responses_path, tmp_path / "representations")

    assert summary["completed"] == 1
    assert summary["failure_count"] == 1
    markdown_path = tmp_path / "representations" / "paddleocr_vl_markdown" / "cv_00001.md"
    json_path = tmp_path / "representations" / "paddleocr_vl_json" / "cv_00001.json"
    assert markdown_path.read_text(encoding="utf-8") == "# Jane Doe"
    assert read_json(json_path) == [{"page": 1, "blocks": ["Jane Doe"]}]
    summary_path = (
        tmp_path / "representations" / "paddleocr_vl_markdown" / "generation_summary.json"
    )
    assert json.loads(summary_path.read_text(encoding="utf-8"))["failure_count"] == 1
