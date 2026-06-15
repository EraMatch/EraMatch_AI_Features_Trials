import json
from pathlib import Path

import pytest
from PIL import Image

from eraparse.donut import (
    TASK_PROMPT,
    build_donut_records,
    build_visual_trial_requests,
    compose_page_images,
    donut_native_special_tokens,
    serialize_donut_native_target,
    serialize_donut_target,
)
from eraparse.io import atomic_write_jsonl
from eraparse.models import ArtifactReference, ManifestRow
from eraparse.training_utils import gradient_accumulation_step


def test_serialize_donut_target_is_compact_deterministic_json(
    reduced_target: dict[str, object],
) -> None:
    serialized = serialize_donut_target(reduced_target)
    assert not serialized.startswith(TASK_PROMPT)
    assert "\n" not in serialized
    assert json.loads(serialized) == reduced_target


def test_native_donut_target_uses_schema_tokens_and_list_separator() -> None:
    target = {"name": "Jane", "skills": ["Python", "SQL"]}
    serialized = serialize_donut_native_target(target)
    assert serialized == "<s_name>Jane</s_name><s_skills>Python<sep/>SQL</s_skills>"
    assert donut_native_special_tokens(target) == {
        "<s_eraparse>",
        "<s_name>",
        "</s_name>",
        "<s_skills>",
        "</s_skills>",
        "<sep/>",
    }


def test_compose_page_images_stacks_pages_in_reading_order(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (20, 10), "red").save(first)
    Image.new("RGB", (10, 20), "blue").save(second)

    composed = compose_page_images([first, second], separator_pixels=2)

    assert composed.size == (20, 52)
    assert composed.getpixel((0, 0)) == (255, 0, 0)
    assert composed.getpixel((0, 51)) == (0, 0, 255)


def test_build_donut_records_rejects_locked_manifest(
    tmp_path: Path,
    reduced_target: dict[str, object],
) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "targets").mkdir(parents=True)
    (dataset_root / "pages").mkdir()
    target_path = dataset_root / "targets" / "cv_00001.json"
    target_path.write_text(json.dumps(reduced_target), encoding="utf-8")
    page_path = dataset_root / "pages" / "cv_00001-1.png"
    Image.new("RGB", (20, 10), "white").save(page_path)
    row = ManifestRow(
        cv_id="cv_00001",
        tier="T1",
        template="T1_classic",
        primary_domain="Backend Engineering",
        split="locked_confirmation",
        selection_seed=20260609,
        artifacts={
            "schema_reduced": ArtifactReference(
                kind="schema_reduced",
                path="targets/cv_00001.json",
                sha256="a",
                size_bytes=target_path.stat().st_size,
            )
        },
        page_images=[
            ArtifactReference(
                kind="page_image",
                path="pages/cv_00001-1.png",
                sha256="b",
                size_bytes=page_path.stat().st_size,
            )
        ],
    )
    manifest_path = tmp_path / "locked_confirmation.jsonl"
    atomic_write_jsonl(manifest_path, [row.model_dump(mode="json")])

    with pytest.raises(PermissionError):
        build_donut_records(manifest_path, dataset_root)


def test_build_visual_trial_requests_includes_full_schema_context(
    tmp_path: Path,
    reduced_target: dict[str, object],
) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "targets").mkdir(parents=True)
    (dataset_root / "text").mkdir()
    (dataset_root / "pages").mkdir()
    target_path = dataset_root / "targets" / "cv_00002.json"
    target_path.write_text(json.dumps(reduced_target), encoding="utf-8")
    text_path = dataset_root / "text" / "cv_00002.txt"
    text_path.write_text("Extracted PDF text", encoding="utf-8")
    page_path = dataset_root / "pages" / "cv_00002-1.png"
    Image.new("RGB", (20, 10), "white").save(page_path)
    row = ManifestRow(
        cv_id="cv_00002",
        tier="T2",
        template="T2_modern",
        primary_domain="Data Engineering",
        split="validation",
        selection_seed=20260609,
        artifacts={
            "schema_reduced": ArtifactReference(
                kind="schema_reduced",
                path="targets/cv_00002.json",
                sha256="a",
                size_bytes=target_path.stat().st_size,
            ),
            "pymupdf_text": ArtifactReference(
                kind="pymupdf_text",
                path="text/cv_00002.txt",
                sha256="b",
                size_bytes=text_path.stat().st_size,
            ),
        },
        page_images=[
            ArtifactReference(
                kind="page_image",
                path="pages/cv_00002-1.png",
                sha256="c",
                size_bytes=page_path.stat().st_size,
            )
        ],
    )
    manifest_path = tmp_path / "validation.jsonl"
    atomic_write_jsonl(manifest_path, [row.model_dump(mode="json")])

    requests = build_visual_trial_requests(
        manifest_path,
        dataset_root,
        model_family="nuextract3",
    )

    assert len(requests) == 1
    assert requests[0]["cv_id"] == "cv_00002"
    assert requests[0]["model_family"] == "nuextract3"
    assert requests[0]["page_images"] == ["pages/cv_00002-1.png"]
    assert requests[0]["truth"] == reduced_target
    assert requests[0]["evidence_text"] == "Extracted PDF text"
    assert "work_experience" in requests[0]["schema_template"]


def test_build_visual_trial_requests_supports_compact_schema(
    tmp_path: Path,
    reduced_target: dict[str, object],
) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "targets").mkdir(parents=True)
    (dataset_root / "text").mkdir()
    (dataset_root / "pages").mkdir()
    target_path = dataset_root / "targets" / "cv_00002.json"
    target_path.write_text(json.dumps(reduced_target), encoding="utf-8")
    text_path = dataset_root / "text" / "cv_00002.txt"
    text_path.write_text("Extracted PDF text", encoding="utf-8")
    page_path = dataset_root / "pages" / "cv_00002-1.png"
    Image.new("RGB", (20, 10), "white").save(page_path)
    row = ManifestRow(
        cv_id="cv_00002",
        tier="T2",
        template="T2_modern",
        primary_domain="Data Engineering",
        split="debug_250",
        selection_seed=20260609,
        artifacts={
            "schema_reduced": ArtifactReference(
                kind="schema_reduced",
                path="targets/cv_00002.json",
                sha256="a",
                size_bytes=target_path.stat().st_size,
            ),
            "pymupdf_text": ArtifactReference(
                kind="pymupdf_text",
                path="text/cv_00002.txt",
                sha256="b",
                size_bytes=text_path.stat().st_size,
            ),
        },
        page_images=[
            ArtifactReference(
                kind="page_image",
                path="pages/cv_00002-1.png",
                sha256="c",
                size_bytes=page_path.stat().st_size,
            )
        ],
    )
    manifest_path = tmp_path / "debug_250.jsonl"
    atomic_write_jsonl(manifest_path, [row.model_dump(mode="json")])

    requests = build_visual_trial_requests(
        manifest_path,
        dataset_root,
        model_family="nuextract3",
        compact_schema=True,
    )

    assert set(requests[0]["schema_template"]) == {
        "n",
        "e",
        "l",
        "ph",
        "li",
        "gh",
        "su",
        "s",
        "w",
        "d",
        "p",
        "c",
    }
    assert requests[0]["compact_schema"] is True


def test_build_visual_trial_requests_falls_back_to_ocr_text(
    tmp_path: Path,
    reduced_target: dict[str, object],
) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "targets").mkdir(parents=True)
    (dataset_root / "text").mkdir()
    (dataset_root / "ocr").mkdir()
    (dataset_root / "pages").mkdir()
    target_path = dataset_root / "targets" / "cv_00003.json"
    target_path.write_text(json.dumps(reduced_target), encoding="utf-8")
    text_path = dataset_root / "text" / "cv_00003.txt"
    text_path.write_text("   ", encoding="utf-8")
    ocr_path = dataset_root / "ocr" / "cv_00003.txt"
    ocr_path.write_text("OCR fallback text", encoding="utf-8")
    page_path = dataset_root / "pages" / "cv_00003-1.png"
    Image.new("RGB", (20, 10), "white").save(page_path)
    row = ManifestRow(
        cv_id="cv_00003",
        tier="T4",
        template="T4_scan",
        primary_domain="Operations",
        split="debug_50",
        selection_seed=20260609,
        artifacts={
            "schema_reduced": ArtifactReference(
                kind="schema_reduced",
                path="targets/cv_00003.json",
                sha256="a",
                size_bytes=target_path.stat().st_size,
            ),
            "pymupdf_text": ArtifactReference(
                kind="pymupdf_text",
                path="text/cv_00003.txt",
                sha256="b",
                size_bytes=text_path.stat().st_size,
            ),
            "tesseract_ocr": ArtifactReference(
                kind="tesseract_ocr",
                path="ocr/cv_00003.txt",
                sha256="c",
                size_bytes=ocr_path.stat().st_size,
            ),
        },
        page_images=[
            ArtifactReference(
                kind="page_image",
                path="pages/cv_00003-1.png",
                sha256="d",
                size_bytes=page_path.stat().st_size,
            )
        ],
    )
    manifest_path = tmp_path / "debug_50.jsonl"
    atomic_write_jsonl(manifest_path, [row.model_dump(mode="json")])

    requests = build_visual_trial_requests(
        manifest_path,
        dataset_root,
        model_family="paddleocr_vl",
    )

    assert requests[0]["model_family"] == "paddleocr_vl"
    assert requests[0]["evidence_text"] == "OCR fallback text"


def test_gradient_accumulation_steps_and_rescales_final_remainder() -> None:
    schedule = [
        gradient_accumulation_step(step, total_micro_steps=19, accumulation_steps=8)
        for step in range(1, 20)
    ]
    assert [index for index, decision in enumerate(schedule, start=1) if decision[0]] == [8, 16, 19]
    assert schedule[-1] == (True, pytest.approx(8 / 3))
