import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from eraparse.constants import REDUCED_SCHEMA_TEMPLATE
from eraparse.data import load_manifest
from eraparse.io import atomic_write_jsonl, read_json

TASK_PROMPT = "<s_eraparse>"


def serialize_donut_target(target: Mapping[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _donut_json2token(value: Any) -> str:
    if isinstance(value, Mapping):
        return "".join(
            f"<s_{key}>{_donut_json2token(value[key])}</s_{key}>" for key in sorted(value)
        )
    if isinstance(value, list):
        return "<sep/>".join(_donut_json2token(item) for item in value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def serialize_donut_native_target(target: Mapping[str, Any]) -> str:
    return _donut_json2token(target)


def donut_native_special_tokens(target: Mapping[str, Any]) -> set[str]:
    tokens = {TASK_PROMPT}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                tokens.update({f"<s_{key}>", f"</s_{key}>"})
                visit(nested)
        elif isinstance(value, list):
            tokens.add("<sep/>")
            for nested in value:
                visit(nested)

    visit(target)
    return tokens


def compose_page_images(
    page_paths: Sequence[Path],
    *,
    separator_pixels: int = 16,
) -> Image.Image:
    if not page_paths:
        raise ValueError("At least one page image is required")
    if separator_pixels < 0:
        raise ValueError("separator_pixels must not be negative")

    pages = [Image.open(path).convert("RGB") for path in page_paths]
    width = max(page.width for page in pages)
    resized = []
    for page in pages:
        if page.width == width:
            resized.append(page)
            continue
        height = round(page.height * width / page.width)
        resized.append(page.resize((width, height), Image.Resampling.LANCZOS))
    height = sum(page.height for page in resized) + separator_pixels * (len(resized) - 1)
    composed = Image.new("RGB", (width, height), "white")
    top = 0
    for page in resized:
        composed.paste(page, (0, top))
        top += page.height + separator_pixels
    return composed


def build_donut_records(
    manifest_path: Path,
    dataset_root: Path,
    *,
    target_format: Literal["raw_json", "native_tokens"] = "raw_json",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in load_manifest(manifest_path):
        truth = read_json(dataset_root / row.artifacts["schema_reduced"].path)
        evidence_text = (dataset_root / row.artifacts["pymupdf_text"].path).read_text(
            encoding="utf-8"
        )
        if not evidence_text.strip() and "tesseract_ocr" in row.artifacts:
            evidence_text = (dataset_root / row.artifacts["tesseract_ocr"].path).read_text(
                encoding="utf-8"
            )
        target = (
            serialize_donut_native_target(truth)
            if target_format == "native_tokens"
            else serialize_donut_target(truth)
        )
        records.append(
            {
                "cv_id": row.cv_id,
                "split": row.split,
                "tier": row.tier,
                "template": row.template,
                "page_images": [artifact.path for artifact in row.page_images],
                "page_image_sha256s": [artifact.sha256 for artifact in row.page_images],
                "target_format": target_format,
                "target": target,
                "special_tokens": sorted(donut_native_special_tokens(truth))
                if target_format == "native_tokens"
                else [TASK_PROMPT],
                "evidence_text": evidence_text,
                "truth": truth,
            }
        )
    return records


def write_donut_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_jsonl(path, [dict(record) for record in records])


def build_visual_trial_requests(
    manifest_path: Path,
    dataset_root: Path,
    *,
    model_family: Literal["nuextract3", "paddleocr_vl"],
    compact_schema: bool = False,
) -> list[dict[str, Any]]:
    from eraparse.compact_schema import COMPACT_SCHEMA_TEMPLATE

    requests: list[dict[str, Any]] = []
    for row in load_manifest(manifest_path):
        truth = read_json(dataset_root / row.artifacts["schema_reduced"].path)
        evidence_text = (dataset_root / row.artifacts["pymupdf_text"].path).read_text(
            encoding="utf-8"
        )
        if not evidence_text.strip() and "tesseract_ocr" in row.artifacts:
            evidence_text = (dataset_root / row.artifacts["tesseract_ocr"].path).read_text(
                encoding="utf-8"
            )
        requests.append(
            {
                "cv_id": row.cv_id,
                "split": row.split,
                "tier": row.tier,
                "template": row.template,
                "primary_domain": row.primary_domain,
                "model_family": model_family,
                "page_images": [artifact.path for artifact in row.page_images],
                "page_image_sha256s": [artifact.sha256 for artifact in row.page_images],
                "schema_template": (
                    COMPACT_SCHEMA_TEMPLATE if compact_schema else REDUCED_SCHEMA_TEMPLATE
                ),
                "compact_schema": compact_schema,
                "truth": truth,
                "evidence_text": evidence_text,
            }
        )
    return requests


def write_visual_trial_requests(path: Path, requests: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_jsonl(path, [dict(request) for request in requests])
