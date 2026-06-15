from pathlib import Path

import pytest

from eraparse.data import _manifest_hash, hamilton_allocate, load_manifest
from eraparse.io import atomic_write_jsonl
from eraparse.models import ArtifactReference, ManifestRow


def test_hamilton_allocate_is_exact_and_capacity_safe() -> None:
    allocation = hamilton_allocate({("a",): 7, ("b",): 3}, 5)
    assert allocation == {("a",): 4, ("b",): 1}
    assert sum(allocation.values()) == 5


def test_hamilton_allocate_rejects_impossible_total() -> None:
    with pytest.raises(ValueError):
        hamilton_allocate({("a",): 2}, 3)


def test_locked_manifest_requires_explicit_permission(tmp_path: Path) -> None:
    path = tmp_path / "locked_confirmation.jsonl"
    atomic_write_jsonl(path, [])
    with pytest.raises(PermissionError):
        load_manifest(path)
    assert load_manifest(path, allow_locked_confirmation=True) == []


def test_manifest_hash_includes_page_images() -> None:
    artifact = ArtifactReference(kind="ground_truth", path="truth.json", sha256="a", size_bytes=1)
    page = ArtifactReference(kind="page_image", path="page.png", sha256="b", size_bytes=1)
    row = ManifestRow(
        cv_id="cv_00001",
        tier="T1",
        template="T1_classic",
        primary_domain="Backend Engineering",
        split="completed",
        selection_seed=20260609,
        artifacts={"ground_truth": artifact},
        page_images=[page],
    )
    changed = row.model_copy(
        update={
            "page_images": [
                page.model_copy(update={"sha256": "changed"}),
            ]
        }
    )
    assert _manifest_hash([row]) != _manifest_hash([changed])
