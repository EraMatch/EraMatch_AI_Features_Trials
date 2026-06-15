from collections import Counter
from pathlib import Path

import pytest

from eraparse.constants import OOD_TEMPLATE_QUOTAS, WORKING_TIER_QUOTAS
from eraparse.data import _build_split_rows, _validate_split_rows, scan_dataset


@pytest.mark.dataset
def test_real_dataset_contract_and_deterministic_splits(dataset_root: Path) -> None:
    if not dataset_root.is_dir():
        pytest.skip(f"dataset is unavailable: {dataset_root}")
    sentinel_paths = [
        dataset_root / "ground_truth" / "cv_00001.json",
        dataset_root / "layout_annotations" / "cv_03501_layout.json",
        dataset_root / "pdfs" / "cv_04951.pdf",
    ]
    sentinel_metadata = {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sentinel_paths
    }
    rows, issues = scan_dataset(dataset_root, hash_artifacts=False)
    assert not issues
    assert len(rows) == 4_950

    first = _build_split_rows(rows)
    second = _build_split_rows(rows)
    assert _validate_split_rows(first)["passed"] is True
    assert [row.cv_id for row in first["working"]] == [row.cv_id for row in second["working"]]
    assert Counter(row.tier for row in first["working"]) == WORKING_TIER_QUOTAS
    assert Counter(row.template for row in first["template_ood_test"]) == OOD_TEMPLATE_QUOTAS
    assert all(row.cv_id != "cv_04951" for row in first["completed"])
    assert sentinel_metadata == {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sentinel_paths
    }
