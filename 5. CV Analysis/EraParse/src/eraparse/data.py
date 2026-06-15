import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from eraparse.constants import (
    EXPECTED_COMPLETED,
    EXPECTED_LOCKED,
    EXPECTED_SPLITS,
    EXPECTED_WORKING,
    JSON_ARTIFACT_KINDS,
    OOD_TEMPLATE_QUOTAS,
    REQUIRED_ARTIFACT_PATTERNS,
    SEED,
    T4_EXPECTED_COUNT,
    WORKING_TIER_QUOTAS,
)
from eraparse.io import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
)
from eraparse.models import (
    ArtifactReference,
    AuditIssue,
    AuditReport,
    ManifestRow,
    ReducedCVTarget,
)

T = TypeVar("T")
Stratum = tuple[str, ...]


def _artifact_reference(
    path: Path,
    dataset_root: Path,
    kind: str,
    hash_artifacts: bool,
    hash_cache: Mapping[Path, str],
) -> ArtifactReference:
    return ArtifactReference(
        kind=kind,
        path=str(path.relative_to(dataset_root)),
        sha256=hash_cache[path] if hash_artifacts else "",
        size_bytes=path.stat().st_size,
    )


def _validate_json_type(kind: str, value: Any) -> bool:
    expected_array = {
        "field_annotations",
        "section_annotations",
        "word_annotations",
        "token_labels",
    }
    if kind in expected_array:
        return isinstance(value, list)
    return isinstance(value, dict)


def _validate_json_container(path: Path, kind: str) -> bool:
    expected = (
        "["
        if kind
        in {
            "field_annotations",
            "section_annotations",
            "word_annotations",
            "token_labels",
        }
        else "{"
    )
    with path.open(encoding="utf-8") as handle:
        while character := handle.read(1):
            if not character.isspace():
                return character == expected
    return False


def scan_dataset(
    dataset_root: Path,
    *,
    hash_artifacts: bool = True,
) -> tuple[list[ManifestRow], list[AuditIssue]]:
    dataset_root = dataset_root.resolve()
    rows: list[ManifestRow] = []
    issues: list[AuditIssue] = []
    ground_truth_paths = sorted((dataset_root / "ground_truth").glob("cv_*.json"))
    page_image_paths: dict[str, list[Path]] = defaultdict(list)
    for page_path in sorted((dataset_root / "page_images").glob("cv_*-*.png")):
        page_image_paths[page_path.stem.rsplit("-", maxsplit=1)[0]].append(page_path)
    hash_cache: dict[Path, str] = {}
    if hash_artifacts:
        completed_ids = [path.stem for path in ground_truth_paths]
        hash_paths = {
            dataset_root / pattern.format(cv_id=cv_id)
            for cv_id in completed_ids
            for pattern in REQUIRED_ARTIFACT_PATTERNS.values()
        }
        hash_paths.update(path for paths in page_image_paths.values() for path in paths)
        hash_paths.update((dataset_root / "ocr_baselines" / "tesseract").glob("cv_*.txt"))
        hash_paths.update((dataset_root / "ocr_ground_truth").glob("cv_*.txt"))
        existing_hash_paths = sorted(path for path in hash_paths if path.is_file())
        with ThreadPoolExecutor(max_workers=8) as executor:
            hash_cache = dict(
                zip(
                    existing_hash_paths,
                    executor.map(sha256_file, existing_hash_paths),
                    strict=True,
                )
            )

    for ground_truth_path in ground_truth_paths:
        cv_id = ground_truth_path.stem
        artifact_refs: dict[str, ArtifactReference] = {}
        loaded_json: dict[str, Any] = {}
        sample_failed = False

        for kind, pattern in REQUIRED_ARTIFACT_PATTERNS.items():
            path = dataset_root / pattern.format(cv_id=cv_id)
            if not path.is_file():
                issues.append(
                    AuditIssue(cv_id=cv_id, kind="missing_artifact", message=f"{kind}: {path}")
                )
                sample_failed = True
                continue
            try:
                artifact_refs[kind] = _artifact_reference(
                    path, dataset_root, kind, hash_artifacts, hash_cache
                )
                if kind in {"ground_truth", "schema_reduced", "layout"}:
                    value = read_json(path)
                    loaded_json[kind] = value
                    if not _validate_json_type(kind, value):
                        issues.append(
                            AuditIssue(
                                cv_id=cv_id,
                                kind="invalid_json_type",
                                message=f"{kind} has type {type(value).__name__}",
                            )
                        )
                        sample_failed = True
                elif kind in JSON_ARTIFACT_KINDS and not _validate_json_container(path, kind):
                    issues.append(
                        AuditIssue(
                            cv_id=cv_id,
                            kind="invalid_json_type",
                            message=f"{kind} has an unexpected top-level container",
                        )
                    )
                    sample_failed = True
            except (OSError, json.JSONDecodeError) as error:
                issues.append(
                    AuditIssue(cv_id=cv_id, kind="unreadable_artifact", message=f"{kind}: {error}")
                )
                sample_failed = True

        page_paths = page_image_paths.get(cv_id, [])
        if not page_paths:
            issues.append(
                AuditIssue(cv_id=cv_id, kind="missing_page_images", message="no page images")
            )
            sample_failed = True
        page_refs = [
            _artifact_reference(path, dataset_root, "page_image", hash_artifacts, hash_cache)
            for path in page_paths
        ]

        if sample_failed:
            continue

        layout = loaded_json["layout"]
        ground_truth = loaded_json["ground_truth"]
        reduced = loaded_json["schema_reduced"]
        if layout.get("cv_id") != cv_id:
            issues.append(
                AuditIssue(
                    cv_id=cv_id,
                    kind="id_mismatch",
                    message=f"layout cv_id={layout.get('cv_id')!r}",
                )
            )
            continue
        tier = layout.get("tier")
        template = layout.get("template")
        domain = ground_truth.get("primary_domain")
        if (
            tier not in WORKING_TIER_QUOTAS
            or not isinstance(template, str)
            or not isinstance(domain, str)
        ):
            issues.append(
                AuditIssue(
                    cv_id=cv_id,
                    kind="invalid_metadata",
                    message="missing or invalid tier/template/domain",
                )
            )
            continue
        try:
            ReducedCVTarget.model_validate(reduced)
        except ValueError as error:
            issues.append(
                AuditIssue(cv_id=cv_id, kind="invalid_reduced_schema", message=str(error))
            )
            continue

        if tier == "T4":
            for kind, relative_path in {
                "tesseract_ocr": f"ocr_baselines/tesseract/{cv_id}.txt",
                "ocr_ground_truth": f"ocr_ground_truth/{cv_id}.txt",
            }.items():
                path = dataset_root / relative_path
                if not path.is_file():
                    issues.append(
                        AuditIssue(
                            cv_id=cv_id, kind="missing_t4_artifact", message=f"{kind}: {path}"
                        )
                    )
                    sample_failed = True
                else:
                    artifact_refs[kind] = _artifact_reference(
                        path, dataset_root, kind, hash_artifacts, hash_cache
                    )
        if sample_failed:
            continue

        rows.append(
            ManifestRow(
                cv_id=cv_id,
                tier=tier,
                template=template,
                primary_domain=domain,
                selection_seed=SEED,
                artifacts=artifact_refs,
                page_images=page_refs,
            )
        )
    return rows, issues


def _audit_report(
    dataset_root: Path, rows: Sequence[ManifestRow], issues: list[AuditIssue]
) -> AuditReport:
    tier_counts = Counter(row.tier for row in rows)
    template_counts = Counter(row.template for row in rows)
    if len(rows) != EXPECTED_COMPLETED:
        issues.append(
            AuditIssue(
                kind="completed_count", message=f"expected {EXPECTED_COMPLETED}, found {len(rows)}"
            )
        )
    if tier_counts.get("T4", 0) != T4_EXPECTED_COUNT:
        issues.append(
            AuditIssue(
                kind="t4_count",
                message=f"expected {T4_EXPECTED_COUNT}, found {tier_counts.get('T4', 0)}",
            )
        )
    orphan = dataset_root.resolve() / "pdfs" / "cv_04951.pdf"
    if not orphan.is_file():
        issues.append(AuditIssue(kind="missing_expected_orphan", message=str(orphan)))
    return AuditReport(
        dataset_root=str(dataset_root.resolve()),
        completed_count=len(rows),
        tier_counts=dict(sorted(tier_counts.items())),
        template_counts=dict(sorted(template_counts.items())),
        issues=issues,
        passed=not issues,
    )


def audit_dataset(
    dataset_root: Path,
    *,
    report_path: Path | None = None,
    hash_artifacts: bool = True,
) -> AuditReport:
    rows, issues = scan_dataset(dataset_root, hash_artifacts=hash_artifacts)
    report = _audit_report(dataset_root, rows, issues)
    if report_path is not None:
        atomic_write_json(report_path, report.model_dump(mode="json"))
    return report


def hamilton_allocate(weights: dict[Stratum, int], total: int) -> dict[Stratum, int]:
    if total < 0 or total > sum(weights.values()):
        raise ValueError(f"cannot allocate {total} from capacity {sum(weights.values())}")
    if total == 0:
        return {key: 0 for key in weights}
    weight_total = sum(weights.values())
    exact = {key: total * weight / weight_total for key, weight in weights.items()}
    allocated = {key: min(weights[key], math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - math.floor(exact[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if allocated[key] < weights[key]:
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("allocation exhausted before reaching requested total")
    return allocated


def _select_balanced(
    rows: Sequence[T],
    total: int,
    *,
    strata: Callable[[T], Stratum],
    hash_parts: Callable[[T], tuple[object, ...]],
    salt: str,
) -> tuple[list[T], list[T]]:
    grouped: dict[Stratum, list[T]] = defaultdict(list)
    for row in rows:
        grouped[strata(row)].append(row)
    quotas = hamilton_allocate({key: len(values) for key, values in grouped.items()}, total)
    selected: list[T] = []
    remaining: list[T] = []
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda row: stable_hash(salt, *hash_parts(row), seed=SEED))
        selected.extend(ordered[: quotas[key]])
        remaining.extend(ordered[quotas[key] :])
    return selected, remaining


def _with_split(rows: Iterable[ManifestRow], split: str) -> list[ManifestRow]:
    return [row.model_copy(update={"split": split}) for row in rows]


def _manifest_hash(rows: Sequence[ManifestRow]) -> str:
    digest_parts = [
        stable_hash(
            row.cv_id,
            row.split,
            *(ref.sha256 for ref in [*row.artifacts.values(), *row.page_images]),
            seed=SEED,
        )
        for row in sorted(rows, key=lambda item: item.cv_id)
    ]
    return stable_hash(*digest_parts, seed=SEED)


def _build_split_rows(completed: Sequence[ManifestRow]) -> dict[str, list[ManifestRow]]:
    heldout_templates = set(OOD_TEMPLATE_QUOTAS)
    ood: list[ManifestRow] = []
    ood_remainder: list[ManifestRow] = []
    for template, quota in OOD_TEMPLATE_QUOTAS.items():
        candidates = [row for row in completed if row.template == template]
        chosen, remainder = _select_balanced(
            candidates,
            quota,
            strata=lambda row: (row.primary_domain,),
            hash_parts=lambda row: (row.cv_id,),
            salt=f"ood:{template}",
        )
        ood.extend(chosen)
        ood_remainder.extend(remainder)

    non_heldout = [row for row in completed if row.template not in heldout_templates]
    working_id: list[ManifestRow] = []
    non_heldout_remainder: list[ManifestRow] = []
    for tier, working_quota in WORKING_TIER_QUOTAS.items():
        tier_ood_count = sum(row.tier == tier for row in ood)
        quota = working_quota - tier_ood_count
        candidates = [row for row in non_heldout if row.tier == tier]
        chosen, remainder = _select_balanced(
            candidates,
            quota,
            strata=lambda row: (row.template, row.primary_domain),
            hash_parts=lambda row: (row.cv_id,),
            salt=f"working:{tier}",
        )
        working_id.extend(chosen)
        non_heldout_remainder.extend(remainder)

    validation, after_validation = _select_balanced(
        working_id,
        EXPECTED_SPLITS["validation"],
        strata=lambda row: (row.tier, row.template, row.primary_domain),
        hash_parts=lambda row: (row.cv_id,),
        salt="validation",
    )
    id_test, train = _select_balanced(
        after_validation,
        EXPECTED_SPLITS["id_test"],
        strata=lambda row: (row.tier, row.template, row.primary_domain),
        hash_parts=lambda row: (row.cv_id,),
        salt="id_test",
    )
    locked = ood_remainder + non_heldout_remainder

    debug_50: list[ManifestRow] = []
    debug_250: list[ManifestRow] = []
    for tier in WORKING_TIER_QUOTAS:
        candidates = [row for row in train if row.tier == tier]
        debug_250_tier, _ = _select_balanced(
            candidates,
            50,
            strata=lambda row: (row.template, row.primary_domain),
            hash_parts=lambda row: (row.cv_id,),
            salt=f"debug250:{tier}",
        )
        debug_50_tier, _ = _select_balanced(
            debug_250_tier,
            10,
            strata=lambda row: (row.template, row.primary_domain),
            hash_parts=lambda row: (row.cv_id,),
            salt=f"debug50:{tier}",
        )
        debug_250.extend(debug_250_tier)
        debug_50.extend(debug_50_tier)

    split_rows = {
        "completed": _with_split(completed, "completed"),
        "working": _with_split([*working_id, *ood], "working"),
        "locked_confirmation": _with_split(locked, "locked_confirmation"),
        "train": _with_split(train, "train"),
        "validation": _with_split(validation, "validation"),
        "id_test": _with_split(id_test, "id_test"),
        "template_ood_test": _with_split(ood, "template_ood_test"),
        "debug_50": _with_split(debug_50, "debug_50"),
        "debug_250": _with_split(debug_250, "debug_250"),
    }
    return split_rows


def _rows_to_dicts(rows: Sequence[ManifestRow]) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in sorted(rows, key=lambda item: item.cv_id)]


def build_manifests(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    completed, issues = scan_dataset(dataset_root, hash_artifacts=True)
    report = _audit_report(dataset_root, completed, issues)
    if not report.passed:
        raise ValueError(f"dataset audit failed with {len(report.issues)} issue(s)")

    split_rows = _build_split_rows(completed)
    summary = _validate_split_rows(split_rows)
    if not summary["passed"]:
        raise ValueError(f"generated manifests failed validation: {summary['issues']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in split_rows.items():
        atomic_write_jsonl(output_dir / f"{name}.jsonl", _rows_to_dicts(rows))
    atomic_write_json(output_dir / "audit_report.json", report.model_dump(mode="json"))
    summary["manifest_hash"] = _manifest_hash(split_rows["completed"])
    summary["dataset_root"] = str(dataset_root.resolve())
    atomic_write_json(output_dir / "manifest_summary.json", summary)
    return summary


def load_manifest(path: Path, *, allow_locked_confirmation: bool = False) -> list[ManifestRow]:
    if path.name == "locked_confirmation.jsonl" and not allow_locked_confirmation:
        raise PermissionError(
            "locked confirmation manifest requires allow_locked_confirmation=True"
        )
    return [ManifestRow.model_validate(row) for row in read_jsonl(path)]


def _validate_split_rows(split_rows: Mapping[str, Sequence[ManifestRow]]) -> dict[str, Any]:
    issues: list[str] = []
    expected_counts = {
        "completed": EXPECTED_COMPLETED,
        "working": EXPECTED_WORKING,
        "locked_confirmation": EXPECTED_LOCKED,
        **EXPECTED_SPLITS,
        "debug_50": 50,
        "debug_250": 250,
    }
    counts = {name: len(rows) for name, rows in split_rows.items()}
    for name, expected in expected_counts.items():
        if counts.get(name) != expected:
            issues.append(f"{name}: expected {expected}, found {counts.get(name)}")

    ids = {name: {row.cv_id for row in rows} for name, rows in split_rows.items()}
    if ids.get("working", set()) & ids.get("locked_confirmation", set()):
        issues.append("working and locked confirmation overlap")
    if ids.get("working", set()) | ids.get("locked_confirmation", set()) != ids.get(
        "completed", set()
    ):
        issues.append("working and locked confirmation do not partition completed")
    primary_splits = ["train", "validation", "id_test", "template_ood_test"]
    for index, left in enumerate(primary_splits):
        for right in primary_splits[index + 1 :]:
            if ids.get(left, set()) & ids.get(right, set()):
                issues.append(f"{left} and {right} overlap")
    if set().union(*(ids.get(name, set()) for name in primary_splits)) != ids.get("working", set()):
        issues.append("primary splits do not partition working")

    tier_counts = Counter(row.tier for row in split_rows.get("working", []))
    if dict(tier_counts) != WORKING_TIER_QUOTAS:
        issues.append(f"working tier counts mismatch: {dict(tier_counts)}")
    ood_counts = Counter(row.template for row in split_rows.get("template_ood_test", []))
    if dict(ood_counts) != OOD_TEMPLATE_QUOTAS:
        issues.append(f"OOD template counts mismatch: {dict(ood_counts)}")
    heldout = set(OOD_TEMPLATE_QUOTAS)
    for name in ("train", "validation", "id_test"):
        leaked = sorted({row.template for row in split_rows.get(name, [])} & heldout)
        if leaked:
            issues.append(f"{name} contains held-out templates: {leaked}")
    if not ids.get("debug_50", set()) <= ids.get("train", set()):
        issues.append("debug_50 is not a subset of train")
    if not ids.get("debug_250", set()) <= ids.get("train", set()):
        issues.append("debug_250 is not a subset of train")

    return {
        "passed": not issues,
        "issues": issues,
        "counts": counts,
        "working_tier_counts": dict(sorted(tier_counts.items())),
        "ood_template_counts": dict(sorted(ood_counts.items())),
    }


def validate_manifests(manifest_dir: Path) -> dict[str, Any]:
    names = [
        "completed",
        "working",
        "locked_confirmation",
        "train",
        "validation",
        "id_test",
        "template_ood_test",
        "debug_50",
        "debug_250",
    ]
    split_rows: dict[str, list[ManifestRow]] = {}
    issues: list[str] = []
    for name in names:
        path = manifest_dir / f"{name}.jsonl"
        if not path.is_file():
            issues.append(f"missing {path}")
            split_rows[name] = []
            continue
        try:
            split_rows[name] = load_manifest(
                path, allow_locked_confirmation=name == "locked_confirmation"
            )
        except (ValueError, json.JSONDecodeError) as error:
            issues.append(f"{path}: {error}")
            split_rows[name] = []
    result = _validate_split_rows(split_rows)
    result["issues"] = [*issues, *result["issues"]]
    result["passed"] = not result["issues"]
    return result
