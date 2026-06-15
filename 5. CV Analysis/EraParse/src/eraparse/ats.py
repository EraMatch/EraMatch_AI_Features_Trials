import math
import platform
import re
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eraparse.constants import DEFAULT_ATS_ROOT, DEFAULT_RUN_DB, SEED
from eraparse.data import load_manifest
from eraparse.io import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
)
from eraparse.models import ManifestRow, RunProvenance, RunRecord
from eraparse.run_store import insert_artifact, insert_run

TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+")
LANES = ("pymupdf_text", "pdfminer_text", "oracle_text", "canonical_structured")


def normalize_phrase(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.casefold()))


def tokenize(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value.casefold())


def _read_target(row: ManifestRow, dataset_root: Path) -> dict[str, Any]:
    value = read_json(dataset_root / row.artifacts["schema_reduced"].path)
    if not isinstance(value, dict):
        raise ValueError(f"{row.cv_id} reduced schema must be an object")
    return value


def build_job_profiles(
    train_manifest: Path,
    dataset_root: Path,
    *,
    skills_per_profile: int = 12,
) -> list[dict[str, Any]]:
    rows = load_manifest(train_manifest)
    domain_documents: dict[str, int] = Counter(row.primary_domain for row in rows)
    domain_skill_df: dict[str, Counter[str]] = defaultdict(Counter)
    global_skill_df: Counter[str] = Counter()

    for row in rows:
        target = _read_target(row, dataset_root)
        skills = {
            normalized
            for skill in target.get("skills", [])
            if isinstance(skill, str) and (normalized := normalize_phrase(skill))
        }
        domain_skill_df[row.primary_domain].update(skills)
        global_skill_df.update(skills)

    profiles = []
    for domain in sorted(domain_documents):
        domain_count = domain_documents[domain]
        scored = []
        for skill, domain_df in domain_skill_df[domain].items():
            prevalence = domain_df / domain_count
            specificity = math.log((len(rows) + 1) / (global_skill_df[skill] + 1))
            scored.append((prevalence * specificity, domain_df, skill))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        profiles.append(
            {
                "profile_id": f"domain:{normalize_phrase(domain).replace(' ', '-')}",
                "domain": domain,
                "required_skills": [],
                "optional_skills": [skill for _, _, skill in scored[:skills_per_profile]],
                "excluded_skills": [],
                "label_source": "training_primary_domain_and_canonical_skills",
                "training_documents": domain_count,
            }
        )
    return profiles


def canonical_search_text(target: Mapping[str, Any]) -> str:
    values: list[str] = []
    values.extend(str(value) for value in target.get("skills", []) if value)
    values.append(str(target.get("summary", "")))
    for field in ("work_experience", "education", "projects", "certifications"):
        records = target.get(field) or []
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for key, value in record.items():
                if key in {"company", "institution", "issuer", "url"}:
                    continue
                if isinstance(value, list):
                    values.extend(str(item) for item in value)
                elif value:
                    values.append(str(value))
    return "\n".join(values)


def prediction_search_text(prediction: Mapping[str, Any] | None) -> str:
    return canonical_search_text(prediction) if prediction is not None else ""


def load_documents(
    manifest_path: Path,
    dataset_root: Path,
    lane: str,
) -> list[dict[str, Any]]:
    if lane not in LANES:
        raise ValueError(f"unsupported ATS lane: {lane}")
    documents = []
    for row in load_manifest(manifest_path):
        if lane == "canonical_structured":
            text = canonical_search_text(_read_target(row, dataset_root))
        else:
            artifact = "text_ground_truth" if lane == "oracle_text" else lane
            text = (dataset_root / row.artifacts[artifact].path).read_text(encoding="utf-8")
        documents.append(
            {
                "cv_id": row.cv_id,
                "domain": row.primary_domain,
                "tier": row.tier,
                "template": row.template,
                "text": text,
                "tokens": tokenize(text),
            }
        )
    return documents


def load_prediction_documents(
    manifest_path: Path,
    results_path: Path,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    manifest_ids = {row.cv_id for row in manifest}
    predictions: dict[str, Any] = {}
    for result in read_jsonl(results_path):
        cv_id = str(result["cv_id"])
        if cv_id in predictions:
            raise ValueError(f"Duplicate prediction result for {cv_id}")
        predictions[cv_id] = result.get("prediction")
    prediction_ids = set(predictions)
    if prediction_ids != manifest_ids:
        missing = sorted(manifest_ids - prediction_ids)
        extra = sorted(prediction_ids - manifest_ids)
        raise ValueError(f"Prediction IDs do not match manifest; missing={missing}, extra={extra}")

    documents = []
    for row in manifest:
        prediction = predictions.get(row.cv_id)
        text = prediction_search_text(prediction if isinstance(prediction, Mapping) else None)
        documents.append(
            {
                "cv_id": row.cv_id,
                "domain": row.primary_domain,
                "tier": row.tier,
                "template": row.template,
                "text": text,
                "tokens": tokenize(text),
            }
        )
    return documents


def bm25_scores(documents: Sequence[Mapping[str, Any]], query_tokens: Sequence[str]) -> list[float]:
    k1 = 1.5
    b = 0.75
    lengths = [len(document["tokens"]) for document in documents]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency: Counter[str] = Counter()
    term_frequencies = []
    for document in documents:
        frequency = Counter(document["tokens"])
        term_frequencies.append(frequency)
        document_frequency.update(frequency.keys())

    scores = []
    for length, frequency in zip(lengths, term_frequencies, strict=True):
        score = 0.0
        for term in query_tokens:
            tf = frequency[term]
            if not tf:
                continue
            df = document_frequency[term]
            inverse_df = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * length / average_length)
            score += inverse_df * tf * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _ranking_metrics(ranked: Sequence[Mapping[str, Any]], relevant_domain: str) -> dict[str, float]:
    relevant_count = sum(document["domain"] == relevant_domain for document in ranked)
    metrics: dict[str, float] = {"relevant_count": float(relevant_count)}
    reciprocal_rank = 0.0
    for index, document in enumerate(ranked, start=1):
        if document["domain"] == relevant_domain:
            reciprocal_rank = 1 / index
            break
    metrics["mrr"] = reciprocal_rank
    for k in (10, 25, 50):
        top = ranked[:k]
        hits = sum(document["domain"] == relevant_domain for document in top)
        dcg = sum(
            1 / math.log2(index + 2)
            for index, document in enumerate(top)
            if document["domain"] == relevant_domain
        )
        ideal_hits = min(relevant_count, k)
        ideal_dcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
        metrics[f"precision_at_{k}"] = hits / len(top) if top else 0.0
        metrics[f"recall_at_{k}"] = hits / relevant_count if relevant_count else 0.0
        metrics[f"ndcg_at_{k}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return metrics


def evaluate_lane(
    documents: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    *,
    lane: str,
    split: str,
) -> list[dict[str, Any]]:
    results = []
    for profile in profiles:
        phrases = [*profile["required_skills"], *profile["optional_skills"]]
        query_tokens = tokenize(" ".join(phrases))
        bm25 = bm25_scores(documents, query_tokens)
        boolean_scores = [
            sum(phrase in normalize_phrase(str(document["text"])) for phrase in phrases)
            for document in documents
        ]
        for method, scores in (("boolean", boolean_scores), ("bm25", bm25)):
            ranked = [
                {
                    "cv_id": document["cv_id"],
                    "domain": document["domain"],
                    "tier": document["tier"],
                    "template": document["template"],
                    "score": score,
                }
                for document, score in sorted(
                    zip(documents, scores, strict=True),
                    key=lambda item: (-item[1], item[0]["cv_id"]),
                )
            ]
            metrics = _ranking_metrics(ranked, str(profile["domain"]))
            if method == "boolean":
                relevant = [item for item in ranked if item["domain"] == profile["domain"]]
                metrics["false_rejection_rate"] = (
                    sum(item["score"] == 0 for item in relevant) / len(relevant)
                    if relevant
                    else 0.0
                )
            results.append(
                {
                    "split": split,
                    "lane": lane,
                    "method": method,
                    "profile_id": profile["profile_id"],
                    "domain": profile["domain"],
                    "query_phrases": phrases,
                    "metrics": metrics,
                    "ranking": ranked,
                }
            )
    return results


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row["metrics"] if key != "relevant_count"})
    return {
        key: sum(float(row["metrics"].get(key, 0.0)) for row in rows) / len(rows) for key in keys
    }


def _false_rejection_breakdown(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[bool]] = defaultdict(list)
    for row in rows:
        if row["method"] != "boolean":
            continue
        for candidate in row["ranking"]:
            if candidate["domain"] != row["domain"]:
                continue
            for dimension in ("tier", "template"):
                groups[(row["split"], row["lane"], dimension, candidate[dimension])].append(
                    candidate["score"] == 0
                )
    return [
        {
            "split": split,
            "lane": lane,
            "dimension": dimension,
            "value": value,
            "relevant_candidates": len(rejections),
            "false_rejections": sum(rejections),
            "false_rejection_rate": sum(rejections) / len(rejections),
        }
        for (split, lane, dimension, value), rejections in sorted(groups.items())
    ]


def current_git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() or None


def run_baseline(
    *,
    train_manifest: Path,
    test_manifests: Sequence[Path],
    dataset_root: Path,
    output_root: Path = DEFAULT_ATS_ROOT,
    run_db: Path = DEFAULT_RUN_DB,
    skills_per_profile: int = 12,
) -> dict[str, Any]:
    profiles = build_job_profiles(
        train_manifest, dataset_root, skills_per_profile=skills_per_profile
    )
    manifest_hashes = {path.stem: sha256_file(path) for path in [train_manifest, *test_manifests]}
    config = {
        "contract_version": 2,
        "lanes": list(LANES),
        "methods": ["boolean", "bm25"],
        "skills_per_profile": skills_per_profile,
        "label_policy": "weak_primary_domain",
        "manifest_hashes": manifest_hashes,
    }
    run_id = f"ats-baseline-{stable_hash(config, seed=SEED)[:12]}"
    output_dir = output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f"ATS baseline run already exists: {output_dir}")

    started_at = datetime.now(UTC)
    run = RunRecord(
        run_id=run_id,
        kind="ats_screening_baseline",
        status="running",
        provenance=RunProvenance(
            code_revision=current_git_revision(),
            manifest_hash=stable_hash(manifest_hashes, seed=SEED),
            environment={"platform": platform.platform()},
            seed=SEED,
            resolved_config=config,
        ),
        parser_id="boolean_bm25",
        started_at=started_at,
    )
    insert_run(run_db, run)

    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    lane_diagnostics: list[dict[str, Any]] = []
    for manifest_path in test_manifests:
        split = manifest_path.stem
        for lane in LANES:
            started = time.perf_counter()
            documents = load_documents(manifest_path, dataset_root, lane)
            split_counts[split] = len(documents)
            rows.extend(evaluate_lane(documents, profiles, lane=lane, split=split))
            lane_diagnostics.append(
                {
                    "split": split,
                    "lane": lane,
                    "candidate_count": len(documents),
                    "empty_document_count": sum(not document["tokens"] for document in documents),
                    "mean_token_count": sum(len(document["tokens"]) for document in documents)
                    / len(documents),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    aggregates = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["lane"], row["method"])].append(row)
    for (split, lane, method), group in sorted(groups.items()):
        aggregates.append(
            {
                "split": split,
                "lane": lane,
                "method": method,
                "query_count": len(group),
                "candidate_count": split_counts[split],
                "mean_metrics": _mean_metrics(group),
            }
        )

    output_dir.mkdir(parents=True)
    profiles_path = output_dir / "job_profiles.json"
    results_path = output_dir / "query_results.jsonl"
    summary_path = output_dir / "summary.json"
    atomic_write_json(profiles_path, profiles)
    atomic_write_jsonl(results_path, rows)
    summary = {
        "run_id": run_id,
        "label_policy": "weak_primary_domain",
        "split_counts": split_counts,
        "manifest_hashes": manifest_hashes,
        "profiles_path": str(profiles_path),
        "results_path": str(results_path),
        "aggregates": aggregates,
        "lane_diagnostics": lane_diagnostics,
        "false_rejection_breakdown": _false_rejection_breakdown(rows),
    }
    atomic_write_json(summary_path, summary)
    for kind, path in (
        ("job_profiles", profiles_path),
        ("query_results", results_path),
        ("summary", summary_path),
    ):
        insert_artifact(
            run_db,
            run_id=run_id,
            kind=kind,
            artifact_path=str(path),
            sha256=sha256_file(path),
        )
    insert_run(
        run_db,
        run.model_copy(update={"status": "completed", "completed_at": datetime.now(UTC)}),
    )
    return summary


def run_prediction_comparison(
    *,
    profiles_path: Path,
    result_manifests: Sequence[tuple[Path, Path]],
    lane: str,
    output_root: Path = DEFAULT_ATS_ROOT,
    run_db: Path = DEFAULT_RUN_DB,
) -> dict[str, Any]:
    profiles = read_json(profiles_path)
    if not isinstance(profiles, list):
        raise ValueError("ATS profiles must be a JSON array")
    input_hashes = {
        f"{manifest.stem}_manifest": sha256_file(manifest) for manifest, _ in result_manifests
    }
    input_hashes.update(
        {f"{manifest.stem}_results": sha256_file(results) for manifest, results in result_manifests}
    )
    config = {
        "contract_version": 1,
        "lane": lane,
        "profiles_hash": sha256_file(profiles_path),
        "input_hashes": input_hashes,
        "label_policy": "weak_primary_domain",
    }
    run_id = f"ats-prediction-{stable_hash(config, seed=SEED)[:12]}"
    output_dir = output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f"ATS prediction comparison already exists: {output_dir}")

    started_at = datetime.now(UTC)
    run = RunRecord(
        run_id=run_id,
        kind="ats_prediction_comparison",
        status="running",
        provenance=RunProvenance(
            code_revision=current_git_revision(),
            manifest_hash=stable_hash(input_hashes, seed=SEED),
            environment={"platform": platform.platform()},
            seed=SEED,
            resolved_config=config,
        ),
        model_id=lane,
        parser_id="structured_prediction",
        started_at=started_at,
    )
    insert_run(run_db, run)

    rows: list[dict[str, Any]] = []
    diagnostics = []
    for manifest_path, results_path in result_manifests:
        started = time.perf_counter()
        documents = load_prediction_documents(manifest_path, results_path)
        rows.extend(evaluate_lane(documents, profiles, lane=lane, split=manifest_path.stem))
        diagnostics.append(
            {
                "split": manifest_path.stem,
                "lane": lane,
                "candidate_count": len(documents),
                "empty_document_count": sum(not document["tokens"] for document in documents),
                "mean_token_count": sum(len(document["tokens"]) for document in documents)
                / len(documents),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["method"])].append(row)
    aggregates = [
        {
            "split": split,
            "lane": lane,
            "method": method,
            "query_count": len(group),
            "candidate_count": next(
                item["candidate_count"] for item in diagnostics if item["split"] == split
            ),
            "mean_metrics": _mean_metrics(group),
        }
        for (split, method), group in sorted(groups.items())
    ]

    output_dir.mkdir(parents=True)
    results_output = output_dir / "query_results.jsonl"
    summary_path = output_dir / "summary.json"
    atomic_write_jsonl(results_output, rows)
    summary = {
        "run_id": run_id,
        "lane": lane,
        "label_policy": "weak_primary_domain",
        "profiles_hash": config["profiles_hash"],
        "input_hashes": input_hashes,
        "results_path": str(results_output),
        "aggregates": aggregates,
        "lane_diagnostics": diagnostics,
        "false_rejection_breakdown": _false_rejection_breakdown(rows),
    }
    atomic_write_json(summary_path, summary)
    for kind, path in (("query_results", results_output), ("summary", summary_path)):
        insert_artifact(
            run_db,
            run_id=run_id,
            kind=kind,
            artifact_path=str(path),
            sha256=sha256_file(path),
        )
    insert_run(
        run_db,
        run.model_copy(update={"status": "completed", "completed_at": datetime.now(UTC)}),
    )
    return summary
