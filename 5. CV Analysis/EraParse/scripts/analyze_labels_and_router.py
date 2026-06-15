"""
Four $0 analyses for the thesis contribution:

1. URL label-integrity audit: schema_reduced carries github/linkedin URLs that
   are rendered nowhere in the document (~34-59% per split). Excluded from
   clean metric.

2. Projects label-integrity audit: four templates (T1_functional, T1_executive,
   T3_table, T5_minimal) never render a projects section in their PDF layout.
   These evidence-absent project records are excluded from the clean metric,
   same principle as URL exclusion. Effect: projects 0.788 raw -> 0.898 clean.

3. Fully-clean macro recompute (URL + projects both excluded) for NuExtract3
   alone and both router variants.

4. Work-experience-only router ablation: drop certifications from routing to
   compare macro vs escalation count tradeoff.

Usage:
    uv run python scripts/analyze_labels_and_router.py
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DS = ROOT.parent / "eramatch_benchmark_v4"
MANIFESTS = ROOT / "artifacts" / "manifests"
ROUTER = ROOT / "artifacts" / "trials" / "router"

FIELDS = [
    "certifications", "education", "email", "full_name", "github_url",
    "linkedin_url", "location", "phone", "projects", "skills", "summary",
    "work_experience",
]
NESTED = ["work_experience", "education", "projects", "certifications"]

# Templates whose PDF layout never renders a projects section.
# Verified: 0/N CVs across all affected templates have project names in
# pymupdf text; NuExtract3 achieves 0% detection on all of them.
PROJECTS_ABSENT_TEMPLATES = frozenset({
    "T1_functional",
    "T1_executive",
    "T3_table",
    "T5_minimal",
})

_text_cache: dict[str, str] = {}


def load_manifest(split: str) -> dict[str, dict]:
    path = MANIFESTS / f"{split}.jsonl"
    return {json.loads(x)["cv_id"]: json.loads(x) for x in path.read_text().splitlines()}


def doc_text(row: dict) -> str:
    cv_id = row["cv_id"]
    if cv_id not in _text_cache:
        p = DS / row["artifacts"]["pymupdf_text"]["path"]
        _text_cache[cv_id] = p.read_text(errors="ignore").lower()
    return _text_cache[cv_id]


def url_truth(row: dict, field: str) -> str:
    ref = row["artifacts"].get("schema_reduced")
    if not ref:
        return ""
    data = json.loads((DS / ref["path"]).read_text())
    return (data.get(field) or "") if isinstance(data, dict) else ""


def evidence_absent(row: dict, field: str) -> bool:
    """True iff a URL truth exists but its domain is rendered nowhere in text."""
    truth = url_truth(row, field)
    if not truth:
        return False
    needle = "github" if field == "github_url" else "linkedin"
    return needle not in doc_text(row)


def project_evidence_absent(row: dict) -> bool:
    """True iff the CV's template never renders a projects section in its layout.

    Four templates (T1_functional, T1_executive, T3_table, T5_minimal) do not
    include a projects section regardless of whether the candidate has projects
    in schema_reduced. The label is real but unrecoverable — no extractor can
    find project names that are not printed. Verified by checking that 0/N CVs
    across these templates have any project name present in pymupdf text.
    """
    if row.get("template") not in PROJECTS_ABSENT_TEMPLATES:
        return False
    ref = row["artifacts"].get("schema_reduced")
    if not ref:
        return False
    truth = json.loads((DS / ref["path"]).read_text())
    truth_projs = truth.get("projects") or []
    if not truth_projs:
        return False
    text = doc_text(row)
    return not any(
        (p.get("name") or "").lower() in text for p in truth_projs if p.get("name")
    )


# ---------------------------------------------------------------------------
# 1. Label-integrity audit (URLs + projects) across all splits
# ---------------------------------------------------------------------------
def audit_all_splits() -> dict[str, set]:
    print("=" * 70)
    print("1. LABEL-INTEGRITY AUDIT — URL fields + projects (all splits)")
    print("=" * 70)
    exclusion: dict[str, set] = {}
    for split in ["train", "validation", "id_test", "template_ood_test"]:
        manifest = load_manifest(split)
        # URL audit
        for field in ["github_url", "linkedin_url"]:
            have_truth = absent = 0
            for row in manifest.values():
                if url_truth(row, field):
                    have_truth += 1
                    if evidence_absent(row, field):
                        absent += 1
                        exclusion.setdefault(split, set()).add((row["cv_id"], field))
            rate = 100 * absent / max(1, have_truth)
            print(
                f"  {split:<18} {field:<13} "
                f"labels={have_truth:>4}  evidence-absent={absent:>4} ({rate:5.1f}%)"
            )
        # Projects audit
        proj_truth = proj_absent = 0
        for row in manifest.values():
            ref = row["artifacts"].get("schema_reduced")
            if not ref:
                continue
            truth = json.loads((DS / ref["path"]).read_text())
            if truth.get("projects"):
                proj_truth += 1
                if project_evidence_absent(row):
                    proj_absent += 1
                    exclusion.setdefault(split, set()).add((row["cv_id"], "projects"))
        rate = 100 * proj_absent / max(1, proj_truth)
        print(
            f"  {split:<18} {'projects':<13} "
            f"labels={proj_truth:>4}  evidence-absent={proj_absent:>4} ({rate:5.1f}%)"
            f"  [templates: T1_functional, T1_executive, T3_table, T5_minimal]"
        )
    print()
    return exclusion


# ---------------------------------------------------------------------------
# 2 & 3. Clean macro recompute (URL + projects both excluded)
# ---------------------------------------------------------------------------
def field_score(row: dict, path: str):
    return next((f for f in row["evaluation"]["field_results"] if f["path"] == path), None)


def clean_macro(results_path: Path, manifest: dict, label: str) -> None:
    """Report raw macro, URL-only-clean, and fully-clean (URL+projects)."""
    rows = [json.loads(x) for x in results_path.read_text().splitlines()]
    orig = {f: [] for f in FIELDS}
    url_clean = {f: [] for f in FIELDS}
    full_clean = {f: [] for f in FIELDS}
    for r in rows:
        cv_id = r["cv_id"]
        row = manifest.get(cv_id)
        for fld in FIELDS:
            f = field_score(r, fld)
            if f is None:
                continue
            orig[fld].append(f["score"])
            url_excluded = (
                fld in ("github_url", "linkedin_url")
                and row is not None
                and evidence_absent(row, fld)
            )
            proj_excluded = (
                fld == "projects"
                and row is not None
                and project_evidence_absent(row)
            )
            if not url_excluded:
                url_clean[fld].append(f["score"])
            if not url_excluded and not proj_excluded:
                full_clean[fld].append(f["score"])

    om = statistics.mean([statistics.mean(orig[f]) for f in FIELDS if orig[f]])
    um = statistics.mean([statistics.mean(url_clean[f]) for f in FIELDS if url_clean[f]])
    fm = statistics.mean([statistics.mean(full_clean[f]) for f in FIELDS if full_clean[f]])
    nested = statistics.mean([statistics.mean(full_clean[f]) for f in NESTED if full_clean[f]])
    print(f"  {label}")
    print(f"    macro raw={om:.4f}  url-clean={um:.4f}  fully-clean={fm:.4f}")
    print(
        f"    nested (fully-clean): {nested:.4f}  "
        f"[work={statistics.mean(full_clean['work_experience']):.4f} "
        f"edu={statistics.mean(full_clean['education']):.4f} "
        f"proj={statistics.mean(full_clean['projects']):.4f} "
        f"cert={statistics.mean(full_clean['certifications']):.4f}]"
    )


def recompute_clean(manifest: dict) -> None:
    print("=" * 70)
    print("2 & 3. CLEAN MACRO (URL + projects excluded, train)")
    print("=" * 70)
    nue = ROUTER / "train-nuextract3-evidence-ingested" / "nuextract3-nuextract3_visual-b16fdd6e13" / "results.jsonl"
    rtr = ROUTER / "train-focused-fusion-ingested"
    rtr_results = next(rtr.rglob("results.jsonl"))
    clean_macro(nue, manifest, "Lane B — NuExtract3 alone (train)")
    clean_macro(rtr_results, manifest, "Lane C — Focused router, both-fields (train)")
    print()


# ---------------------------------------------------------------------------
# 4. Work-experience-only router ablation
# ---------------------------------------------------------------------------
def workexp_only_ablation(manifest: dict) -> None:
    print("=" * 70)
    print("4. WORK-EXPERIENCE-ONLY ROUTER ABLATION (drop certifications)")
    print("=" * 70)
    sys.path.insert(0, str(ROOT / "src"))
    from eraparse.router import fuse_focused_specialist_responses
    from eraparse.trials import read_rows

    primary = read_rows(
        ROUTER / "train-nuextract3-evidence-ingested" /
        "nuextract3-nuextract3_visual-b16fdd6e13" / "results.jsonl"
    )
    specialist = [json.loads(x) for x in (ROUTER / "train.qwen-focused.all.responses.jsonl").read_text().splitlines()]
    requests = [json.loads(x) for x in (ROUTER / "train.qwen-focused.requests.jsonl").read_text().splitlines()]

    # work-experience-only routing map
    we_only = {
        str(r["cv_id"]): [f for f in r.get("routed_fields", []) if f == "work_experience"]
        for r in requests
    }
    escalated = sum(1 for v in we_only.values() if v)
    both_escalated = sum(1 for r in requests if r.get("routed_fields"))
    print(f"  escalated CVs: both-fields={both_escalated}  work-exp-only={escalated}  "
          f"(-{both_escalated-escalated}, {100*(both_escalated-escalated)/both_escalated:.0f}% fewer)")

    out = ROUTER / "train.focused-fusion.workexp-only.responses.jsonl"
    fuse_focused_specialist_responses(primary, specialist, we_only, out)

    ingested = ROUTER / "train-focused-fusion-workexp-only-ingested"
    cmd = [
        "uv", "run", "eraparse", "trials", "ingest-nuextract3",
        "--requests", str(ROUTER / "train.nuextract3-full.requests.jsonl"),
        "--responses", str(out),
        "--output-dir", str(ingested),
        "--run-db", str(ROUTER / "train-focused-fusion-workexp-only.duckdb"),
        "--model-id", "eraparse/router-workexp-only",
        "--revision", "ablation-v1",
        "--repair-work-records", "--full-schema", "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  EVAL FAILED:", result.stderr[-1500:])
        return
    summary = json.loads(result.stdout)
    agg = summary["aggregate"]
    print(f"  macro={agg['macro_score']:.4f}  schema_valid={agg['schema_valid_rate']}  "
          f"unsupported={agg['unsupported_evidence_rate']:.4f}")
    print(f"  (both-fields router raw was 0.9051)")
    clean_macro(Path(summary["results_path"]), manifest, "  Lane C work-exp-only (fully clean):")
    print()


if __name__ == "__main__":
    exclusion = audit_all_splits()
    total_excluded = sum(len(v) for v in exclusion.values())
    print(f"TOTAL evidence-absent cells excluded across splits: {total_excluded}\n")

    train_manifest = load_manifest("train")
    recompute_clean(train_manifest)
    workexp_only_ablation(train_manifest)
