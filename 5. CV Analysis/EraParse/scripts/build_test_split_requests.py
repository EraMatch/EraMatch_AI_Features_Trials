"""Build inference requests for id_test and template_ood splits.

Produces:
  - artifacts/trials/ft/id_test.gemma-ft.requests.jsonl
  - artifacts/trials/ft/id_test.smolvlm2-ft.requests.jsonl   (same fields; for future use)
  - artifacts/trials/ft/template_ood.gemma-ft.requests.jsonl
  - artifacts/trials/ft/template_ood.smolvlm2-ft.requests.jsonl
  - artifacts/trials/router/id_test.requests.jsonl
  - artifacts/trials/router/template_ood.requests.jsonl

Format follows validation.gemma-ft.requests.jsonl:
  cv_id, representation, split, template, tier, primary_domain, text, system, truth

Run:
    uv run python scripts/build_test_split_requests.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DS = ROOT.parent / "eramatch_benchmark_v4"
REPS = ROOT / "artifacts" / "representations" / "pymupdf4llm_markdown"

sys.path.insert(0, str(ROOT / "src"))
from eraparse.constants import REDUCED_SCHEMA_TEMPLATE

# ── system prompt (identical to validation.gemma-ft.requests.jsonl) ──────────
SYSTEM = (
    "You are a precise CV information extraction system. Extract the CV into "
    "exactly this JSON schema. Return one valid JSON object. Use empty strings "
    "or empty arrays for missing values. Do not invent values absent from the CV.\n\n"
    "Schema:\n" + json.dumps(REDUCED_SCHEMA_TEMPLATE, indent=2)
)

PROJECTS_ABSENT_TEMPLATES = frozenset({"T1_functional", "T1_executive", "T3_table", "T5_minimal"})


def clean_target(truth: dict, doc_text: str, template: str) -> dict:
    """Remove evidence-absent github/linkedin labels and phantom projects."""
    out = dict(truth)
    low = (doc_text or "").lower()
    for field, needle in (("github_url", "github"), ("linkedin_url", "linkedin")):
        if out.get(field) and needle not in low:
            out[field] = None
    if template in PROJECTS_ABSENT_TEMPLATES and out.get("projects"):
        if not any(
            (p.get("name") or "").lower() in low
            for p in out["projects"]
            if p.get("name")
        ):
            out["projects"] = []
    return out


def load_manifest(manifest_path: Path) -> list[dict]:
    return [json.loads(x) for x in manifest_path.read_text().splitlines() if x.strip()]


def build_gemma_requests(manifest_path: Path, split_name: str) -> list[dict]:
    rows = load_manifest(manifest_path)
    out = []
    missing_reps = 0
    missing_truth = 0
    for row in rows:
        cv_id = row["cv_id"]
        md_path = REPS / f"{cv_id}.md"
        if not md_path.exists():
            missing_reps += 1
            continue

        # schema_reduced path
        ref = row["artifacts"].get("schema_reduced", {})
        truth_path = DS / ref["path"] if ref.get("path") else None
        if truth_path is None or not truth_path.exists():
            missing_truth += 1
            continue

        text = md_path.read_text(errors="ignore")
        truth_raw = json.loads(truth_path.read_text())

        # pymupdf plain text for evidence check
        pymupdf_ref = row["artifacts"].get("pymupdf_text", {})
        pymupdf_text_path = DS / pymupdf_ref["path"] if pymupdf_ref.get("path") else None
        doc_text = pymupdf_text_path.read_text(errors="ignore") if (pymupdf_text_path and pymupdf_text_path.exists()) else text
        truth = clean_target(truth_raw, doc_text, row.get("template", ""))

        out.append({
            "cv_id": cv_id,
            "representation": "pymupdf4llm_markdown",
            "split": split_name,
            "template": row.get("template"),
            "tier": row.get("tier"),
            "primary_domain": row.get("primary_domain"),
            "text": text,
            "system": SYSTEM,
            "truth": truth,
        })

    print(f"  [{split_name}] built {len(out)} requests  |  missing_reps={missing_reps}  missing_truth={missing_truth}")
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path}")


SPLITS = [
    ("id_test", ROOT / "artifacts/manifests/id_test.jsonl"),
    ("template_ood", ROOT / "artifacts/manifests/template_ood_test.jsonl"),
]

FT_OUT = ROOT / "artifacts/trials/ft"
ROUTER_OUT = ROOT / "artifacts/trials/router"


def main():
    for split_name, manifest_path in SPLITS:
        print(f"\n=== {split_name} ===")
        requests = build_gemma_requests(manifest_path, split_name)

        # gemma-ft requests (used by gemma_adapter_infer.py)
        write_jsonl(FT_OUT / f"{split_name}.gemma-ft.requests.jsonl", requests)

        # smolvlm2-ft requests (same text-only format; image_paths omitted since not needed yet)
        write_jsonl(FT_OUT / f"{split_name}.smolvlm2-ft.requests.jsonl", requests)

        # router requests (identical format — same fields)
        write_jsonl(ROUTER_OUT / f"{split_name}.requests.jsonl", requests)

    print("\nAll request files built successfully.")


if __name__ == "__main__":
    main()
