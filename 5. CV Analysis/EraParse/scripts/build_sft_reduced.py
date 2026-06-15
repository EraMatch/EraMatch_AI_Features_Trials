"""Build cleaned reduced-schema SFT pairs (ShareGPT format) from a manifest.

Input  per CV: pymupdf4llm markdown representation.
Target per CV: reduced ground-truth JSON with evidence-absent cells removed
               (phantom github/linkedin URLs + non-rendering project templates)
               so the model learns faithful extraction, not to hallucinate.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DS = ROOT.parent / "eramatch_benchmark_v4"
REPS = ROOT / "artifacts" / "representations" / "pymupdf4llm_markdown"

import sys
sys.path.insert(0, str(ROOT / "src"))
from eraparse.constants import REDUCED_SCHEMA_TEMPLATE

PROJECTS_ABSENT_TEMPLATES = frozenset({"T1_functional", "T1_executive", "T3_table", "T5_minimal"})

SYSTEM = (
    "You are a precise CV information extraction system. Extract the CV into "
    "exactly this JSON schema. Return one valid JSON object. Use empty strings "
    "or empty arrays for missing values. Do not invent values absent from the CV.\n\n"
    "Schema:\n" + json.dumps(REDUCED_SCHEMA_TEMPLATE, indent=2)
)


def clean_target(truth: dict, doc_text: str, template: str) -> dict:
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


def build_example(cv_id: str, markdown: str, target: dict) -> dict:
    return {
        "id": cv_id,
        "conversations": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": markdown},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
    }


def _reduced_truth(row: dict) -> dict:
    ref = row["artifacts"]["schema_reduced"]
    return json.loads((DS / ref["path"]).read_text())


def _doc_text(row: dict) -> str:
    ref = row["artifacts"]["pymupdf_text"]
    return (DS / ref["path"]).read_text(errors="ignore")


def build(manifest_path: Path, out_path: Path) -> int:
    rows = [json.loads(x) for x in manifest_path.read_text().splitlines() if x.strip()]
    written = 0
    with out_path.open("w") as fh:
        for row in rows:
            cv_id = row["cv_id"]
            md = REPS / f"{cv_id}.md"
            if not md.exists():
                continue
            try:
                truth = _reduced_truth(row)
                text = _doc_text(row)
            except Exception:
                continue
            target = clean_target(truth, text, row.get("template", ""))
            fh.write(json.dumps(build_example(cv_id, md.read_text(errors="ignore"), target)) + "\n")
            written += 1
    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "artifacts/manifests/train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/sft/train.reduced.sft.jsonl"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = build(Path(a.manifest), Path(a.out))
    print(f"wrote {n} SFT examples -> {a.out}")
