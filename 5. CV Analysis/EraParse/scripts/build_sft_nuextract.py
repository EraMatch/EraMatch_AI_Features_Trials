"""Build cleaned reduced-schema SFT pairs in NuExtract-1.5 format.

NuExtract-1.5 uses a special extraction format (not chat), matching how it was
pretrained over many schemas. Fine-tuning ON this format preserves its
extraction pretraining — essential for the RQ2b ablation (extraction-pretrained
base vs general SLM when both fine-tuned on identical data).

Format (from NuExtract official):
  <|input|>
  ### Template:
  {schema_json}
  ### Text:
  {document_text}

  <|output|>{extracted_json}<|end-output|>
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DS = ROOT.parent / "eramatch_benchmark_v4"
REPS = ROOT / "artifacts" / "representations" / "pymupdf4llm_markdown"

import sys
sys.path.insert(0, str(ROOT / "src"))
from eraparse.constants import REDUCED_SCHEMA_TEMPLATE, NUEXTRACT_MODEL_ID
from eraparse.representations import build_nuextract_prompt

PROJECTS_ABSENT_TEMPLATES = frozenset({"T1_functional", "T1_executive", "T3_table", "T5_minimal"})

SCHEMA_TEMPLATE_STR = json.dumps(REDUCED_SCHEMA_TEMPLATE, indent=2)


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


def build_nuextract_example(cv_id: str, markdown: str, target: dict) -> dict:
    prompt = build_nuextract_prompt(markdown, REDUCED_SCHEMA_TEMPLATE)
    completion = json.dumps(target, ensure_ascii=False)
    return {
        "id": cv_id,
        "text": prompt + completion + "<|end-output|>",
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
            fh.write(json.dumps(build_nuextract_example(cv_id, md.read_text(errors="ignore"), target)) + "\n")
            written += 1
    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "artifacts/manifests/train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/sft/train.nuextract.sft.jsonl"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = build(Path(a.manifest), Path(a.out))
    print(f"wrote {n} SFT examples -> {a.out}")
