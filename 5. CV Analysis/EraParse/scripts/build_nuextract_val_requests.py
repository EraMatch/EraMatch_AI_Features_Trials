"""Build NuExtract-format inference requests for the validation split.

Output: one JSONL with fields cv_id, prompt, split, template, tier — compatible
with nuextract_adapter_infer.py.

Run:
    uv run python scripts/build_nuextract_val_requests.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DS = ROOT.parent / "eramatch_benchmark_v4"
REPS = ROOT / "artifacts" / "representations" / "pymupdf4llm_markdown"

import sys
sys.path.insert(0, str(ROOT / "src"))
from eraparse.representations import build_nuextract_prompt

MANIFEST = ROOT / "artifacts/manifests/validation.jsonl"
OUT = ROOT / "artifacts/trials/router/validation.nuextract-tiny-ft.requests.jsonl"


def main():
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w") as fh:
        for row in rows:
            cv_id = row["cv_id"]
            md = REPS / f"{cv_id}.md"
            if not md.exists():
                continue
            text = md.read_text(errors="ignore")
            prompt = build_nuextract_prompt(text)
            truth_path = DS / row["artifacts"]["schema_reduced"]["path"]
            truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
            fh.write(json.dumps({
                "cv_id": cv_id,
                "prompt": prompt,
                "split": row.get("split"),
                "template": row.get("template"),
                "tier": row.get("tier"),
                "primary_domain": row.get("primary_domain"),
                "truth": truth,
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} requests -> {OUT}")


if __name__ == "__main__":
    main()
