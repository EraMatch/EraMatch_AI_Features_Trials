"""Build (image, schema, JSON) SFT triples for SmolVLM2-2.2B fine-tuning (Track C).

Each example = first page image of the CV (or concatenated pages for short CVs)
+ schema-as-template in the user turn + gold JSON in the assistant turn.

Schema augmentation: generate N_AUGMENTS random schema subsets per CV so the
VLM learns to follow *arbitrary* target schemas, not just the fixed one.
This is the key ingredient of the NuExtract recipe.

Output: one JSONL per example:
  {
    "id": cv_id,
    "image_paths": ["page_images/cv_xxxxx-1.png", ...],   # relative to dataset_root
    "schema": {...},        # the target schema template used for this example
    "target": {...},        # gold JSON restricted to schema fields
    "conversations": [...]  # HF chat-template-ready messages (no images embedded)
  }

Images are NOT embedded in the JSONL (they're large). The fine-tune script reads
them from disk using image_paths.

Run:
    uv run python scripts/build_sft_vision.py
    uv run python scripts/build_sft_vision.py --manifest artifacts/manifests/validation.jsonl \\
        --out artifacts/sft/validation.vision.sft.jsonl
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DS = ROOT.parent / "eramatch_benchmark_v4"

import sys
sys.path.insert(0, str(ROOT / "src"))
from eraparse.constants import REDUCED_SCHEMA_TEMPLATE

SEED = 42
N_AUGMENTS = 3  # schema-subset variants per CV; keeps JSONL manageable

SYSTEM = (
    "You are a precise CV information extraction system. "
    "Extract the CV into exactly the provided JSON schema. "
    "Return one valid JSON object. Use empty strings or empty arrays for missing values. "
    "Do not invent values absent from the CV."
)

# Fields that can be dropped in schema augmentation (always keep full_name + email)
DROPPABLE_FIELDS = [
    "location", "phone", "linkedin_url", "github_url", "summary",
    "skills", "work_experience", "education", "projects", "certifications",
]


def augment_schema(base_schema: dict, rng: random.Random) -> dict:
    """Return a random subset of fields from base_schema (always keep full_name, email)."""
    keep = {"full_name", "email"}
    n_drop = rng.randint(0, len(DROPPABLE_FIELDS) // 2)
    drop = set(rng.sample(DROPPABLE_FIELDS, n_drop))
    return {k: v for k, v in base_schema.items() if k not in drop}


def restrict_target(target: dict, schema: dict) -> dict:
    """Drop target fields not present in schema."""
    return {k: v for k, v in target.items() if k in schema}


def clean_target(truth: dict, doc_text: str, template: str) -> dict:
    """Same faithfulness cleaning as text SFT."""
    PROJECTS_ABSENT = frozenset({"T1_functional", "T1_executive", "T3_table", "T5_minimal"})
    out = dict(truth)
    low = (doc_text or "").lower()
    for field, needle in (("github_url", "github"), ("linkedin_url", "linkedin")):
        if out.get(field) and needle not in low:
            out[field] = None
    if template in PROJECTS_ABSENT and out.get("projects"):
        if not any(
            (p.get("name") or "").lower() in low
            for p in out["projects"] if p.get("name")
        ):
            out["projects"] = []
    return out


def build_vision_example(
    cv_id: str,
    image_paths: list[str],
    schema: dict,
    target: dict,
) -> dict:
    schema_str = json.dumps(schema, indent=2)
    user_text = f"Extract the CV into this JSON schema:\n{schema_str}\n\nReturn valid JSON only."
    return {
        "id": cv_id,
        "image_paths": image_paths,
        "schema": schema,
        "target": target,
        "conversations": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
    }


def _get_page_images(row: dict) -> list[str]:
    """Return relative page image paths for this CV (relative to DS root)."""
    return [p["path"] for p in row.get("page_images", [])]


def _doc_text(row: dict) -> str:
    ref = row["artifacts"].get("pymupdf_text")
    if not ref:
        return ""
    p = DS / ref["path"]
    return p.read_text(errors="ignore") if p.exists() else ""


def build(manifest_path: Path, out_path: Path, n_augments: int = N_AUGMENTS) -> int:
    rows = [json.loads(x) for x in manifest_path.read_text().splitlines() if x.strip()]
    rng = random.Random(SEED)
    written = 0
    with out_path.open("w") as fh:
        for row in rows:
            cv_id = row["cv_id"]
            image_paths = _get_page_images(row)
            if not image_paths:
                continue

            try:
                ref = row["artifacts"]["schema_reduced"]
                truth = json.loads((DS / ref["path"]).read_text())
            except Exception:
                continue

            doc_text = _doc_text(row)
            template = row.get("template", "")
            cleaned = clean_target(truth, doc_text, template)

            # Full schema example (always included)
            ex = build_vision_example(cv_id, image_paths, REDUCED_SCHEMA_TEMPLATE, cleaned)
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1

            # Schema-augmented variants
            for _ in range(n_augments):
                sub_schema = augment_schema(REDUCED_SCHEMA_TEMPLATE, rng)
                sub_target = restrict_target(cleaned, sub_schema)
                ex = build_vision_example(cv_id, image_paths, sub_schema, sub_target)
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written += 1

    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "artifacts/manifests/train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/sft/train.vision.sft.jsonl"))
    ap.add_argument("--n-augments", type=int, default=N_AUGMENTS)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = build(Path(a.manifest), Path(a.out), a.n_augments)
    print(f"wrote {n} vision SFT examples -> {a.out}")
