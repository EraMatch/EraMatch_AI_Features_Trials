#!/usr/bin/env python3
"""run_batch2.py — Runner script for Batch 2 synthetic CV generation.

Generates ~100 CVs, renders them to PDF, saves JSON + metadata,
and produces a manifest with domain distribution.
"""

import json
import logging
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from cv_generator_batch2 import DOMAINS, generate_synthetic_cv
from pdf_renderer import TIER_TEMPLATES, render_cv_pdf
from schema import CVSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("batches/batch_2_synthetic")
TARGET_COUNT = 100

# Domain distribution targets (matches default in generate_batch)
DOMAIN_DISTRIBUTION: Dict[str, int] = {
    "backend": 20,
    "frontend": 20,
    "data_science": 15,
    "devops": 15,
    "fullstack": 10,
    "mobile": 10,
    "security": 5,
    "product_management": 5,
}


def get_template_for_cv(cv_index: int, tier: str) -> str:
    """Rotate through tier templates based on CV index."""
    templates = TIER_TEMPLATES.get(tier, TIER_TEMPLATES["T1"])
    return templates[cv_index % len(templates)]


def validate_pdf_with_pdfinfo(pdf_path: Path) -> bool:
    """Check a PDF is valid using pdfinfo."""
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Resolve domain distribution — map "security" alias to "cybersecurity"
    resolved: Dict[str, int] = {}
    for key, val in DOMAIN_DISTRIBUTION.items():
        resolved_key = "cybersecurity" if key == "security" else key
        resolved[resolved_key] = resolved.get(resolved_key, 0) + val

    # Scale if totals don't match TARGET_COUNT
    dist_total = sum(resolved.values())
    if dist_total != TARGET_COUNT and dist_total > 0:
        scale = TARGET_COUNT / dist_total
        resolved = {k: max(1, round(v * scale)) for k, v in resolved.items()}

    results: List[Dict[str, Any]] = []
    cv_id = 1
    pdf_ok = 0
    pdf_fail = 0
    json_ok = 0
    json_fail = 0
    domain_counter: Counter = Counter()
    template_counter: Counter = Counter()

    total_cvs = sum(resolved.values())
    logger.info(
        f"Starting batch 2 generation: {total_cvs} CVs across {len(resolved)} domains"
    )

    with tqdm(total=total_cvs, desc="Generating CVs", unit="cv") as pbar:
        for domain_key, n in resolved.items():
            if domain_key not in DOMAINS:
                logger.warning(f"Unknown domain '{domain_key}', skipping")
                continue
            for i in range(n):
                # Generate CVSchema object
                cv: CVSchema = generate_synthetic_cv(cv_id=cv_id, domain_key=domain_key)
                tier = cv.parsing_metadata.get("tier", "T1")
                template_name = get_template_for_cv(i, tier)

                # Save JSON
                json_path = OUTPUT_DIR / f"cv_{cv_id:05d}.json"
                cv_dict = cv.model_dump()
                try:
                    json_path.write_text(
                        json.dumps(cv_dict, indent=2, default=str), encoding="utf-8"
                    )
                    json_ok += 1
                except Exception as e:
                    logger.error(f"Failed to write JSON for cv_{cv_id:05d}: {e}")
                    json_fail += 1

                # Render PDF
                pdf_path = OUTPUT_DIR / f"cv_{cv_id:05d}.pdf"
                ok = render_cv_pdf(cv, template_name, pdf_path)
                if ok:
                    pdf_ok += 1
                else:
                    pdf_fail += 1
                    logger.error(
                        f"PDF render failed for cv_{cv_id:05d} template={template_name}"
                    )

                # Save per-CV metadata
                meta_path = OUTPUT_DIR / f"cv_{cv_id:05d}_meta.json"
                meta = {
                    "cv_id": cv_id,
                    "domain": domain_key,
                    "seniority_level": cv.seniority_level,
                    "tier": tier,
                    "template_used": template_name,
                    "pdf_path": str(pdf_path),
                    "json_path": str(json_path),
                    "pdf_rendered": ok,
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                domain_counter[domain_key] += 1
                template_counter[template_name] += 1
                results.append(cv_dict)
                cv_id += 1
                pbar.update(1)

    elapsed = time.time() - start_time

    # Build manifest
    manifest = {
        "total_cvs": len(results),
        "domain_distribution": dict(domain_counter),
        "template_usage": dict(template_counter),
        "pdf_ok": pdf_ok,
        "pdf_fail": pdf_fail,
        "json_ok": json_ok,
        "json_fail": json_fail,
        "generator_version": "cv_generator_batch2",
        "model_used": "synthetic_faker",
        "generation_timestamp": datetime.now().isoformat(),
        "generation_seconds": round(elapsed, 2),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Batch 2 Generation Complete")
    print(f"{'=' * 60}")
    print(f"Total CVs generated: {len(results)}")
    print(f"PDFs: {pdf_ok} ok, {pdf_fail} failed")
    print(f"JSONs: {json_ok} ok, {json_fail} failed")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nDomain distribution:")
    for domain, count in sorted(domain_counter.items()):
        print(f"  {domain}: {count}")
    print(f"\nTemplate usage:")
    for tmpl, count in sorted(template_counter.items()):
        print(f"  {tmpl}: {count}")
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
