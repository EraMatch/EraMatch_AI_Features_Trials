#!/usr/bin/env python3
"""Run Batch 1 Premium CV Generation (20 emails).

Generates premium CVs using deepseek-v4-pro:cloud + real GitHub repos,
renders them to PDF, and saves all outputs to batches/batch_1_premium/.

File naming: cv_premium_{N:03d}.pdf, cv_premium_{N:03d}.json, cv_premium_{N:03d}_metadata.json

Usage:
    python3 run_batch1.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from cv_generator_batch1 import generate_premium_cv, load_specialization_map
from pdf_renderer import render_cv_pdf, get_template_for_tier, TIER_TEMPLATES
from schema import CVSchema

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = BASE_DIR / "batches" / "batch_1_premium"
EVIDENCE_DIR = BASE_DIR / ".sisyphus" / "evidence"

# Specialization → tier mapping for template selection
SPECIALIZATION_TIER: Dict[str, str] = {
    "backend": "T1",
    "frontend": "T2",
    "fullstack": "T1",
    "data_science": "T3",
    "devops": "T2",
    "mobile": "T2",
    "security": "T3",
    "cybersecurity": "T3",
    "cloud": "T2",
    "ml_engineering": "T3",
    "game_development": "T5",
    "product_management": "T4",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------


def _select_template(specialization: str, idx: int) -> str:
    """Select a template for a specialization.

    Uses the first template for the tier unless there are enough CVs
    to justify variety (then rotate through available templates).

    Args:
        specialization: The specialization string.
        idx: CV index (0-based) for rotation.

    Returns:
        Template name string (e.g. "T1_classic").
    """
    tier = SPECIALIZATION_TIER.get(specialization, "T1")
    templates = TIER_TEMPLATES.get(tier, TIER_TEMPLATES["T1"])
    # Rotate through templates for variety
    return templates[idx % len(templates)]


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------


def run_batch1() -> Tuple[List[Dict], List[Dict]]:
    """Run full Batch 1 Premium CV generation.

    Returns:
        Tuple of (results_list, failures_list).
        results_list: list of dicts with email, cv_file, pdf_file, metadata_file, template
        failures_list: list of dicts with email, reason
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    spec_map = load_specialization_map()
    emails = list(spec_map.keys())

    results: List[Dict] = []
    failures: List[Dict] = []
    cv_counter = 1  # 1-based for file naming

    # Track specializations for template rotation
    spec_counter: Dict[str, int] = {}

    logger.info("Starting Batch 1 Premium CV generation for %d emails", len(emails))
    logger.info("Output directory: %s", OUTPUT_DIR)

    batch_start = time.time()

    for email in tqdm(emails, desc="Batch 1 Premium CVs", unit="cv"):
        entry = spec_map[email]
        specialization = entry["specialization"]
        career_stage = entry["career_stage"]

        # Track rotation per specialization
        if specialization not in spec_counter:
            spec_counter[specialization] = 0
        spec_idx = spec_counter[specialization]
        spec_counter[specialization] += 1

        # Select template
        template_name = _select_template(specialization, spec_idx)

        logger.info(
            "[%d/%d] Generating: %s → %s (%s) template=%s",
            cv_counter,
            len(emails),
            email,
            specialization,
            career_stage,
            template_name,
        )

        try:
            # Generate CV (internally retries up to 2 times)
            cv_data, metadata = generate_premium_cv(email)

            # Convert dict → CVSchema object for PDF rendering
            cv_schema = CVSchema(**cv_data)

            # File naming: cv_premium_{N:03d}
            file_prefix = f"cv_premium_{cv_counter:03d}"
            cv_json_path = OUTPUT_DIR / f"{file_prefix}.json"
            pdf_path = OUTPUT_DIR / f"{file_prefix}.pdf"
            meta_path = OUTPUT_DIR / f"{file_prefix}_metadata.json"

            # Save CV JSON
            cv_json_path.write_text(
                json.dumps(cv_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Save metadata JSON
            metadata["template_used"] = template_name
            metadata["file_prefix"] = file_prefix
            meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Render PDF
            pdf_ok = render_cv_pdf(cv_schema, template_name, pdf_path)
            if not pdf_ok:
                logger.warning(
                    "PDF rendering failed for %s, continuing without PDF", email
                )
                pdf_path_str = "FAILED"
            else:
                pdf_path_str = str(pdf_path.name)

            results.append(
                {
                    "email": email,
                    "specialization": specialization,
                    "career_stage": career_stage,
                    "cv_file": f"{file_prefix}.json",
                    "pdf_file": pdf_path_str,
                    "metadata_file": f"{file_prefix}_metadata.json",
                    "template": template_name,
                    "generation_time": metadata.get("generation_time", 0),
                }
            )

            logger.info(
                "  ✓ Saved: %s (PDF=%s, time=%.1fs)",
                file_prefix,
                pdf_ok,
                metadata.get("generation_time", 0),
            )

            cv_counter += 1

        except Exception as exc:
            reason = str(exc)
            failures.append(
                {
                    "email": email,
                    "specialization": specialization,
                    "career_stage": career_stage,
                    "reason": reason,
                }
            )
            logger.error("  ✗ FAILED: %s — %s", email, reason)

    batch_time = time.time() - batch_start

    # Summary
    logger.info("=" * 60)
    logger.info(
        "Batch 1 complete: %d/%d succeeded, %d failed (%.1f min total)",
        len(results),
        len(emails),
        len(failures),
        batch_time / 60,
    )

    if failures:
        logger.info("Failed emails:")
        for f in failures:
            logger.info("  - %s: %s", f["email"], f["reason"])

    # Generate manifest
    manifest = _generate_manifest(results, failures, batch_time)

    # Save run log
    _save_run_log(results, failures, manifest)

    return results, failures


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------


def _generate_manifest(
    results: List[Dict],
    failures: List[Dict],
    batch_time: float,
) -> Dict[str, Any]:
    """Generate manifest.json for the batch."""
    # Collect repository provenance
    repo_provenance: Dict[str, List[str]] = {}
    for r in results:
        meta_path = OUTPUT_DIR / r["metadata_file"]
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for gh_repo in meta.get("github_projects", []):
                    spec = r["specialization"]
                    url = gh_repo.get("url", "")
                    if spec not in repo_provenance:
                        repo_provenance[spec] = []
                    if url and url not in repo_provenance[spec]:
                        repo_provenance[spec].append(url)
            except Exception:
                pass

    # Count specializations
    spec_counts: Dict[str, int] = {}
    for r in results:
        spec = r["specialization"]
        spec_counts[spec] = spec_counts.get(spec, 0) + 1

    manifest = {
        "batch_name": "batch_1_premium",
        "total_cvs": len(results),
        "total_attempted": len(results) + len(failures),
        "model_used": "deepseek-v4-pro:cloud",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_generation_time_seconds": round(batch_time, 2),
        "emails_processed": [r["email"] for r in results],
        "emails_failed": [
            {"email": f["email"], "reason": f["reason"]} for f in failures
        ],
        "specializations": spec_counts,
        "repository_provenance": repo_provenance,
        "output_directory": str(OUTPUT_DIR),
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Manifest saved to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# Run log (evidence)
# ---------------------------------------------------------------------------


def _save_run_log(
    results: List[Dict],
    failures: List[Dict],
    manifest: Dict[str, Any],
) -> None:
    """Save detailed run log as evidence."""
    log_path = EVIDENCE_DIR / "task-10-batch1-run.txt"

    lines = [
        "=" * 60,
        "BATCH 1 PREMIUM CV GENERATION — RUN LOG",
        "=" * 60,
        f"Timestamp: {manifest['generation_timestamp']}",
        f"Total attempted: {manifest['total_attempted']}",
        f"Total succeeded: {manifest['total_cvs']}",
        f"Total failed: {len(failures)}",
        f"Batch time: {manifest['batch_generation_time_seconds']:.1f}s ({manifest['batch_generation_time_seconds'] / 60:.1f} min)",
        f"Model: {manifest['model_used']}",
        "",
        "SPECIALIZATIONS:",
    ]
    for spec, count in sorted(manifest["specializations"].items()):
        lines.append(f"  {spec}: {count}")

    lines.append("")
    lines.append("RESULTS:")
    for i, r in enumerate(results, 1):
        lines.append(
            f"  {i:2d}. {r['email']} → {r['specialization']} ({r['career_stage']}) "
            f"→ {r['cv_file']} / {r['pdf_file']} [{r['template']}] ({r['generation_time']:.1f}s)"
        )

    if failures:
        lines.append("")
        lines.append("FAILURES:")
        for i, f in enumerate(failures, 1):
            lines.append(
                f"  {i:2d}. {f['email']} ({f['specialization']}): {f['reason']}"
            )

    lines.append("")
    lines.append("=" * 60)

    log_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Run log saved to %s", log_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results, failures = run_batch1()
    print(f"\n{'=' * 40}")
    print(f"Batch 1 Premium: {len(results)} succeeded, {len(failures)} failed")
    print(f"{'=' * 40}")
