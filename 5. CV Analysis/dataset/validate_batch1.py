#!/usr/bin/env python3
"""Validate Batch 1 Premium CV outputs."""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "batches" / "batch_1_premium"
EVIDENCE_DIR = BASE_DIR / ".sisyphus" / "evidence"

sys.path.insert(0, str(BASE_DIR))

from schema import CVSchema


def validate_batch():
    errors = []
    warnings = []
    checks = []

    # 1. Count files
    pdfs = sorted(OUTPUT_DIR.glob("cv_premium_*.pdf"))
    cv_jsons = sorted(OUTPUT_DIR.glob("cv_premium_[0-9]*.json"))
    meta_jsons = sorted(OUTPUT_DIR.glob("cv_premium_*_metadata.json"))

    checks.append(f"PDF count: {len(pdfs)} (expected: 18-20)")
    checks.append(f"CV JSON count: {len(cv_jsons)} (expected: 18-20)")
    checks.append(f"Metadata JSON count: {len(meta_jsons)} (expected: 18-20)")
    checks.append(f"Manifest exists: {(OUTPUT_DIR / 'manifest.json').exists()}")

    if len(pdfs) < 18:
        errors.append(f"Only {len(pdfs)} PDFs, need at least 18")
    if len(cv_jsons) < 18:
        errors.append(f"Only {len(cv_jsons)} CV JSONs, need at least 18")

    # 2. Validate each CV JSON against CVSchema
    github_urls = []
    email_matches = 0
    schema_errors = []

    for i, cv_json in enumerate(cv_jsons, 1):
        try:
            data = json.loads(cv_json.read_text(encoding="utf-8"))
            cv = CVSchema(**data)

            # Check email match with metadata
            meta_path = OUTPUT_DIR / f"cv_premium_{i:03d}_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if cv.email == meta.get("source_email"):
                    email_matches += 1
                else:
                    warnings.append(
                        f"CV {i:03d}: email mismatch — cv.email={cv.email}, meta={meta.get('source_email')}"
                    )

            # Collect GitHub URLs
            for proj in cv.projects:
                if proj.url and "github.com" in proj.url:
                    github_urls.append((i, proj.url))

        except Exception as exc:
            schema_errors.append(f"CV {i:03d} ({cv_json.name}): {exc}")
            errors.append(f"CV {i:03d} schema validation failed: {exc}")

    checks.append(
        f"Schema-valid CVs: {len(cv_jsons) - len(schema_errors)}/{len(cv_jsons)}"
    )
    checks.append(
        f"Email matches between CV and metadata: {email_matches}/{len(cv_jsons)}"
    )

    # 3. Check GitHub URLs are real (not synthetic-*)
    synthetic_urls = [(i, url) for i, url in github_urls if "synthetic-" in url]
    real_urls = [
        (i, url)
        for i, url in github_urls
        if "synthetic-" not in url and url.startswith("https://github.com/")
    ]
    bad_urls = [
        (i, url) for i, url in github_urls if not url.startswith("https://github.com/")
    ]

    checks.append(f"GitHub URLs (real): {len(real_urls)}")
    checks.append(f"GitHub URLs (synthetic): {len(synthetic_urls)}")
    checks.append(f"GitHub URLs (bad format): {len(bad_urls)}")

    if synthetic_urls:
        for i, url in synthetic_urls:
            warnings.append(f"CV {i:03d}: synthetic URL found: {url}")
    if bad_urls:
        for i, url in bad_urls:
            errors.append(f"CV {i:03d}: bad GitHub URL format: {url}")

    # 4. Validate PDFs with pdfinfo
    pdf_valid = 0
    pdf_invalid = []
    for pdf_path in pdfs:
        try:
            result = subprocess.run(
                ["pdfinfo", str(pdf_path)], capture_output=True, text=True, timeout=10
            )
            output = result.stdout
            if "Pages:" in output:
                pages_line = [l for l in output.split("\n") if l.startswith("Pages:")]
                if pages_line:
                    pages = pages_line[0].split(":")[1].strip()
                    if pages in ("1", "2"):
                        pdf_valid += 1
                    else:
                        pdf_invalid.append((pdf_path.name, f"pages={pages}"))
                        warnings.append(
                            f"PDF {pdf_path.name}: {pages} pages (expected 1-2)"
                        )
                else:
                    pdf_invalid.append((pdf_path.name, "no Pages line"))
                    errors.append(f"PDF {pdf_path.name}: no Pages line in pdfinfo")
            else:
                pdf_invalid.append((pdf_path.name, "no Pages in output"))
                errors.append(f"PDF {pdf_path.name}: pdfinfo output missing Pages")
        except Exception as exc:
            pdf_invalid.append((pdf_path.name, str(exc)))
            errors.append(f"PDF {pdf_path.name}: pdfinfo error: {exc}")

    checks.append(f"Valid PDFs (1-2 pages): {pdf_valid}/{len(pdfs)}")
    if pdf_invalid:
        checks.append(f"Invalid PDFs: {len(pdf_invalid)}")

    # 5. Check manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks.append(f"Manifest total_cvs: {manifest.get('total_cvs')}")
        checks.append(f"Manifest total_attempted: {manifest.get('total_attempted')}")
        checks.append(
            f"Manifest emails_failed count: {len(manifest.get('emails_failed', []))}"
        )
        checks.append(f"Manifest specializations: {manifest.get('specializations')}")
    else:
        errors.append("manifest.json not found")

    # Build report
    lines = [
        "=" * 60,
        "BATCH 1 PREMIUM CV VALIDATION REPORT",
        "=" * 60,
        "",
        "CHECKS:",
    ]
    for c in checks:
        lines.append(f"  {c}")

    lines.append("")
    lines.append(f"ERRORS ({len(errors)}):")
    if errors:
        for e in errors:
            lines.append(f"  ✗ {e}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"WARNINGS ({len(warnings)}):")
    if warnings:
        for w in warnings:
            lines.append(f"  ⚠ {w}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"VERDICT: {'PASS' if not errors else 'FAIL'}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    # Save evidence
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "task-10-batch1-validation.txt").write_text(
        report, encoding="utf-8"
    )
    print(
        f"\nValidation report saved to {EVIDENCE_DIR / 'task-10-batch1-validation.txt'}"
    )

    return len(errors) == 0


if __name__ == "__main__":
    ok = validate_batch()
    sys.exit(0 if ok else 1)
