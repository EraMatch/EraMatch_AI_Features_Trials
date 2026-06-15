"""GRAVEX-200K dataset content audit — Modal CPU function.

Downloads 1000 random images from the GRAVEX dataset, runs MediaPipe
face detection, and determines whether the dataset is suitable for
the avatar detection training pipeline.

Decision rule (from AGENTS.md):
  face_rate >= 0.60  →  INCLUDE (supplement training data)
  face_rate <  0.60  →  EXCLUDE (low domain relevance)

Usage:
    modal run modal/audit_gravex.py

Requires pre-created Modal secrets and volumes (see Task 2):
    modal secret create kaggle-creds KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
    modal volume create avatar-data-vol
    modal volume create avatar-results-vol
"""

import random
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/data")
RESULTS_ROOT = Path("/results")
GRAVEX_DIR = DATA_ROOT / "gravex"
DB_PATH = RESULTS_ROOT / "experiments.db"

KAGGLE_SLUG = "muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal"
N_SAMPLE = 1000
FACE_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# App + named resources
# ---------------------------------------------------------------------------
app = modal.App("avatar-audit-gravex")

data_vol = modal.Volume.from_name("avatar-data-vol", create_if_missing=False)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=False)

kaggle_creds = modal.Secret.from_name("kaggle-creds")
hf_secret = modal.Secret.from_name("huggingface-secret")

# ---------------------------------------------------------------------------
# Image — CPU-only, no GPU needed for face detection on 1K images
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "pytorch-metric-learning>=2.0.0",
        "albumentations>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "kaggle>=1.5.0",
        "mediapipe>=0.10.0",
        "opencv-python-headless>=4.8.0",
        "umap-learn>=0.5.0",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("src")
)


# ---------------------------------------------------------------------------
# Smoke test — verify imports and DB schema
# ---------------------------------------------------------------------------
@app.function(image=image, cpu=2, memory=4096, timeout=2 * 60)
def validate_remote_imports() -> dict:
    """Cheap CPU smoke test: verify all imports work inside the container."""
    import mediapipe
    import cv2
    from src.utils.db import init_db, log_dataset_audit

    # kaggle API auto-authenticates on import in newer versions, so we check
    # the package metadata via importlib without triggering auth.
    import importlib.metadata

    try:
        kaggle_version = importlib.metadata.version("kaggle")
        kaggle_available = True
    except Exception:
        kaggle_version = "unavailable"
        kaggle_available = False

    return {
        "mediapipe": mediapipe.__version__,
        "opencv": cv2.__version__,
        "kaggle": kaggle_version,
        "kaggle_available": kaggle_available,
        "db_init": callable(init_db),
        "log_audit": callable(log_dataset_audit),
    }


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,  # 30 min — download + inference on 1K images
    volumes={
        "/data": data_vol,
        "/results": results_vol,
    },
    secrets=[kaggle_creds, hf_secret],
)
def run_audit() -> dict:
    """Download 1000 random GRAVEX images, run face detection, log decision."""
    import os
    import glob

    import cv2
    import numpy as np
    import mediapipe as mp
    from PIL import Image
    from kaggle.api.kaggle_api_extended import KaggleApi

    from src.utils.db import init_db, log_dataset_audit

    random.seed(42)

    # -----------------------------------------------------------------------
    # 1. Download GRAVEX dataset via Kaggle API
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("GRAVEX-200K Dataset Audit")
    print("=" * 60)

    os.makedirs(GRAVEX_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"\n[1/5] Downloading dataset: {KAGGLE_SLUG}")
    print(f"       Destination: {GRAVEX_DIR}")
    api.dataset_download_files(KAGGLE_SLUG, path=str(GRAVEX_DIR), unzip=True)

    # Commit so downloaded data persists on the volume for later tasks
    data_vol.commit()
    print("       Data volume committed.")

    # -----------------------------------------------------------------------
    # 2. Collect all image file paths
    # -----------------------------------------------------------------------
    print("\n[2/5] Scanning for image files...")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    all_image_paths = [
        p
        for p in GRAVEX_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    print(f"       Found {len(all_image_paths)} total images.")

    if len(all_image_paths) == 0:
        print(
            "       ERROR: No images found after download. Check dataset slug or unzip."
        )
        return {"error": "no_images_found", "n_total": 0}

    # -----------------------------------------------------------------------
    # 3. Sample N_SAMPLE random images
    # -----------------------------------------------------------------------
    n_sample = min(N_SAMPLE, len(all_image_paths))
    sampled = random.sample(all_image_paths, n_sample)
    print(f"\n[3/5] Sampled {n_sample} images for face detection.")

    # -----------------------------------------------------------------------
    # 4. Run MediaPipe face detection
    # -----------------------------------------------------------------------
    print("\n[4/5] Running MediaPipe face detection...")
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,  # short-range model (faces within ~2m)
        min_detection_confidence=0.5,
    )

    n_with_face = 0
    n_processed = 0
    resolutions = []

    for i, img_path in enumerate(sampled):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            resolutions.append((w, h))

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if results.detections and len(results.detections) > 0:
                n_with_face += 1

            n_processed += 1
        except Exception as e:
            print(f"       Warning: failed to process {img_path.name}: {e}")
            continue

        if (i + 1) % 200 == 0:
            print(f"       Processed {i + 1}/{n_sample}...")

    face_detector.close()

    # -----------------------------------------------------------------------
    # 5. Compute statistics and make decision
    # -----------------------------------------------------------------------
    print("\n[5/5] Computing statistics...")

    face_rate = n_with_face / n_processed if n_processed > 0 else 0.0
    avg_w = int(np.mean([r[0] for r in resolutions])) if resolutions else 0
    avg_h = int(np.mean([r[1] for r in resolutions])) if resolutions else 0
    avg_resolution = f"{avg_w}x{avg_h}"

    decision = "INCLUDE" if face_rate >= FACE_THRESHOLD else "EXCLUDE"

    print(f"\n{'=' * 60}")
    print(f"GRAVEX DECISION: {decision}")
    print(f"  face_rate     = {face_rate:.4f}  (threshold: {FACE_THRESHOLD})")
    print(f"  avg_resolution = {avg_resolution}")
    print(f"  n_with_face   = {n_with_face} / {n_processed}")
    print(f"  n_total_found = {len(all_image_paths)}")
    print(f"{'=' * 60}\n")

    # -----------------------------------------------------------------------
    # 6. Log to SQLite database
    # -----------------------------------------------------------------------
    init_db(db_path=str(DB_PATH))

    notes = (
        f"Decision: {decision}. "
        f"Face rate {face_rate:.4f} vs threshold {FACE_THRESHOLD}. "
        f"Avg resolution {avg_resolution}. "
        f"Sampled {n_processed}/{len(all_image_paths)} total images."
    )

    log_dataset_audit(
        db_path=str(DB_PATH),
        dataset_name="gravex",
        n_total=n_processed,
        n_real=None,  # unknown — dataset labels not inspected
        n_fake=None,  # unknown — dataset labels not inspected
        face_rate=face_rate,
        avg_resolution=avg_resolution,
        generators="unknown",
        notes=notes,
    )

    print(f"  Audit row logged to {DB_PATH}")

    # Commit results volume so the DB row persists
    results_vol.commit()
    print(f"  Results volume committed.")

    return {
        "decision": decision,
        "face_rate": face_rate,
        "n_with_face": n_with_face,
        "n_processed": n_processed,
        "n_total_found": len(all_image_paths),
        "avg_resolution": avg_resolution,
    }


# ---------------------------------------------------------------------------
# Local entrypoint — invoked via `modal run modal/audit_gravex.py`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    """Run the GRAVEX content audit. Requires pre-created volumes and secrets."""
    print("Starting GRAVEX dataset audit...")
    print("  Volumes: avatar-data-vol, avatar-results-vol")
    print("  Secrets: kaggle-creds, huggingface-secret")

    # Step 1: smoke test imports
    print("\n[Smoke Test] Validating remote imports...")
    smoke = validate_remote_imports.remote()
    print(
        f"  mediapipe={smoke['mediapipe']}, opencv={smoke['opencv']}, "
        f"kaggle={smoke['kaggle']}"
    )
    print(f"  db_init={smoke['db_init']}, log_audit={smoke['log_audit']}")

    # Step 2: run the full audit
    print("\n[Main] Running audit on 1000 random GRAVEX images...")
    result = run_audit.remote()

    if "error" in result:
        print(f"\nAudit failed: {result}")
    else:
        print(f"\nAudit complete:")
        print(f"  Decision:       {result['decision']}")
        print(f"  Face rate:      {result['face_rate']:.4f}")
        print(f"  Face count:     {result['n_with_face']} / {result['n_processed']}")
        print(f"  Total images:   {result['n_total_found']}")
        print(f"  Avg resolution: {result['avg_resolution']}")
