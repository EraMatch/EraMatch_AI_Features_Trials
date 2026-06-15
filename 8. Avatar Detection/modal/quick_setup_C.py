import csv
import os
import random
from pathlib import Path

import modal

IMG_SIZE = 256
DATA_ROOT = Path("/data")
RESULTS_ROOT = Path("/results")
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

app = modal.App("avatar-quick-setup")

data_vol = modal.Volume.from_name("avatar-data-vol", create_if_missing=True)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={
        "/data": data_vol,
        "/results": results_vol,
    },
)
def resize_and_manifest_C() -> dict:
    from PIL import Image
    from sklearn.model_selection import train_test_split

    random.seed(42)

    DIR_C = DATA_ROOT / "dataset_C"
    SRC_REAL = DIR_C / "Human Faces Dataset" / "Real Images"
    SRC_FAKE = DIR_C / "Human Faces Dataset" / "AI-Generated Images"
    DST_REAL = DIR_C / "256x256" / "real"
    DST_FAKE = DIR_C / "256x256" / "fake"

    DST_REAL.mkdir(parents=True, exist_ok=True)
    DST_FAKE.mkdir(parents=True, exist_ok=True)

    def collect_images(root):
        return sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def resize_image(src, dst):
        try:
            img = Image.open(src)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img_resized.save(dst, quality=95)
            return True
        except Exception as e:
            print(f"  WARNING: failed to resize {src.name}: {e}")
            return False

    real_src = collect_images(SRC_REAL)
    fake_src = collect_images(SRC_FAKE)
    print(f"Found {len(real_src)} real, {len(fake_src)} fake in Dataset C")

    real_pairs = []
    for i, src in enumerate(real_src):
        dst = DST_REAL / f"real_{i:06d}{src.suffix}"
        if resize_image(src, dst):
            real_pairs.append(dst)
        if (i + 1) % 1000 == 0:
            print(f"  real: resized {i + 1}/{len(real_src)}")

    fake_pairs = []
    for i, src in enumerate(fake_src):
        dst = DST_FAKE / f"fake_{i:06d}{src.suffix}"
        if resize_image(src, dst):
            fake_pairs.append(dst)
        if (i + 1) % 1000 == 0:
            print(f"  fake: resized {i + 1}/{len(fake_src)}")

    print(f"Resized: {len(real_pairs)} real, {len(fake_pairs)} fake")

    data_vol.commit()
    print("Volume committed after resize.")

    real_labels = [0] * len(real_pairs)
    fake_labels = [1] * len(fake_pairs)

    real_train, real_temp, rl_train, rl_temp = train_test_split(
        real_pairs, real_labels, test_size=0.30, random_state=42, stratify=real_labels
    )
    real_val, real_test, rl_val, rl_test = train_test_split(
        real_temp, rl_temp, test_size=0.50, random_state=42, stratify=rl_temp
    )

    fake_train, fake_temp, fl_train, fl_temp = train_test_split(
        fake_pairs, fake_labels, test_size=0.30, random_state=42, stratify=fake_labels
    )
    fake_val, fake_test, fl_val, fl_test = train_test_split(
        fake_temp, fl_temp, test_size=0.50, random_state=42, stratify=fl_temp
    )

    manifest_rows = []

    for path in real_train:
        manifest_rows.append(
            {
                "path": str(path.relative_to(DATA_ROOT)),
                "label": 0,
                "dataset": "C",
                "split": "train",
                "generator": "unknown",
            }
        )
    for path in real_val:
        manifest_rows.append(
            {
                "path": str(path.relative_to(DATA_ROOT)),
                "label": 0,
                "dataset": "C",
                "split": "val",
                "generator": "unknown",
            }
        )
    for path in real_test:
        manifest_rows.append(
            {
                "path": str(path.relative_to(DATA_ROOT)),
                "label": 0,
                "dataset": "C",
                "split": "test",
                "generator": "unknown",
            }
        )
    for path in fake_train:
        manifest_rows.append(
            {
                "path": str(path.relative_to(DATA_ROOT)),
                "label": 1,
                "dataset": "C",
                "split": "train",
                "generator": "GAN_unknown",
            }
        )
    for path in fake_val:
        manifest_rows.append(
            {
                "path": str(path.relative_to(DATA_ROOT)),
                "label": 1,
                "dataset": "C",
                "split": "val",
                "generator": "GAN_unknown",
            }
        )
    for path in fake_test:
        manifest_rows.append(
            {
                "path": str(path.relative_to(DATA_ROOT)),
                "label": 1,
                "dataset": "C",
                "split": "test",
                "generator": "GAN_unknown",
            }
        )

    random.shuffle(manifest_rows)

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "dataset", "split", "generator"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} rows to {MANIFEST_PATH}")

    from collections import Counter

    split_counts = Counter(r["split"] for r in manifest_rows)
    label_counts = Counter(
        ("real" if r["label"] == 0 else "fake") for r in manifest_rows
    )
    print(f"Splits: {dict(split_counts)}")
    print(f"Labels: {dict(label_counts)}")

    data_vol.commit()
    print("Volume committed after manifest.")

    return {
        "manifest_rows": len(manifest_rows),
        "real_resized": len(real_pairs),
        "fake_resized": len(fake_pairs),
        "splits": dict(split_counts),
    }


@app.local_entrypoint()
def main() -> None:
    print("Quick setup: Dataset C resize + manifest...")
    result = resize_and_manifest_C.remote()
    print(f"\nDone: {result}")
