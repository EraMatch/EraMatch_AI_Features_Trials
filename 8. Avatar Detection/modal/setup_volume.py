"""Dataset download + resize + manifest pipeline — Modal CPU function.

Downloads Datasets A (130K), B (SFHQ-T2I), and C (9.6K existing) via Kaggle
API, resizes all images to 256x256 LANCZOS, samples 40K from SFHQ-T2I,
builds manifest.csv, and writes everything to avatar-data-vol.

Usage:
    modal run modal/setup_volume.py

Requires pre-created Modal secrets and volumes:
    modal secret create kaggle-creds KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
    modal volume create avatar-data-vol
    modal volume create avatar-results-vol
"""

import csv
import os
import random
from pathlib import Path

import modal

IMG_SIZE = 256
DATA_ROOT = Path("/data")
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
SPLIT_RATIOS = {"train": 0.80, "val": 0.05, "test": 0.15}
SFHQ_SAMPLE_TOTAL = 40000
SFHQ_PER_GENERATOR = 8000
RESULTS_ROOT = Path("/results")

KAGGLE_SLUG_A = "shreyanshpatel1/130k-real-vs-fake-face"
KAGGLE_SLUG_B = "selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models"
KAGGLE_SLUG_C = "kaustubhdhote/human-faces-dataset"

DIR_A = DATA_ROOT / "dataset_A"
DIR_B = DATA_ROOT / "dataset_B"
DIR_C = DATA_ROOT / "dataset_C"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# CSV may use various casings; normalise to config.py naming
GENERATOR_NORMALIZE = {
    "flux1.pro": "FLUX1.PRO",
    "flux1.dev": "FLUX1.DEV",
    "flux1.schnell": "FLUX1.SCHNELL",
    "sdxl": "SDXL",
    "dall-e 3": "DALL-E3",
    "dall-e3": "DALL-E3",
    "dalle-3": "DALL-E3",
    "dalle3": "DALL-E3",
}

app = modal.App("avatar-setup-volume")

data_vol = modal.Volume.from_name("avatar-data-vol", create_if_missing=False)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=False)

kaggle_creds = modal.Secret.from_name("kaggle-creds")
hf_secret = modal.Secret.from_name("huggingface-secret")

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


def collect_images(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_directory(root: Path, max_depth: int = 3) -> str:
    lines: list[str] = []
    prefix = str(root)
    for dirpath, dirnames, filenames in os.walk(str(root)):
        depth = dirpath.replace(prefix, "").count(os.sep)
        if depth > max_depth:
            continue
        indent = "  " * depth
        dirname = os.path.basename(dirpath)
        lines.append(f"{indent}{dirname}/  ({len(filenames)} files)")
        for fname in sorted(filenames)[:5]:
            lines.append(f"{indent}  {fname}")
        if len(filenames) > 5:
            lines.append(f"{indent}  ... +{len(filenames) - 5} more")
    return "\n".join(lines)


def resize_image(src: Path, dst: Path) -> bool:
    from PIL import Image

    try:
        img = Image.open(src)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img_resized.save(dst, quality=95)
        return True
    except Exception as e:
        print(f"  WARNING: failed to resize {src.name}: {e}")
        return False


def quick_verify_shape(img_path: Path) -> bool:
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(img_path)
        arr = np.array(img)
        if img.mode == "RGB":
            return arr.shape == (IMG_SIZE, IMG_SIZE, 3)
        elif img.mode == "L":
            return arr.shape == (IMG_SIZE, IMG_SIZE)
        arr_rgb = np.array(img.convert("RGB"))
        return arr_rgb.shape == (IMG_SIZE, IMG_SIZE, 3)
    except Exception as e:
        print(f"  VERIFY FAILED for {img_path}: {e}")
        return False


@app.function(image=image, cpu=2, memory=4096, timeout=2 * 60)
def validate_remote_imports() -> dict:
    import importlib.metadata
    import numpy
    import pandas
    from PIL import Image
    from src.utils.db import init_db, log_dataset_audit

    try:
        kaggle_version = importlib.metadata.version("kaggle")
    except Exception:
        kaggle_version = "unavailable"
    return {
        "kaggle": kaggle_version,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "PIL": Image.__version__,
        "db_init": callable(init_db),
        "log_audit": callable(log_dataset_audit),
    }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=5 * 60,
    volumes={"/data": data_vol},
)
def quick_test() -> dict:
    import numpy as np
    from PIL import Image

    results = {}
    for tag, base_dir in [("A", DIR_A), ("B_sfhq", DIR_B), ("C", DIR_C)]:
        resized_dir = base_dir / "256x256"
        images = collect_images(resized_dir)
        if not images:
            results[tag] = {"status": "NO_IMAGES", "count": 0}
            continue
        sample = random.choice(images)
        img = Image.open(sample)
        arr = np.array(img)
        results[tag] = {
            "path": str(sample.relative_to(DATA_ROOT)),
            "mode": img.mode,
            "shape": arr.shape,
            "correct": arr.shape == (IMG_SIZE, IMG_SIZE, 3)
            or (img.mode == "L" and arr.shape == (IMG_SIZE, IMG_SIZE)),
        }
    return results


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3 * 60 * 60,
    volumes={
        "/data": data_vol,
        "/results": results_vol,
    },
    secrets=[kaggle_creds, hf_secret],
)
def run_setup() -> dict:
    import numpy as np
    import pandas as pd
    from PIL import Image
    from kaggle.api.kaggle_api_extended import KaggleApi
    from src.utils.db import init_db, log_dataset_audit

    random.seed(42)

    manifest_rows: list[dict] = []

    # ==== 1. Download datasets via Kaggle API ====
    print("=" * 70)
    print("Step 1: Download datasets via Kaggle API")
    print("=" * 70)

    api = KaggleApi()
    api.authenticate()

    print(f"\n[1/3] Downloading Dataset A: {KAGGLE_SLUG_A}")
    DIR_A.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_SLUG_A, path=str(DIR_A), unzip=True)
    print("       Dataset A downloaded.")

    print(f"\n[2/3] Downloading Dataset B: {KAGGLE_SLUG_B}")
    DIR_B.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_SLUG_B, path=str(DIR_B), unzip=True)
    print("       Dataset B downloaded.")

    print(f"\n[3/3] Downloading Dataset C: {KAGGLE_SLUG_C}")
    DIR_C.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_SLUG_C, path=str(DIR_C), unzip=True)
    print("       Dataset C downloaded.")

    data_vol.commit()
    print("\n  Data volume committed after downloads.")

    # ==== 2. Inspect directory structures ====
    print("\n" + "=" * 70)
    print("Step 2: Inspect directory structures")
    print("=" * 70)

    for tag, base_dir in [("A", DIR_A), ("B", DIR_B), ("C", DIR_C)]:
        print(f"\n--- Dataset {tag} structure ---")
        print(inspect_directory(base_dir, max_depth=3))

    # ==== 3. Classify images by label (real=0 / fake=1) ====
    print("\n" + "=" * 70)
    print("Step 3: Classify images by label (real=0 / fake=1)")
    print("=" * 70)

    a_real_paths: list[Path] = []
    a_fake_paths: list[Path] = []

    for root, dirs, files in os.walk(str(DIR_A)):
        root_lower = root.lower()
        if "256x256" in root:
            continue
        for f in files:
            fpath = Path(root) / f
            if fpath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if "real" in root_lower:
                a_real_paths.append(fpath)
            elif "fake" in root_lower:
                a_fake_paths.append(fpath)

    print(f"  Dataset A: {len(a_real_paths)} real, {len(a_fake_paths)} fake")

    if not a_real_paths and not a_fake_paths:
        all_a = [p for p in collect_images(DIR_A) if "256x256" not in str(p)]
        print(f"  Dataset A: keyword search found nothing, {len(all_a)} total images")

    b_csv_paths = list(DIR_B.rglob("*.csv"))
    b_meta: pd.DataFrame | None = None
    if b_csv_paths:
        print(f"  Dataset B: found CSV files: {[p.name for p in b_csv_paths]}")
        for csv_path in b_csv_paths:
            try:
                df = pd.read_csv(csv_path, nrows=5)
                if "model" in df.columns or "generator" in df.columns:
                    b_meta = pd.read_csv(csv_path)
                    print(f"    Using {csv_path.name} ({len(b_meta)} rows)")
                    break
            except Exception:
                continue

    b_all_images = [p for p in collect_images(DIR_B) if "256x256" not in str(p)]
    print(f"  Dataset B: {len(b_all_images)} total images")

    c_real_paths: list[Path] = []
    c_fake_paths: list[Path] = []

    for root, dirs, files in os.walk(str(DIR_C)):
        root_lower = root.lower()
        if "256x256" in root:
            continue
        for f in files:
            fpath = Path(root) / f
            if fpath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if "real" in root_lower and "ai-generated" not in root_lower:
                c_real_paths.append(fpath)
            elif "ai-generated" in root_lower or "fake" in root_lower:
                c_fake_paths.append(fpath)

    print(f"  Dataset C: {len(c_real_paths)} real, {len(c_fake_paths)} fake")

    if not c_real_paths and not c_fake_paths:
        all_c = [p for p in collect_images(DIR_C) if "256x256" not in str(p)]
        print(f"  Dataset C: keyword search found nothing, {len(all_c)} total images")

    # ==== 4. Resize all images to 256x256 LANCZOS ====
    print("\n" + "=" * 70)
    print("Step 4: Resize images to 256x256 LANCZOS")
    print("=" * 70)

    def batch_resize(
        src_paths: list[Path], dst_dir: Path, label_prefix: str
    ) -> list[tuple[Path, Path]]:
        resized_pairs = []
        failed = 0
        dst_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(src_paths):
            rel = src.relative_to(src.parents[3]) if len(src.parts) > 4 else src.name
            dst = dst_dir / rel
            if len(dst.parts) > 6:
                dst = dst_dir / f"{label_prefix}_{i:06d}{src.suffix}"

            if resize_image(src, dst):
                resized_pairs.append((src, dst))
            else:
                failed += 1

            if (i + 1) % 5000 == 0:
                print(f"    {label_prefix}: resized {i + 1}/{len(src_paths)}")

        print(
            f"    {label_prefix}: done — {len(resized_pairs)} success, {failed} failed"
        )
        return resized_pairs

    a_real_resized_dir = DIR_A / "256x256" / "real"
    a_real_pairs = batch_resize(a_real_paths, a_real_resized_dir, "A_real")

    a_fake_resized_dir = DIR_A / "256x256" / "fake"
    a_fake_pairs = batch_resize(a_fake_paths, a_fake_resized_dir, "A_fake")

    b_resized_dir = DIR_B / "256x256"
    b_pairs = batch_resize(b_all_images, b_resized_dir, "B")

    c_real_resized_dir = DIR_C / "256x256" / "real"
    c_real_pairs = batch_resize(c_real_paths, c_real_resized_dir, "C_real")

    c_fake_resized_dir = DIR_C / "256x256" / "fake"
    c_fake_pairs = batch_resize(c_fake_paths, c_fake_resized_dir, "C_fake")

    data_vol.commit()
    print("\n  Data volume committed after resize.")

    # ==== 5. Sample SFHQ-T2I (40K total, ~8K per generator) ====
    print("\n" + "=" * 70)
    print("Step 5: Sample 40K from SFHQ-T2I (~8K per generator)")
    print("=" * 70)

    b_sampled_paths: list[Path] = []
    b_generators: list[str] = []

    if b_meta is not None:
        gen_col = None
        for col_name in ["model", "generator", "generator_name", "source"]:
            if col_name in b_meta.columns:
                gen_col = col_name
                break

        if gen_col is not None:
            b_meta["gen_norm"] = b_meta[gen_col].astype(str).str.lower().str.strip()
            b_meta["gen_norm"] = b_meta["gen_norm"].map(
                lambda x: GENERATOR_NORMALIZE.get(x, x.upper())
            )

            generators_found = sorted(b_meta["gen_norm"].unique())
            print(f"  Generators found: {generators_found}")

            path_col = None
            for col_name in ["image_path", "path", "file_name", "filename", "img"]:
                if col_name in b_meta.columns:
                    path_col = col_name
                    break

            if path_col is not None:
                for gen in generators_found:
                    gen_df = b_meta[b_meta["gen_norm"] == gen]
                    n_sample = min(SFHQ_PER_GENERATOR, len(gen_df))
                    sampled_df = gen_df.sample(n=n_sample, random_state=42)

                    for _, row in sampled_df.iterrows():
                        img_name = str(row[path_col])
                        candidates = list(b_resized_dir.rglob(img_name))
                        if not candidates:
                            candidates = list(b_resized_dir.rglob(Path(img_name).name))
                        if candidates:
                            b_sampled_paths.append(candidates[0])
                            b_generators.append(gen)

                print(
                    f"  Sampled {len(b_sampled_paths)} from SFHQ-T2I "
                    f"(target: {SFHQ_SAMPLE_TOTAL})"
                )

                if len(b_sampled_paths) < SFHQ_SAMPLE_TOTAL:
                    remaining_needed = SFHQ_SAMPLE_TOTAL - len(b_sampled_paths)
                    all_b_resized = collect_images(b_resized_dir)
                    already_sampled = set(b_sampled_paths)
                    pool = [p for p in all_b_resized if p not in already_sampled]
                    extra = random.sample(pool, min(remaining_needed, len(pool)))
                    for p in extra:
                        b_sampled_paths.append(p)
                        b_generators.append("unknown")
                    print(f"  Added {len(extra)} random samples to reach target.")
            else:
                print("  No path column in CSV, using random sample.")
                all_b_resized = collect_images(b_resized_dir)
                n_sample = min(SFHQ_SAMPLE_TOTAL, len(all_b_resized))
                b_sampled_paths = random.sample(all_b_resized, n_sample)
                b_generators = ["unknown"] * n_sample
        else:
            print("  No generator column in CSV, using random sample.")
            all_b_resized = collect_images(b_resized_dir)
            n_sample = min(SFHQ_SAMPLE_TOTAL, len(all_b_resized))
            b_sampled_paths = random.sample(all_b_resized, n_sample)
            b_generators = ["unknown"] * n_sample
    else:
        print("  No metadata CSV found, using random sample.")
        all_b_resized = collect_images(b_resized_dir)
        n_sample = min(SFHQ_SAMPLE_TOTAL, len(all_b_resized))
        b_sampled_paths = random.sample(all_b_resized, n_sample)
        b_generators = ["unknown"] * n_sample

    # ==== 6. Build manifest.csv ====
    print("\n" + "=" * 70)
    print("Step 6: Build manifest.csv")
    print("=" * 70)

    print("  Building Dataset A rows...")

    a_all_labeled: list[tuple[Path, int]] = []
    for _, resized in a_real_pairs:
        a_all_labeled.append((resized, 0))
    for _, resized in a_fake_pairs:
        a_all_labeled.append((resized, 1))

    a_real_list = [(p, 0) for p, l in a_all_labeled if l == 0]
    a_fake_list = [(p, 1) for p, l in a_all_labeled if l == 1]

    random.shuffle(a_real_list)
    random.shuffle(a_fake_list)

    def stratified_split(
        items: list[tuple[Path, int]], ratios: dict[str, float]
    ) -> dict[str, list[tuple[Path, int]]]:
        n = len(items)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        return {
            "train": items[:n_train],
            "val": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }

    a_real_splits = stratified_split(a_real_list, SPLIT_RATIOS)
    a_fake_splits = stratified_split(a_fake_list, SPLIT_RATIOS)

    for split_name in ["train", "val", "test"]:
        for path, label in a_real_splits[split_name]:
            manifest_rows.append(
                {
                    "path": str(path.relative_to(DATA_ROOT)),
                    "label": label,
                    "dataset": "A",
                    "split": split_name,
                    "generator": "Flickr" if label == 0 else "unknown",
                }
            )
        for path, label in a_fake_splits[split_name]:
            gen = "unknown"
            path_str = str(path).lower()
            for gen_key in [
                "flux1.dev",
                "flux1.pro",
                "sdxl",
                "flux",
                "dall-e",
                "stable",
            ]:
                if gen_key in path_str:
                    gen = GENERATOR_NORMALIZE.get(gen_key, gen_key.upper())
                    break
            manifest_rows.append(
                {
                    "path": str(path.relative_to(DATA_ROOT)),
                    "label": label,
                    "dataset": "A",
                    "split": split_name,
                    "generator": gen,
                }
            )

    print("  Building Dataset B rows...")
    for img_path, gen in zip(b_sampled_paths, b_generators):
        manifest_rows.append(
            {
                "path": str(img_path.relative_to(DATA_ROOT)),
                "label": 1,
                "dataset": "B_sfhq",
                "split": "train",
                "generator": gen,
            }
        )

    print("  Building Dataset C rows...")
    for _, resized in c_real_pairs:
        manifest_rows.append(
            {
                "path": str(resized.relative_to(DATA_ROOT)),
                "label": 0,
                "dataset": "C",
                "split": "ood_test",
                "generator": "unknown",
            }
        )
    for _, resized in c_fake_pairs:
        manifest_rows.append(
            {
                "path": str(resized.relative_to(DATA_ROOT)),
                "label": 1,
                "dataset": "C",
                "split": "ood_test",
                "generator": "GAN_unknown",
            }
        )

    # ==== 7. Write manifest.csv ====
    print("\n" + "=" * 70)
    print("Step 7: Write manifest.csv")
    print("=" * 70)

    random.shuffle(manifest_rows)

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "dataset", "split", "generator"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"  Wrote {len(manifest_rows)} rows to {MANIFEST_PATH}")

    # ==== 8. Print summary ====
    print("\n" + "=" * 70)
    print("Step 8: Summary")
    print("=" * 70)

    from collections import Counter

    ds_label_counts: dict[tuple[str, str], int] = Counter()
    split_counts: dict[str, int] = Counter()
    for row in manifest_rows:
        ds_label_counts[(row["dataset"], "real" if row["label"] == 0 else "fake")] += 1
        split_counts[row["split"]] += 1

    print(f"\n{'Dataset':<10} {'Label':<8} {'Count':>8}")
    print(f"{'-' * 10} {'-' * 8} {'-' * 8}")
    for (ds, label), count in sorted(ds_label_counts.items()):
        print(f"{ds:<10} {label:<8} {count:>8}")

    print(f"\n{'Split':<12} {'Count':>8}")
    print(f"{'-' * 12} {'-' * 8}")
    for split, count in sorted(split_counts.items()):
        print(f"{split:<12} {count:>8}")

    print(f"\n  Total manifest rows: {len(manifest_rows)}")

    # ==== 9. Quick verification (1 sample per dataset) ====
    print("\n" + "=" * 70)
    print("Step 9: Quick verification (1 sample per dataset)")
    print("=" * 70)

    verify_ok = True
    for tag, base_dir in [("A", DIR_A), ("B_sfhq", DIR_B), ("C", DIR_C)]:
        resized_dir = base_dir / "256x256"
        images = collect_images(resized_dir)
        if not images:
            print(f"  {tag}: NO IMAGES found in {resized_dir}")
            verify_ok = False
            continue
        sample = random.choice(images)
        img = Image.open(sample)
        arr = np.array(img)
        if img.mode == "RGB":
            expected = (IMG_SIZE, IMG_SIZE, 3)
        elif img.mode == "L":
            expected = (IMG_SIZE, IMG_SIZE)
        else:
            arr = np.array(img.convert("RGB"))
            expected = (IMG_SIZE, IMG_SIZE, 3)

        shape_ok = arr.shape == expected
        print(
            f"  {tag}: {sample.name} — mode={img.mode}, "
            f"shape={arr.shape}, expected={expected}, "
            f"{'OK' if shape_ok else 'MISMATCH'}"
        )
        if not shape_ok:
            verify_ok = False

    if verify_ok:
        print("\n  All quick verification checks passed.")
    else:
        print("\n  Some verification checks FAILED — review logs above.")

    # ==== 10. Log dataset audit to SQLite ====
    print("\n" + "=" * 70)
    print("Step 10: Log dataset audit to SQLite")
    print("=" * 70)

    init_db(db_path=str(RESULTS_ROOT / "experiments.db"))

    for ds_tag, ds_name, n_real, n_fake in [
        ("A", "130k_real_vs_fake_face", len(a_real_pairs), len(a_fake_pairs)),
        ("B_sfhq", "sfhq_t2i_synthetic_faces", 0, len(b_sampled_paths)),
        ("C", "human_faces_dataset_9.6k", len(c_real_pairs), len(c_fake_pairs)),
    ]:
        n_total = n_real + n_fake
        log_dataset_audit(
            db_path=str(RESULTS_ROOT / "experiments.db"),
            dataset_name=ds_name,
            n_total=n_total,
            n_real=n_real,
            n_fake=n_fake,
            face_rate=None,
            avg_resolution=f"{IMG_SIZE}x{IMG_SIZE}",
            generators=",".join(sorted(set(b_generators)))
            if ds_tag == "B_sfhq"
            else "unknown",
            notes=f"Setup volume pipeline. Dataset {ds_tag}, resized to {IMG_SIZE}x{IMG_SIZE}.",
        )
        print(
            f"  Logged audit for {ds_name}: {n_total} total ({n_real} real, {n_fake} fake)"
        )

    # ==== 11. Commit volumes ====
    print("\n" + "=" * 70)
    print("Step 11: Commit volumes")
    print("=" * 70)

    data_vol.commit()
    print("  avatar-data-vol committed.")
    results_vol.commit()
    print("  avatar-results-vol committed.")

    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)

    return {
        "manifest_rows": len(manifest_rows),
        "a_real": len(a_real_pairs),
        "a_fake": len(a_fake_pairs),
        "b_sampled": len(b_sampled_paths),
        "c_real": len(c_real_pairs),
        "c_fake": len(c_fake_pairs),
        "verify_ok": verify_ok,
    }


@app.local_entrypoint()
def main() -> None:
    print("Starting avatar dataset setup pipeline...")
    print("  Volumes: avatar-data-vol, avatar-results-vol")
    print("  Secrets: kaggle-creds, huggingface-secret")

    print("\n[Smoke Test] Validating remote imports...")
    smoke = validate_remote_imports.remote()
    print(
        f"  kaggle={smoke['kaggle']}, numpy={smoke['numpy']}, "
        f"pandas={smoke['pandas']}, PIL={smoke['PIL']}"
    )
    print(f"  db_init={smoke['db_init']}, log_audit={smoke['log_audit']}")

    print("\n[Main] Running setup pipeline...")
    result = run_setup.remote()

    if "error" in result:
        print(f"\nSetup FAILED: {result}")
    else:
        print(f"\nSetup complete:")
        print(f"  Manifest rows:   {result['manifest_rows']}")
        print(f"  Dataset A real:  {result['a_real']}")
        print(f"  Dataset A fake:  {result['a_fake']}")
        print(f"  Dataset B sfhq:  {result['b_sampled']}")
        print(f"  Dataset C real:  {result['c_real']} (ood_test)")
        print(f"  Dataset C fake:  {result['c_fake']} (ood_test)")
        print(f"  Quick verify:    {'PASS' if result['verify_ok'] else 'FAIL'}")

    print("\n[Quick Test] Verifying resized image shapes on volume...")
    qt = quick_test.remote()
    for tag, info in qt.items():
        if isinstance(info, dict) and "shape" in info:
            print(
                f"  {tag}: shape={info['shape']}, "
                f"mode={info['mode']}, "
                f"correct={info['correct']}"
            )
        else:
            print(f"  {tag}: {info}")
