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

app = modal.App("avatar-setup-staged")

data_vol = modal.Volume.from_name("avatar-data-vol", create_if_missing=True)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=True)

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

    print(f"    {label_prefix}: done — {len(resized_pairs)} success, {failed} failed")
    return resized_pairs


def classify_paths(base_dir: Path) -> tuple[list[Path], list[Path]]:
    real_paths: list[Path] = []
    fake_paths: list[Path] = []

    for root, dirs, files in os.walk(str(base_dir)):
        root_lower = root.lower()
        if "256x256" in root:
            continue
        for f in files:
            fpath = Path(root) / f
            if fpath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if (
                "real" in root_lower
                and "ai-generated" not in root_lower
                and "fake" not in root_lower
            ):
                real_paths.append(fpath)
            elif "fake" in root_lower or "ai-generated" in root_lower:
                fake_paths.append(fpath)

    return real_paths, fake_paths


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


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=90 * 60,
    volumes={"/data": data_vol},
    secrets=[kaggle_creds, hf_secret],
)
def download_and_resize_A_and_C() -> dict:
    from kaggle.api.kaggle_api_extended import KaggleApi

    random.seed(42)

    print("=" * 70)
    print("STAGE 1: Download and resize Datasets A + C")
    print("=" * 70)

    api = KaggleApi()
    api.authenticate()

    print(f"\n[1/2] Downloading Dataset A: {KAGGLE_SLUG_A}")
    DIR_A.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_SLUG_A, path=str(DIR_A), unzip=True)
    print("       Dataset A downloaded.")

    print(f"\n[2/2] Downloading Dataset C: {KAGGLE_SLUG_C}")
    DIR_C.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_SLUG_C, path=str(DIR_C), unzip=True)
    print("       Dataset C downloaded.")

    print("\n--- Dataset A structure ---")
    print(inspect_directory(DIR_A, max_depth=3))
    print("\n--- Dataset C structure ---")
    print(inspect_directory(DIR_C, max_depth=3))

    a_real_paths, a_fake_paths = classify_paths(DIR_A)
    print(f"\n  Dataset A: {len(a_real_paths)} real, {len(a_fake_paths)} fake")

    if not a_real_paths and not a_fake_paths:
        all_a = [p for p in collect_images(DIR_A) if "256x256" not in str(p)]
        print(f"  Dataset A: keyword search found nothing, {len(all_a)} total images")

    c_real_paths, c_fake_paths = classify_paths(DIR_C)
    print(f"  Dataset C: {len(c_real_paths)} real, {len(c_fake_paths)} fake")

    if not c_real_paths and not c_fake_paths:
        all_c = [p for p in collect_images(DIR_C) if "256x256" not in str(p)]
        print(f"  Dataset C: keyword search found nothing, {len(all_c)} total images")

    print("\nResizing Dataset A...")
    a_real_pairs = batch_resize(a_real_paths, DIR_A / "256x256" / "real", "A_real")
    a_fake_pairs = batch_resize(a_fake_paths, DIR_A / "256x256" / "fake", "A_fake")

    print("\nResizing Dataset C...")
    c_real_pairs = batch_resize(c_real_paths, DIR_C / "256x256" / "real", "C_real")
    c_fake_pairs = batch_resize(c_fake_paths, DIR_C / "256x256" / "fake", "C_fake")

    data_vol.commit()
    print("\n  Data volume committed after Stage 1.")

    print(f"\n  Dataset A: {len(a_real_pairs)} real, {len(a_fake_pairs)} fake resized")
    print(f"  Dataset C: {len(c_real_pairs)} real, {len(c_fake_pairs)} fake resized")

    return {
        "a_real": len(a_real_pairs),
        "a_fake": len(a_fake_pairs),
        "c_real": len(c_real_pairs),
        "c_fake": len(c_fake_pairs),
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=90 * 60,
    volumes={"/data": data_vol},
    secrets=[kaggle_creds, hf_secret],
)
def download_and_resize_B_sfhq() -> dict:
    import pandas as pd
    from kaggle.api.kaggle_api_extended import KaggleApi

    random.seed(42)

    print("=" * 70)
    print("STAGE 2: Download and resize Dataset B (SFHQ-T2I)")
    print("=" * 70)

    api = KaggleApi()
    api.authenticate()

    print(f"\n[1/1] Downloading Dataset B: {KAGGLE_SLUG_B}")
    DIR_B.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_SLUG_B, path=str(DIR_B), unzip=True)
    print("       Dataset B downloaded.")

    print("\n--- Dataset B structure ---")
    print(inspect_directory(DIR_B, max_depth=3))

    b_all_images = [p for p in collect_images(DIR_B) if "256x256" not in str(p)]
    print(f"  Dataset B: {len(b_all_images)} total images")

    print("\nResizing Dataset B...")
    b_pairs = batch_resize(b_all_images, DIR_B / "256x256", "B")

    b_meta: pd.DataFrame | None = None
    b_csv_paths = list(DIR_B.rglob("*.csv"))
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

    b_sampled_paths: list[Path] = []
    b_generators: list[str] = []

    b_resized_dir = DIR_B / "256x256"

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

    data_vol.commit()
    print("\n  Data volume committed after Stage 2.")

    print(f"\n  Dataset B: {len(b_pairs)} resized, {len(b_sampled_paths)} sampled")

    return {
        "b_resized": len(b_pairs),
        "b_sampled": len(b_sampled_paths),
        "b_generators": len(set(b_generators)),
    }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={
        "/data": data_vol,
        "/results": results_vol,
    },
)
def build_manifest() -> dict:
    from collections import Counter

    random.seed(42)

    print("=" * 70)
    print("STAGE 3: Build manifest.csv")
    print("=" * 70)

    manifest_rows: list[dict] = []

    a_real_dir = DIR_A / "256x256" / "real"
    a_fake_dir = DIR_A / "256x256" / "fake"

    a_real_images = collect_images(a_real_dir) if a_real_dir.exists() else []
    a_fake_images = collect_images(a_fake_dir) if a_fake_dir.exists() else []
    print(f"  Dataset A: {len(a_real_images)} real, {len(a_fake_images)} fake")

    a_real_list = [(p, 0) for p in a_real_images]
    a_fake_list = [(p, 1) for p in a_fake_images]

    random.shuffle(a_real_list)
    random.shuffle(a_fake_list)

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
                    "generator": "Flickr",
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

    b_resized_dir = DIR_B / "256x256"
    b_sampled_paths: list[Path] = []
    b_generators: list[str] = []

    import pandas as pd

    b_csv_paths = list(DIR_B.rglob("*.csv"))
    b_meta: pd.DataFrame | None = None
    if b_csv_paths:
        for csv_path in b_csv_paths:
            try:
                df = pd.read_csv(csv_path, nrows=5)
                if "model" in df.columns or "generator" in df.columns:
                    b_meta = pd.read_csv(csv_path)
                    break
            except Exception:
                continue

    all_b_resized = collect_images(b_resized_dir) if b_resized_dir.exists() else []

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

            path_col = None
            for col_name in ["image_path", "path", "file_name", "filename", "img"]:
                if col_name in b_meta.columns:
                    path_col = col_name
                    break

            if path_col is not None:
                for gen in sorted(b_meta["gen_norm"].unique()):
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

                if len(b_sampled_paths) < SFHQ_SAMPLE_TOTAL:
                    remaining_needed = SFHQ_SAMPLE_TOTAL - len(b_sampled_paths)
                    already_sampled = set(b_sampled_paths)
                    pool = [p for p in all_b_resized if p not in already_sampled]
                    extra = random.sample(pool, min(remaining_needed, len(pool)))
                    for p in extra:
                        b_sampled_paths.append(p)
                        b_generators.append("unknown")
            else:
                n_sample = min(SFHQ_SAMPLE_TOTAL, len(all_b_resized))
                b_sampled_paths = random.sample(all_b_resized, n_sample)
                b_generators = ["unknown"] * n_sample
        else:
            n_sample = min(SFHQ_SAMPLE_TOTAL, len(all_b_resized))
            b_sampled_paths = random.sample(all_b_resized, n_sample)
            b_generators = ["unknown"] * n_sample
    else:
        if all_b_resized:
            n_sample = min(SFHQ_SAMPLE_TOTAL, len(all_b_resized))
            b_sampled_paths = random.sample(all_b_resized, n_sample)
            b_generators = ["unknown"] * n_sample

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

    c_real_dir = DIR_C / "256x256" / "real"
    c_fake_dir = DIR_C / "256x256" / "fake"

    c_real_images = collect_images(c_real_dir) if c_real_dir.exists() else []
    c_fake_images = collect_images(c_fake_dir) if c_fake_dir.exists() else []
    print(f"  Dataset C: {len(c_real_images)} real, {len(c_fake_images)} fake")

    for path in c_real_images:
        manifest_rows.append(
            {
                "path": str(path.relative_to(DATA_ROOT)),
                "label": 0,
                "dataset": "C",
                "split": "ood_test",
                "generator": "unknown",
            }
        )
    for path in c_fake_images:
        manifest_rows.append(
            {
                "path": str(path.relative_to(DATA_ROOT)),
                "label": 1,
                "dataset": "C",
                "split": "ood_test",
                "generator": "GAN_unknown",
            }
        )

    random.shuffle(manifest_rows)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "dataset", "split", "generator"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n  Wrote {len(manifest_rows)} rows to {MANIFEST_PATH}")

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

    data_vol.commit()
    print("  avatar-data-vol committed after Stage 3.")

    return {"manifest_rows": len(manifest_rows)}


@app.local_entrypoint()
def main(stage: str = "all"):
    if stage in ("1", "all"):
        print("Running Stage 1: Download and resize Datasets A + C...")
        result_1 = download_and_resize_A_and_C.remote()
        print(f"Stage 1 complete: {result_1}")

    if stage in ("2", "all"):
        print("Running Stage 2: Download and resize Dataset B (SFHQ-T2I)...")
        result_2 = download_and_resize_B_sfhq.remote()
        print(f"Stage 2 complete: {result_2}")

    if stage in ("3", "all"):
        print("Running Stage 3: Build manifest.csv...")
        result_3 = build_manifest.remote()
        print(f"Stage 3 complete: {result_3}")
