"""
Build manifest.csv for the 3 Kaggle datasets.

Datasets:
  DS1 — kaustubhdhote/human-faces-dataset          (9.6K GAN, real+fake)
  DS2 — muhammadbilal6305/200k-real-vs-ai-visuals  (71.8K diffusion fakes ONLY)
  DS3 — shreyanshpatel1/130k-real-vs-fake-face      (133K diffusion, real+fake)

Split strategy (cross-dataset OOD evaluation):
  - DS3 → ood_test entirely (held out, never seen during training)
  - DS1+DS2 → train/val/in_dist_test (70/15/15), subsampled to 1/3

Usage:
    modal run modal/setup_datasets.py
"""

from pathlib import Path
import modal

DATA_ROOT    = Path("/data")
RESULTS_ROOT = Path("/results")
SEED         = 42
SAMPLE_FRAC  = 1 / 3
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp"}

data_vol    = modal.Volume.from_name("avatar-data-vol",    create_if_missing=False)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kaggle==1.7.4", "pandas==2.2.3", "scikit-learn==1.6.1")
)

app = modal.App("avatar-setup-datasets")


@app.function(
    image=image,
    cpu=4,
    memory=32768,
    timeout=90 * 60,
    volumes={
        str(DATA_ROOT):    data_vol,
        str(RESULTS_ROOT): results_vol,
    },
    secrets=[modal.Secret.from_name("kaggle-creds")],
)
def setup() -> dict:
    import os, random
    import pandas as pd
    from sklearn.model_selection import train_test_split

    random.seed(SEED)
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # ── 1. Download any missing datasets ─────────────────────────────────────
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    datasets = [
        ("kaustubhdhote/human-faces-dataset",                    DATA_ROOT / "ds1"),
        ("muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal",  DATA_ROOT / "ds2"),
        ("shreyanshpatel1/130k-real-vs-fake-face",               DATA_ROOT / "ds3"),
    ]
    for slug, dst in datasets:
        if dst.exists() and any(dst.rglob("*")):
            print(f"  [SKIP] {slug} — already on volume")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        print(f"  [DL] {slug}…", flush=True)
        api.dataset_download_files(slug, path=str(dst), unzip=True, quiet=False)
        print(f"  [DL] {slug} done", flush=True)

    data_vol.commit()

    # ── 2. Collect images with labels ─────────────────────────────────────────
    def collect(folder: Path, label: int, source: str) -> list[dict]:
        rows = []
        for f in folder.rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                rows.append({
                    "path":   str(f.relative_to(DATA_ROOT)),
                    "label":  label,
                    "source": source,
                })
        return rows

    rows = []

    # DS1: Real Images / AI-Generated Images subdirs
    ds1 = DATA_ROOT / "ds1"
    for sub in sorted(ds1.rglob("*")):
        if not sub.is_dir(): continue
        n = sub.name.lower()
        if "real" in n and "ai" not in n and "generated" not in n:
            before = len(rows); rows.extend(collect(sub, 0, "ds1"))
            print(f"  [DS1] real:  {sub.name}  +{len(rows)-before}")
        elif "ai" in n or "generated" in n or "fake" in n:
            before = len(rows); rows.extend(collect(sub, 1, "ds1"))
            print(f"  [DS1] fake:  {sub.name}  +{len(rows)-before}")

    # DS2: fakes only — path confirmed: ds2/my_real_vs_ai_dataset/my_real_vs_ai_dataset/ai_images/
    ds2_fake = DATA_ROOT / "ds2" / "my_real_vs_ai_dataset" / "my_real_vs_ai_dataset" / "ai_images"
    if ds2_fake.exists():
        before = len(rows); rows.extend(collect(ds2_fake, 1, "ds2"))
        print(f"  [DS2] fake (ai_images):  +{len(rows)-before}")
    else:
        # fallback: scan for any dir named ai_images
        for sub in (DATA_ROOT / "ds2").rglob("*"):
            if sub.is_dir() and sub.name.lower() == "ai_images":
                before = len(rows); rows.extend(collect(sub, 1, "ds2"))
                print(f"  [DS2] fake fallback: {sub}  +{len(rows)-before}")
                break

    # DS3: images/real and images/fake
    ds3 = DATA_ROOT / "ds3"
    images_dir = ds3 / "images" if (ds3 / "images").exists() else ds3
    for lbl, dname in [(0, "real"), (1, "fake")]:
        d = images_dir / dname
        if d.exists():
            before = len(rows); rows.extend(collect(d, lbl, "ds3"))
            print(f"  [DS3] {dname}:  +{len(rows)-before}")

    df = pd.DataFrame(rows)
    by_src = df.groupby("source")["label"].value_counts().to_dict()
    print(f"\nTotal images: {len(df):,}")
    for src, grp in df.groupby("source"):
        n0 = (grp.label==0).sum(); n1 = (grp.label==1).sum()
        print(f"  {src}: {len(grp):,}  real={n0:,}  fake={n1:,}")

    # ── 3. Split strategy ──────────────────────────────────────────────────────
    # DS3 → all tagged ood_test (never touches training)
    df_ood  = df[df["source"] == "ds3"].copy()
    df_ood["split"] = "ood_test"
    print(f"\nOOD test (DS3): {len(df_ood):,}  real={(df_ood.label==0).sum():,}  fake={(df_ood.label==1).sum():,}")

    # DS1+DS2 → 1/3 subsample then 70/15/15
    df_pool = df[df["source"] != "ds3"].reset_index(drop=True)
    df_pool, _ = train_test_split(
        df_pool, train_size=SAMPLE_FRAC,
        stratify=df_pool["label"], random_state=SEED)

    df_train, df_tmp  = train_test_split(df_pool, test_size=0.30, stratify=df_pool["label"], random_state=SEED)
    df_val,   df_test = train_test_split(df_tmp,  test_size=0.50, stratify=df_tmp["label"],  random_state=SEED)

    df_train = df_train.copy(); df_train["split"] = "train"
    df_val   = df_val.copy();   df_val["split"]   = "val"
    df_test  = df_test.copy();  df_test["split"]  = "in_dist_test"

    n0_tr = (df_train.label==0).sum(); n1_tr = (df_train.label==1).sum()
    print(f"\nTrain (DS1+DS2, 1/3):  {len(df_train):,}  real={n0_tr:,}  fake={n1_tr:,}  imbalance={n1_tr/n0_tr:.1f}×")
    print(f"Val:                   {len(df_val):,}")
    print(f"In-dist test:          {len(df_test):,}")

    manifest = pd.concat([df_train, df_val, df_test, df_ood], ignore_index=True)
    manifest.to_csv(DATA_ROOT / "manifest.csv", index=False)
    print(f"\nmanifest.csv written: {len(manifest):,} rows total")

    data_vol.commit()
    results_vol.commit()
    print("Done.", flush=True)

    return {
        "train":       len(df_train),
        "val":         len(df_val),
        "in_dist_test": len(df_test),
        "ood_test":    len(df_ood),
        "total":       len(manifest),
        "real_train":  int(n0_tr),
        "fake_train":  int(n1_tr),
    }


@app.local_entrypoint()
def main():
    print("Rebuilding manifest…")
    r = setup.remote()
    print("\nManifest stats:")
    for k, v in r.items():
        print(f"  {k}: {v:,}")
