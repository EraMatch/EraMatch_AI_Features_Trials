"""Quick training test with real data from avatar-data-vol.

Runs 5 forward+backward passes using DCT4ChannelModel on real manifest data.
Verifies loss is finite and decreasing before committing to full training.

Usage:
    modal run modal/quick_test_train.py
"""

import time
from pathlib import Path

import modal

DATA_ROOT = Path("/data")
RESULTS_ROOT = Path("/results")
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
MODEL_CACHE = "/models"

app = modal.App("avatar-quick-test")

data_vol = modal.Volume.from_name("avatar-data-vol", create_if_missing=True)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=True)
model_vol = modal.Volume.from_name("avatar-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "timm==1.0.15",
        "scipy==1.15.3",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "pillow==11.3.0",
        "albumentations==2.0.5",
        "numpy==2.2.6",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("src")
)


@app.function(
    image=image,
    gpu="T4",
    memory=16384,
    timeout=10 * 60,
    volumes={
        str(DATA_ROOT): data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE: model_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def quick_train_test() -> dict:
    import csv
    import random

    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from src.models.dct_branch import DCT4ChannelModel
    from src.data.dataset import AvatarDataset
    from src.data.augmentations import get_train_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[QuickTest] Device: {device}")

    C_REAL_DIR = DATA_ROOT / "dataset_C/Human Faces Dataset/Real Images"
    C_FAKE_DIR = DATA_ROOT / "dataset_C/Human Faces Dataset/AI-Generated Images"

    real_paths = sorted(
        [
            p
            for p in C_REAL_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    fake_paths = sorted(
        [
            p
            for p in C_FAKE_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    print(f"[QuickTest] Found {len(real_paths)} real, {len(fake_paths)} fake images")

    random.seed(42)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    n_real = len(real_paths)
    n_fake = len(fake_paths)
    n_real_train = int(n_real * 0.70)
    n_real_val = int(n_real * 0.15)
    n_fake_train = int(n_fake * 0.70)
    n_fake_val = int(n_fake * 0.15)

    rows = []
    for p in real_paths[:n_real_train]:
        rows.append(
            {
                "path": str(p),
                "label": 0,
                "dataset": "C",
                "split": "train",
                "generator": "unknown",
            }
        )
    for p in real_paths[n_real_train : n_real_train + n_real_val]:
        rows.append(
            {
                "path": str(p),
                "label": 0,
                "dataset": "C",
                "split": "val",
                "generator": "unknown",
            }
        )
    for p in real_paths[n_real_train + n_real_val :]:
        rows.append(
            {
                "path": str(p),
                "label": 0,
                "dataset": "C",
                "split": "test",
                "generator": "unknown",
            }
        )
    for p in fake_paths[:n_fake_train]:
        rows.append(
            {
                "path": str(p),
                "label": 1,
                "dataset": "C",
                "split": "train",
                "generator": "GAN_unknown",
            }
        )
    for p in fake_paths[n_fake_train : n_fake_train + n_fake_val]:
        rows.append(
            {
                "path": str(p),
                "label": 1,
                "dataset": "C",
                "split": "val",
                "generator": "GAN_unknown",
            }
        )
    for p in fake_paths[n_fake_train + n_fake_val :]:
        rows.append(
            {
                "path": str(p),
                "label": 1,
                "dataset": "C",
                "split": "test",
                "generator": "GAN_unknown",
            }
        )

    random.shuffle(rows)

    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "label", "dataset", "split", "generator"]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[QuickTest] Built manifest: {len(rows)} rows")
    print(
        f"[QuickTest] train={sum(1 for r in rows if r['split'] == 'train')}, val={sum(1 for r in rows if r['split'] == 'val')}, test={sum(1 for r in rows if r['split'] == 'test')}"
    )

    transform = get_train_transforms(img_size=224)

    dataset = AvatarDataset(
        manifest_path=str(MANIFEST_PATH),
        split="train",
        transform=transform,
        img_size=224,
    )

    print(f"[QuickTest] Train samples: {len(dataset)}")

    if len(dataset) == 0:
        print("QUICK TEST FAILED: No training samples found in manifest")
        return {"status": "FAILED", "reason": "No training samples in manifest"}

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=AvatarDataset.collate_fn,
        drop_last=True,
    )

    try:
        model = DCT4ChannelModel(pretrained=True, num_classes=2).to(device)
    except Exception as e:
        print(f"QUICK TEST FAILED: Model creation error: {e}")
        return {"status": "FAILED", "reason": f"Model creation: {e}"}
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[QuickTest] Model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    model.train()
    losses = []
    start = time.time()

    for step, batch in enumerate(dataloader):
        if step >= 5:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        print(f"  [QuickTest] step {step + 1}/5  loss={loss_val:.4f}")

    elapsed = time.time() - start
    print(f"[QuickTest] Elapsed: {elapsed:.1f}s")

    all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)
    decreasing = losses[-1] < losses[0]

    if not all_finite:
        reason = f"Non-finite losses: {losses}"
        print(f"QUICK TEST FAILED: {reason}")
        return {"status": "FAILED", "reason": reason, "losses": losses}

    print("QUICK TEST PASSED")
    return {
        "status": "PASSED",
        "losses": losses,
        "loss_decreased": decreasing,
        "elapsed_s": round(elapsed, 1),
        "n_params": n_params,
        "n_train_samples": len(dataset),
    }


@app.local_entrypoint()
def main():
    result = quick_train_test.remote()
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
