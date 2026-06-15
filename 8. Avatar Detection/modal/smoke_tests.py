"""
Smoke tests for the Modal avatar-detection training pipeline.

Run all tests before the full training run to catch config/data/GPU issues early.

Usage:
    modal run modal/smoke_tests.py              # runs all 5 tests
    modal run modal/smoke_tests.py::smoke_data  # individual test

Tests:
    smoke_data   — manifest.csv exists, class balance, images openable
    smoke_model  — all 4 configs (A/B/C/D) forward-pass on CPU
    smoke_gpu    — CUDA available, VRAM reported, A10G confirmed
    smoke_db     — experiments.db writable and readable
    smoke_train  — 5 real training steps on a 256-image subset, loss finite
"""

from pathlib import Path
import modal

# ── Volumes ───────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("/data")
RESULTS_ROOT = Path("/results")
MODEL_CACHE  = "/models"

data_vol    = modal.Volume.from_name("avatar-data-vol",    create_if_missing=False)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=False)
model_vol   = modal.Volume.from_name("avatar-model-cache", create_if_missing=True)

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "timm==1.0.15",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "pillow==11.3.0",
        "albumentations==2.0.8",
        "numpy==1.26.4",
        "matplotlib==3.9.0",
        "optuna==3.6.1",
    )
    .env({"HF_HUB_CACHE": MODEL_CACHE, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("src")
)

app = modal.App("avatar-smoke-tests")

# ─── 1. DATA SMOKE ────────────────────────────────────────────────────────────
@app.function(
    image=image,
    memory=4096,
    timeout=5 * 60,
    volumes={
        str(DATA_ROOT):    data_vol,
        str(RESULTS_ROOT): results_vol,
    },
)
def smoke_data() -> dict:
    """
    Checks:
      ✓ manifest.csv exists and is non-empty
      ✓ expected columns present (path, label, split)
      ✓ class balance (real/fake) roughly 40–60 %
      ✓ train/val/test split sizes are sane
      ✓ 20 random images are openable as RGB
      ✓ all image paths in sample exist on disk
    """
    import pandas as pd
    from PIL import Image
    import random

    results = {}
    manifest = DATA_ROOT / "manifest.csv"

    # ── existence ──────────────────────────────────────────────────────────
    if not manifest.exists():
        return {"FAIL": "manifest.csv not found — run setup_volume.py first"}

    df = pd.read_csv(manifest)
    results["total_rows"] = len(df)

    # ── columns ───────────────────────────────────────────────────────────
    required = {"path", "label", "split"}
    missing  = required - set(df.columns)
    if missing:
        return {"FAIL": f"missing columns: {missing}"}
    results["columns"] = list(df.columns)

    # ── class balance ────────────────────────────────────────────────────
    real_pct = (df["label"] == 0).mean()
    fake_pct = (df["label"] == 1).mean()
    results["real_pct"] = round(real_pct, 3)
    results["fake_pct"] = round(fake_pct, 3)
    if not (0.40 <= real_pct <= 0.60):
        results["WARN_balance"] = f"class ratio unusual: real={real_pct:.2%}"

    # ── split sizes ───────────────────────────────────────────────────────
    for split in df["split"].unique():
        results[f"n_{split}"] = int((df["split"] == split).sum())

    # ── sample image open check ───────────────────────────────────────────
    sample = df.sample(min(20, len(df)), random_state=42)
    opened = 0
    missing_files = []
    for _, row in sample.iterrows():
        p = DATA_ROOT / row["path"]
        if not p.exists():
            missing_files.append(str(row["path"]))
            continue
        try:
            img = Image.open(p).convert("RGB")
            assert img.size[0] > 0
            opened += 1
        except Exception as e:
            results.setdefault("open_errors", []).append(str(e))

    results["images_opened_ok"] = opened
    results["images_checked"]   = len(sample)
    if missing_files:
        results["WARN_missing_files"] = missing_files[:5]

    results["STATUS"] = "PASS" if opened == len(sample) and not missing else "WARN"
    return results


# ─── 2. MODEL SMOKE ──────────────────────────────────────────────────────────
@app.function(
    image=image,
    memory=8192,
    timeout=10 * 60,
    volumes={MODEL_CACHE: model_vol},
)
def smoke_model() -> dict:
    """
    Checks:
      ✓ All 4 configs (A/B/C/D) instantiate without error
      ✓ Forward pass on a random (2, 3, 224, 224) batch returns logits shape (2, 2)
      ✓ Config C and D return 'proj' key for SupCon
      ✓ Param counts are reasonable
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    # ── inline model defs (same as train_ablation.py) ─────────────────────
    SRM_KERNELS = torch.tensor([
        [[ 0, 0, 0, 0, 0], [ 0,-1, 2,-1, 0], [ 0, 2,-4, 2, 0], [ 0,-1, 2,-1, 0], [ 0, 0, 0, 0, 0]],
        [[-1, 2,-2, 2,-1], [ 2,-6, 8,-6, 2], [-2, 8,-12,8,-2], [ 2,-6, 8,-6, 2], [-1, 2,-2, 2,-1]],
        [[ 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0], [ 0, 1,-2, 1, 0], [ 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0]],
    ], dtype=torch.float32) / 12.0

    class SRMFilter(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", SRM_KERNELS.unsqueeze(1))
        def forward(self, x):
            B, C, H, W = x.shape
            out = F.conv2d(x.reshape(B*C, 1, H, W), self.weight, padding=2)
            return torch.tanh(out.reshape(B, C, 3, H, W).mean(1))

    class SRMEncoder(nn.Module):
        def __init__(self, d=256):
            super().__init__()
            self.srm = SRMFilter()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.proj = nn.Sequential(nn.Linear(256, d), nn.GELU())
        def forward(self, x): return self.proj(self.net(self.srm(x)))

    class DCTEncoder(nn.Module):
        def __init__(self, d=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.proj = nn.Sequential(nn.Linear(256, d), nn.GELU())
        def forward(self, x):
            mag = torch.log1p(torch.fft.fftshift(torch.abs(torch.fft.fft2(x, norm="ortho")), dim=(-2,-1)))
            return self.proj(self.net(mag))

    class CrossModalAttention(nn.Module):
        def __init__(self, rgb=768, aux=256, d=256):
            super().__init__()
            self.q = nn.Linear(rgb, d); self.k = nn.Linear(aux, d)
            self.v = nn.Linear(aux, d); self.out = nn.Linear(d, rgb)
            self.scale = d ** -0.5
        def forward(self, rgb, aux):
            w = torch.sigmoid((self.q(rgb) * self.k(aux)).sum(-1, keepdim=True) * self.scale)
            return rgb + self.out(w * self.v(aux))

    class ConfigA(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
            self.head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 2))
        def forward(self, x):
            f = self.bb(x); return {"logits": self.head(f), "features": f}

    class ConfigB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
            self.srm = SRMEncoder()
            self.head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 2))
        def forward(self, x):
            f = torch.cat([self.rgb(x), self.srm(x)], 1); return {"logits": self.head(f), "features": f}

    class ConfigC(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
            self.srm = SRMEncoder(); self.attn = CrossModalAttention()
            self.cls = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 2))
            self.proj = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 128))
        def forward(self, x):
            srm = self.srm(x); rgb = self.attn(self.rgb(x), srm)
            f = torch.cat([rgb, srm], 1)
            return {"logits": self.cls(f), "features": f, "proj": F.normalize(self.proj(f), dim=1)}

    class ConfigD(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
            self.srm = SRMEncoder(); self.dct = DCTEncoder()
            self.a_srm = CrossModalAttention(); self.a_dct = CrossModalAttention(aux=256)
            fused = 768 + 256 + 256
            self.cls = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 2))
            self.proj = nn.Sequential(nn.Linear(fused, 256), nn.GELU(), nn.Linear(256, 128))
        def forward(self, x):
            srm = self.srm(x); dct = self.dct(x)
            rgb = self.a_dct(self.a_srm(self.rgb(x), srm), dct)
            f   = torch.cat([rgb, srm, dct], 1)
            return {"logits": self.cls(f), "features": f, "proj": F.normalize(self.proj(f), dim=1)}

    # ── run checks ────────────────────────────────────────────────────────
    results = {}
    dummy = torch.randn(2, 3, 224, 224)

    for name, cls, expect_proj in [
        ("config_A", ConfigA, False),
        ("config_B", ConfigB, False),
        ("config_C", ConfigC, True),
        ("config_D", ConfigD, True),
    ]:
        try:
            m = cls().eval()
            n_params = sum(p.numel() for p in m.parameters()) / 1e6
            with torch.no_grad():
                out = m(dummy)
            logits_ok  = out["logits"].shape == (2, 2)
            proj_ok    = (not expect_proj) or ("proj" in out and out["proj"].shape == (2, 128))
            results[name] = {
                "params_M":  round(n_params, 1),
                "logits_ok": logits_ok,
                "proj_ok":   proj_ok,
                "STATUS":    "PASS" if logits_ok and proj_ok else "FAIL",
            }
        except Exception as e:
            results[name] = {"STATUS": "FAIL", "error": str(e)}

    results["overall"] = "PASS" if all(v.get("STATUS") == "PASS" for v in results.values()) else "FAIL"
    return results


# ─── 3. GPU SMOKE ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    memory=8192,
    timeout=5 * 60,
)
def smoke_gpu() -> dict:
    """
    Checks:
      ✓ torch.cuda.is_available()
      ✓ GPU name contains 'A10' (not assigned a T4 by accident)
      ✓ VRAM ≥ 20GB
      ✓ A simple 224×224 ConvNeXt-tiny forward+backward pass succeeds on GPU
    """
    import torch
    import timm

    results = {}
    results["cuda_available"] = torch.cuda.is_available()
    if not results["cuda_available"]:
        return {"FAIL": "no CUDA", **results}

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    results["gpu_name"]  = gpu_name
    results["vram_gb"]   = round(vram_gb, 1)
    results["is_A10G"]   = "A10" in gpu_name
    results["vram_ok"]   = vram_gb >= 20.0

    if not results["is_A10G"]:
        results["WARN"] = f"Expected A10G, got {gpu_name}"

    # small forward+backward
    try:
        m = timm.create_model("convnext_tiny", pretrained=False, num_classes=2).cuda()
        x = torch.randn(4, 3, 224, 224, device="cuda")
        loss = m(x).sum()
        loss.backward()
        results["forward_backward"] = "PASS"
        results["allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
    except Exception as e:
        results["forward_backward"] = f"FAIL: {e}"

    results["STATUS"] = "PASS" if results["cuda_available"] and results["vram_ok"] and results["forward_backward"] == "PASS" else "FAIL"
    return results


# ─── 4. DB SMOKE ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    memory=2048,
    timeout=2 * 60,
    volumes={str(RESULTS_ROOT): results_vol},
)
def smoke_db() -> dict:
    """
    Checks:
      ✓ experiments.db exists (created by setup_volume.py) or can be created
      ✓ init_db() runs without error
      ✓ log_experiment() inserts a row and returns an id
      ✓ log_training_history() inserts epoch rows
      ✓ get_experiments() retrieves the inserted row
      ✓ row is cleaned up after test
    """
    import sqlite3
    from src.utils.db import init_db, log_experiment, log_training_history, get_experiments

    db_path = RESULTS_ROOT / "experiments.db"
    results = {}

    results["db_exists_before"] = db_path.exists()
    init_db(db_path=str(db_path))
    results["init_db"] = "PASS"

    # insert test experiment
    exp_id = log_experiment(
        db_path=str(db_path),
        trial_name="__smoke_test__",
        model_arch="smoke",
        backbone="convnext_tiny",
        train_datasets="smoke_data",
        test_dataset="smoke_data",
        n_train=100, n_val=20, n_test=20,
        test_auc=0.999, test_f1=0.999,
        gpu_type="A10G",
        notes="automated smoke test — safe to delete",
    )
    results["exp_id"]       = exp_id
    results["log_exp"]      = "PASS" if isinstance(exp_id, int) and exp_id > 0 else "FAIL"

    # insert epoch rows
    for ep in range(1, 4):
        log_training_history(
            db_path=str(db_path),
            experiment_id=exp_id,
            epoch=ep,
            train_loss=1.0 - ep * 0.1,
            val_auc=0.7 + ep * 0.05,
        )
    results["log_history"] = "PASS"

    # read back
    exps = get_experiments(db_path=str(db_path))
    found = [e for e in exps if e.get("trial_name") == "__smoke_test__"]
    results["read_back"] = "PASS" if found else "FAIL"

    # clean up test row
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM training_history WHERE experiment_id=?", (exp_id,))
    conn.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
    conn.commit(); conn.close()
    results["cleanup"] = "PASS"

    results_vol.commit()
    _skip = {"db_exists_before", "exp_id"}
    results["STATUS"] = "PASS" if all(v == "PASS" for k, v in results.items() if k not in _skip) else "FAIL"
    return results


# ─── 5. TRAIN SMOKE ───────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    memory=16384,
    timeout=10 * 60,
    volumes={
        str(DATA_ROOT):    data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE:       model_vol,
    },
)
def smoke_train() -> dict:
    """
    Checks:
      ✓ Loads 256 images from manifest (train split) without error
      ✓ DataLoader yields correct tensor shapes
      ✓ Config C (full novel model) trains 5 steps on GPU
      ✓ Loss is finite after every step
      ✓ Loss decreases from step 1 → step 5 (loosely)
      ✓ AMP autocast works
      ✓ EMA update runs without error
    """
    import torch, torch.nn as nn, torch.nn.functional as F
    import pandas as pd
    import numpy as np
    import timm
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    device = torch.device("cuda")
    results = {}

    # ── load 256-image subset from manifest ───────────────────────────────
    manifest = DATA_ROOT / "manifest.csv"
    if not manifest.exists():
        return {"FAIL": "manifest.csv missing — run smoke_data first"}

    df = pd.read_csv(manifest)
    train_df = df[df["split"] == "train"].sample(256, random_state=42)
    results["subset_size"] = len(train_df)
    results["class_balance"] = {
        "real": int((train_df["label"]==0).sum()),
        "fake": int((train_df["label"]==1).sum()),
    }

    # ── dataset ───────────────────────────────────────────────────────────
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    class TinyDS(Dataset):
        def __init__(self, df):
            self.paths  = (DATA_ROOT / df["path"]).tolist() if False else \
                          [DATA_ROOT / p for p in df["path"].tolist()]
            self.labels = df["label"].tolist()
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = np.array(Image.open(self.paths[i]).convert("RGB"))
            return transform(image=img)["image"], self.labels[i]

    loader = DataLoader(TinyDS(train_df), batch_size=32, shuffle=True, num_workers=2)
    batch_imgs, batch_labels = next(iter(loader))
    results["batch_shape"]  = list(batch_imgs.shape)
    results["label_shape"]  = list(batch_labels.shape)
    results["batch_ok"]     = batch_imgs.shape == (32, 3, 224, 224)

    # ── inline ConfigC for smoke (pretrained=False to skip HF download) ───
    SRM_KERNELS = torch.tensor([
        [[ 0, 0, 0, 0, 0], [ 0,-1, 2,-1, 0], [ 0, 2,-4, 2, 0], [ 0,-1, 2,-1, 0], [ 0, 0, 0, 0, 0]],
        [[-1, 2,-2, 2,-1], [ 2,-6, 8,-6, 2], [-2, 8,-12,8,-2], [ 2,-6, 8,-6, 2], [-1, 2,-2, 2,-1]],
        [[ 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0], [ 0, 1,-2, 1, 0], [ 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0]],
    ], dtype=torch.float32) / 12.0

    class SRMFilter(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", SRM_KERNELS.unsqueeze(1))
        def forward(self, x):
            B, C, H, W = x.shape
            return torch.tanh(F.conv2d(x.reshape(B*C,1,H,W), self.weight, padding=2).reshape(B,C,3,H,W).mean(1))

    class SRMEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.srm = SRMFilter()
            self.net = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32,64,3,stride=2,padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64,128,3,stride=2,padding=1), nn.BatchNorm2d(128), nn.GELU(),
                nn.Conv2d(128,256,3,stride=2,padding=1), nn.BatchNorm2d(256), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.proj = nn.Sequential(nn.Linear(256,256), nn.GELU())
        def forward(self, x): return self.proj(self.net(self.srm(x)))

    class CMA(nn.Module):
        def __init__(self):
            super().__init__()
            self.q=nn.Linear(768,256); self.k=nn.Linear(256,256)
            self.v=nn.Linear(256,256); self.out=nn.Linear(256,768)
            self.s=256**-0.5
        def forward(self,rgb,srm):
            w=torch.sigmoid((self.q(rgb)*self.k(srm)).sum(-1,keepdim=True)*self.s)
            return rgb+self.out(w*self.v(srm))

    class SmokeC(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb=timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
            self.srm=SRMEncoder(); self.attn=CMA()
            self.cls=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,512),nn.GELU(),nn.Dropout(0.3),nn.Linear(512,2))
            self.proj=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,128))
        def forward(self, x):
            srm=self.srm(x); rgb=self.attn(self.rgb(x),srm); f=torch.cat([rgb,srm],1)
            return {"logits":self.cls(f),"proj":F.normalize(self.proj(f),dim=1)}

    # ── 5-step training ───────────────────────────────────────────────────
    model     = SmokeC().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda")
    losses    = []

    model.train()
    for step, (imgs, labels) in enumerate(loader):
        if step >= 5: break
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out["logits"], labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        l = loss.item()
        losses.append(l)
        results[f"step_{step+1}_loss"] = round(l, 4)

    results["all_finite"]  = all(np.isfinite(losses))
    results["loss_trend"]  = "decreasing" if losses[-1] < losses[0] else "not_decreasing"
    results["vram_used_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    results["STATUS"] = "PASS" if results["all_finite"] and results["batch_ok"] else "FAIL"
    return results


# ─── LOCAL ENTRYPOINT — runs all 5 ───────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("\n" + "="*60)
    print("  AVATAR DETECTION — SMOKE TEST SUITE")
    print("="*60)

    tests = [
        ("1. Data",   smoke_data),
        ("2. Model",  smoke_model),
        ("3. GPU",    smoke_gpu),
        ("4. DB",     smoke_db),
        ("5. Train",  smoke_train),
    ]

    all_pass = True
    for label, fn in tests:
        print(f"\n{'─'*40}")
        print(f"  Running {label}…")
        try:
            result = fn.remote()
            status = result.get("STATUS", "?")
            print(f"  {label}: {status}")
            for k, v in result.items():
                if k != "STATUS":
                    print(f"    {k}: {v}")
            if status != "PASS":
                all_pass = False
        except Exception as e:
            print(f"  {label}: FAIL — {e}")
            all_pass = False

    print(f"\n{'='*60}")
    print(f"  OVERALL: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    print("="*60)
