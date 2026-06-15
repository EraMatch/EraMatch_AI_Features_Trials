"""
AI Avatar Detection — SRM + ConvNeXt Ablation Trial
====================================================
Ablation over 3 configs:
  A  — RGB-only ConvNeXt-tiny (baseline)
  B  — ConvNeXt-tiny + SRM branch (concat, no attention)
  C  — ConvNeXt-tiny + SRM branch + cross-modal attention + SupCon  ← NOVEL

Datasets (3 combined, 70/15/15 stratified split):
  • kaustubhdhote/human-faces-dataset          →  9,630 images (GAN fakes)
  • muhammadbilal6305/200k-real-vs-ai-visuals  → 200,000 images (video deepfakes)
  • shreyanshpatel1/130k-real-vs-fake-face     → 133,569 images (diffusion fakes)

GPU: 2× T4 via DataParallel  (requested via push.py → machineShape=GPU_T4_X2)
"""

# ─── 0. GPU COMPATIBILITY GUARD ─────────────────────────────────────────────
# Must run before torch import so we can re-exec with a compatible build
import subprocess, sys, os

def _check_and_fix_gpu():
    # Guard: if we already re-launched once, skip — prevents infinite loop
    if os.environ.get("_AVATAR_COMPAT_DONE") == "1":
        import torch
        print(f"[COMPAT] Using PyTorch {torch.__version__} (compat build)")
        return

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return  # no GPU at all
    cap = float(r.stdout.strip().split("\n")[0])
    if cap < 7.0:
        print(f"[COMPAT] GPU sm_{cap:.0f} detected — PyTorch 2.10+cu128 needs sm_70+")
        print("[COMPAT] Installing torch==2.4.1+cu118 (last build with P100 support)…")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.4.1+cu118", "torchvision==0.19.1+cu118",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118",
        ], check=True)
        print("[COMPAT] Re-launching with compatible PyTorch…")
        # Set flag in environment so the re-exec'd process skips this block
        os.environ["_AVATAR_COMPAT_DONE"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

_check_and_fix_gpu()

# ─── 1. IMPORTS & CONFIG ─────────────────────────────────────────────────────
import json, time, random, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

SEED        = 42
IMG_SIZE    = 224
BATCH_SIZE  = 32          # per GPU; effective = 64 with 2× T4
EPOCHS      = 8
LR          = 1e-4
WD          = 1e-5
SUPCON_W    = 0.3         # weight of SupCon loss in config C
SUPCON_TEMP = 0.07
SRM_DIM     = 256         # SRM encoder output dim
RGB_DIM     = 768         # ConvNeXt-tiny output dim
PROJ_DIM    = 128         # SupCon projection head dim
OUT_DIR     = Path("/kaggle/working")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Device: {device}  |  GPUs: {n_gpus}")
if n_gpus > 0:
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)")

# ─── 2. DATASET DISCOVERY ────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def find_dataset_root(candidates: list[str]) -> Path | None:
    """Search common Kaggle mount prefixes for the dataset folder."""
    prefixes = [
        "/kaggle/input/datasets",
        "/kaggle/input",
    ]
    for prefix in prefixes:
        for cand in candidates:
            p = Path(prefix) / cand
            if p.exists():
                return p
    return None


def collect_images(folder: Path, label: int) -> list[dict]:
    rows = []
    for f in folder.rglob("*"):
        if f.suffix.lower() in IMG_EXTS:
            rows.append({"path": str(f), "label": label})
    return rows


def build_manifest() -> pd.DataFrame:
    rows: list[dict] = []

    # ── Dataset 1: kaustubhdhote/human-faces-dataset (9.6K) ─────────────────
    ds1 = find_dataset_root([
        "kaustubhdhote/human-faces-dataset",
        "human-faces-dataset",
    ])
    if ds1:
        # Subfolders: "Real Images/" and "AI-Generated Images/"
        for sub in ds1.rglob("*"):
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if "real" in name and "ai" not in name:
                rows.extend(collect_images(sub, label=0))
                print(f"  [DS1] real     : {sub} ({len(rows)} so far)")
                break
        for sub in ds1.rglob("*"):
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if "ai" in name or "generated" in name or "fake" in name:
                before = len(rows)
                rows.extend(collect_images(sub, label=1))
                print(f"  [DS1] fake/ai  : {sub} (+{len(rows)-before})")
                break
    else:
        print("  [DS1] kaustubhdhote/human-faces-dataset — NOT FOUND")

    # ── Dataset 2: GRAVEX 200K (muhammadbilal6305) ───────────────────────────
    ds2 = find_dataset_root([
        "muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal",
        "200k-real-vs-ai-visuals-by-mbilal",
    ])
    if ds2:
        # Structure: my_real_vs_ai_dataset/my_real_vs_ai_dataset/{real,ai_images}
        for sub in ds2.rglob("*"):
            if not sub.is_dir():
                continue
            if sub.name.lower() == "real":
                before = len(rows)
                rows.extend(collect_images(sub, label=0))
                print(f"  [DS2] real     : {sub} (+{len(rows)-before})")
            elif sub.name.lower() == "ai_images":
                before = len(rows)
                rows.extend(collect_images(sub, label=1))
                print(f"  [DS2] ai_images: {sub} (+{len(rows)-before})")
    else:
        print("  [DS2] muhammadbilal6305/200k — NOT FOUND")

    # ── Dataset 3: 130K real-vs-fake-face (shreyanshpatel1) ──────────────────
    ds3 = find_dataset_root([
        "shreyanshpatel1/130k-real-vs-fake-face",
        "130k-real-vs-fake-face",
    ])
    if ds3:
        images_dir = ds3 / "images"
        if not images_dir.exists():
            images_dir = ds3
        real_dir = images_dir / "real"
        fake_dir = images_dir / "fake"
        if real_dir.exists():
            before = len(rows)
            rows.extend(collect_images(real_dir, label=0))
            print(f"  [DS3] real     : {real_dir} (+{len(rows)-before})")
        if fake_dir.exists():
            before = len(rows)
            rows.extend(collect_images(fake_dir, label=1))
            print(f"  [DS3] fake     : {fake_dir} (+{len(rows)-before})")
    else:
        print("  [DS3] shreyanshpatel1/130k — NOT FOUND")

    df = pd.DataFrame(rows)
    n0 = (df.label == 0).sum()
    n1 = (df.label == 1).sum()
    print(f"\nManifest: {len(df):,} total  |  real={n0:,}  fake={n1:,}")
    return df


print("\n[MANIFEST] Discovering datasets…", flush=True)
df_all = build_manifest()

# 70/15/15 stratified split
df_train, df_tmp = train_test_split(
    df_all, test_size=0.30, stratify=df_all["label"], random_state=SEED)
df_val, df_test = train_test_split(
    df_tmp, test_size=0.50, stratify=df_tmp["label"], random_state=SEED)

print(f"Split  →  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

# ─── 3. AUGMENTATIONS (albumentations 2.0.8 API) ─────────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),                                   # was RandomHorizontalFlip
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE),             # size must be tuple
                            scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.05, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.ImageCompression(quality_range=(50, 95), p=0.3),         # was JpegCompression
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

# ─── 4. DATASET CLASS ─────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths   = df["path"].tolist()
        self.labels  = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx]


def make_loaders(batch_size: int):
    t_tr = get_train_transform()
    t_vl = get_val_transform()
    kw   = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(FaceDataset(df_train, t_tr),
                              batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(FaceDataset(df_val,   t_vl),
                              batch_size=batch_size * 2, shuffle=False, **kw)
    test_loader  = DataLoader(FaceDataset(df_test,  t_vl),
                              batch_size=batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader, test_loader

# ─── 5. SRM FILTER BANK ──────────────────────────────────────────────────────
# 3 fixed kernels that extract noise residuals (Rich-Model / Fridrich 2012)
SRM_KERNELS = torch.tensor([
    # Kernel 1: centre-minus-mean 3×3
    [[ 0, 0, 0, 0, 0],
     [ 0,-1, 2,-1, 0],
     [ 0, 2,-4, 2, 0],
     [ 0,-1, 2,-1, 0],
     [ 0, 0, 0, 0, 0]],
    # Kernel 2: edge detection variant
    [[-1, 2,-2, 2,-1],
     [ 2,-6, 8,-6, 2],
     [-2, 8,-12,8,-2],
     [ 2,-6, 8,-6, 2],
     [-1, 2,-2, 2,-1]],
    # Kernel 3: diagonal / 3×3 Laplacian
    [[ 0, 0, 0, 0, 0],
     [ 0, 0, 0, 0, 0],
     [ 0, 1,-2, 1, 0],
     [ 0, 0, 0, 0, 0],
     [ 0, 0, 0, 0, 0]],
], dtype=torch.float32) / 12.0   # normalise


class SRMFilter(nn.Module):
    """3 fixed high-pass kernels applied channel-independently → 3-ch noise map."""
    def __init__(self):
        super().__init__()
        # weight shape: (out_ch=3, in_ch=1, 5, 5) — applied per colour channel
        w = SRM_KERNELS.unsqueeze(1)   # (3,1,5,5)
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)  →  noise residuals per channel, then mean across colours
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        out    = F.conv2d(x_flat, self.weight, padding=2)   # (B*3, 3, H, W)
        out    = out.reshape(B, C, 3, H, W).mean(dim=1)     # (B, 3, H, W)
        return torch.tanh(out)                              # soft clip


class SRMEncoder(nn.Module):
    """Lightweight CNN that encodes the SRM noise map to a feature vector."""
    def __init__(self, out_dim: int = SRM_DIM):
        super().__init__()
        self.srm = SRMFilter()
        self.net = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,256,3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(256, out_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.srm(x)          # (B, 3, H, W)
        feats = self.net(noise)      # (B, 256)
        return self.proj(feats)      # (B, SRM_DIM)

# ─── 6. CROSS-MODAL ATTENTION ─────────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    """
    SRM features gate the RGB feature vector.
    Query = RGB  (B, rgb_dim)
    Key/Value = SRM (B, srm_dim)
    Output = RGB + residual conditioned on SRM
    """
    def __init__(self, rgb_dim: int = RGB_DIM, srm_dim: int = SRM_DIM,
                 attn_dim: int = 256):
        super().__init__()
        self.q   = nn.Linear(rgb_dim, attn_dim)
        self.k   = nn.Linear(srm_dim, attn_dim)
        self.v   = nn.Linear(srm_dim, attn_dim)
        self.out = nn.Linear(attn_dim, rgb_dim)
        self.scale = attn_dim ** -0.5

    def forward(self, rgb: torch.Tensor, srm: torch.Tensor) -> torch.Tensor:
        q   = self.q(rgb)                              # (B, attn_dim)
        k   = self.k(srm)
        v   = self.v(srm)
        # scalar attention per sample
        w   = torch.sigmoid((q * k).sum(-1, keepdim=True) * self.scale)  # (B,1)
        out = self.out(w * v)                          # (B, rgb_dim)
        return rgb + out                               # residual

# ─── 7. MODEL DEFINITIONS ─────────────────────────────────────────────────────
class ConfigA_RGB(nn.Module):
    """Baseline: ConvNeXt-tiny, RGB only."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.LayerNorm(RGB_DIM),
            nn.Linear(RGB_DIM, 2),
        )

    def forward(self, x):
        f   = self.backbone(x)          # (B, 768)
        out = self.head(f)
        return {"logits": out, "features": f}


class ConfigB_SRM(nn.Module):
    """RGB ConvNeXt + SRM branch, concatenated."""
    def __init__(self):
        super().__init__()
        self.rgb_branch = timm.create_model(
            "convnext_tiny", pretrained=True, num_classes=0)
        self.srm_branch = SRMEncoder(SRM_DIM)
        fused = RGB_DIM + SRM_DIM
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        rgb  = self.rgb_branch(x)       # (B, 768)
        srm  = self.srm_branch(x)       # (B, 256)
        f    = torch.cat([rgb, srm], 1) # (B, 1024)
        out  = self.head(f)
        return {"logits": out, "features": f}


class ConfigC_Full(nn.Module):
    """Full novel model: ConvNeXt + SRM + cross-modal attention + SupCon head."""
    def __init__(self):
        super().__init__()
        self.rgb_branch = timm.create_model(
            "convnext_tiny", pretrained=True, num_classes=0)
        self.srm_branch  = SRMEncoder(SRM_DIM)
        self.attention   = CrossModalAttention(RGB_DIM, SRM_DIM)
        fused = RGB_DIM + SRM_DIM
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
        self.proj_head = nn.Sequential(    # for SupCon loss
            nn.Linear(fused, 256), nn.GELU(),
            nn.Linear(256, PROJ_DIM),
        )

    def forward(self, x):
        rgb  = self.rgb_branch(x)                  # (B, 768)
        srm  = self.srm_branch(x)                  # (B, 256)
        rgb  = self.attention(rgb, srm)             # (B, 768) — SRM-guided
        f    = torch.cat([rgb, srm], 1)             # (B, 1024)
        logits = self.classifier(f)
        proj   = F.normalize(self.proj_head(f), dim=1)  # L2-norm for SupCon
        return {"logits": logits, "features": f, "proj": proj}

# ─── 8. SUPERVISED CONTRASTIVE LOSS ──────────────────────────────────────────
class SupConLoss(nn.Module):
    """
    Khosla et al. (NeurIPS 2020) — eq. 2
    temperature = 0.07 (original paper default for face/image tasks)
    """
    def __init__(self, temperature: float = SUPCON_TEMP):
        super().__init__()
        self.T = temperature

    def forward(self, proj: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = proj.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=proj.device)

        # Similarity matrix
        sim = torch.einsum("id,jd->ij", proj, proj) / self.T   # (B, B)

        # Mask: same class, excluding self
        labels  = labels.unsqueeze(1)                           # (B, 1)
        pos_mask = (labels == labels.T).float()                 # (B, B)
        pos_mask.fill_diagonal_(0.0)

        # Log-softmax trick for numerical stability (exclude self from denominator)
        self_mask = torch.eye(B, device=proj.device).bool()
        sim.masked_fill_(self_mask, float("-inf"))

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B, B)

        n_pos = pos_mask.sum(1).clamp(min=1)
        loss  = -(pos_mask * log_prob).sum(1) / n_pos
        return loss.mean()

# ─── 9. TRAINING LOOP ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, use_supcon: bool,
                    scaler, epoch: int) -> tuple[float, float]:
    model.train()
    supcon = SupConLoss()
    total_loss = total_ce = total_sc = 0.0
    correct = total = 0

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.amp.autocast("cuda"):
            out     = model(imgs)
            logits  = out["logits"]
            ce_loss = criterion(logits, labels)

            if use_supcon and "proj" in out:
                sc_loss = supcon(out["proj"], labels)
                loss    = ce_loss + SUPCON_W * sc_loss
                total_sc += sc_loss.item()
            else:
                loss = ce_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_ce   += ce_loss.item()
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (step + 1) % 50 == 0:
            print(f"    step {step+1}/{len(loader)}  "
                  f"loss={total_loss/(step+1):.4f}  "
                  f"acc={(correct/total)*100:.1f}%", flush=True)

    n = len(loader)
    return total_loss / n, correct / total


@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    all_logits, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        # DataParallel returns gathered tensor on GPU0
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits, 1)[:, 1].numpy()
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1  = f1_score(labels, preds, zero_division=0)
    cm  = confusion_matrix(labels, preds)

    # TPR @ FPR=1%
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    idx = np.searchsorted(fpr_arr, 0.01)
    tpr_at_1 = float(tpr_arr[min(idx, len(tpr_arr)-1)])

    return {
        "auc": float(auc), "f1": float(f1),
        "tpr_at_fpr1": tpr_at_1,
        "probs": probs, "labels": labels, "cm": cm,
        "fpr": fpr_arr, "tpr": tpr_arr,
    }

# ─── 10. VISUALISATIONS ──────────────────────────────────────────────────────
def save_training_plot(history: dict, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, ax, title in [
        ("train_loss", axes[0], "Training Loss"),
        ("val_auc",    axes[1], "Validation AUC"),
    ]:
        ax.plot(history[key], marker="o")
        ax.set_title(f"{name} — {title}")
        ax.set_xlabel("Epoch"); ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_training.png", dpi=120)
    plt.close()


def save_roc_plot(metrics: dict, name: str):
    plt.figure(figsize=(6, 6))
    plt.plot(metrics["fpr"], metrics["tpr"],
             label=f"AUC={metrics['auc']:.4f}")
    plt.plot([0,1],[0,1],"--", color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{name} — ROC Curve")
    plt.legend(); plt.grid(True)
    plt.savefig(OUT_DIR / f"{name}_roc.png", dpi=120)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, name: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["real","fake"]); ax.set_yticklabels(["real","fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_cm.png", dpi=120)
    plt.close()

# ─── 11. RUN ONE CONFIG ──────────────────────────────────────────────────────
def run_config(config_name: str, model_cls, use_supcon: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"  CONFIG {config_name}")
    print(f"{'='*60}")

    model = model_cls()
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.1f}M")

    effective_bs = BATCH_SIZE * max(1, n_gpus)
    train_loader, val_loader, test_loader = make_loaders(BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda")

    history = defaultdict(list)
    best_auc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, use_supcon, scaler, epoch)
        val_m = evaluate(model, val_loader)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_auc"].append(val_m["auc"])
        history["val_f1"].append(val_m["f1"])

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={tr_loss:.4f}  acc={tr_acc*100:.1f}%  "
              f"val_auc={val_m['auc']:.4f}  "
              f"val_f1={val_m['f1']:.4f}  "
              f"({time.time()-t0:.0f}s)")

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            # Save only model weights (unwrap DataParallel if needed)
            m = model.module if hasattr(model, "module") else model
            best_state = {k: v.cpu() for k, v in m.state_dict().items()}

    # ── Test evaluation with best checkpoint ──────────────────────────────
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(best_state)

    test_m = evaluate(model, test_loader)
    print(f"\n  TEST  AUC={test_m['auc']:.4f}  F1={test_m['f1']:.4f}  "
          f"TPR@FPR1%={test_m['tpr_at_fpr1']:.4f}")

    # ── Save artefacts ───────────────────────────────────────────────────
    save_training_plot(history, config_name)
    save_roc_plot(test_m, config_name)
    save_confusion_matrix(test_m["cm"], config_name)

    # Save weights
    ckpt_path = OUT_DIR / f"{config_name}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    return {
        "config":       config_name,
        "test_auc":     test_m["auc"],
        "test_f1":      test_m["f1"],
        "tpr_at_fpr1":  test_m["tpr_at_fpr1"],
        "best_val_auc": best_auc,
        "epochs":       EPOCHS,
        "n_params_M":   round(n_params / 1e6, 2),
    }

# ─── 12. MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    configs = [
        ("config_A_rgb_only",   ConfigA_RGB,  False),
        ("config_B_srm_concat", ConfigB_SRM,  False),
        ("config_C_full_novel", ConfigC_Full, True),
    ]

    all_results = []
    for name, cls, supcon in configs:
        result = run_config(name, cls, supcon)
        all_results.append(result)
        # Free VRAM between configs
        torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<26} {'AUC':>6} {'F1':>6} {'TPR@1%':>8} {'Params':>8}")
    print(f"  {'-'*56}")
    for r in all_results:
        print(f"  {r['config']:<26} "
              f"{r['test_auc']:>6.4f} "
              f"{r['test_f1']:>6.4f} "
              f"{r['tpr_at_fpr1']:>8.4f} "
              f"{r['n_params_M']:>6.1f}M")

    results_path = OUT_DIR / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")
    print("DONE.")
