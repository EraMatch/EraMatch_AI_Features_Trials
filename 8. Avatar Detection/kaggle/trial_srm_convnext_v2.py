"""
AI Avatar Detection — SRM + ConvNeXt Ablation Trial  v2
========================================================
Ablation over 4 configs:
  A — RGB-only ConvNeXt-tiny                              (baseline)
  B — ConvNeXt-tiny + SRM branch (concat)                (+ noise residual)
  C — ConvNeXt-tiny + SRM + cross-modal attention + SupCon  ← NOVEL
  D — Config C + DCT frequency branch (3-branch fusion)  ← EXTENDED NOVEL

Training improvements over v1:
  • 1/3 stratified subsample for faster iteration (~114K images)
  • CosineAnnealingWarmRestarts LR schedule (T_0=4, T_mult=1)
  • Label smoothing = 0.1 on CE loss
  • Mixup augmentation (α=0.2) in training loop
  • EMA model weights (decay=0.999) applied at evaluation time
  • Optuna HPO on Config C (15 trials × 3-epoch quick search, then full retrain)

Datasets (3 combined, 70/15/15 stratified split):
  • kaustubhdhote/human-faces-dataset          →  9,630 images (GAN fakes)
  • muhammadbilal6305/200k-real-vs-ai-visuals  → 200,000 images (diffusion fakes)
  • shreyanshpatel1/130k-real-vs-fake-face     → 133,569 images (diffusion fakes)

GPU: 2× T4 via DataParallel (requested via push.py → machineShape=GPU_T4_X2)
"""

# ─── 0. GPU COMPATIBILITY GUARD ─────────────────────────────────────────────
import subprocess, sys, os

def _check_and_fix_gpu():
    if os.environ.get("_AVATAR_COMPAT_DONE") == "1":
        import torch
        print(f"[COMPAT] Using PyTorch {torch.__version__} (compat build)")
        return
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return
    cap = float(r.stdout.strip().split("\n")[0])
    if cap < 7.0:
        print(f"[COMPAT] GPU sm_{cap:.0f} — installing torch==2.4.1+cu118…")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.4.1+cu118", "torchvision==0.19.1+cu118",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118",
        ], check=True)
        os.environ["_AVATAR_COMPAT_DONE"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

_check_and_fix_gpu()

# ─── 1. IMPORTS & CONFIG ─────────────────────────────────────────────────────
import json, time, random, warnings, copy
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
    roc_auc_score, f1_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEED         = 42
IMG_SIZE     = 224
BATCH_SIZE   = 32          # per GPU; effective = 64 with 2× T4
EPOCHS       = 8
LR           = 1e-4
WD           = 1e-5
LABEL_SMOOTH = 0.1
MIXUP_ALPHA  = 0.2         # 0 = disable mixup
EMA_DECAY    = 0.999
SUPCON_W     = 0.3
SUPCON_TEMP  = 0.07
SRM_DIM      = 256
DCT_DIM      = 256
RGB_DIM      = 768         # ConvNeXt-tiny output dim
PROJ_DIM     = 128         # SupCon projection head dim
SAMPLE_FRAC  = 1 / 3      # subsample the full manifest to this fraction

# ── Optuna HPO settings (runs on Config C only) ──────────────────────────────
HPO_ENABLED  = True
HPO_TRIALS   = 15
HPO_EPOCHS   = 3           # quick search: 3 epochs per trial

OUT_DIR = Path("/kaggle/working")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Device: {device}  |  GPUs: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
          f"({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)")

# ─── 2. DATASET DISCOVERY ────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def find_dataset_root(candidates):
    for prefix in ["/kaggle/input/datasets", "/kaggle/input"]:
        for cand in candidates:
            p = Path(prefix) / cand
            if p.exists():
                return p
    return None


def collect_images(folder: Path, label: int):
    return [{"path": str(f), "label": label}
            for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS]


def build_manifest() -> pd.DataFrame:
    rows = []

    ds1 = find_dataset_root(["kaustubhdhote/human-faces-dataset", "human-faces-dataset"])
    if ds1:
        for sub in ds1.rglob("*"):
            if not sub.is_dir(): continue
            n = sub.name.lower()
            if "real" in n and "ai" not in n:
                rows.extend(collect_images(sub, 0))
                print(f"  [DS1] real    : {sub} ({len(rows)} so far)")
                break
        for sub in ds1.rglob("*"):
            if not sub.is_dir(): continue
            n = sub.name.lower()
            if "ai" in n or "generated" in n or "fake" in n:
                before = len(rows)
                rows.extend(collect_images(sub, 1))
                print(f"  [DS1] fake/ai : {sub} (+{len(rows)-before})")
                break
    else:
        print("  [DS1] NOT FOUND")

    ds2 = find_dataset_root(["muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal",
                              "200k-real-vs-ai-visuals-by-mbilal"])
    if ds2:
        for sub in ds2.rglob("*"):
            if not sub.is_dir(): continue
            if sub.name.lower() == "real":
                before = len(rows); rows.extend(collect_images(sub, 0))
                print(f"  [DS2] real    : {sub} (+{len(rows)-before})")
            elif sub.name.lower() == "ai_images":
                before = len(rows); rows.extend(collect_images(sub, 1))
                print(f"  [DS2] ai      : {sub} (+{len(rows)-before})")
    else:
        print("  [DS2] NOT FOUND")

    ds3 = find_dataset_root(["shreyanshpatel1/130k-real-vs-fake-face",
                              "130k-real-vs-fake-face"])
    if ds3:
        images_dir = ds3 / "images" if (ds3 / "images").exists() else ds3
        for lbl, dirname in [(0, "real"), (1, "fake")]:
            d = images_dir / dirname
            if d.exists():
                before = len(rows); rows.extend(collect_images(d, lbl))
                print(f"  [DS3] {dirname:<5} : {d} (+{len(rows)-before})")
    else:
        print("  [DS3] NOT FOUND")

    df = pd.DataFrame(rows)
    print(f"\nManifest: {len(df):,} total  |  real={( df.label==0).sum():,}  "
          f"fake={(df.label==1).sum():,}")
    return df


print("\n[MANIFEST] Discovering datasets…", flush=True)
df_all = build_manifest()

# ─── 2b. SUBSAMPLE TO 1/3 ────────────────────────────────────────────────────
df_all, _ = train_test_split(
    df_all, train_size=SAMPLE_FRAC, stratify=df_all["label"], random_state=SEED)
print(f"[SUBSAMPLE] Using {len(df_all):,} images ({SAMPLE_FRAC:.0%} of full set)  "
      f"|  real={(df_all.label==0).sum():,}  fake={(df_all.label==1).sum():,}")

# 70 / 15 / 15 stratified split
df_train, df_tmp = train_test_split(
    df_all, test_size=0.30, stratify=df_all["label"], random_state=SEED)
df_val, df_test = train_test_split(
    df_tmp, test_size=0.50, stratify=df_tmp["label"], random_state=SEED)
print(f"Split  →  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

# ─── 3. AUGMENTATIONS ────────────────────────────────────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.ImageCompression(quality_range=(50, 95), p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

# ─── 4. DATASET ───────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx]


def make_loaders(batch_size: int):
    # num_workers=2 keeps CPU RAM lower; no persistent_workers to avoid accumulation
    # across configs when loaders are recreated each run_config call
    kw = dict(num_workers=2, pin_memory=True, persistent_workers=False)
    return (
        DataLoader(FaceDataset(df_train, get_train_transform()),
                   batch_size=batch_size, shuffle=True, **kw),
        DataLoader(FaceDataset(df_val,   get_val_transform()),
                   batch_size=batch_size * 2, shuffle=False, **kw),
        DataLoader(FaceDataset(df_test,  get_val_transform()),
                   batch_size=batch_size * 2, shuffle=False, **kw),
    )

# ─── 5. SRM FILTER BANK ──────────────────────────────────────────────────────
SRM_KERNELS = torch.tensor([
    [[ 0, 0, 0, 0, 0], [ 0,-1, 2,-1, 0], [ 0, 2,-4, 2, 0], [ 0,-1, 2,-1, 0], [ 0, 0, 0, 0, 0]],
    [[-1, 2,-2, 2,-1], [ 2,-6, 8,-6, 2], [-2, 8,-12,8,-2], [ 2,-6, 8,-6, 2], [-1, 2,-2, 2,-1]],
    [[ 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0], [ 0, 1,-2, 1, 0], [ 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0]],
], dtype=torch.float32) / 12.0


class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", SRM_KERNELS.unsqueeze(1))  # (3,1,5,5)

    def forward(self, x):
        B, C, H, W = x.shape
        out = F.conv2d(x.reshape(B*C, 1, H, W), self.weight, padding=2)  # (B*C, 3, H, W)
        return torch.tanh(out.reshape(B, C, 3, H, W).mean(dim=1))        # (B, 3, H, W)


class SRMEncoder(nn.Module):
    def __init__(self, out_dim=SRM_DIM):
        super().__init__()
        self.srm = SRMFilter()
        self.net = nn.Sequential(
            nn.Conv2d(3,  32,  3, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(256, out_dim), nn.GELU())

    def forward(self, x):
        return self.proj(self.net(self.srm(x)))

# ─── 5b. DCT / FREQUENCY BRANCH ──────────────────────────────────────────────
class DCTEncoder(nn.Module):
    """
    Frequency-domain branch via 2-D FFT magnitude spectrum (log-scaled).
    GAN / diffusion models leave characteristic high-frequency artefacts
    that spatial RGB features miss; this branch captures them explicitly.
    """
    def __init__(self, out_dim=DCT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  32,  3, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(256, out_dim), nn.GELU())

    def forward(self, x):
        # x: (B, 3, H, W) — normalised RGB  (values roughly in [-2, 2])
        freq = torch.fft.fft2(x, norm="ortho")           # complex (B,3,H,W)
        mag  = torch.fft.fftshift(torch.abs(freq), dim=(-2, -1))  # centre DC
        mag  = torch.log1p(mag)                           # compress dynamic range
        return self.proj(self.net(mag))                   # (B, out_dim)

# ─── 6. CROSS-MODAL ATTENTION ─────────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    """SRM (or any aux) features gate the RGB feature vector via residual attention."""
    def __init__(self, rgb_dim=RGB_DIM, aux_dim=SRM_DIM, attn_dim=256):
        super().__init__()
        self.q   = nn.Linear(rgb_dim,  attn_dim)
        self.k   = nn.Linear(aux_dim,  attn_dim)
        self.v   = nn.Linear(aux_dim,  attn_dim)
        self.out = nn.Linear(attn_dim, rgb_dim)
        self.scale = attn_dim ** -0.5

    def forward(self, rgb, aux):
        q   = self.q(rgb)
        k   = self.k(aux)
        v   = self.v(aux)
        w   = torch.sigmoid((q * k).sum(-1, keepdim=True) * self.scale)
        return rgb + self.out(w * v)

# ─── 7. MODEL DEFINITIONS ─────────────────────────────────────────────────────
class ConfigA_RGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        self.head = nn.Sequential(nn.LayerNorm(RGB_DIM), nn.Linear(RGB_DIM, 2))

    def forward(self, x):
        f = self.backbone(x)
        return {"logits": self.head(f), "features": f}


class ConfigB_SRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_branch = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        self.srm_branch = SRMEncoder(SRM_DIM)
        fused = RGB_DIM + SRM_DIM
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        f = torch.cat([self.rgb_branch(x), self.srm_branch(x)], 1)
        return {"logits": self.head(f), "features": f}


class ConfigC_Full(nn.Module):
    """Novel: ConvNeXt + SRM + cross-modal attention + SupCon projection head."""
    def __init__(self, supcon_temp=SUPCON_TEMP):
        super().__init__()
        self.supcon_temp = supcon_temp
        self.rgb_branch  = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        self.srm_branch  = SRMEncoder(SRM_DIM)
        self.attention   = CrossModalAttention(RGB_DIM, SRM_DIM)
        fused = RGB_DIM + SRM_DIM
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(fused, 256), nn.GELU(), nn.Linear(256, PROJ_DIM))

    def forward(self, x):
        srm  = self.srm_branch(x)               # compute once, reuse below
        rgb  = self.attention(self.rgb_branch(x), srm)
        f    = torch.cat([rgb, srm], 1)          # fixed: was calling srm_branch twice
        proj = F.normalize(self.proj_head(f), dim=1)
        return {"logits": self.classifier(f), "features": f, "proj": proj}


class ConfigD_DCT(nn.Module):
    """
    Extended novel: ConvNeXt + SRM + DCT (3-branch fusion).
    SRM captures pixel-domain noise residuals; DCT captures frequency-domain artefacts.
    Both gate the RGB backbone via cascaded cross-modal attention.
    """
    def __init__(self):
        super().__init__()
        self.rgb_branch = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        self.srm_branch = SRMEncoder(SRM_DIM)
        self.dct_branch = DCTEncoder(DCT_DIM)
        # SRM gates RGB first, then DCT gates the result
        self.attn_srm = CrossModalAttention(RGB_DIM, SRM_DIM)
        self.attn_dct = CrossModalAttention(RGB_DIM, DCT_DIM)
        fused = RGB_DIM + SRM_DIM + DCT_DIM
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(fused, 256), nn.GELU(), nn.Linear(256, PROJ_DIM))

    def forward(self, x):
        rgb  = self.rgb_branch(x)
        srm  = self.srm_branch(x)
        dct  = self.dct_branch(x)
        rgb  = self.attn_srm(rgb, srm)
        rgb  = self.attn_dct(rgb, dct)
        f    = torch.cat([rgb, srm, dct], 1)
        proj = F.normalize(self.proj_head(f), dim=1)
        return {"logits": self.classifier(f), "features": f, "proj": proj}

# ─── 8. LOSSES ────────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temperature=SUPCON_TEMP):
        super().__init__()
        self.T = temperature

    def forward(self, proj, labels):
        B = proj.shape[0]
        if B < 2: return torch.tensor(0.0, device=proj.device)
        sim = torch.einsum("id,jd->ij", proj, proj) / self.T
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0.0)
        sim.masked_fill_(torch.eye(B, device=proj.device).bool(), float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)
        return loss.mean()

# ─── 9. MIXUP UTILITIES ──────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=MIXUP_ALPHA):
    """Returns mixed inputs, label pairs, and mixing coefficient."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_ce(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)

# ─── 10. EMA ──────────────────────────────────────────────────────────────────
class EMA:
    """
    Exponential moving average of model weights stored on CPU.
    Keeping shadow on CPU saves ~111MB GPU VRAM per model.
    """
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        # store on CPU to keep GPU VRAM free
        self.shadow = {n: p.detach().cpu().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(
                    p.data.detach().cpu(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        """Copy EMA (CPU) weights into model (GPU) for evaluation; return originals."""
        orig = {n: p.data.clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device))
        return orig

    @torch.no_grad()
    def restore(self, model, orig):
        for n, p in model.named_parameters():
            if n in orig:
                p.data.copy_(orig[n])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]

# ─── 11. TRAINING LOOP ───────────────────────────────────────────────────────
def train_one_epoch(model, ema, loader, optimizer, criterion,
                    use_supcon, supcon_w, scaler) -> tuple[float, float]:
    model.train()
    supcon_fn = SupConLoss()
    total_loss = correct = total = 0

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, ya, yb, lam = mixup_batch(imgs, labels)

        with torch.amp.autocast("cuda"):
            out    = model(imgs)
            logits = out["logits"]
            loss   = mixup_ce(criterion, logits, ya, yb, lam)

            if use_supcon and "proj" in out:
                # SupCon uses the dominant label (ya) since mixup blends embeddings
                loss = loss + supcon_w * supcon_fn(out["proj"], ya)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model.module if hasattr(model, "module") else model)

        total_loss += loss.item()
        preds   = logits.argmax(1)
        correct += (preds == ya).sum().item()
        total   += ya.size(0)

        if (step + 1) % 50 == 0:
            print(f"    step {step+1}/{len(loader)}  "
                  f"loss={total_loss/(step+1):.4f}  "
                  f"acc={(correct/total)*100:.1f}%", flush=True)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, ema, loader) -> dict:
    """Evaluate using EMA weights."""
    raw_model = model.module if hasattr(model, "module") else model
    orig = ema.apply(raw_model)

    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        out = model(imgs.to(device))
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels)

    ema.restore(raw_model, orig)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits, 1)[:, 1].numpy()
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1  = f1_score(labels, preds, zero_division=0)
    cm  = confusion_matrix(labels, preds)
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    idx = np.searchsorted(fpr_arr, 0.01)
    tpr_at_1 = float(tpr_arr[min(idx, len(tpr_arr)-1)])

    return {"auc": float(auc), "f1": float(f1), "tpr_at_fpr1": tpr_at_1,
            "probs": probs, "labels": labels, "cm": cm,
            "fpr": fpr_arr, "tpr": tpr_arr}

# ─── 12. VISUALISATIONS ──────────────────────────────────────────────────────
def save_training_plot(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, ax, title in [("train_loss", axes[0], "Training Loss"),
                            ("val_auc",    axes[1], "Validation AUC")]:
        ax.plot(history[key], marker="o"); ax.set_title(f"{name} — {title}")
        ax.set_xlabel("Epoch"); ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_training.png", dpi=120); plt.close()


def save_roc_plot(metrics, name):
    plt.figure(figsize=(6, 6))
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"AUC={metrics['auc']:.4f}")
    plt.plot([0,1],[0,1],"--", color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} — ROC")
    plt.legend(); plt.grid(True)
    plt.savefig(OUT_DIR / f"{name}_roc.png", dpi=120); plt.close()


def save_confusion_matrix(cm, name):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["real","fake"]); ax.set_yticklabels(["real","fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_cm.png", dpi=120); plt.close()

# ─── 13. RUN ONE CONFIG (with epoch-level checkpoint / resume) ───────────────
def run_config(config_name, model_cls, use_supcon,
               lr=LR, wd=WD, supcon_w=SUPCON_W,
               epochs=EPOCHS, save_artefacts=True) -> dict:
    print(f"\n{'='*60}\n  CONFIG {config_name}\n{'='*60}")

    resume_path = OUT_DIR / f"{config_name}_resume.pt"

    model = model_cls()
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    raw_model = model.module if hasattr(model, "module") else model

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.1f}M")

    train_loader, val_loader, test_loader = make_loaders(BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=4, T_mult=1, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler    = torch.amp.GradScaler("cuda")
    ema       = EMA(raw_model, decay=EMA_DECAY)

    history        = defaultdict(list)
    best_auc       = 0.0
    best_ema_shadow = None
    start_epoch    = 1

    # ── Resume from last completed epoch if checkpoint exists ─────────────────
    if resume_path.exists():
        print(f"  [RESUME] Loading checkpoint {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        ema.load_state_dict(ckpt["ema"])
        history         = defaultdict(list, ckpt["history"])
        best_auc        = ckpt["best_auc"]
        best_ema_shadow = ckpt.get("best_ema_shadow")
        start_epoch     = ckpt["epoch"] + 1
        print(f"  [RESUME] Resuming from epoch {start_epoch}  (best_val_auc={best_auc:.4f})")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, ema, train_loader, optimizer, criterion,
            use_supcon, supcon_w, scaler)
        val_m = evaluate(model, ema, val_loader)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_auc"].append(val_m["auc"])
        history["val_f1"].append(val_m["f1"])

        print(f"  Epoch {epoch:02d}/{epochs}  loss={tr_loss:.4f}  "
              f"acc={tr_acc*100:.1f}%  val_auc={val_m['auc']:.4f}  "
              f"val_f1={val_m['f1']:.4f}  ({time.time()-t0:.0f}s)")

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            best_ema_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Save resume checkpoint after every epoch
        torch.save({
            "epoch":           epoch,
            "model":           raw_model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "scaler":          scaler.state_dict(),
            "ema":             ema.state_dict(),
            "history":         dict(history),
            "best_auc":        best_auc,
            "best_ema_shadow": best_ema_shadow,
        }, resume_path)

    # ── Final test evaluation with best EMA weights ───────────────────────────
    if best_ema_shadow:
        ema.shadow = best_ema_shadow

    test_m = evaluate(model, ema, test_loader)
    print(f"\n  TEST  AUC={test_m['auc']:.4f}  F1={test_m['f1']:.4f}  "
          f"TPR@FPR1%={test_m['tpr_at_fpr1']:.4f}")

    if save_artefacts:
        save_training_plot(history, config_name)
        save_roc_plot(test_m, config_name)
        save_confusion_matrix(test_m["cm"], config_name)

        ckpt_path = OUT_DIR / f"{config_name}_best_ema.pt"
        torch.save(best_ema_shadow or ema.shadow, ckpt_path)
        print(f"  EMA checkpoint → {ckpt_path}")

        # Clean up resume file once training is complete
        resume_path.unlink(missing_ok=True)

    return {
        "config":       config_name,
        "test_auc":     test_m["auc"],
        "test_f1":      test_m["f1"],
        "tpr_at_fpr1":  test_m["tpr_at_fpr1"],
        "best_val_auc": best_auc,
        "epochs":       epochs,
        "n_params_M":   round(n_params / 1e6, 2),
        "lr":           lr, "wd": wd, "supcon_w": supcon_w,
    }

# ─── 14. OPTUNA HPO (Config C only) ──────────────────────────────────────────
def run_hpo():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[HPO] optuna not found — skipping")
        return None

    print(f"\n{'='*60}\n  OPTUNA HPO — Config C  ({HPO_TRIALS} trials × {HPO_EPOCHS} epochs)\n{'='*60}")

    def objective(trial):
        lr_       = trial.suggest_float("lr",         1e-5,  5e-4, log=True)
        wd_       = trial.suggest_float("wd",         1e-6,  1e-3, log=True)
        supcon_w_ = trial.suggest_float("supcon_w",   0.1,   0.5)
        supcon_t_ = trial.suggest_float("supcon_temp",0.05,  0.2)

        # Quick run — no artefact saving
        try:
            r = run_config(
                f"hpo_trial_{trial.number}", ConfigC_Full, True,
                lr=lr_, wd=wd_, supcon_w=supcon_w_,
                epochs=HPO_EPOCHS, save_artefacts=False)
            return r["best_val_auc"]
        except Exception as e:
            print(f"  [HPO] trial {trial.number} failed: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=HPO_TRIALS, show_progress_bar=False)

    best = study.best_params
    print(f"\n[HPO] Best params: {best}")
    print(f"[HPO] Best val AUC: {study.best_value:.4f}")

    # Save Optuna study results
    hpo_rows = [{"trial": t.number, "val_auc": t.value, **t.params}
                for t in study.trials if t.value is not None]
    pd.DataFrame(hpo_rows).sort_values("val_auc", ascending=False).to_csv(
        OUT_DIR / "hpo_results.csv", index=False)

    # Visualise param importances
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(list(importances.keys()), list(importances.values()))
        ax.set_title("Optuna — Hyperparameter Importance (Config C)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "hpo_importance.png", dpi=120); plt.close()
    except Exception:
        pass

    return best


# ─── 15. MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Phase 1: Ablation (A → B → C → D) ───────────────────────────────────
    configs = [
        ("config_A_rgb_only",    ConfigA_RGB,   False),
        ("config_B_srm_concat",  ConfigB_SRM,   False),
        ("config_C_full_novel",  ConfigC_Full,  True),
        ("config_D_dct_3branch", ConfigD_DCT,   True),
    ]

    all_results = []
    for name, cls, supcon in configs:
        result = run_config(name, cls, supcon)
        all_results.append(result)
        torch.cuda.empty_cache()

    # ── Phase 2: Optuna HPO on Config C ──────────────────────────────────────
    best_hpo_params = None
    if HPO_ENABLED:
        best_hpo_params = run_hpo()
        torch.cuda.empty_cache()

        if best_hpo_params:
            print("\n[HPO] Full retrain of Config C with best params…")
            hpo_result = run_config(
                "config_C_hpo_optimised", ConfigC_Full, True,
                lr=best_hpo_params["lr"],
                wd=best_hpo_params["wd"],
                supcon_w=best_hpo_params["supcon_w"],
                epochs=EPOCHS,
                save_artefacts=True,
            )
            all_results.append(hpo_result)
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  ABLATION SUMMARY\n{'='*60}")
    print(f"  {'Config':<28} {'AUC':>6} {'F1':>6} {'TPR@1%':>8} {'Params':>8}")
    print(f"  {'-'*58}")
    for r in all_results:
        print(f"  {r['config']:<28} "
              f"{r['test_auc']:>6.4f} "
              f"{r['test_f1']:>6.4f} "
              f"{r['tpr_at_fpr1']:>8.4f} "
              f"{r['n_params_M']:>6.1f}M")

    results_path = OUT_DIR / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")
    print("DONE.")
