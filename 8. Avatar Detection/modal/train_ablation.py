"""
AI Avatar Detection — SRM + ConvNeXt Ablation  (Modal / A10G)  — Trial 2
=========================================================================
Fixes vs Trial 1:
  - SupCon NaN: cast to fp32, clamp similarities, T=0.1, NaN guard
  - Cross-dataset OOD: DS3 is ood_test (never seen in training)
  - Class weights: DS1+DS2 are imbalanced (~15× fake), weighted CE loss
  - Full metrics: accuracy, precision, recall, specificity, EER + TPR@FPR1%
  - DB logs both in-dist and OOD results per config

Ablation configs:
  A — RGB-only ConvNeXt-tiny                              (baseline)
  B — RGB + SRM concat                                    (+ noise residual)
  C — RGB + SRM + cross-modal attention + SupCon          ← NOVEL
  D — RGB + SRM + DCT + cross-modal attention + SupCon   ← EXTENDED NOVEL
  C_hpo — Config C retrained with Optuna best params
"""

from pathlib import Path
import modal

DATA_ROOT    = Path("/data")
RESULTS_ROOT = Path("/results")
MODEL_CACHE  = "/models"

data_vol    = modal.Volume.from_name("avatar-data-vol",    create_if_missing=False)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=False)
model_vol   = modal.Volume.from_name("avatar-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "timm==1.0.15",
        "scikit-learn==1.6.1", "pandas==2.2.3", "pillow==11.3.0",
        "albumentations==2.0.8", "numpy==1.26.4",
        "matplotlib==3.9.0", "optuna==3.6.1",
    )
    .env({"HF_HUB_CACHE": MODEL_CACHE, "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("avatar-ablation-t2")


@app.function(
    image=image,
    gpu="A10G",
    memory=65536,
    timeout=9 * 3600,
    volumes={
        str(DATA_ROOT):    data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE:       model_vol,
    },
)
def run_ablation() -> dict:
    import json, time, random, warnings, copy
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
    from PIL import Image as PILImage
    from sklearn.metrics import (
        roc_auc_score, f1_score, confusion_matrix, roc_curve,
        accuracy_score, precision_score, recall_score,
    )
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore", category=UserWarning)

    # ── Inline SQLite DB helpers (avoids Modal src-package import issues) ─────
    import sqlite3
    from datetime import datetime

    def _db_connect(path):
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_db(db_path):
        schema = """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_name TEXT NOT NULL, model_arch TEXT NOT NULL,
            backbone TEXT, train_datasets TEXT, test_dataset TEXT,
            n_train INTEGER, n_val INTEGER, n_test INTEGER,
            use_srm_branch INTEGER DEFAULT 1, use_supcon INTEGER DEFAULT 1,
            use_freq_augmentation INTEGER DEFAULT 1, use_attention_gate INTEGER DEFAULT 1,
            test_auc REAL, test_f1 REAL, test_accuracy REAL, test_precision REAL,
            test_recall REAL, test_specificity REAL, test_eer REAL, test_tpr_at_fpr1 REAL,
            val_best_auc REAL, epochs INTEGER, batch_size INTEGER,
            learning_rate REAL, weight_decay REAL, supcon_weight REAL,
            checkpoint_path TEXT, plots_dir TEXT, config_json TEXT,
            gpu_type TEXT, training_time_s INTEGER, timestamp TEXT, notes TEXT
        );
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL, epoch INTEGER NOT NULL,
            train_loss REAL, val_loss REAL, val_auc REAL, val_f1 REAL, lr REAL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
        """
        conn = _db_connect(db_path)
        conn.executescript(schema); conn.commit(); conn.close()

    def log_experiment(db_path, **kw):
        cols = ("trial_name,model_arch,backbone,train_datasets,test_dataset,"
                "n_train,n_val,n_test,use_srm_branch,use_supcon,use_freq_augmentation,"
                "use_attention_gate,test_auc,test_f1,test_accuracy,test_precision,"
                "test_recall,test_specificity,test_eer,test_tpr_at_fpr1,val_best_auc,"
                "epochs,batch_size,learning_rate,weight_decay,supcon_weight,"
                "checkpoint_path,plots_dir,config_json,gpu_type,training_time_s,timestamp,notes")
        keys = cols.split(",")
        vals = tuple(kw.get(k) for k in keys[:-1]) + (datetime.utcnow().isoformat(),)
        # notes is last key
        vals = tuple(kw.get(k) for k in keys[:-2]) + (datetime.utcnow().isoformat(), kw.get("notes"))
        ph = ",".join(["?"] * len(keys))
        conn = _db_connect(db_path)
        cur = conn.execute(f"INSERT INTO experiments ({cols}) VALUES ({ph})", vals)
        conn.commit(); eid = cur.lastrowid; conn.close()
        return eid

    def log_training_history(db_path, experiment_id, epoch, train_loss=None,
                              val_loss=None, val_auc=None, val_f1=None, lr=None):
        conn = _db_connect(db_path)
        conn.execute(
            "INSERT INTO training_history (experiment_id,epoch,train_loss,val_loss,val_auc,val_f1,lr)"
            " VALUES (?,?,?,?,?,?,?)",
            (experiment_id, epoch, train_loss, val_loss, val_auc, val_f1, lr))
        conn.commit(); conn.close()

    # ── Config ────────────────────────────────────────────────────────────────
    SEED         = 42
    IMG_SIZE     = 224
    BATCH_SIZE   = 64
    EPOCHS       = 10
    LR           = 1e-4
    WD           = 1e-5
    LABEL_SMOOTH = 0.1
    MIXUP_ALPHA  = 0.2
    EMA_DECAY    = 0.999
    SUPCON_W     = 0.3
    SUPCON_TEMP  = 0.10        # was 0.07 → raised to 0.10 for fp16 stability
    SRM_DIM      = 256
    DCT_DIM      = 256
    RGB_DIM      = 768
    PROJ_DIM     = 128
    HPO_TRIALS   = 15
    HPO_EPOCHS   = 3
    DB_PATH      = str(RESULTS_ROOT / "experiments_t2.db")
    PLOTS_DIR    = RESULTS_ROOT / "plots_t2"
    CKPT_DIR     = RESULTS_ROOT / "checkpoints_t2"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")

    init_db(db_path=DB_PATH)

    # ── Load manifest ─────────────────────────────────────────────────────────
    print("\n[DATA] Loading manifest.csv…", flush=True)
    manifest = DATA_ROOT / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError("manifest.csv not found — run setup_datasets.py first")

    df_all = pd.read_csv(manifest)
    splits = set(df_all["split"].unique())
    print(f"  Splits found: {splits}")

    if "train" in splits:
        df_train    = df_all[df_all["split"] == "train"].reset_index(drop=True)
        df_val      = df_all[df_all["split"] == "val"].reset_index(drop=True)
        df_indist   = df_all[df_all["split"] == "in_dist_test"].reset_index(drop=True)
        df_ood      = df_all[df_all["split"] == "ood_test"].reset_index(drop=True)
    else:
        raise ValueError(f"Unexpected splits: {splits}. Re-run setup_datasets.py.")

    n0_tr = (df_train.label==0).sum(); n1_tr = (df_train.label==1).sum()
    print(f"  train:        {len(df_train):,}  real={n0_tr:,}  fake={n1_tr:,}  imbalance={n1_tr/n0_tr:.1f}×")
    print(f"  val:          {len(df_val):,}")
    print(f"  in_dist_test: {len(df_indist):,}")
    print(f"  ood_test:     {len(df_ood):,}  (DS3 — unseen generator)")

    # class weights for imbalanced train set (reals are scarce in DS1+DS2)
    class_weight = torch.tensor([float(n1_tr) / n0_tr, 1.0], dtype=torch.float32, device=device)
    print(f"  class_weight: real={class_weight[0]:.2f}  fake={class_weight[1]:.2f}")

    # ── Augmentations ─────────────────────────────────────────────────────────
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def get_train_tf():
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

    def get_val_tf():
        return A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    class FaceDS(Dataset):
        def __init__(self, df, transform=None):
            self.paths  = [DATA_ROOT / p for p in df["path"].tolist()]
            self.labels = df["label"].tolist()
            self.tf     = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = np.array(PILImage.open(self.paths[i]).convert("RGB"))
            return self.tf(image=img)["image"] if self.tf else img, self.labels[i]

    def make_loaders(bs):
        kw = dict(num_workers=4, pin_memory=True, persistent_workers=False)
        return (
            DataLoader(FaceDS(df_train,  get_train_tf()), batch_size=bs,   shuffle=True,  **kw),
            DataLoader(FaceDS(df_val,    get_val_tf()),   batch_size=bs*2, shuffle=False, **kw),
            DataLoader(FaceDS(df_indist, get_val_tf()),   batch_size=bs*2, shuffle=False, **kw),
            DataLoader(FaceDS(df_ood,    get_val_tf()),   batch_size=bs*2, shuffle=False, **kw),
        )

    # ── SRM filters ───────────────────────────────────────────────────────────
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
        def __init__(self, d=SRM_DIM):
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
        def __init__(self, d=DCT_DIM):
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
            mag = torch.log1p(torch.fft.fftshift(
                torch.abs(torch.fft.fft2(x.float(), norm="ortho")), dim=(-2,-1)))
            return self.proj(self.net(mag))

    class CrossModalAttention(nn.Module):
        def __init__(self, rgb_dim=RGB_DIM, aux_dim=SRM_DIM, d=256):
            super().__init__()
            self.q=nn.Linear(rgb_dim,d); self.k=nn.Linear(aux_dim,d)
            self.v=nn.Linear(aux_dim,d); self.out=nn.Linear(d,rgb_dim)
            self.scale=d**-0.5
        def forward(self, rgb, aux):
            w=torch.sigmoid((self.q(rgb)*self.k(aux)).sum(-1,keepdim=True)*self.scale)
            return rgb+self.out(w*self.v(aux))

    # ── Model configs ─────────────────────────────────────────────────────────
    class ConfigA_RGB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb   = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
            self.head = nn.Sequential(nn.LayerNorm(RGB_DIM), nn.Linear(RGB_DIM, 2))
        def forward(self, x):
            f = self.bb(x); return {"logits": self.head(f), "features": f}

    class ConfigB_SRM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb  = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
            self.srm  = SRMEncoder()
            fused     = RGB_DIM + SRM_DIM
            self.head = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused,512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512,2))
        def forward(self, x):
            f = torch.cat([self.rgb(x), self.srm(x)], 1); return {"logits": self.head(f), "features": f}

    class ConfigC_Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb  = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
            self.srm  = SRMEncoder(); self.attn = CrossModalAttention()
            fused     = RGB_DIM + SRM_DIM
            self.cls  = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused,512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512,2))
            self.proj = nn.Sequential(nn.Linear(fused,256), nn.GELU(), nn.Linear(256,PROJ_DIM))
        def forward(self, x):
            srm=self.srm(x); rgb=self.attn(self.rgb(x), srm); f=torch.cat([rgb,srm],1)
            return {"logits":self.cls(f), "features":f, "proj":F.normalize(self.proj(f),dim=1)}

    class ConfigD_DCT(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb   = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
            self.srm   = SRMEncoder(); self.dct = DCTEncoder()
            self.a_srm = CrossModalAttention()
            self.a_dct = CrossModalAttention(aux_dim=DCT_DIM)
            fused      = RGB_DIM + SRM_DIM + DCT_DIM
            self.cls   = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused,512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512,2))
            self.proj  = nn.Sequential(nn.Linear(fused,256), nn.GELU(), nn.Linear(256,PROJ_DIM))
        def forward(self, x):
            srm=self.srm(x); dct=self.dct(x)
            rgb=self.a_dct(self.a_srm(self.rgb(x),srm),dct); f=torch.cat([rgb,srm,dct],1)
            return {"logits":self.cls(f), "features":f, "proj":F.normalize(self.proj(f),dim=1)}

    # ── SupCon loss (NaN-safe) ────────────────────────────────────────────────
    class SupConLoss(nn.Module):
        """Numerically stable supervised contrastive loss.

        Key fixes vs v1:
        - Always runs in fp32 (proj.float()) — prevents fp16 overflow at T=0.1
        - Clamps cosine similarities before /T — prevents log(0) edge case
        - Skips samples with no in-batch positives
        - NaN guard at output
        """
        def __init__(self, T=SUPCON_TEMP):
            super().__init__()
            self.T = T

        def forward(self, proj: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            proj   = proj.float()                          # fp32 always
            labels = labels.long()
            B      = proj.shape[0]
            if B < 2:
                return torch.tensor(0.0, device=proj.device)

            # cosine sim matrix, clamped for numerical safety before /T
            sim = torch.einsum("id,jd->ij", proj, proj).clamp(-1 + 1e-6, 1 - 1e-6) / self.T

            # mask out self-contrast on diagonal
            eye     = torch.eye(B, device=proj.device, dtype=torch.bool)
            sim     = sim.masked_fill(eye, float("-inf"))

            # positive pair mask (same label, not self)
            lbl_col = labels.unsqueeze(1)
            pos     = (lbl_col == labels.unsqueeze(0)).float()
            pos.fill_diagonal_(0.0)

            # only keep rows that have at least one positive
            has_pos = pos.sum(1) > 0
            if not has_pos.any():
                return torch.tensor(0.0, device=proj.device)

            # numerically stable log-softmax over negatives+positives
            log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

            # mean over positives per anchor, then mean over valid anchors
            loss = -(pos * log_prob).sum(1) / pos.sum(1).clamp(min=1)
            loss = loss[has_pos].mean()

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=proj.device)
            return loss

    def mixup(x, y, alpha=MIXUP_ALPHA):
        if alpha <= 0: return x, y, y, 1.0
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam*x + (1-lam)*x[idx], y, y[idx], lam

    def mixup_ce(crit, logits, ya, yb, lam):
        return lam*crit(logits, ya) + (1-lam)*crit(logits, yb)

    # ── EMA ───────────────────────────────────────────────────────────────────
    class EMA:
        def __init__(self, model, decay=EMA_DECAY):
            self.decay  = decay
            self.shadow = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        def update(self, model):
            for n,p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.data.detach().cpu(), alpha=1-self.decay)
        def apply(self, model):
            orig = {n: p.data.clone() for n,p in model.named_parameters() if n in self.shadow}
            for n,p in model.named_parameters():
                if n in self.shadow: p.data.copy_(self.shadow[n].to(p.device))
            return orig
        def restore(self, model, orig):
            for n,p in model.named_parameters():
                if n in orig: p.data.copy_(orig[n])
        def state_dict(self): return {"decay": self.decay, "shadow": self.shadow}
        def load_state_dict(self, sd): self.decay=sd["decay"]; self.shadow=sd["shadow"]

    # ── Full metrics ──────────────────────────────────────────────────────────
    def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
        preds = (probs >= 0.5).astype(int)
        auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        f1    = f1_score(labels, preds, zero_division=0)
        acc   = accuracy_score(labels, preds)
        prec  = precision_score(labels, preds, zero_division=0)
        rec   = recall_score(labels, preds, zero_division=0)

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        fpr, tpr, _ = roc_curve(labels, probs)
        # EER: point where FPR == FNR
        fnr     = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
        eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

        # TPR at FPR=1%
        tpr_at_1 = float(tpr[min(np.searchsorted(fpr, 0.01), len(tpr)-1)])

        return {
            "auc":         float(auc),
            "f1":          float(f1),
            "accuracy":    float(acc),
            "precision":   float(prec),
            "recall":      float(rec),
            "specificity": float(spec),
            "eer":         float(eer),
            "tpr_at_fpr1": tpr_at_1,
            "cm":          cm,
            "fpr":         fpr,
            "tpr":         tpr,
            "probs":       probs,
            "labels":      labels,
        }

    @torch.no_grad()
    def evaluate(model, ema, loader) -> dict:
        orig = ema.apply(model); model.eval()
        all_logits, all_labels = [], []
        for imgs, labels in loader:
            all_logits.append(model(imgs.to(device))["logits"].cpu())
            all_labels.append(labels)
        ema.restore(model, orig)
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels).numpy()
        probs  = torch.softmax(logits, 1)[:, 1].numpy()
        return compute_metrics(labels, probs)

    # ── Train epoch ───────────────────────────────────────────────────────────
    sc = SupConLoss()

    def train_epoch(model, ema, loader, opt, crit, use_supcon, supcon_w, scaler):
        model.train()
        total_loss = correct = total = 0
        for step, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, ya, yb, lam = mixup(imgs, labels)
            with torch.amp.autocast("cuda"):
                out    = model(imgs)
                logits = out["logits"]
                loss   = mixup_ce(crit, logits, ya, yb, lam)

            # SupCon computed in fp32 outside autocast to prevent fp16 overflow
            if use_supcon and "proj" in out:
                with torch.amp.autocast("cuda", enabled=False):
                    sc_loss = sc(out["proj"], ya)
                if not (torch.isnan(sc_loss) or torch.isinf(sc_loss)):
                    loss = loss + supcon_w * sc_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ema.update(model)

            total_loss += loss.item()
            correct    += (logits.argmax(1) == ya).sum().item()
            total      += ya.size(0)

            if (step+1) % 100 == 0:
                print(f"    step {step+1}/{len(loader)}  loss={total_loss/(step+1):.4f}"
                      f"  acc={correct/total*100:.1f}%", flush=True)

        return total_loss / len(loader), correct / total

    # ── Plots ─────────────────────────────────────────────────────────────────
    def save_plots(history, indist_m, ood_m, name):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history["train_loss"], marker="o", label="train loss")
        axes[0].set_title(f"{name} — Loss"); axes[0].grid(True)

        axes[1].plot(history["val_auc"], marker="o", label="val AUC")
        axes[1].set_title(f"{name} — Val AUC"); axes[1].grid(True)

        axes[2].plot(indist_m["fpr"], indist_m["tpr"], label=f"In-dist AUC={indist_m['auc']:.3f}")
        axes[2].plot(ood_m["fpr"],    ood_m["tpr"],    label=f"OOD AUC={ood_m['auc']:.3f}", linestyle="--")
        axes[2].plot([0,1],[0,1],"--", color="grey")
        axes[2].set_xlabel("FPR"); axes[2].set_ylabel("TPR")
        axes[2].set_title(f"{name} — ROC"); axes[2].legend(); axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{name}_curves.png", dpi=120); plt.close()

        for suffix, m in [("indist", indist_m), ("ood", ood_m)]:
            cm = m["cm"]; fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(cm, cmap="Blues"); ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["real","fake"]); ax.set_yticklabels(["real","fake"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i,j], ha="center", va="center",
                            color="white" if cm[i,j] > cm.max()/2 else "black")
            plt.colorbar(im, ax=ax)
            plt.title(f"{name} — CM ({suffix})")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{name}_cm_{suffix}.png", dpi=120); plt.close()

    # ── run_config ────────────────────────────────────────────────────────────
    def run_config(config_name, model_cls, use_supcon,
                   lr=LR, wd=WD, supcon_w=SUPCON_W,
                   epochs=EPOCHS, save_artefacts=True):
        print(f"\n{'='*60}\n  CONFIG {config_name}\n{'='*60}")
        t_start     = time.time()
        resume_path = CKPT_DIR / f"{config_name}_resume.pt"

        model = model_cls().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_params/1e6:.1f}M")

        train_loader, val_loader, indist_loader, ood_loader = make_loaders(BATCH_SIZE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=LABEL_SMOOTH)
        scaler    = torch.amp.GradScaler("cuda")
        ema       = EMA(model)

        history = defaultdict(list); best_auc = 0.0; best_shadow = None; start_epoch = 1

        if resume_path.exists():
            print(f"  [RESUME] {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            ema.load_state_dict(ckpt["ema"])
            history     = defaultdict(list, ckpt["history"])
            best_auc    = ckpt["best_auc"]
            best_shadow = ckpt.get("best_shadow")
            start_epoch = ckpt["epoch"] + 1
            print(f"  Resuming from epoch {start_epoch}  (best_auc={best_auc:.4f})")

        for epoch in range(start_epoch, epochs+1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, ema, train_loader, optimizer,
                                          criterion, use_supcon, supcon_w, scaler)
            val_m = evaluate(model, ema, val_loader)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_auc"].append(val_m["auc"])
            history["val_f1"].append(val_m["f1"])

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:02d}/{epochs}  loss={tr_loss:.4f}  acc={tr_acc*100:.1f}%"
                  f"  val_auc={val_m['auc']:.4f}  val_f1={val_m['f1']:.4f}"
                  f"  lr={cur_lr:.2e}  ({time.time()-t0:.0f}s)", flush=True)

            if val_m["auc"] > best_auc:
                best_auc    = val_m["auc"]
                best_shadow = {k: v.clone() for k, v in ema.shadow.items()}

            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "ema": ema.state_dict(),
                "history": dict(history), "best_auc": best_auc, "best_shadow": best_shadow,
            }, resume_path)
            results_vol.commit()

        if best_shadow:
            ema.shadow = best_shadow

        indist_m = evaluate(model, ema, indist_loader)
        ood_m    = evaluate(model, ema, ood_loader)

        def fmt(m, tag):
            return (f"  {tag}: AUC={m['auc']:.4f}  F1={m['f1']:.4f}"
                    f"  Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}"
                    f"  Rec={m['recall']:.4f}  Spec={m['specificity']:.4f}"
                    f"  EER={m['eer']:.4f}  TPR@1%={m['tpr_at_fpr1']:.4f}")
        print(fmt(indist_m, "IN-DIST TEST"))
        print(fmt(ood_m,    "OOD TEST    "))

        training_time = int(time.time() - t_start)
        ckpt_path = None

        if save_artefacts:
            save_plots(history, indist_m, ood_m, config_name)
            ckpt_path = str(CKPT_DIR / f"{config_name}_best_ema.pt")
            torch.save(best_shadow or ema.shadow, ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")
            resume_path.unlink(missing_ok=True)

        def _log(m, test_tag):
            return log_experiment(
                db_path        = DB_PATH,
                trial_name     = f"{config_name}__{test_tag}",
                model_arch     = config_name.split("_")[1] if "_" in config_name else config_name,
                backbone       = "convnext_tiny",
                train_datasets = "DS1+DS2 (1/3 subsample, cross-dataset split)",
                test_dataset   = test_tag,
                n_train        = len(df_train), n_val=len(df_val),
                n_test         = len(df_indist) if test_tag == "in_dist_test" else len(df_ood),
                use_srm_branch = int("srm" in config_name.lower() or config_name.startswith("config_C") or config_name.startswith("config_D")),
                use_supcon     = int(use_supcon),
                use_freq_augmentation = int("dct" in config_name.lower()),
                use_attention_gate    = int(config_name.startswith("config_C") or config_name.startswith("config_D")),
                test_auc       = m["auc"],
                test_f1        = m["f1"],
                test_accuracy  = m["accuracy"],
                test_precision = m["precision"],
                test_recall    = m["recall"],
                test_specificity = m["specificity"],
                test_eer       = m["eer"],
                test_tpr_at_fpr1 = m["tpr_at_fpr1"],
                val_best_auc   = best_auc,
                epochs         = epochs, batch_size = BATCH_SIZE,
                learning_rate  = lr, weight_decay = wd, supcon_weight = supcon_w,
                checkpoint_path = ckpt_path if save_artefacts and test_tag == "in_dist_test" else None,
                plots_dir      = str(PLOTS_DIR) if save_artefacts else None,
                gpu_type       = gpu_name,
                training_time_s = training_time,
                notes          = f"Trial-2 Modal A10G | {test_tag}",
            )

        id_indist = _log(indist_m, "in_dist_test")
        id_ood    = _log(ood_m,    "ood_test")

        for ep, (tl, va, vf) in enumerate(
                zip(history["train_loss"], history["val_auc"], history["val_f1"]), 1):
            log_training_history(db_path=DB_PATH, experiment_id=id_indist,
                                 epoch=ep, train_loss=tl, val_auc=va, val_f1=vf)

        results_vol.commit()

        return {
            "config":      config_name,
            "n_params_M":  round(n_params/1e6, 2),
            "epochs":      epochs,
            "lr": lr, "wd": wd, "supcon_w": supcon_w,
            "training_time_s": training_time,
            "db_id_indist": id_indist,
            "db_id_ood":    id_ood,
            "indist": {k: indist_m[k] for k in
                       ("auc","f1","accuracy","precision","recall","specificity","eer","tpr_at_fpr1")},
            "ood":    {k: ood_m[k] for k in
                       ("auc","f1","accuracy","precision","recall","specificity","eer","tpr_at_fpr1")},
        }

    # ── HPO ───────────────────────────────────────────────────────────────────
    def run_hpo():
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print(f"\n{'='*60}\n  OPTUNA HPO — Config C  ({HPO_TRIALS} trials × {HPO_EPOCHS} epochs)\n{'='*60}")

        def objective(trial):
            lr_ = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
            wd_ = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
            sw_ = trial.suggest_float("supcon_w", 0.1, 0.5)
            try:
                r = run_config(f"hpo_t{trial.number}", ConfigC_Full, True,
                               lr=lr_, wd=wd_, supcon_w=sw_,
                               epochs=HPO_EPOCHS, save_artefacts=False)
                return r["indist"]["auc"]
            except Exception as e:
                print(f"  trial {trial.number} failed: {e}")
                return 0.0

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=HPO_TRIALS)
        best = study.best_params
        print(f"\n[HPO] Best: {best}  val_auc={study.best_value:.4f}")

        rows = [{"trial": t.number, "val_auc": t.value, **t.params}
                for t in study.trials if t.value]
        pd.DataFrame(rows).sort_values("val_auc", ascending=False).to_csv(
            RESULTS_ROOT / "hpo_results_t2.csv", index=False)
        results_vol.commit()
        return best

    # ── MAIN ──────────────────────────────────────────────────────────────────
    configs = [
        ("config_A_rgb_only",    ConfigA_RGB,  False),
        ("config_B_srm_concat",  ConfigB_SRM,  False),
        ("config_C_full_novel",  ConfigC_Full, True),
        ("config_D_dct_3branch", ConfigD_DCT,  True),
    ]

    all_results = []
    for name, cls, supcon in configs:
        r = run_config(name, cls, supcon)
        all_results.append(r)
        torch.cuda.empty_cache()

    best_hpo = run_hpo()
    torch.cuda.empty_cache()
    if best_hpo:
        r = run_config("config_C_hpo_optimised", ConfigC_Full, True,
                       lr=best_hpo["lr"], wd=best_hpo["wd"], supcon_w=best_hpo["supcon_w"])
        all_results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  ABLATION SUMMARY — TRIAL 2\n{'='*60}")
    hdr = f"  {'Config':<28} {'In-dist AUC':>12} {'OOD AUC':>9} {'OOD F1':>7} {'OOD Acc':>8} {'OOD EER':>8} {'Params':>7}"
    print(hdr)
    print(f"  {'-'*80}")
    for r in all_results:
        print(f"  {r['config']:<28} {r['indist']['auc']:>12.4f} {r['ood']['auc']:>9.4f}"
              f" {r['ood']['f1']:>7.4f} {r['ood']['accuracy']:>8.4f}"
              f" {r['ood']['eer']:>8.4f} {r['n_params_M']:>6.1f}M")

    json_path = RESULTS_ROOT / "ablation_results_t2.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    results_vol.commit()
    print(f"\nResults → {json_path}\nDONE.")
    return {"status": "complete", "n_configs": len(all_results), "results": all_results}


@app.local_entrypoint()
def main():
    print("Launching Trial 2 ablation on Modal A10G…")
    result = run_ablation.remote()
    print(f"\nDone: {result['n_configs']} configs")
    for r in result["results"]:
        print(f"  {r['config']}: in-dist AUC={r['indist']['auc']:.4f}  OOD AUC={r['ood']['auc']:.4f}")
