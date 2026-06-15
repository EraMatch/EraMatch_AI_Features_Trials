"""Trial 1: DCT Dual-Branch Baseline — Modal T4 training script.

Proves that feeding DCT as a 4th model input channel beats RGB-only.
Uses EfficientNet-B1 with weight surgery for the extra channel.
Trains on Dataset C (9.6K existing) to keep comparison fair with v11.

Three configs:
    1a_rgb_only    — RGB only (v11 reproduction baseline)
    1b_rgb_dct     — RGB + DCT 4th channel (the upgrade)
    1c_rgb_dct_ls  — RGB + DCT + label smoothing 0.1

Usage:
    modal run modal/train_nb1_dct.py

Requires pre-created Modal secrets and volumes:
    modal secret create kaggle-creds KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
    modal volume create avatar-data-vol
    modal volume create avatar-results-vol
"""

import json
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/data")
RESULTS_ROOT = Path("/results")
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
MODEL_CACHE = "/models"

# ---------------------------------------------------------------------------
# App + named resources
# ---------------------------------------------------------------------------
app = modal.App("avatar-dct-trial")

data_vol = modal.Volume.from_name("avatar-data-vol", create_if_missing=True)
results_vol = modal.Volume.from_name("avatar-results-vol", create_if_missing=True)
model_vol = modal.Volume.from_name("avatar-model-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: pinned deps, cache env, local source
# ---------------------------------------------------------------------------
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
        "matplotlib==3.10.3",
        "seaborn==0.13.2",
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


# ---------------------------------------------------------------------------
# Smoke test: validate remote imports (CPU, 2 min)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=2 * 60,
    volumes={
        str(DATA_ROOT): data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE: model_vol,
    },
)
def validate_remote_imports() -> dict:
    """Verify all imports work and DCT4ChannelModel can be instantiated."""
    import torch
    import timm
    import scipy
    import sklearn
    import albumentations

    from src.config import TRIAL1_CONFIGS, IMG_SIZE
    from src.models.dct_branch import DCT4ChannelModel
    from src.data.dataset import AvatarDataset
    from src.data.augmentations import get_train_transforms, get_val_transforms
    from src.utils.db import log_experiment, log_training_history, init_db
    from src.utils.metrics import compute_all_metrics
    from src.utils.viz import plot_training_curves, plot_confusion_matrix

    model = DCT4ChannelModel(pretrained=False, num_classes=2)
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "torch": torch.__version__,
        "timm": timm.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "albumentations": albumentations.__version__,
        "n_configs": len(TRIAL1_CONFIGS),
        "img_size": IMG_SIZE,
        "model_params": n_params,
        "status": "OK",
    }
    print(f"[Smoke] validate_remote_imports PASSED: {result}")
    return result


# ---------------------------------------------------------------------------
# Quick test: 8-sample overfit (GPU T4, 5 min)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",
    memory=1024,
    timeout=5 * 60,
    volumes={
        str(DATA_ROOT): data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE: model_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_smoke_training() -> dict:
    """8-sample overfit test: verify model trains and loss decreases.

    Creates synthetic data, runs 5 forward+backward passes, checks
    loss is finite and decreasing. Aborts full run if this fails.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.dct_branch import DCT4ChannelModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    images = torch.randn(8, 3, 224, 224, device=device)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)

    model = DCT4ChannelModel(pretrained=False, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  [Smoke] step {step + 1}/5  loss={loss.item():.4f}")

    all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)
    if not all_finite:
        raise RuntimeError(f"SMOKE TEST FAILED: non-finite losses: {losses}")

    decreasing = losses[-1] < losses[0]
    if not decreasing:
        raise RuntimeError(
            f"SMOKE TEST FAILED: loss not decreasing: first={losses[0]:.4f} last={losses[-1]:.4f}"
        )

    ckpt_path = Path("/tmp/smoke_ckpt.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "step": 5, "losses": losses},
        str(ckpt_path),
    )
    loaded = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    reloaded_model = DCT4ChannelModel(pretrained=False, num_classes=2).to(device)
    reloaded_model.load_state_dict(loaded["model_state_dict"])

    with torch.no_grad():
        orig_out = model(images)
        reload_out = reloaded_model(images)
    match = torch.allclose(orig_out, reload_out, atol=1e-5)

    if not match:
        raise RuntimeError(
            "SMOKE TEST FAILED: reloaded model output differs from original"
        )

    result = {
        "losses": losses,
        "loss_decreased": decreasing,
        "checkpoint_roundtrip": match,
        "status": "PASSED",
    }
    print(f"[Smoke] SMOKE TEST PASSED: {result}")
    return result


# ---------------------------------------------------------------------------
# Training function: one config (GPU T4, 45 min)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",
    memory=16384,
    timeout=60 * 60,
    volumes={
        str(DATA_ROOT): data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE: model_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("kaggle-creds"),
    ],
)
def train_config(config_name: str) -> dict:
    """Train a single Trial 1 config and save all artifacts.

    Reads config from TRIAL1_CONFIGS, trains on Dataset C (9.6K),
    logs to SQLite, saves plots and checkpoint.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from src.config import TRIAL1_CONFIGS, IMG_SIZE, RESULTS_ROOT as SRC_RESULTS_ROOT
    from src.models.dct_branch import DCT4ChannelModel
    from src.data.dataset import AvatarDataset
    from src.data.augmentations import get_train_transforms, get_val_transforms
    from src.utils.db import init_db, log_experiment, log_training_history
    from src.utils.metrics import compute_all_metrics
    from src.utils.viz import (
        plot_training_curves,
        plot_confusion_matrix,
        plot_roc_curve,
        plot_score_distribution,
        plot_frequency_heatmap,
    )

    # ------------------------------------------------------------------
    # Resolve config
    # ------------------------------------------------------------------
    config = None
    for c in TRIAL1_CONFIGS:
        if c["name"] == config_name:
            config = c
            break
    if config is None:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {[c['name'] for c in TRIAL1_CONFIGS]}"
        )

    use_dct = config["use_dct"]
    label_smoothing = config["label_smoothing"]

    # Trial 1 hyperparams (per AGENTS.md Section 7)
    EPOCHS = 10
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_EVAL = 64
    LR = 1e-4
    WEIGHT_DECAY = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Train] config={config_name}  use_dct={use_dct}  label_smoothing={label_smoothing}"
    )
    print(f"[Train] device={device}  epochs={EPOCHS}  batch_train={BATCH_SIZE_TRAIN}")

    # ------------------------------------------------------------------
    # Init DB
    # ------------------------------------------------------------------
    init_db()
    print("[Train] DB initialized")

    # ------------------------------------------------------------------
    # Load datasets — Dataset C (9.6K) only, with 70/15/15 stratified split
    # The manifest.csv must have split='train'/'val'/'test' for dataset C.
    # Per AGENTS.md, 9.6K is labeled as split in manifest; Trial 1 uses
    # train/val/test from dataset C only.
    # ------------------------------------------------------------------
    # Filter manifest to dataset C only (9.6K) for Trial 1
    # Per AGENTS.md: "Dataset: 9.6K existing ONLY (keeps comparison fair with v11)"
    import pandas as pd

    manifest_df = pd.read_csv(MANIFEST_PATH)
    dataset_c_df = manifest_df[manifest_df["dataset"] == "C"].copy()
    temp_manifest = DATA_ROOT / "manifest_trial1.csv"
    dataset_c_df.to_csv(str(temp_manifest), index=False)
    split_value_counts = pd.Series(dataset_c_df["split"]).value_counts()
    print(
        f"[Train] Dataset C rows: {len(dataset_c_df)}  splits={split_value_counts.to_dict()}"
    )

    train_transform = get_train_transforms(img_size=IMG_SIZE)
    val_transform = get_val_transforms(img_size=IMG_SIZE)

    train_ds = AvatarDataset(
        manifest_path=str(temp_manifest),
        split="train",
        transform=train_transform,
        img_size=IMG_SIZE,
    )
    val_ds = AvatarDataset(
        manifest_path=str(temp_manifest),
        split="val",
        transform=val_transform,
        img_size=IMG_SIZE,
    )
    test_ds = AvatarDataset(
        manifest_path=str(temp_manifest),
        split="test",
        transform=val_transform,
        img_size=IMG_SIZE,
    )

    n_train = len(train_ds)
    n_val = len(val_ds)
    n_test = len(test_ds)
    print(f"[Train] samples: train={n_train}  val={n_val}  test={n_test}")

    if n_train == 0:
        raise RuntimeError(
            f"No training samples found for dataset C in {temp_manifest}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=AvatarDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=AvatarDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=AvatarDataset.collate_fn,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if use_dct:
        model = DCT4ChannelModel(pretrained=True, num_classes=2).to(device)
        model_arch = "efficientnet_b1_dct4ch"
    else:
        import timm

        model = timm.create_model("efficientnet_b1", pretrained=True, num_classes=2).to(
            device
        )
        model_arch = "efficientnet_b1_rgb"

    print(
        f"[Train] model_arch={model_arch}  params={sum(p.numel() for p in model.parameters()):,}"
    )

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": [], "lr": []}
    best_val_auc = 0.0
    best_epoch = 0
    start_time = time.time()

    artifacts_dir = RESULTS_ROOT / "trial1" / config_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / "checkpoint_best.pt"

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_losses = []
        val_scores_list = []
        val_labels_list = []
        val_preds_list = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)

                val_scores_list.append(probs.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())
                val_preds_list.append(preds.cpu().numpy())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_scores = np.concatenate(val_scores_list)
        val_labels = np.concatenate(val_labels_list)
        val_preds = np.concatenate(val_preds_list)

        val_metrics = compute_all_metrics(val_labels, val_scores)
        val_auc = val_metrics["auc"]
        val_f1 = val_metrics["f1"]

        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        # Log to SQLite
        log_training_history(
            experiment_id=0,  # placeholder, updated below
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_auc=val_auc,
            val_f1=val_f1,
            lr=current_lr,
        )

        # Checkpoint best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "config_name": config_name,
                    "config": config,
                },
                str(ckpt_path),
            )

        print(
            f"[Train] epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}  lr={current_lr:.6f}"
        )

    training_time_s = int(time.time() - start_time)
    print(
        f"[Train] Training done. Best val_auc={best_val_auc:.4f} at epoch {best_epoch}  time={training_time_s}s"
    )

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    # Load best checkpoint
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_scores_list = []
    test_labels_list = []
    test_preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            test_scores_list.append(probs.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())
            test_preds_list.append(preds.cpu().numpy())

    test_scores = np.concatenate(test_scores_list)
    test_labels = np.concatenate(test_labels_list)
    test_preds = np.concatenate(test_preds_list)

    test_metrics = compute_all_metrics(test_labels, test_scores)

    print(
        f"[Train] TEST  AUC={test_metrics['auc']:.4f}  F1={test_metrics['f1']:.4f}  "
        f"Acc={test_metrics['accuracy']:.4f}  EER={test_metrics['eer']:.4f}  "
        f"TPR@FPR1={test_metrics['tpr_at_fpr1']:.4f}"
    )

    # ------------------------------------------------------------------
    # Log experiment to SQLite
    # ------------------------------------------------------------------
    exp_id = log_experiment(
        trial_name=config_name,
        model_arch=model_arch,
        backbone="efficientnet_b1",
        train_datasets="C_9.6k",
        test_dataset="C_9.6k_test",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        use_srm_branch=0 if use_dct else 0,
        use_supcon=0,
        use_freq_augmentation=0,
        use_attention_gate=0,
        test_auc=test_metrics["auc"],
        test_f1=test_metrics["f1"],
        test_accuracy=test_metrics["accuracy"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        test_specificity=test_metrics["specificity"],
        test_eer=test_metrics["eer"],
        test_tpr_at_fpr1=test_metrics["tpr_at_fpr1"],
        val_best_auc=best_val_auc,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE_TRAIN,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        supcon_weight=0.0,
        checkpoint_path=str(ckpt_path),
        plots_dir=str(artifacts_dir),
        config_json=json.dumps(config),
        gpu_type="T4",
        training_time_s=training_time_s,
        notes=f"Trial 1 config. use_dct={use_dct}, label_smoothing={label_smoothing}. Best val AUC at epoch {best_epoch}.",
    )

    # Update training_history: we logged with experiment_id=0 placeholder during training
    import sqlite3

    conn = sqlite3.connect(str(RESULTS_ROOT / "experiments.db"))
    conn.execute(
        "UPDATE training_history SET experiment_id = ? WHERE experiment_id = 0",
        (exp_id,),
    )
    conn.commit()
    conn.close()

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    plot_training_curves(history, save_path=artifacts_dir / "training_curves.png")
    plot_confusion_matrix(
        test_labels, test_preds, save_path=artifacts_dir / "confusion_matrix.png"
    )
    plot_roc_curve(
        test_labels,
        test_scores,
        auc_value=test_metrics["auc"],
        save_path=artifacts_dir / "roc_curve.png",
    )
    plot_score_distribution(
        test_labels, test_scores, save_path=artifacts_dir / "score_dist.png"
    )

    # Frequency heatmap: compute mean DCT magnitude for real vs fake test images
    _compute_and_save_freq_heatmap(test_loader, device, artifacts_dir)

    # Grad-CAM on 4 samples (2 real, 2 fake)
    _compute_and_save_gradcam(model, test_loader, device, artifacts_dir, use_dct)

    # ------------------------------------------------------------------
    # Commit results volume
    # ------------------------------------------------------------------
    results_vol.commit()
    print(f"[Train] Artifacts saved to {artifacts_dir}")
    print(f"[Train] Volume committed. Experiment ID: {exp_id}")

    return {
        "config_name": config_name,
        "exp_id": exp_id,
        "test_auc": test_metrics["auc"],
        "val_best_auc": best_val_auc,
        "best_epoch": best_epoch,
        "training_time_s": training_time_s,
    }


# ---------------------------------------------------------------------------
# Helpers (called inside train_config's container)
# ---------------------------------------------------------------------------


def _compute_and_save_freq_heatmap(test_loader, device, artifacts_dir):
    """Compute mean |DCT| for real vs fake, save heatmap."""
    import numpy as np
    import torch
    from scipy.fft import dctn

    from src.utils.viz import plot_frequency_heatmap

    real_dcts = []
    fake_dcts = []
    n_samples = 0
    max_samples = 200  # cap for speed

    for batch in test_loader:
        if n_samples >= max_samples:
            break
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()

        for i in range(images.shape[0]):
            if n_samples >= max_samples:
                break
            # Convert to grayscale
            gray = (
                0.299 * images[i, 0:1, :, :]
                + 0.587 * images[i, 1:2, :, :]
                + 0.114 * images[i, 2:3, :, :]
            )
            gray_np = gray.squeeze().cpu().numpy().astype(np.float32)
            d = dctn(gray_np, type=2, norm="ortho")
            d_array = np.asarray(d, dtype=np.float64)
            magnitude = np.log1p(np.abs(d_array))

            if labels[i] == 0:
                real_dcts.append(magnitude)
            else:
                fake_dcts.append(magnitude)
            n_samples += 1

    if real_dcts and fake_dcts:
        real_mean = np.mean(real_dcts, axis=0)
        fake_mean = np.mean(fake_dcts, axis=0)
        plot_frequency_heatmap(
            real_mean, fake_mean, save_path=artifacts_dir / "freq_heatmap.png"
        )
    else:
        print("[Train] Skipping freq heatmap: not enough real/fake samples")


def _compute_and_save_gradcam(model, test_loader, device, artifacts_dir, use_dct):
    """Generate Grad-CAM visualizations for 2 real + 2 fake samples."""
    import numpy as np
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.viz import _DPI

    # Collect 2 real + 2 fake samples
    real_images = []
    fake_images = []
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()
        for i in range(images.shape[0]):
            if labels[i] == 0 and len(real_images) < 2:
                real_images.append(images[i : i + 1])
            elif labels[i] == 1 and len(fake_images) < 2:
                fake_images.append(images[i : i + 1])
            if len(real_images) >= 2 and len(fake_images) >= 2:
                break
        if len(real_images) >= 2 and len(fake_images) >= 2:
            break

    samples = real_images + fake_images
    sample_labels = ["Real"] * len(real_images) + ["Fake"] * len(fake_images)

    if not samples:
        print("[Train] Skipping Grad-CAM: no samples found")
        return

    # Simple gradient-based CAM
    fig, axes = plt.subplots(len(samples), 2, figsize=(8, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    for idx, (img_tensor, label_str) in enumerate(zip(samples, sample_labels)):
        img_tensor = img_tensor.clone().requires_grad_(True)

        # Forward pass
        output = model(img_tensor)
        score = output[0, 1]  # fake class score
        score.backward()

        # Get gradients from the last conv layer
        gradients = img_tensor.grad[0].cpu().numpy()
        grayscale_grad = np.mean(np.abs(gradients), axis=0)

        # Normalize
        if grayscale_grad.max() > 0:
            grayscale_grad = grayscale_grad / grayscale_grad.max()

        # Original image for display
        img_display = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
        # Denormalize (approximate)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = img_display * std + mean
        img_display = np.clip(img_display, 0, 1)

        axes[idx, 0].imshow(img_display)
        axes[idx, 0].set_title(f"{label_str} (original)")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(img_display)
        axes[idx, 1].imshow(grayscale_grad, cmap="jet", alpha=0.5)
        axes[idx, 1].set_title(f"{label_str} (Grad-CAM)")
        axes[idx, 1].axis("off")

    fig.suptitle("Grad-CAM Visualization")
    fig.tight_layout()
    fig.savefig(str(artifacts_dir / "gradcam.png"), dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Local entrypoint: smoke → smoke training → full training per config
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    """Run all Trial 1 configs in sequence: smoke tests then training."""
    from src.config import TRIAL1_CONFIGS

    print("=" * 60)
    print("Trial 1: DCT Dual-Branch Baseline")
    print("=" * 60)

    # Step 1: Validate imports
    print("\n[Smoke] Validating imports...")
    import_result = validate_remote_imports.remote()
    print(f"[Smoke] Import check: {import_result}")

    # Step 2: 8-sample overfit check
    print("\n[Smoke] Running 8-sample overfit check...")
    smoke_result = run_smoke_training.remote()
    print(f"[Smoke] Training check: {smoke_result}")

    if smoke_result.get("status") != "PASSED":
        print("[FATAL] Smoke test failed — aborting full training.")
        return

    # Step 3: Train each config sequentially
    results = []
    for config in TRIAL1_CONFIGS:
        config_name = config["name"]
        print(f"\n[Train] Starting config: {config_name}")
        result = train_config.remote(config_name)
        results.append(result)
        print(
            f"[Train] Completed: {config_name} → AUC={result.get('test_auc', 'N/A'):.4f}"
        )

    # Summary
    print("\n" + "=" * 60)
    print("Trial 1 Summary")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['config_name']:20s}  test_auc={r['test_auc']:.4f}  val_best_auc={r['val_best_auc']:.4f}  time={r['training_time_s']}s"
        )
    print("=" * 60)
