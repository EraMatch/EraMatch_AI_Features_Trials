"""Trial 2: SRM + ConvNeXt with SupCon — Modal A10G training script.

Main contribution: dual-branch SRM+RGB with cross-modal attention and
supervised contrastive regularization trained on modern diffusion-model fakes.

Three configs (budget-reduced ablation):
    2a_baseline     — RGB-only ConvNeXt (ablation baseline)
    2b_srm_concat   — SRM + RGB naive concat (no attention, no SupCon)
    2e_full_model   — SRM + attention + SupCon (full model)

Usage:
    modal run modal/train_nb2_srm.py

Requires pre-created Modal secrets and volumes:
    modal secret create kaggle-creds KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
    modal volume create avatar-data-vol
    modal volume create avatar-results-vol
    modal volume create avatar-model-cache
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
app = modal.App("avatar-srm-trial")

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
        "pytorch-metric-learning==2.8.1",
        "scipy==1.15.3",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "pillow==11.3.0",
        "albumentations==2.0.5",
        "matplotlib==3.10.3",
        "seaborn==0.13.2",
        "numpy==2.2.6",
        "umap-learn==0.5.7",
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
    """Verify all imports work and DualBranchSRM can be instantiated."""
    import torch
    import timm
    import pytorch_metric_learning
    import scipy
    import sklearn
    import albumentations

    from src.config import TRIAL2_CONFIGS, TRIAL2_HYPERPARAMS, IMG_SIZE
    from src.models.dual_branch import DualBranchSRM
    from src.models.losses import AvatarDetectionLoss
    from src.data.dataset import AvatarDataset
    from src.data.augmentations import get_train_transforms, get_val_transforms
    from src.utils.db import log_experiment, log_training_history, init_db
    from src.utils.metrics import compute_all_metrics
    from src.utils.viz import plot_training_curves, plot_confusion_matrix

    # Instantiate with pretrained=False for CPU (no GPU, no download needed)
    model = DualBranchSRM(
        use_srm=True,
        use_attention=True,
        use_supcon=True,
        pretrained=False,
    )
    n_params = sum(p.numel() for p in model.parameters())

    # Quick forward pass sanity check
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out["logits"].shape == (2, 2), f"Bad logits shape: {out['logits'].shape}"
    assert out["embedding"].shape == (2, 128), (
        f"Bad embedding shape: {out['embedding'].shape}"
    )

    # Verify loss function
    criterion = AvatarDetectionLoss(lambda_supcon=0.3, supcon_temperature=0.07)
    loss = criterion(out["logits"], out["embedding"], torch.tensor([0, 1]))
    assert loss.isfinite(), f"Non-finite loss: {loss.item()}"

    result = {
        "torch": torch.__version__,
        "timm": timm.__version__,
        "pytorch_metric_learning": pytorch_metric_learning.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "albumentations": albumentations.__version__,
        "n_configs": len(TRIAL2_CONFIGS),
        "img_size": IMG_SIZE,
        "hyperparams": TRIAL2_HYPERPARAMS,
        "model_params": n_params,
        "logits_shape": list(out["logits"].shape),
        "embedding_shape": list(out["embedding"].shape),
        "loss_value": loss.item(),
        "status": "OK",
    }
    print(f"[Smoke] validate_remote_imports PASSED: {result}")
    return result


# ---------------------------------------------------------------------------
# Quick test: 8-sample overfit (GPU A10G, 5 min)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    memory=24 * 1024,
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

    Creates synthetic data, runs 5 forward+backward passes with
    AvatarDetectionLoss, checks loss is finite and decreasing.
    Also tests checkpoint save/reload roundtrip.
    """
    import torch
    from torch.optim import AdamW

    from src.models.dual_branch import DualBranchSRM
    from src.models.losses import AvatarDetectionLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    images = torch.randn(8, 3, 256, 256, device=device)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)

    model = DualBranchSRM(
        use_srm=True,
        use_attention=True,
        use_supcon=True,
        pretrained=False,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = AvatarDetectionLoss(
        lambda_supcon=0.3, supcon_temperature=0.07, label_smoothing=0.05
    )

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out["logits"], out["embedding"], labels)
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

    # Checkpoint roundtrip
    ckpt_path = Path("/tmp/smoke_ckpt_srm.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "step": 5, "losses": losses},
        str(ckpt_path),
    )
    loaded = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    reloaded_model = DualBranchSRM(
        use_srm=True, use_attention=True, use_supcon=True, pretrained=False
    ).to(device)
    reloaded_model.load_state_dict(loaded["model_state_dict"])

    with torch.no_grad():
        orig_out = model(images)
        reload_out = reloaded_model(images)
    match = torch.allclose(orig_out["logits"], reload_out["logits"], atol=1e-5)

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
# Memory check: verify DualBranchSRM fits in A10G 24GB (GPU A10G, 2 min)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    memory=24 * 1024,
    timeout=2 * 60,
    volumes={
        str(DATA_ROOT): data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE: model_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def check_gpu_memory() -> dict:
    """Check peak GPU memory for each config at batch_size=64, img_size=256."""
    import torch

    from src.config import TRIAL2_CONFIGS
    from src.models.dual_branch import DualBranchSRM

    device = torch.device("cuda")
    results = {}
    SAFETY_LIMIT_GB = 22.0

    for config in TRIAL2_CONFIGS:
        name = config["name"]
        use_srm = config["use_srm"]
        use_attention = config["use_attention"]
        use_supcon = config["use_supcon"]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = DualBranchSRM(
            use_srm=use_srm,
            use_attention=use_attention,
            use_supcon=use_supcon,
            pretrained=True,
        ).to(device)

        # Forward pass at full batch size
        dummy = torch.randn(64, 3, 256, 256, device=device)
        with torch.no_grad():
            _ = model(dummy)

        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        results[name] = {
            "peak_memory_gb": round(peak_mem, 2),
            "fits": peak_mem < SAFETY_LIMIT_GB,
            "use_srm": use_srm,
            "use_attention": use_attention,
            "use_supcon": use_supcon,
        }
        print(
            f"[MemCheck] {name}: peak={peak_mem:.2f}GB  "
            f"fits={'YES' if peak_mem < SAFETY_LIMIT_GB else 'NO'}"
        )

        del model, dummy
        torch.cuda.empty_cache()

    # Abort check
    any_over = any(not r["fits"] for r in results.values())
    if any_over:
        over_configs = [n for n, r in results.items() if not r["fits"]]
        print(
            f"[MemCheck] ABORT: configs {over_configs} exceed {SAFETY_LIMIT_GB}GB safety limit"
        )
    else:
        print(f"[MemCheck] All configs fit within {SAFETY_LIMIT_GB}GB safety limit")

    results["status"] = "OK" if not any_over else "OVER_LIMIT"
    return results


# ---------------------------------------------------------------------------
# Training function: one config (GPU A10G, 4 hours)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    memory=24 * 1024,
    timeout=4 * 60 * 60,
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
    """Train a single Trial 2 config and save all artifacts.

    Reads config from TRIAL2_CONFIGS, trains on Dataset A (+ SFHQ sample
    for full model), logs to SQLite, saves plots, checkpoint, and embeddings.
    """
    import numpy as np
    import pandas as pd
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from torch.utils.data import DataLoader

    from src.config import (
        TRIAL2_CONFIGS,
        TRIAL2_HYPERPARAMS,
        IMG_SIZE,
        RESULTS_ROOT as SRC_RESULTS_ROOT,
    )
    from src.models.dual_branch import DualBranchSRM
    from src.models.losses import AvatarDetectionLoss
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
        plot_attention_maps,
    )

    # ------------------------------------------------------------------
    # Resolve config
    # ------------------------------------------------------------------
    config = None
    for c in TRIAL2_CONFIGS:
        if c["name"] == config_name:
            config = c
            break
    if config is None:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {[c['name'] for c in TRIAL2_CONFIGS]}"
        )

    use_srm = config["use_srm"]
    use_attention = config["use_attention"]
    use_supcon = config["use_supcon"]
    use_freq_aug = config["use_freq_aug"]

    # Hyperparams from central config
    hp = TRIAL2_HYPERPARAMS
    EPOCHS = hp["epochs"]
    BATCH_SIZE_TRAIN = hp["batch_size"]
    BATCH_SIZE_EVAL = hp["batch_size"]
    LR = hp["lr"]
    WEIGHT_DECAY = hp["weight_decay"]
    WARMUP_EPOCHS = hp["warmup_epochs"]
    LAMBDA_SUPCON = hp["lambda_supcon"]
    SUPCON_TEMP = hp["supcon_temperature"]
    LABEL_SMOOTHING = hp["label_smoothing"]
    DROPOUT = hp["dropout"]
    BACKBONE = hp["backbone"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Train] config={config_name}  use_srm={use_srm}  "
        f"use_attention={use_attention}  use_supcon={use_supcon}"
    )
    print(
        f"[Train] device={device}  epochs={EPOCHS}  batch={BATCH_SIZE_TRAIN}  backbone={BACKBONE}"
    )

    # ------------------------------------------------------------------
    # Init DB
    # ------------------------------------------------------------------
    init_db()

    # ------------------------------------------------------------------
    # Load datasets — Dataset A (130K) train/val/test
    # Per AGENTS.md: "Dataset: Full training corpus (130K + SFHQ-T2I 40K sample)"
    # For 2a and 2b (ablation baselines), train on A only.
    # For 2e (full model), train on A + SFHQ sample.
    # ------------------------------------------------------------------
    manifest_df = pd.read_csv(MANIFEST_PATH)

    # Filter to training datasets: always include A; for full model also include B_sfhq
    train_datasets_filter = ["A"]
    if config_name == "2e_full_model":
        train_datasets_filter.append("B_sfhq")

    train_df = manifest_df[
        (manifest_df["dataset"].isin(train_datasets_filter))
        & (manifest_df["split"] == "train")
    ].copy()
    val_df = manifest_df[
        (manifest_df["dataset"].isin(train_datasets_filter))
        & (manifest_df["split"] == "val")
    ].copy()
    # Test on Dataset A test split only (within-dataset evaluation)
    test_df = manifest_df[
        (manifest_df["dataset"] == "A") & (manifest_df["split"] == "test")
    ].copy()

    # Write filtered manifests
    train_manifest = DATA_ROOT / f"manifest_trial2_{config_name}_train.csv"
    val_manifest = DATA_ROOT / f"manifest_trial2_{config_name}_val.csv"
    test_manifest = DATA_ROOT / f"manifest_trial2_{config_name}_test.csv"
    train_df.to_csv(str(train_manifest), index=False)
    val_df.to_csv(str(val_manifest), index=False)
    test_df.to_csv(str(test_manifest), index=False)

    print(
        f"[Train] train_datasets={train_datasets_filter}  "
        f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}"
    )

    train_transform = get_train_transforms(img_size=IMG_SIZE)
    val_transform = get_val_transforms(img_size=IMG_SIZE)

    train_ds = AvatarDataset(
        manifest_path=str(train_manifest),
        split="train",
        transform=train_transform,
        img_size=IMG_SIZE,
    )
    val_ds = AvatarDataset(
        manifest_path=str(val_manifest),
        split="val",
        transform=val_transform,
        img_size=IMG_SIZE,
    )
    test_ds = AvatarDataset(
        manifest_path=str(test_manifest),
        split="test",
        transform=val_transform,
        img_size=IMG_SIZE,
    )

    n_train = len(train_ds)
    n_val = len(val_ds)
    n_test = len(test_ds)
    print(f"[Train] dataset sizes: train={n_train}  val={n_val}  test={n_test}")

    if n_train == 0:
        raise RuntimeError(f"No training samples found for config {config_name}")

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
    model = DualBranchSRM(
        use_srm=use_srm,
        use_attention=use_attention,
        use_supcon=use_supcon,
        backbone=BACKBONE,
        pretrained=True,
        dropout=DROPOUT,
    ).to(device)

    model_arch = "dual_branch_srm"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] model_arch={model_arch}  backbone={BACKBONE}  params={n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup + CosineAnnealing: 2 epochs linear warmup, then cosine
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    criterion = AvatarDetectionLoss(
        lambda_supcon=LAMBDA_SUPCON,
        supcon_temperature=SUPCON_TEMP,
        label_smoothing=LABEL_SMOOTHING,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": [], "lr": []}
    best_val_auc = 0.0
    best_epoch = 0
    start_time = time.time()

    artifacts_dir = RESULTS_ROOT / "trial2" / config_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    # Save full config for reproducibility
    full_config = {**config, **hp}
    (artifacts_dir / "config.json").write_text(json.dumps(full_config, indent=2))

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out["logits"], out["embedding"], labels)
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

                out = model(images)
                loss = criterion(out["logits"], out["embedding"], labels)
                val_losses.append(loss.item())

                probs = torch.softmax(out["logits"], dim=1)[:, 1]
                preds = out["logits"].argmax(dim=1)

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
                    "config": full_config,
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
        f"[Train] Training done. Best val_auc={best_val_auc:.4f} at epoch {best_epoch}  "
        f"time={training_time_s}s"
    )

    # ------------------------------------------------------------------
    # Test evaluation + embedding collection
    # ------------------------------------------------------------------
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_scores_list = []
    test_labels_list = []
    test_preds_list = []
    # Collect embeddings for UMAP (configs 2a and 2e)
    collect_embeddings = config_name in ("2a_baseline", "2e_full_model")
    test_embeddings_list = []
    test_datasets_list = []
    test_paths_list = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            out = model(images)
            probs = torch.softmax(out["logits"], dim=1)[:, 1]
            preds = out["logits"].argmax(dim=1)

            test_scores_list.append(probs.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())
            test_preds_list.append(preds.cpu().numpy())

            if collect_embeddings:
                test_embeddings_list.append(out["embedding"].cpu().numpy())
                test_datasets_list.extend(batch["dataset"])
                test_paths_list.extend(batch["path"])

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
    # Save embeddings for UMAP visualization
    # ------------------------------------------------------------------
    if collect_embeddings and test_embeddings_list:
        embeddings_array = np.concatenate(test_embeddings_list, axis=0)
        np.savez(
            str(artifacts_dir / "test_embeddings.npz"),
            embeddings=embeddings_array,
            labels=test_labels,
            datasets=np.array(test_datasets_list),
            paths=np.array(test_paths_list),
            scores=test_scores,
        )
        print(
            f"[Train] Saved {embeddings_array.shape[0]} embeddings "
            f"(dim={embeddings_array.shape[1]}) for UMAP"
        )

    # ------------------------------------------------------------------
    # Log experiment to SQLite
    # ------------------------------------------------------------------
    exp_id = log_experiment(
        trial_name=config_name,
        model_arch=model_arch,
        backbone=BACKBONE,
        train_datasets=",".join(train_datasets_filter),
        test_dataset="A_test",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        use_srm_branch=int(use_srm),
        use_supcon=int(use_supcon),
        use_freq_augmentation=int(use_freq_aug),
        use_attention_gate=int(use_attention),
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
        supcon_weight=LAMBDA_SUPCON if use_supcon else 0.0,
        checkpoint_path=str(ckpt_path),
        plots_dir=str(artifacts_dir),
        config_json=json.dumps(full_config),
        gpu_type="A10G",
        training_time_s=training_time_s,
        notes=(
            f"Trial 2 config. use_srm={use_srm}, use_attention={use_attention}, "
            f"use_supcon={use_supcon}. Best val AUC at epoch {best_epoch}. "
            f"lambda_supcon={LAMBDA_SUPCON}, temp={SUPCON_TEMP}, "
            f"label_smoothing={LABEL_SMOOTHING}, dropout={DROPOUT}."
        ),
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

    # Frequency heatmap: mean SRM residual for real vs fake
    _compute_and_save_srm_heatmap(model, test_loader, device, artifacts_dir, use_srm)

    # Attention map visualization (only if attention is enabled)
    if use_attention and use_srm:
        _compute_and_save_attention_maps(model, test_loader, device, artifacts_dir)

    # Grad-CAM on 4 samples (2 real, 2 fake)
    _compute_and_save_gradcam(model, test_loader, device, artifacts_dir)

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


def _compute_and_save_srm_heatmap(model, test_loader, device, artifacts_dir, use_srm):
    """Compute mean |SRM residual| for real vs fake, save heatmap."""
    import numpy as np
    import torch

    from src.utils.viz import plot_frequency_heatmap

    if not use_srm:
        print("[Train] Skipping SRM heatmap: SRM branch disabled")
        return

    real_residuals = []
    fake_residuals = []
    n_samples = 0
    max_samples = 200

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if n_samples >= max_samples:
                break
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()

            for i in range(images.shape[0]):
                if n_samples >= max_samples:
                    break
                # Get SRM residual from the model's internal SRM branch
                img_single = images[i : i + 1]
                if model.srm_branch is not None:
                    residual = model.srm_branch(img_single)
                    # Average across 3 filter channels for a 2D map
                    residual_map = residual.mean(dim=1).squeeze().cpu().numpy()
                else:
                    continue

                if labels[i] == 0:
                    real_residuals.append(np.abs(residual_map))
                else:
                    fake_residuals.append(np.abs(residual_map))
                n_samples += 1

    if real_residuals and fake_residuals:
        real_mean = np.mean(real_residuals, axis=0)
        fake_mean = np.mean(fake_residuals, axis=0)
        plot_frequency_heatmap(
            real_mean, fake_mean, save_path=artifacts_dir / "srm_heatmap.png"
        )
    else:
        print("[Train] Skipping SRM heatmap: not enough real/fake samples")


def _compute_and_save_attention_maps(model, test_loader, device, artifacts_dir):
    """Generate attention map overlay visualizations."""
    import torch

    from src.utils.viz import plot_attention_maps

    samples_images = []
    samples_attention = []
    n_real = 0
    n_fake = 0
    target_per_class = 2

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if n_real >= target_per_class and n_fake >= target_per_class:
                break
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()

            out = model(images)
            attention = out["attention"]

            for i in range(images.shape[0]):
                if labels[i] == 0 and n_real < target_per_class:
                    samples_images.append(images[i : i + 1])
                    samples_attention.append(attention[i : i + 1])
                    n_real += 1
                elif labels[i] == 1 and n_fake < target_per_class:
                    samples_images.append(images[i : i + 1])
                    samples_attention.append(attention[i : i + 1])
                    n_fake += 1

                if n_real >= target_per_class and n_fake >= target_per_class:
                    break

    if samples_images:
        all_images = torch.cat(samples_images, dim=0)
        all_attention = torch.cat(samples_attention, dim=0)
        plot_attention_maps(
            all_images, all_attention, save_path=artifacts_dir / "attention_maps.png"
        )
    else:
        print("[Train] Skipping attention maps: no samples found")


def _compute_and_save_gradcam(model, test_loader, device, artifacts_dir):
    """Generate gradient-based CAM visualizations for 2 real + 2 fake samples."""
    import numpy as np
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.viz import _DPI

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

    fig, axes = plt.subplots(len(samples), 2, figsize=(8, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    for idx, (img_tensor, label_str) in enumerate(zip(samples, sample_labels)):
        img_tensor = img_tensor.clone().requires_grad_(True)

        out = model(img_tensor)
        score = out["logits"][0, 1]  # fake class logit
        score.backward()

        gradients = img_tensor.grad[0].cpu().numpy()
        grayscale_grad = np.mean(np.abs(gradients), axis=0)

        if grayscale_grad.max() > 0:
            grayscale_grad = grayscale_grad / grayscale_grad.max()

        img_display = img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
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
# Local entrypoint: smoke → smoke training → memory check → train configs
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    """Run all Trial 2 configs in sequence: smoke tests then training."""
    from src.config import TRIAL2_CONFIGS

    print("=" * 60)
    print("Trial 2: SRM + ConvNeXt with SupCon (Main Contribution)")
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

    # Step 3: GPU memory check
    print("\n[MemCheck] Verifying GPU memory fits...")
    mem_result = check_gpu_memory.remote()
    print(f"[MemCheck] Memory check: {mem_result}")

    if mem_result.get("status") != "OK":
        print("[FATAL] Memory check failed — aborting full training.")
        return

    # Step 4: Train each config sequentially
    results = []
    for config in TRIAL2_CONFIGS:
        config_name = config["name"]
        print(f"\n[Train] Starting config: {config_name} (estimated 3-4 hours)...")
        result = train_config.remote(config_name)
        results.append(result)
        print(
            f"[Train] Completed: {config_name} -> AUC={result.get('test_auc', 'N/A'):.4f}"
        )

    # Summary
    print("\n" + "=" * 60)
    print("Trial 2 Summary")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['config_name']:20s}  test_auc={r['test_auc']:.4f}  "
            f"val_best_auc={r['val_best_auc']:.4f}  time={r['training_time_s']}s"
        )
    print("=" * 60)
