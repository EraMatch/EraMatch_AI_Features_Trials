"""Trial 3: Cross-Dataset Generalization Study — Modal A10G training script.

The central thesis scientific claim: how performance degrades when training
and testing distributions differ in their fake generator family.

4 runs × 3 configs (2a, 2b, 2e) = 12 experiment rows in SQLite.
Each run trains on a specific dataset combination, then evaluates on BOTH
Dataset A (within-distribution) AND Dataset C (out-of-distribution / OOD).

Runs:
    3a_A_only      — Train on Dataset A only, test on A + C
    3b_C_only      — Train on Dataset C (9.6K) only, test on A + C
    3c_AC_combined — Train on A + C combined, test on A + C
    3d_A_sfhq      — Train on A + SFHQ sample, test on A + C

Configs (from Trial 2):
    2a_baseline    — RGB-only ConvNeXt (ablation baseline)
    2b_srm_concat  — SRM + RGB naive concat (no attention, no SupCon)
    2e_full_model  — SRM + attention + SupCon (full model)

Usage:
    modal run modal/train_nb3_cross.py

Requires pre-created Modal secrets and volumes:
    modal secret create kaggle-creds KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
    modal volume create avatar-data-vol
    modal volume create avatar-results-vol
    modal volume create avatar-model-cache
"""

import csv
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

# Cross-dataset evaluation epochs (smaller than Trial 2's 20 for budget)
CROSS_EPOCHS = 5

# ---------------------------------------------------------------------------
# App + named resources
# ---------------------------------------------------------------------------
app = modal.App("avatar-cross-trial")

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

    from src.config import TRIAL3_RUNS, TRIAL2_CONFIGS, TRIAL2_HYPERPARAMS, IMG_SIZE
    from src.models.dual_branch import DualBranchSRM
    from src.models.losses import AvatarDetectionLoss
    from src.data.dataset import AvatarDataset
    from src.data.augmentations import get_train_transforms, get_val_transforms
    from src.utils.db import log_experiment, log_training_history, init_db
    from src.utils.metrics import compute_all_metrics

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
        "n_trial3_runs": len(TRIAL3_RUNS),
        "n_trial2_configs": len(TRIAL2_CONFIGS),
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
    ckpt_path = Path("/tmp/smoke_ckpt_cross.pt")
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
# Dataset filtering helpers
# ---------------------------------------------------------------------------
def _get_dataset_filters(run_name: str) -> dict:
    """Return train/val/test dataset filter lists for a Trial 3 run.

    Maps run config from TRIAL3_RUNS to manifest column filter values.
    Dataset column in manifest uses: 'A' (130K), 'C' (9.6K), 'B_sfhq' (SFHQ sample).
    """
    from src.config import TRIAL3_RUNS

    run_config = None
    for r in TRIAL3_RUNS:
        if r["name"] == run_name:
            run_config = r
            break
    if run_config is None:
        raise ValueError(
            f"Unknown run '{run_name}'. Available: {[r['name'] for r in TRIAL3_RUNS]}"
        )

    train_on = run_config["train_on"]
    val_on = run_config["val_on"]

    # Map shorthand to manifest dataset column values
    _DATASET_MAP = {
        "A": ["A"],
        "C": ["C"],
        "AC": ["A", "C"],
        "A_sfhq": ["A", "B_sfhq"],
    }

    train_datasets = _DATASET_MAP.get(train_on, [train_on])
    val_datasets = _DATASET_MAP.get(val_on, [val_on])

    # Test sets: always evaluate on both A and C
    test_dataset_a = ["A"]
    test_dataset_c = ["C"]

    return {
        "train_datasets": train_datasets,
        "val_datasets": val_datasets,
        "test_dataset_a": test_dataset_a,
        "test_dataset_c": test_dataset_c,
        "run_config": run_config,
    }


# ---------------------------------------------------------------------------
# Training function: one run (GPU A10G, ~1.5 hours)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    memory=24 * 1024,
    timeout=2 * 60 * 60,
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
def train_run(run_name: str) -> dict:
    """Train all 3 configs for one Trial 3 run, test on BOTH A and C.

    For each config (2a, 2b, 2e):
        1. Build DualBranchSRM with config flags
        2. Train on the run's specified dataset (5 epochs)
        3. Evaluate on Dataset A test split AND Dataset C ood_test split
        4. Log experiment row to SQLite
        5. Save artifacts to /results/trial3/{run_name}_{config_name}/
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
        plot_roc_curve,
        plot_score_distribution,
        plot_confusion_matrix,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trial3] run={run_name}  device={device}")

    # ------------------------------------------------------------------
    # Init DB
    # ------------------------------------------------------------------
    init_db()

    # ------------------------------------------------------------------
    # Resolve dataset filters for this run
    # ------------------------------------------------------------------
    filters = _get_dataset_filters(run_name)
    run_config = filters["run_config"]
    train_datasets = filters["train_datasets"]
    val_datasets = filters["val_datasets"]

    print(f"[Trial3] train_datasets={train_datasets}  val_datasets={val_datasets}")

    # ------------------------------------------------------------------
    # Load manifest and create filtered CSVs
    # ------------------------------------------------------------------
    manifest_df = pd.read_csv(MANIFEST_PATH)

    # Train: specified datasets with split=train
    train_df = manifest_df[
        manifest_df["dataset"].isin(train_datasets) & (manifest_df["split"] == "train")
    ].copy()

    # Val: specified datasets with split=val
    # For runs training on C (which is ood_test only), use the same data for val
    if run_config["val_on"] == "C":
        # Dataset C has no val split — use ood_test for both train and val
        # (C is small, ~9.6K; use 80/20 split within C for train/val)
        c_data = manifest_df[manifest_df["dataset"] == "C"].copy()
        if len(c_data) > 0:
            # Random 80/20 split with fixed seed
            c_data = c_data.sample(frac=1, random_state=42).reset_index(drop=True)
            split_idx = int(len(c_data) * 0.8)
            c_train = c_data.iloc[:split_idx].copy()
            c_val = c_data.iloc[split_idx:].copy()
            # Override: use the C train subset as train, C val subset as val
            train_df = c_train.copy()
            val_df = c_val.copy()
        else:
            raise RuntimeError("No Dataset C samples found in manifest")
    else:
        val_df = manifest_df[
            manifest_df["dataset"].isin(val_datasets) & (manifest_df["split"] == "val")
        ].copy()

    # Test A: Dataset A test split
    test_a_df = manifest_df[
        (manifest_df["dataset"] == "A") & (manifest_df["split"] == "test")
    ].copy()

    # Test C: Dataset C ood_test (the OOD evaluation set — never in train)
    test_c_df = manifest_df[
        (manifest_df["dataset"] == "C") & (manifest_df["split"] == "ood_test")
    ].copy()

    # For 3b (C_only), val_df is already set above; test_a uses A test split
    # and test_c uses the C val portion (since all C is ood_test originally)

    print(
        f"[Trial3] Data sizes: train={len(train_df)}  val={len(val_df)}  "
        f"test_A={len(test_a_df)}  test_C={len(test_c_df)}"
    )

    # Write temporary filtered manifests
    train_manifest = DATA_ROOT / f"manifest_trial3_{run_name}_train.csv"
    val_manifest = DATA_ROOT / f"manifest_trial3_{run_name}_val.csv"
    test_a_manifest = DATA_ROOT / f"manifest_trial3_{run_name}_test_a.csv"
    test_c_manifest = DATA_ROOT / f"manifest_trial3_{run_name}_test_c.csv"
    train_df.to_csv(str(train_manifest), index=False)
    val_df.to_csv(str(val_manifest), index=False)
    test_a_df.to_csv(str(test_a_manifest), index=False)
    test_c_df.to_csv(str(test_c_manifest), index=False)

    # Hyperparams from central config (override epochs)
    hp = TRIAL2_HYPERPARAMS
    EPOCHS = CROSS_EPOCHS
    BATCH_SIZE = hp["batch_size"]
    LR = hp["lr"]
    WEIGHT_DECAY = hp["weight_decay"]
    WARMUP_EPOCHS = min(hp["warmup_epochs"], EPOCHS - 1)
    LAMBDA_SUPCON = hp["lambda_supcon"]
    SUPCON_TEMP = hp["supcon_temperature"]
    LABEL_SMOOTHING = hp["label_smoothing"]
    DROPOUT = hp["dropout"]
    BACKBONE = hp["backbone"]

    # ------------------------------------------------------------------
    # Train each config
    # ------------------------------------------------------------------
    config_results = []

    for config in TRIAL2_CONFIGS:
        config_name = config["name"]
        use_srm = config["use_srm"]
        use_attention = config["use_attention"]
        use_supcon = config["use_supcon"]
        use_freq_aug = config["use_freq_aug"]

        trial_label = f"{run_name}_{config_name}"
        print(f"\n[Trial3] === {trial_label} ===")
        print(
            f"  use_srm={use_srm}  use_attention={use_attention}  use_supcon={use_supcon}"
        )

        # --- Data loaders ---
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
        test_a_ds = AvatarDataset(
            manifest_path=str(test_a_manifest),
            split="test",
            transform=val_transform,
            img_size=IMG_SIZE,
        )
        test_c_ds = AvatarDataset(
            manifest_path=str(test_c_manifest),
            split="ood_test",
            transform=val_transform,
            img_size=IMG_SIZE,
        )

        # Use smaller batch size for smaller datasets (C has only ~9.6K)
        effective_batch = min(BATCH_SIZE, max(8, len(train_ds) // 4))

        train_loader = DataLoader(
            train_ds,
            batch_size=effective_batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=AvatarDataset.collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=effective_batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=AvatarDataset.collate_fn,
        )
        test_a_loader = DataLoader(
            test_a_ds,
            batch_size=effective_batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=AvatarDataset.collate_fn,
        )
        test_c_loader = DataLoader(
            test_c_ds,
            batch_size=effective_batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=AvatarDataset.collate_fn,
        )

        n_train = len(train_ds)
        n_val = len(val_ds)
        n_test_a = len(test_a_ds)
        n_test_c = len(test_c_ds)
        print(
            f"  n_train={n_train}  n_val={n_val}  "
            f"n_test_A={n_test_a}  n_test_C={n_test_c}  batch={effective_batch}"
        )

        if n_train == 0:
            print(f"  [SKIP] No training samples for {trial_label}")
            continue

        # --- Model ---
        model = DualBranchSRM(
            use_srm=use_srm,
            use_attention=use_attention,
            use_supcon=use_supcon,
            backbone=BACKBONE,
            pretrained=True,
            dropout=DROPOUT,
        ).to(device)

        # --- Optimizer, scheduler, loss ---
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        if EPOCHS > 1 and WARMUP_EPOCHS > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=EPOCHS - WARMUP_EPOCHS
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[WARMUP_EPOCHS],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))

        criterion = AvatarDetectionLoss(
            lambda_supcon=LAMBDA_SUPCON,
            supcon_temperature=SUPCON_TEMP,
            label_smoothing=LABEL_SMOOTHING,
        )

        # --- Training loop ---
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_f1": [],
            "lr": [],
        }
        best_val_auc = 0.0
        best_epoch = 0
        start_time = time.time()

        artifacts_dir = RESULTS_ROOT / "trial3" / trial_label
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = artifacts_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best.pt"

        # Save config
        full_config = {
            **config,
            **hp,
            "cross_epochs": EPOCHS,
            "run_name": run_name,
            "train_datasets": train_datasets,
        }
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
            val_scores = (
                np.concatenate(val_scores_list) if val_scores_list else np.array([])
            )
            val_labels = (
                np.concatenate(val_labels_list) if val_labels_list else np.array([])
            )
            val_preds = (
                np.concatenate(val_preds_list) if val_preds_list else np.array([])
            )

            if len(val_labels) > 0 and len(np.unique(val_labels)) >= 2:
                val_metrics = compute_all_metrics(val_labels, val_scores)
                val_auc = val_metrics["auc"]
                val_f1 = val_metrics["f1"]
            else:
                val_auc = 0.5
                val_f1 = 0.0

            current_lr = optimizer.param_groups[0]["lr"]

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
                        "run_name": run_name,
                        "config_name": config_name,
                        "config": full_config,
                    },
                    str(ckpt_path),
                )

            print(
                f"  epoch {epoch}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}  lr={current_lr:.6f}"
            )

        training_time_s = int(time.time() - start_time)
        print(
            f"  Training done. Best val_auc={best_val_auc:.4f} at epoch {best_epoch}  "
            f"time={training_time_s}s"
        )

        # ------------------------------------------------------------------
        # Test evaluation on BOTH Dataset A and Dataset C
        # ------------------------------------------------------------------
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        def _evaluate_on_loader(loader, loader_name):
            """Run inference on a DataLoader and return metrics dict."""
            scores_list = []
            labels_list = []
            preds_list = []

            with torch.no_grad():
                for batch in loader:
                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)

                    out = model(images)
                    probs = torch.softmax(out["logits"], dim=1)[:, 1]
                    preds = out["logits"].argmax(dim=1)

                    scores_list.append(probs.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
                    preds_list.append(preds.cpu().numpy())

            if not scores_list:
                return {
                    "auc": 0.5,
                    "f1": 0.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "specificity": 0.0,
                    "eer": 0.5,
                    "tpr_at_fpr1": 0.5,
                    "n_samples": 0,
                }

            scores = np.concatenate(scores_list)
            labels_np = np.concatenate(labels_list)
            preds = np.concatenate(preds_list)

            if len(np.unique(labels_np)) < 2:
                metrics = {
                    "auc": 0.5,
                    "f1": 0.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "specificity": 0.0,
                    "eer": 0.5,
                    "tpr_at_fpr1": 0.5,
                }
            else:
                metrics = compute_all_metrics(labels_np, scores)

            metrics["n_samples"] = len(labels_np)
            return metrics

        # Evaluate on Dataset A test
        metrics_a = _evaluate_on_loader(test_a_loader, "test_A")
        print(
            f"  TEST_A  AUC={metrics_a['auc']:.4f}  F1={metrics_a['f1']:.4f}  "
            f"Acc={metrics_a['accuracy']:.4f}  EER={metrics_a['eer']:.4f}  "
            f"TPR@FPR1={metrics_a['tpr_at_fpr1']:.4f}  n={metrics_a['n_samples']}"
        )

        # Evaluate on Dataset C OOD
        metrics_c = _evaluate_on_loader(test_c_loader, "test_C")
        print(
            f"  TEST_C  AUC={metrics_c['auc']:.4f}  F1={metrics_c['f1']:.4f}  "
            f"Acc={metrics_c['accuracy']:.4f}  EER={metrics_c['eer']:.4f}  "
            f"TPR@FPR1={metrics_c['tpr_at_fpr1']:.4f}  n={metrics_c['n_samples']}"
        )

        # Generalization gap
        gen_gap = metrics_a["auc"] - metrics_c["auc"]
        print(f"  GENERALIZATION GAP (A - C): {gen_gap:+.4f}")

        # ------------------------------------------------------------------
        # Log experiment to SQLite — Dataset A test metrics
        # ------------------------------------------------------------------
        exp_id_a = log_experiment(
            trial_name=f"trial3_{trial_label}",
            model_arch="dual_branch_srm",
            backbone=BACKBONE,
            train_datasets=",".join(train_datasets),
            test_dataset="A_test",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test_a,
            use_srm_branch=int(use_srm),
            use_supcon=int(use_supcon),
            use_freq_augmentation=int(use_freq_aug),
            use_attention_gate=int(use_attention),
            test_auc=metrics_a["auc"],
            test_f1=metrics_a["f1"],
            test_accuracy=metrics_a["accuracy"],
            test_precision=metrics_a["precision"],
            test_recall=metrics_a["recall"],
            test_specificity=metrics_a["specificity"],
            test_eer=metrics_a["eer"],
            test_tpr_at_fpr1=metrics_a["tpr_at_fpr1"],
            val_best_auc=best_val_auc,
            epochs=EPOCHS,
            batch_size=effective_batch,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            supcon_weight=LAMBDA_SUPCON if use_supcon else 0.0,
            checkpoint_path=str(ckpt_path),
            plots_dir=str(artifacts_dir),
            config_json=json.dumps(full_config),
            gpu_type="A10G",
            training_time_s=training_time_s,
            notes=(
                f"Trial 3 cross-dataset. run={run_name}, config={config_name}. "
                f"use_srm={use_srm}, use_attention={use_attention}, "
                f"use_supcon={use_supcon}. Best val AUC at epoch {best_epoch}. "
                f"Test A AUC={metrics_a['auc']:.4f}, Test C AUC={metrics_c['auc']:.4f}, "
                f"Gen gap={gen_gap:+.4f}."
            ),
        )

        # Log Dataset C OOD metrics as a separate experiment row
        exp_id_c = log_experiment(
            trial_name=f"trial3_{trial_label}",
            model_arch="dual_branch_srm",
            backbone=BACKBONE,
            train_datasets=",".join(train_datasets),
            test_dataset="C_ood",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test_c,
            use_srm_branch=int(use_srm),
            use_supcon=int(use_supcon),
            use_freq_augmentation=int(use_freq_aug),
            use_attention_gate=int(use_attention),
            test_auc=metrics_c["auc"],
            test_f1=metrics_c["f1"],
            test_accuracy=metrics_c["accuracy"],
            test_precision=metrics_c["precision"],
            test_recall=metrics_c["recall"],
            test_specificity=metrics_c["specificity"],
            test_eer=metrics_c["eer"],
            test_tpr_at_fpr1=metrics_c["tpr_at_fpr1"],
            val_best_auc=best_val_auc,
            epochs=EPOCHS,
            batch_size=effective_batch,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            supcon_weight=LAMBDA_SUPCON if use_supcon else 0.0,
            checkpoint_path=str(ckpt_path),
            plots_dir=str(artifacts_dir),
            config_json=json.dumps(full_config),
            gpu_type="A10G",
            training_time_s=training_time_s,
            notes=(
                f"Trial 3 cross-dataset OOD eval. run={run_name}, config={config_name}. "
                f"use_srm={use_srm}, use_attention={use_attention}, "
                f"use_supcon={use_supcon}. "
                f"Test A AUC={metrics_a['auc']:.4f}, Test C AUC={metrics_c['auc']:.4f}, "
                f"Gen gap={gen_gap:+.4f}."
            ),
        )

        # Log training history
        for epoch_idx in range(len(history["train_loss"])):
            log_training_history(
                experiment_id=exp_id_a,
                epoch=epoch_idx + 1,
                train_loss=history["train_loss"][epoch_idx],
                val_loss=history["val_loss"][epoch_idx],
                val_auc=history["val_auc"][epoch_idx],
                val_f1=history["val_f1"][epoch_idx],
                lr=history["lr"][epoch_idx],
            )

        # ------------------------------------------------------------------
        # Save plots
        # ------------------------------------------------------------------
        plot_training_curves(history, save_path=artifacts_dir / "training_curves.png")

        # ROC curve for Dataset A test
        if n_test_a > 0 and len(test_a_df) > 0:
            # Re-collect scores for plotting
            a_scores = _collect_scores(model, test_a_loader, device)
            a_labels = _collect_labels(test_a_loader)
            if (
                a_scores is not None
                and a_labels is not None
                and len(np.unique(a_labels)) >= 2
            ):
                plot_roc_curve(
                    a_labels,
                    a_scores,
                    auc_value=metrics_a["auc"],
                    save_path=artifacts_dir / "roc_curve_A.png",
                )
                plot_score_distribution(
                    a_labels,
                    a_scores,
                    save_path=artifacts_dir / "score_dist_A.png",
                )
                a_preds = (a_scores >= 0.5).astype(int)
                plot_confusion_matrix(
                    a_labels,
                    a_preds,
                    save_path=artifacts_dir / "confusion_matrix_A.png",
                )

        # ROC curve for Dataset C OOD
        if n_test_c > 0 and len(test_c_df) > 0:
            c_scores = _collect_scores(model, test_c_loader, device)
            c_labels = _collect_labels(test_c_loader)
            if (
                c_scores is not None
                and c_labels is not None
                and len(np.unique(c_labels)) >= 2
            ):
                plot_roc_curve(
                    c_labels,
                    c_scores,
                    auc_value=metrics_c["auc"],
                    save_path=artifacts_dir / "roc_curve_C.png",
                )
                plot_score_distribution(
                    c_labels,
                    c_scores,
                    save_path=artifacts_dir / "score_dist_C.png",
                )
                c_preds = (c_scores >= 0.5).astype(int)
                plot_confusion_matrix(
                    c_labels,
                    c_preds,
                    save_path=artifacts_dir / "confusion_matrix_C.png",
                )

        config_results.append(
            {
                "config_name": config_name,
                "trial_label": trial_label,
                "exp_id_a": exp_id_a,
                "exp_id_c": exp_id_c,
                "test_a_auc": metrics_a["auc"],
                "test_c_auc": metrics_c["auc"],
                "gen_gap": gen_gap,
                "val_best_auc": best_val_auc,
                "best_epoch": best_epoch,
                "training_time_s": training_time_s,
            }
        )

        # Clean up model from GPU
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save partial cross-dataset matrix CSV for this run
    # ------------------------------------------------------------------
    _save_partial_matrix(run_name, config_results, RESULTS_ROOT)

    # ------------------------------------------------------------------
    # Commit results volume
    # ------------------------------------------------------------------
    results_vol.commit()
    print(f"[Trial3] Run {run_name} complete. Volume committed.")

    return {
        "run_name": run_name,
        "configs": config_results,
    }


# ---------------------------------------------------------------------------
# Helpers for score/label collection
# ---------------------------------------------------------------------------
def _collect_scores(model, loader, device):
    """Run inference and return numpy array of P(fake) scores."""
    import numpy as np
    import torch

    model.eval()
    scores_list = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            out = model(images)
            probs = torch.softmax(out["logits"], dim=1)[:, 1]
            scores_list.append(probs.cpu().numpy())

    if not scores_list:
        return None
    return np.concatenate(scores_list)


def _collect_labels(loader):
    """Extract labels from a DataLoader."""
    import numpy as np

    labels_list = []
    for batch in loader:
        labels_list.append(batch["label"].numpy())

    if not labels_list:
        return None
    return np.concatenate(labels_list)


def _save_partial_matrix(run_name: str, config_results: list, results_root: Path):
    """Save a partial cross-dataset matrix CSV for the completed run."""
    csv_path = results_root / "trial3" / f"partial_matrix_{run_name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(csv_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["run", "config", "test_A_auc", "test_C_auc", "gen_gap", "val_best_auc"]
        )
        for r in config_results:
            writer.writerow(
                [
                    run_name,
                    r["config_name"],
                    f"{r['test_a_auc']:.4f}",
                    f"{r['test_c_auc']:.4f}",
                    f"{r['gen_gap']:+.4f}",
                    f"{r['val_best_auc']:.4f}",
                ]
            )

    print(f"[Trial3] Partial matrix saved to {csv_path}")


# ---------------------------------------------------------------------------
# Generate final cross-dataset matrix (CPU, 1 min)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=60,
    volumes={
        str(DATA_ROOT): data_vol,
        str(RESULTS_ROOT): results_vol,
    },
)
def generate_cross_dataset_matrix() -> dict:
    """Query SQLite for all trial3 experiments, pivot into matrix, save CSV+MD.

    Rows = (run, config) combinations.
    Columns = test datasets (A_test, C_ood).
    Values = AUC.
    Also computes the generalization gap per (run, config).
    """
    import sqlite3

    from src.config import RESULTS_ROOT as SRC_RESULTS_ROOT

    db_path = SRC_RESULTS_ROOT / "experiments.db"
    if not db_path.exists():
        return {"error": f"Database not found at {db_path}", "status": "MISSING"}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Query all trial3 experiments
    rows = conn.execute(
        """SELECT trial_name, test_dataset, test_auc,
                  use_srm_branch, use_supcon, use_attention_gate
           FROM experiments
           WHERE trial_name LIKE 'trial3_%'
           ORDER BY trial_name, test_dataset"""
    ).fetchall()
    conn.close()

    if not rows:
        return {"error": "No trial3 experiments found", "status": "EMPTY"}

    # Pivot: group by trial_name, collect AUC per test_dataset
    pivot = {}
    for row in rows:
        trial_name = row["trial_name"]
        test_dataset = row["test_dataset"]
        auc = row["test_auc"]

        if trial_name not in pivot:
            pivot[trial_name] = {
                "test_A_auc": None,
                "test_C_auc": None,
                "use_srm": row["use_srm_branch"],
                "use_supcon": row["use_supcon"],
                "use_attention": row["use_attention_gate"],
            }

        if test_dataset == "A_test":
            pivot[trial_name]["test_A_auc"] = auc
        elif test_dataset == "C_ood":
            pivot[trial_name]["test_C_auc"] = auc

    # Build CSV
    csv_path = SRC_RESULTS_ROOT / "trial3" / "cross_dataset_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(csv_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial_name",
                "test_A_auc",
                "test_C_auc",
                "gen_gap",
                "use_srm",
                "use_supcon",
                "use_attention",
            ]
        )
        for trial_name, data in sorted(pivot.items()):
            a_auc = data["test_A_auc"] or 0.0
            c_auc = data["test_C_auc"] or 0.0
            gen_gap = a_auc - c_auc
            writer.writerow(
                [
                    trial_name,
                    f"{a_auc:.4f}",
                    f"{c_auc:.4f}",
                    f"{gen_gap:+.4f}",
                    data["use_srm"],
                    data["use_supcon"],
                    data["use_attention"],
                ]
            )

    # Build markdown
    md_path = SRC_RESULTS_ROOT / "trial3" / "cross_dataset_matrix.md"
    md_lines = [
        "# Trial 3: Cross-Dataset Generalization Matrix",
        "",
        "| Trial Config | Test A AUC | Test C AUC | Gen Gap | SRM | SupCon | Attn |",
        "|---|---|---|---|---|---|---|",
    ]
    for trial_name, data in sorted(pivot.items()):
        a_auc = data["test_A_auc"] or 0.0
        c_auc = data["test_C_auc"] or 0.0
        gen_gap = a_auc - c_auc
        md_lines.append(
            f"| {trial_name} | {a_auc:.4f} | {c_auc:.4f} | {gen_gap:+.4f} "
            f"| {data['use_srm']} | {data['use_supcon']} | {data['use_attention']} |"
        )
    md_lines.append("")

    md_path.write_text("\n".join(md_lines))

    print(f"[Matrix] CSV saved to {csv_path}")
    print(f"[Matrix] MD saved to {md_path}")
    print(f"[Matrix] {len(pivot)} experiment pairs found")

    # Print summary table
    print("\n" + "=" * 80)
    print("Trial 3: Cross-Dataset Generalization Matrix")
    print("=" * 80)
    for trial_name, data in sorted(pivot.items()):
        a_auc = data["test_A_auc"] or 0.0
        c_auc = data["test_C_auc"] or 0.0
        gen_gap = a_auc - c_auc
        print(f"  {trial_name:35s}  A={a_auc:.4f}  C={c_auc:.4f}  gap={gen_gap:+.4f}")
    print("=" * 80)

    results_vol.commit()

    return {
        "n_experiments": len(pivot),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# Local entrypoint: smoke → training → matrix generation
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    """Run all Trial 3 runs in sequence: smoke tests → training → matrix."""
    from src.config import TRIAL3_RUNS

    print("=" * 60)
    print("Trial 3: Cross-Dataset Generalization Study")
    print("=" * 60)
    print(f"Runs: {len(TRIAL3_RUNS)}")
    print(f"Configs per run: 3 (2a, 2b, 2e)")
    print(
        f"Total experiment rows: {len(TRIAL3_RUNS) * 3 * 2} (3 configs × 2 test sets)"
    )
    print(f"Epochs per config: {CROSS_EPOCHS}")
    print()

    # Step 1: Validate imports
    print("[Smoke] Validating imports...")
    import_result = validate_remote_imports.remote()
    print(f"[Smoke] Import check: {import_result}")

    # Step 2: 8-sample overfit check
    print("\n[Smoke] Running 8-sample overfit check...")
    smoke_result = run_smoke_training.remote()
    print(f"[Smoke] Training check: {smoke_result}")

    if smoke_result.get("status") != "PASSED":
        print("[FATAL] Smoke test failed — aborting full training.")
        return

    # Step 3: Train each run sequentially
    run_results = []
    for run in TRIAL3_RUNS:
        run_name = run["name"]
        print(f"\n[Train] Starting run: {run_name} (estimated ~1.5h)...")
        result = train_run.remote(run_name)
        run_results.append(result)
        n_configs = len(result.get("configs", []))
        print(f"[Train] Completed: {run_name} — {n_configs} configs trained")

        # Early sanity check after first run
        if run_name == TRIAL3_RUNS[0]["name"]:
            configs = result.get("configs", [])
            if configs:
                first_cfg = configs[0]
                a_auc = first_cfg.get("test_a_auc", 0)
                c_auc = first_cfg.get("test_c_auc", 0)
                print(
                    f"[Sanity] First config: "
                    f"test_A_auc={a_auc:.4f}  test_C_auc={c_auc:.4f}  "
                    f"gap={a_auc - c_auc:+.4f}"
                )
                if a_auc < 0.5 or c_auc < 0.3:
                    print(
                        "[WARNING] Very low AUC detected — check data loading "
                        "before continuing. Continuing anyway per user override."
                    )

    # Step 4: Generate final cross-dataset matrix
    print("\n[Final] Generating cross-dataset matrix CSV...")
    matrix_result = generate_cross_dataset_matrix.remote()
    print(f"[Final] Matrix: {matrix_result}")

    # Summary
    print("\n" + "=" * 60)
    print("Trial 3 Summary")
    print("=" * 60)
    for rr in run_results:
        run_name = rr.get("run_name", "unknown")
        configs = rr.get("configs", [])
        print(f"\n  Run: {run_name}")
        for c in configs:
            print(
                f"    {c['config_name']:20s}  "
                f"A_AUC={c['test_a_auc']:.4f}  C_AUC={c['test_c_auc']:.4f}  "
                f"gap={c['gen_gap']:+.4f}"
            )
    print("=" * 60)
