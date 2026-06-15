"""Trial 4: Video-Level Aggregation Simulation — Modal T4 inference script.

Loads the best Trial 2 checkpoint (config 2e, full model) and evaluates
3 temporal aggregation strategies on simulated video clips:

1. mean_score       — Baseline: P(ai) = mean(frame_scores)
2. variance_gated    — Flags eerily consistent (low std) sequences
3. temporal_drift    — Flags non-decreasing score trajectories

Since we have static images (not video), temporal consistency is simulated by:
- Real faces: slight augmentation per frame (jitter, blur) → HIGH score variance
- AI faces: no augmentation or brightness-only shift → LOW score variance

Usage:
    modal run modal/train_nb4_video.py

Requires pre-created Modal volumes:
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

BEST_CHECKPOINT = RESULTS_ROOT / "trial2" / "2e_full_model" / "checkpoints" / "best.pt"

N_CLIPS_PER_CLASS = 50
CLIP_LENGTH = 10

# ---------------------------------------------------------------------------
# App + named resources
# ---------------------------------------------------------------------------
app = modal.App("avatar-video-trial")

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
    """Verify all imports work and DualBranchSRM + video_aggregation load."""
    import torch
    import timm
    import sklearn
    import albumentations
    import numpy as np

    from src.config import TRIAL4_STRATEGIES, IMG_SIZE
    from src.models.dual_branch import DualBranchSRM
    from src.utils.video_aggregation import (
        mean_score,
        variance_gated,
        temporal_drift,
        STRATEGY_REGISTRY,
    )
    from src.utils.db import log_experiment, init_db
    from src.utils.metrics import compute_all_metrics

    model = DualBranchSRM(
        use_srm=True,
        use_attention=True,
        use_supcon=True,
        pretrained=False,
    )
    n_params = sum(p.numel() for p in model.parameters())

    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out["logits"].shape == (2, 2), f"Bad logits shape: {out['logits'].shape}"

    assert abs(mean_score([0.3, 0.5, 0.7]) - 0.5) < 1e-9
    assert abs(variance_gated([0.5, 0.5, 0.5], threshold=0.1) - 0.7) < 1e-9
    assert temporal_drift([0.3, 0.5, 0.7]) > np.mean([0.3, 0.5, 0.7])

    assert set(TRIAL4_STRATEGIES) == set(STRATEGY_REGISTRY.keys())

    result = {
        "torch": torch.__version__,
        "timm": timm.__version__,
        "sklearn": sklearn.__version__,
        "albumentations": albumentations.__version__,
        "n_params": n_params,
        "logits_shape": list(out["logits"].shape),
        "strategies": TRIAL4_STRATEGIES,
        "status": "OK",
    }
    print(f"[Smoke] validate_remote_imports PASSED: {result}")
    return result


# ---------------------------------------------------------------------------
# Quick test: aggregation smoke (T4, 3 min)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",
    memory=16 * 1024,
    timeout=3 * 60,
    volumes={
        str(DATA_ROOT): data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE: model_vol,
    },
)
def run_smoke_aggregation() -> dict:
    """Quick test: 3 synthetic sequences, verify strategy behavior."""
    import numpy as np

    from src.utils.video_aggregation import mean_score, variance_gated, temporal_drift

    stable_high = [0.85, 0.84, 0.86, 0.85, 0.85, 0.84, 0.86, 0.85, 0.84, 0.85]
    varying = [0.2, 0.6, 0.3, 0.8, 0.1, 0.7, 0.4, 0.5, 0.3, 0.6]
    increasing = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    results = {}

    for name, seq in [
        ("stable_high", stable_high),
        ("varying", varying),
        ("increasing", increasing),
    ]:
        m = mean_score(seq)
        vg = variance_gated(seq, threshold=0.1)
        td = temporal_drift(seq)
        results[name] = {"mean": m, "variance_gated": vg, "temporal_drift": td}

    assert results["stable_high"]["variance_gated"] >= 0.7, (
        "Stable high sequence should trigger variance gate"
    )
    assert results["varying"]["variance_gated"] == results["varying"]["mean"], (
        "Varying sequence should bypass variance gate"
    )
    assert results["increasing"]["temporal_drift"] > results["increasing"]["mean"], (
        "Increasing sequence should boost temporal drift above mean"
    )

    print("[Smoke] SMOKE TEST PASSED")
    print(f"[Smoke] Results: {json.dumps(results, indent=2)}")
    return {"status": "PASSED", "results": results}


# ---------------------------------------------------------------------------
# Simulated clip generation
# ---------------------------------------------------------------------------
def _generate_simulated_clips(
    images: list,
    labels: list,
    clip_length: int,
    is_real: bool,
):
    """Generate simulated video clips from static images.

    Real faces: slight augmentation per frame (jitter, blur) → high variance.
    AI faces: no augmentation or brightness-only shift → low variance.
    """
    import albumentations as A
    import numpy as np
    from PIL import Image

    if is_real:
        frame_aug = A.Compose(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.5),
                A.GaussNoise(std_range=(0.2, 0.44), p=0.3),
            ]
        )
    else:
        frame_aug = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.02, contrast_limit=0.0, p=0.3
                ),
            ]
        )

    clips = []
    for img_pil in images:
        clip_frames = []
        img_np = np.array(img_pil)
        for _ in range(clip_length):
            augmented = frame_aug(image=img_np)["image"]
            clip_frames.append(augmented)
        clips.append(clip_frames)

    return clips


# ---------------------------------------------------------------------------
# Main evaluation: run_video_evaluation (T4, 30 min)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",
    memory=16 * 1024,
    timeout=60 * 60,
    volumes={
        str(DATA_ROOT): data_vol,
        str(RESULTS_ROOT): results_vol,
        MODEL_CACHE: model_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_video_evaluation() -> dict:
    """Load best Trial 2 checkpoint, evaluate 3 aggregation strategies.

    For each strategy:
      1. Sample N real and N fake clips from the test set
      2. Simulate temporal variation (augmentation per frame)
      3. Run per-frame inference → P(ai) scores
      4. Apply aggregation strategy → video-level score
      5. Compute precision, recall, F1, AUC

    Generates:
      - score_trajectories.png
      - aggregation_roc_comparison.png
      - consistency_threshold_sweep.png
    """
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
    from torchvision import transforms
    from PIL import Image

    from src.config import IMG_SIZE, TRIAL4_STRATEGIES
    from src.models.dual_branch import DualBranchSRM
    from src.utils.video_aggregation import (
        mean_score,
        variance_gated,
        temporal_drift,
        STRATEGY_REGISTRY,
    )
    from src.utils.db import init_db, log_experiment
    from src.utils.metrics import compute_all_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trial4] device={device}")

    init_db()

    # ------------------------------------------------------------------
    # Load model from best Trial 2 checkpoint
    # ------------------------------------------------------------------
    ckpt_path = str(BEST_CHECKPOINT)
    model = DualBranchSRM(
        use_srm=True,
        use_attention=True,
        use_supcon=True,
        pretrained=True,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Trial4] Loaded checkpoint from {ckpt_path}")

    # ------------------------------------------------------------------
    # Read manifest, sample test images
    # ------------------------------------------------------------------
    manifest_df = pd.read_csv(MANIFEST_PATH)

    test_df = manifest_df[manifest_df["split"].isin(["test", "ood_test"])].copy()
    real_df = test_df[test_df["label"].str.strip().str.lower() == "real"]
    fake_df = test_df[test_df["label"].str.strip().str.lower() == "fake"]

    n_real = min(N_CLIPS_PER_CLASS, len(real_df))
    n_fake = min(N_CLIPS_PER_CLASS, len(fake_df))

    rng = np.random.RandomState(42)
    real_sample = real_df.sample(n=n_real, random_state=rng).reset_index(drop=True)
    fake_sample = fake_df.sample(n=n_fake, random_state=rng).reset_index(drop=True)

    print(f"[Trial4] Sampling {n_real} real + {n_fake} fake images for clips")

    # ------------------------------------------------------------------
    # Load PIL images
    # ------------------------------------------------------------------
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def _load_pil(path_str):
        try:
            return Image.open(path_str).convert("RGB")
        except Exception:
            return None

    real_images = [_load_pil(p) for p in real_sample["path"]]
    real_images = [img for img in real_images if img is not None]
    fake_images = [_load_pil(p) for p in fake_sample["path"]]
    fake_images = [img for img in fake_images if img is not None]

    print(f"[Trial4] Loaded {len(real_images)} real + {len(fake_images)} fake images")

    # ------------------------------------------------------------------
    # Generate simulated clips
    # ------------------------------------------------------------------
    real_clips = _generate_simulated_clips(
        real_images, [0] * len(real_images), CLIP_LENGTH, is_real=True
    )
    fake_clips = _generate_simulated_clips(
        fake_images, [1] * len(fake_images), CLIP_LENGTH, is_real=False
    )

    # ------------------------------------------------------------------
    # Per-frame inference: get P(ai) scores for every frame
    # ------------------------------------------------------------------
    def _score_frame(frame_np):
        """Score a single numpy frame through the model."""
        img_pil = Image.fromarray(frame_np)
        tensor = val_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            prob = torch.softmax(out["logits"], dim=1)[0, 1].item()
        return prob

    def _score_clip(clip_frames):
        """Score all frames in a clip."""
        return [_score_frame(f) for f in clip_frames]

    print("[Trial4] Scoring real clips...")
    real_clip_scores = []
    for i, clip in enumerate(real_clips):
        scores = _score_clip(clip)
        real_clip_scores.append(scores)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(real_clips)} real clips scored")

    print("[Trial4] Scoring fake clips...")
    fake_clip_scores = []
    for i, clip in enumerate(fake_clips):
        scores = _score_clip(clip)
        fake_clip_scores.append(scores)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(fake_clips)} fake clips scored")

    # ------------------------------------------------------------------
    # Apply aggregation strategies
    # ------------------------------------------------------------------
    all_clip_scores = real_clip_scores + fake_clip_scores
    all_labels = [0] * len(real_clip_scores) + [1] * len(fake_clip_scores)
    all_labels_np = np.array(all_labels)

    strategy_results = {}
    artifacts_dir = RESULTS_ROOT / "trial4"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for strategy_name in TRIAL4_STRATEGIES:
        strategy_fn = STRATEGY_REGISTRY[strategy_name]
        video_scores = []

        for clip_scores in all_clip_scores:
            video_score = strategy_fn(clip_scores)
            video_scores.append(video_score)

        video_scores_np = np.array(video_scores)
        y_pred = (video_scores_np >= 0.5).astype(int)

        if len(np.unique(all_labels_np)) >= 2:
            metrics = compute_all_metrics(all_labels_np, video_scores_np)
        else:
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

        strategy_results[strategy_name] = {
            "metrics": metrics,
            "video_scores": video_scores_np,
            "y_pred": y_pred,
        }

        print(
            f"[Trial4] {strategy_name:20s}  AUC={metrics['auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  EER={metrics['eer']:.4f}  "
            f"TPR@FPR1={metrics['tpr_at_fpr1']:.4f}"
        )

        # Log to SQLite
        log_experiment(
            trial_name=f"trial4_{strategy_name}",
            model_arch="dual_branch_srm",
            backbone="convnext_tiny.fb_in1k",
            test_dataset="video_aggregation_sim",
            n_test=len(all_labels_np),
            use_srm_branch=1,
            use_supcon=1,
            use_freq_augmentation=0,
            use_attention_gate=1,
            test_auc=metrics["auc"],
            test_f1=metrics["f1"],
            test_accuracy=metrics["accuracy"],
            test_precision=metrics["precision"],
            test_recall=metrics["recall"],
            test_specificity=metrics["specificity"],
            test_eer=metrics["eer"],
            test_tpr_at_fpr1=metrics["tpr_at_fpr1"],
            checkpoint_path=str(BEST_CHECKPOINT),
            plots_dir=str(artifacts_dir),
            config_json=json.dumps(
                {
                    "strategy": strategy_name,
                    "clip_length": CLIP_LENGTH,
                    "n_clips_per_class": N_CLIPS_PER_CLASS,
                }
            ),
            gpu_type="T4",
            training_time_s=int(time.time() - start_time),
            notes=(
                f"Trial 4 video aggregation. Strategy={strategy_name}. "
                f"Clip length={CLIP_LENGTH}, "
                f"n_real={len(real_clip_scores)}, n_fake={len(fake_clip_scores)}. "
                f"Real clips: augmentations (blur+jitter+noise). "
                f"Fake clips: brightness-only shift."
            ),
        )

    # ------------------------------------------------------------------
    # Plot 1: score_trajectories.png
    # ------------------------------------------------------------------
    _plot_score_trajectories(
        real_clip_scores,
        fake_clip_scores,
        save_path=artifacts_dir / "score_trajectories.png",
    )

    # ------------------------------------------------------------------
    # Plot 2: aggregation_roc_comparison.png
    # ------------------------------------------------------------------
    _plot_aggregation_roc(
        strategy_results,
        all_labels_np,
        save_path=artifacts_dir / "aggregation_roc_comparison.png",
    )

    # ------------------------------------------------------------------
    # Plot 3: consistency_threshold_sweep.png
    # ------------------------------------------------------------------
    _plot_threshold_sweep(
        all_clip_scores,
        all_labels_np,
        save_path=artifacts_dir / "consistency_threshold_sweep.png",
    )

    # ------------------------------------------------------------------
    # Commit results volume
    # ------------------------------------------------------------------
    results_vol.commit()
    print("[Trial4] Evaluation complete. Volume committed.")

    summary = {}
    for sname, sdata in strategy_results.items():
        summary[sname] = {k: v for k, v in sdata["metrics"].items()}

    return {"status": "OK", "strategies": summary}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_score_trajectories(real_scores, fake_scores, save_path):
    """Plot per-frame score trajectories for 5 real and 5 fake clips."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(5, len(real_scores), len(fake_scores))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(n_show):
        frames = list(range(len(real_scores[i])))
        axes[0].plot(frames, real_scores[i], alpha=0.7, label=f"Clip {i + 1}")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("P(AI)")
    axes[0].set_title("Real Face Clips — Score Trajectories")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for i in range(n_show):
        frames = list(range(len(fake_scores[i])))
        axes[1].plot(frames, fake_scores[i], alpha=0.7, label=f"Clip {i + 1}")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("P(AI)")
    axes[1].set_title("AI Face Clips — Score Trajectories")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Trial 4: Per-Frame Score Trajectories")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[Trial4] Saved score_trajectories.png")


def _plot_aggregation_roc(strategy_results, y_true, save_path):
    """Plot ROC curves for all 3 strategies on one figure."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {
        "mean": "tab:blue",
        "variance_gated": "tab:orange",
        "temporal_drift": "tab:green",
    }

    for strategy_name, sdata in strategy_results.items():
        video_scores = sdata["video_scores"]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, video_scores)
        auc_val = auc(fpr, tpr)
        color = colors.get(strategy_name, "tab:gray")
        ax.plot(
            fpr, tpr, color=color, lw=2, label=f"{strategy_name} (AUC={auc_val:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Trial 4: Aggregation Strategy ROC Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[Trial4] Saved aggregation_roc_comparison.png")


def _plot_threshold_sweep(all_clip_scores, all_labels, save_path):
    """Plot variance-gated AUC over threshold range [0.05, 0.30]."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    from src.utils.video_aggregation import variance_gated

    thresholds = np.arange(0.05, 0.31, 0.01)
    aucs = []

    for thr in thresholds:
        video_scores = []
        for clip_scores in all_clip_scores:
            video_scores.append(variance_gated(clip_scores, threshold=float(thr)))
        video_scores_np = np.array(video_scores)

        if len(np.unique(all_labels)) >= 2:
            try:
                a = float(roc_auc_score(all_labels, video_scores_np))
            except ValueError:
                a = 0.5
        else:
            a = 0.5
        aucs.append(a)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, aucs, "o-", color="tab:orange", lw=2)
    ax.set_xlabel("Consistency Threshold (std)")
    ax.set_ylabel("AUC")
    ax.set_title("Trial 4: Variance-Gated AUC vs Consistency Threshold")
    ax.grid(alpha=0.3)

    best_idx = int(np.argmax(aucs))
    best_thr = thresholds[best_idx]
    best_auc = aucs[best_idx]
    ax.axvline(x=best_thr, color="red", linestyle="--", alpha=0.7)
    ax.annotate(
        f"Best: {best_auc:.4f} @ {best_thr:.2f}",
        xy=(best_thr, best_auc),
        xytext=(best_thr + 0.03, best_auc - 0.02),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[Trial4] Saved consistency_threshold_sweep.png")


# ---------------------------------------------------------------------------
# Local entrypoint: smoke → aggregation smoke → evaluation
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    """Run all Trial 4 steps: import validation → aggregation smoke → evaluation."""
    from src.config import TRIAL4_STRATEGIES as _STRATEGIES

    print("=" * 60)
    print("Trial 4: Video-Level Aggregation Simulation")
    print("=" * 60)
    print(f"Strategies: {_STRATEGIES}")
    print(f"Clips per class: {N_CLIPS_PER_CLASS}")
    print(f"Clip length: {CLIP_LENGTH} frames")
    print()

    print("[Smoke] Validating imports...")
    import_result = validate_remote_imports.remote()
    print(f"[Smoke] Import check: {import_result}")

    print("\n[Smoke] Running aggregation smoke test...")
    smoke_result = run_smoke_aggregation.remote()
    print(f"[Smoke] Aggregation check: {smoke_result}")

    if smoke_result.get("status") != "PASSED":
        print("[FATAL] Aggregation smoke test failed — aborting evaluation.")
        return

    print("\n[Eval] Running video-level evaluation...")
    eval_result = run_video_evaluation.remote()
    print(f"[Eval] Evaluation: {eval_result}")

    print("\n" + "=" * 60)
    print("Trial 4 Summary")
    print("=" * 60)
    strategies = eval_result.get("strategies", {})
    for sname, smetrics in strategies.items():
        print(
            f"  {sname:20s}  AUC={smetrics.get('auc', 0):.4f}  "
            f"F1={smetrics.get('f1', 0):.4f}  "
            f"Prec={smetrics.get('precision', 0):.4f}  "
            f"Rec={smetrics.get('recall', 0):.4f}"
        )
    print("=" * 60)
