"""Visualization utilities for avatar detection experiments.

Generates publication-quality plots: training curves, confusion matrices,
ROC curves, score distributions, frequency heatmaps, ablation comparisons,
and attention map overlays. All functions save to file (no display).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
from torch.nn.functional import interpolate


_DPI = 150


def _resolve_save_path(save_path):
    return Path(save_path)


def plot_training_curves(history, save_path):
    save_path = _resolve_save_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_auc = history["val_auc"]
    epochs = list(range(1, len(train_loss) + 1))

    best_epoch = int(np.argmax(val_auc)) + 1
    best_auc = float(max(val_auc))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="tab:blue")
    ax1.plot(epochs, val_loss, "s--", label="Val Loss", color="tab:cyan")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("AUC", color="tab:red")
    ax2.plot(
        epochs, val_auc, "^-", label=f"Val AUC (best={best_auc:.4f})", color="tab:red"
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0.0, 1.05)

    ax1.axvline(x=best_epoch, color="gray", linestyle=":", alpha=0.7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Training Curves")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, save_path, labels=None):
    if labels is None:
        labels = ["Real", "Fake"]

    save_path = _resolve_save_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix (Normalized by Row)")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI)
    plt.close(fig)


def plot_roc_curve(y_true, y_score, save_path, auc_value=None):
    save_path = _resolve_save_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    if auc_value is None:
        auc_value = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="tab:blue", lw=2, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI)
    plt.close(fig)


def plot_score_distribution(y_true, y_score, save_path):
    save_path = _resolve_save_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    real_scores = y_score[y_true == 0]
    fake_scores = y_score[y_true == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        real_scores, bins=50, alpha=0.5, color="tab:blue", label="Real", density=True
    )
    ax.hist(
        fake_scores, bins=50, alpha=0.5, color="tab:red", label="Fake", density=True
    )
    ax.set_xlabel("P(AI)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI)
    plt.close(fig)


def plot_frequency_heatmap(real_magnitudes, fake_magnitudes, save_path):
    save_path = _resolve_save_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    real_m = np.asarray(real_magnitudes, dtype=np.float64)
    fake_m = np.asarray(fake_magnitudes, dtype=np.float64)
    epsilon = 1e-10
    ratio = fake_m / (real_m + epsilon)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(real_m, cmap="viridis", aspect="auto")
    axes[0].set_title("Real |Magnitude|")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(fake_m, cmap="viridis", aspect="auto")
    axes[1].set_title("Fake |Magnitude|")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(ratio, cmap="inferno", aspect="auto")
    axes[2].set_title("Fake / Real Ratio")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI)
    plt.close(fig)


def plot_ablation_comparison(configs, auc_values, save_path):
    save_path = _resolve_save_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(configs, auc_values, color="tab:blue", edgecolor="black")

    for bar, val in zip(bars, auc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("AUC")
    ax.set_xlabel("Configuration")
    ax.set_title("Ablation Comparison")
    ax.set_ylim(0.8, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI)
    plt.close(fig)


def plot_attention_maps(images, attention_weights, save_path, n_samples=4):
    save_path = _resolve_save_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    n_samples = min(n_samples, images.shape[0])
    h, w = images.shape[2], images.shape[3]

    fig, axes = plt.subplots(n_samples, 2, figsize=(6, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        img = images[i].transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img[:, :, 0]

        attn = attention_weights[i, 0]
        attn_upsampled = (
            interpolate(
                torch.tensor(attn[np.newaxis, np.newaxis], dtype=torch.float32),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        axes[i, 0].imshow(img if img.ndim == 2 else img)
        axes[i, 0].set_title(f"Sample {i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img if img.ndim == 2 else img)
        axes[i, 1].imshow(attn_upsampled, cmap="Reds", alpha=0.5)
        axes[i, 1].set_title(f"Attention {i}")
        axes[i, 1].axis("off")

    fig.suptitle("Attention Map Overlay")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI)
    plt.close(fig)
