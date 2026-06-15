"""
Avatar Detection Module — Configuration Constants

Central configuration for all trials. No credentials, no local paths.
All paths are Modal volume mount points.
"""

from pathlib import Path

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Paths (Modal volume mount points)
# ---------------------------------------------------------------------------
MODEL_CACHE = "/models"
DATA_ROOT = Path("/data")
RESULTS_ROOT = Path("/results")

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
IMG_SIZE = 256

# ---------------------------------------------------------------------------
# Trial 1: DCT Dual-Branch Baseline
# ---------------------------------------------------------------------------
TRIAL1_CONFIGS = [
    {"name": "1a_rgb_only", "use_dct": False, "label_smoothing": 0.0},
    {"name": "1b_rgb_dct", "use_dct": True, "label_smoothing": 0.0},
    {"name": "1c_rgb_dct_ls", "use_dct": True, "label_smoothing": 0.1},
]

# ---------------------------------------------------------------------------
# Trial 2: SRM + ConvNeXt with SupCon (Main Contribution)
# Budget reduction: 3 configs only (2a, 2b, 2e)
# ---------------------------------------------------------------------------
TRIAL2_CONFIGS = [
    {
        "name": "2a_baseline",
        "use_srm": False,
        "use_attention": False,
        "use_supcon": False,
        "use_freq_aug": False,
    },
    {
        "name": "2b_srm_concat",
        "use_srm": True,
        "use_attention": False,
        "use_supcon": False,
        "use_freq_aug": False,
    },
    {
        "name": "2e_full_model",
        "use_srm": True,
        "use_attention": True,
        "use_supcon": True,
        "use_freq_aug": False,
    },
]

TRIAL2_HYPERPARAMS = {
    "backbone": "convnext_tiny.fb_in1k",
    "img_size": 256,
    "batch_size": 64,
    "epochs": 20,
    "optimizer": "AdamW",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "warmup_epochs": 2,
    "lambda_supcon": 0.3,
    "supcon_temperature": 0.07,
    "dropout": 0.4,
    "label_smoothing": 0.05,
}

# ---------------------------------------------------------------------------
# Trial 3: Cross-Dataset Generalization Study
# ---------------------------------------------------------------------------
TRIAL3_RUNS = [
    {
        "name": "3a_A_only",
        "train_on": "A",
        "val_on": "A",
        "test_on_A": True,
        "test_on_C": True,
    },
    {
        "name": "3b_C_only",
        "train_on": "C",
        "val_on": "C",
        "test_on_A": True,
        "test_on_C": True,
    },
    {
        "name": "3c_AC_combined",
        "train_on": "AC",
        "val_on": "AC",
        "test_on_A": True,
        "test_on_C": True,
    },
    {
        "name": "3d_A_sfhq",
        "train_on": "A_sfhq",
        "val_on": "A",
        "test_on_A": True,
        "test_on_C": True,
    },
]

# ---------------------------------------------------------------------------
# Trial 4: Video-Level Aggregation Simulation
# ---------------------------------------------------------------------------
TRIAL4_STRATEGIES = ["mean", "variance_gated", "temporal_drift"]

# ---------------------------------------------------------------------------
# Dataset Catalog
# ---------------------------------------------------------------------------
DATASETS = {
    "130k": {
        "kaggle_slug": "shreyanshpatel1/130k-real-vs-fake-face",
        "real_count": 70000,
        "fake_count": 60000,
        "generators": ["FLUX1.DEV", "FLUX1.PRO", "SDXL"],
        "resolution": "varies",
        "role": "primary_train_val",
    },
    "sfhq": {
        "kaggle_slug": "selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models",
        "real_count": 0,
        "fake_count": 122726,
        "generators": ["FLUX1.pro", "FLUX1.dev", "FLUX1.schnell", "SDXL", "DALL-E3"],
        "resolution": "1024px",
        "role": "fake_diversity_40k_sample",
    },
    "9.6k": {
        "kaggle_slug": "kaustubhdhote/human-faces-dataset",
        "real_count": 5000,
        "fake_count": 4630,
        "generators": ["GAN_unknown"],
        "resolution": "varies",
        "role": "ood_test_only",
    },
    "gravex": {
        "kaggle_slug": "muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal",
        "real_count": None,  # unknown until audit
        "fake_count": None,  # unknown until audit
        "generators": ["unknown"],
        "resolution": "small",
        "role": "pending_audit",
    },
}

# ---------------------------------------------------------------------------
# SRM Filter Kernels (Fridrich & Kodovsky, IEEE TIFS 2012)
# 3 high-pass filters for steganalysis residual extraction
# ---------------------------------------------------------------------------
SRM_KERNEL_1 = [
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0],
]

SRM_KERNEL_2 = [
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1],
]

SRM_KERNEL_3 = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, -2, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

# Shape: (3, 1, 5, 5) — 3 filters, 1 input channel, 5×5 kernel
SRM_KERNELS = torch.tensor(
    [SRM_KERNEL_1, SRM_KERNEL_2, SRM_KERNEL_3], dtype=torch.float32
).unsqueeze(1)
