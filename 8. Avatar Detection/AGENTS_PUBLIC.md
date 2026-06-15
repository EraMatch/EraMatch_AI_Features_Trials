# AGENT PROMPT — EraMatch Avatar Detection Module

## AI-Generated Interview Video Detection Research Pipeline

> ⚠️  **SECURITY** : This file contains credentials. DO NOT commit to any public or shared git repository.
> Store in a private local directory only.

---

## 0. Who You Are and What This Is

You are an AI coding agent building the **avatar detection module** for  **EraMatch** , an AI-powered
end-to-end recruitment platform developed as a graduation project (CSAI 498/499) at
Zewail City of Science and Technology.

EraMatch runs live video interviews. The specific threat you are defending against:
candidates using real-time AI face-swap tools (HeyGen, Synthesia avatars), OBS virtual camera
loops, or diffusion-model-generated face overlays during the interview session.

**Your module is called per-frame (every ~30 frames = 2 seconds of video) during a live interview.**
It returns a probability score P(ai). The interview session aggregates these per-frame scores
into a video-level risk signal. This architecture means all your training and evaluation
is on  **static face crops from frames** , not full video clips.

 **Academic goal** : prove that a frequency-domain dual-branch architecture (SRM residual + RGB spatial)
with supervised contrastive loss achieves better cross-generator generalization than a plain CNN baseline,
and that generalization degrades predictably when trained on GAN-style fakes and tested on diffusion-model fakes.

---

## 1. Skills and Tools You Have Access To

### Modal Skill

Before writing any Modal code, read the skill file:

```
/mnt/skills/user/modal/SKILL.md
```

This gives you the exact patterns, cost-safety gates, and design rules. Follow them precisely.
 **Key rule from the skill** : always run a smoke test (1 forward pass, save checkpoint, reload) before
any full-dataset GPU run.

### Context7 MCP — Package Version Verification

Before using ANY Python package in training code, call Context7 MCP to verify the correct
import syntax and API for the current version. Specifically verify these before using them:

* `timm` — model creation API changes across versions
* `torch` / `torchvision` — transform API, `v2` vs legacy
* `modal` — decorator syntax changes frequently
* `albumentations` — augmentation API
* `pytorch-metric-learning` — SupCon loss API

Call `resolve-library-id` then `get-library-docs` for each. Do not assume API signatures
from training data — verify them.

---

## 2. One-Time Environment Setup

Run these commands once from your local machine before any other work.

### 2a. Create Kaggle Credentials Secret

```bash
modal secret create kaggle-creds \
  [REDACTED] \
  [REDACTED]
```

### 2b. Create Persistent Data Volume

```bash
modal volume create avatar-data-vol
```

### 2c. Create Results Volume

```bash
modal volume create avatar-results-vol
```

These volumes persist forever across runs. You pay ~$0.10/GB/month for storage.
`avatar-data-vol` holds all dataset images (processed to 256×256).
`avatar-results-vol` holds checkpoints, plots, the SQLite experiment database, and CSV reports.

---

## 3. Repository Structure to Create

```
avatar-detection/
├── modal/
│   ├── setup_volume.py          # Stage 0: dataset download + resize pipeline
│   ├── audit_gravex.py          # Stage 0b: GRAVEX-200K content inspection
│   ├── train_nb1_dct.py         # Trial 1: DCT dual-branch upgrade (T4)
│   ├── train_nb2_srm.py         # Trial 2: SRM + ConvNeXt main model (A10G)
│   ├── train_nb3_cross.py       # Trial 3: cross-dataset generalization matrix (A10G)
│   └── train_nb4_video.py       # Trial 4: video-level aggregation sim (T4)
├── src/
│   ├── models/
│   │   ├── srm_filters.py       # Fixed SRM 5×5 kernels as non-trainable conv layer
│   │   ├── dual_branch.py       # Full SRM-RGB dual-branch model
│   │   ├── dct_branch.py        # DCT magnitude map branch (Trial 1)
│   │   └── losses.py            # BCE + Supervised Contrastive loss
│   ├── data/
│   │   ├── dataset.py           # AvatarDataset reading from manifest.csv
│   │   └── augmentations.py     # Forensic-aware augmentation pipeline
│   └── utils/
│       ├── metrics.py           # AUC, F1, EER, TPR@FPR, calibration
│       ├── db.py                # SQLite experiment logger
│       ├── viz.py               # All standard plots
│       └── freq_analysis.py     # DCT/SRM frequency visualization tools
├── notebooks/
│   └── nb1_dct_baseline.ipynb   # Kaggle T4 notebook (Trial 1 only)
├── results/                     # Local copy of results downloaded from Volume
│   ├── plots/
│   ├── checkpoints/
│   └── experiments.db
├── AGENT_PROMPT_AVATAR_DETECTION.md
└── README.md
```

---

## 4. Scientific Problem Background

> Read this section before writing any code. Understanding the problem determines
> every architecture and dataset decision.

### 4a. Why plain CNNs fail across generators

AI face generators fall into two major families with different artifact signatures:

**GAN-based** (StyleGAN, StyleGAN2, PGGAN): produce checkerboard upsampling artifacts at
specific DCT frequencies. These come from transposed convolution operations. Classic
deepfake detectors (Xception, MesoNet) learned to detect these frequency peaks.

**Diffusion-based** (FLUX1, SDXL, DALL-E 3): use UNet + attention decoder. Their
upsampling artifacts appear at different frequencies. Detectors trained on GANs often
**fail completely** on diffusion model outputs because the frequency signature they learned
is absent.

The 9.6K existing dataset (`kaustubhdhote/human-faces-dataset`) contains GAN-style fakes.
The 130K dataset (`shreyanshpatel1/130k-real-vs-fake-face`) contains diffusion-style fakes.
**Training on one and testing on the other measures this exact generalization gap.**

Reference: *"From Specificity to Generality: Revisiting Generalizable Artifacts in Detecting
Face Deepfakes"* (arXiv 2504.04827, 2025). Key finding: no single artifact type generalizes
across all generators; the best approach learns multiple artifact types simultaneously.

### 4b. Why frequency domain features help

Video generator decoders must upsample from a compressed latent space to full resolution.
This upsampling — whether bilinear, transposed conv, or pixel-shuffle — leaves periodic
grid-like traces in the frequency domain. These are invisible to human eyes but appear
as distinct high-frequency spikes under DCT or FFT analysis.

The **SRM (Steganalysis Rich Model)** approach applies fixed high-pass filter kernels that
were originally designed for image steganography detection. They suppress low-frequency
image content (the "natural" part) and amplify high-frequency residuals (the "manipulation" part).
A small CNN on these residuals learns which manipulation patterns are diagnostic.

Reference: *"Rich Models for Steganalysis of Digital Images"* (Fridrich & Kodovsky, IEEE TIFS 2012).
Original SRM paper. The 30 filter kernels are available as fixed weights.

Reference: *"Leveraging Frequency Analysis for Deep Fake Image Recognition"* (Frank et al., ICML 2020).
First systematic study showing that GAN outputs have characteristic DCT peaks that CNNs can exploit.

### 4c. Why contrastive loss improves generalization

Standard cross-entropy loss only asks "is this real or fake?" It may learn dataset-specific
shortcuts (e.g. JPEG quality level, background color distribution) rather than genuine
manipulation artifacts. These shortcuts fail when you change datasets.

**Supervised Contrastive Loss (SupCon)** additionally asks: "do two real images embed close
together? do a real and a fake embed far apart?" This regularizes the embedding space so
the model learns generator-agnostic features of "real-ness" rather than generator-specific
features of "GAN-ness" or "diffusion-ness".

Reference: *"Supervised Contrastive Learning"* (Khosla et al., NeurIPS 2020).
Reference: *"Contrastive learning-based general Deepfake detection with multi-scale RGB
frequency clues"* (CLGD, Pattern Recognition 2023). Applies SupCon to deepfake detection
specifically; shows ~4-6% AUC improvement on cross-dataset protocols.

### 4d. What makes cross-modal attention non-trivial

The SRM branch detects WHERE in the image frequency anomalies appear (blending boundaries,
upsampling grid artifacts, hairline edges). The RGB branch detects WHAT the anomaly looks like
as a texture or color discontinuity. A naive concatenation treats them as independent signals.

**Cross-modal attention** lets SRM features generate a spatial attention mask that tells
the RGB branch which regions deserve scrutiny. This is important because a full face has
many regions that look normal (forehead, chin); the interesting artifacts concentrate at
the face boundary, eye-sclera boundary, and hair-skin transition.

Reference: *"LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and
Generalizable Deepfake Detection"* (CVPR 2024). Uses localized attention to focus
detection on the most artifact-rich regions.

---

## 5. Dataset Catalog

### ⚠️ CRITICAL RULE: Inspect Before Coding

**For every dataset, before writing any Dataset class or data loader:**

1. Download a small sample (or just list file structure via `os.walk`)
2. Print the full directory tree to depth 3
3. Check one random image: `PIL.Image.open(path).size`, format, mode
4. Check label distribution: count files per folder
5. If a CSV is present, print `df.head(10)` and `df['label'].value_counts()`
6. Only then write the `AvatarDataset` class

The directory structures below are **predicted** based on what is known about each dataset.
They may be wrong. Verify before writing code. Print what you actually find.

---

### Dataset A: `shreyanshpatel1/130k-real-vs-fake-face`

 **Role** : Primary training + validation set
 **Size** : ~130K images — 70K real (Flickr photos), 60K fake (FLUX1.DEV, FLUX1.PRO, SDXL)
 **Why it matters** : Uses modern diffusion models that are actually deployed for interview fraud today.
Flickr real photos are naturalistic and demographically diverse.

**Expected structure after unzip** (verify this — may have nested folders):

```
130k-real-vs-fake-face/
├── Train/
│   ├── Real/          # ~56K images
│   └── Fake/          # ~48K images
└── Test/
    ├── Real/          # ~14K images
    └── Fake/          # ~12K images
```

Or it may be flat:

```
real/    fake/
```

Print what you find. The manifest.csv will normalize this.

 **Split protocol** : Use the dataset's own Train/Test split if present. If not, do 80/15/5
stratified split yourself. Never use Test images for any hyperparameter tuning.

---

### Dataset B: `selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models`

 **Role** : Generator diversity supplement — adds DALL-E 3 and FLUX-schnell to training fake space
 **Size** : 122,726 images at 1024×1024 (large — resize to 256×256 immediately on download)
 **Generators covered** : FLUX1.pro, FLUX1.dev, FLUX1.schnell, SDXL, DALL-E 3
 **Why it matters** : Expands the number of fake generators from 3 (in Dataset A) to 5.
DALL-E 3 uses different upsampling than FLUX/SDXL; exposing the model to it improves robustness.

 **Important** : This dataset is **fakes only** — no real images. Sample 40,000 images from it
(random stratified by generator if a metadata CSV is provided) and add them to the fake pool.
The real pool is already sufficient from Dataset A.

 **Expected structure** :

```
sfhq-t2i-synthetic-faces-from-text-2-image-models/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── SFHQ_T2I_dataset.csv   # has columns: image_path, model (flux1_pro, sdxl, etc.)
```

Use the CSV to sample 8K per generator for diversity balance.

---

### Dataset C: `kaustubhdhote/human-faces-dataset` (existing 9.6K)

 **Role** : Out-of-distribution (OOD) test set — NEVER used for training
 **Size** : 5,000 real + 4,630 AI-generated = 9,630 images
 **Generators** : Unknown/older, likely GAN-based (StyleGAN era)
 **Why it matters** : Testing on this dataset after training on Dataset A measures the
GAN-drift problem — how badly does a diffusion-trained model perform on GAN fakes?

 **Structure** : already familiar from v11 notebook:

```
Human Faces Dataset/
├── Real Images/       # 5,000 images
└── AI-Generated Images/  # 4,630 images
```

**Do not train on this dataset.** It is your OOD evaluation set for the thesis generalization claim.

---

### Dataset D: `muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal` (GRAVEX-200K)

 **Role** : Conditional — pending content audit
 **Size** : ~200K images, ~2GB total (~10KB/image average = very small resolution)
 **Status** : UNKNOWN content — "visuals" not "faces". Must audit before use.

 **The audit determines its role** :

* Face prevalence ≥ 60% → include as supplemental training data (face-filter first)
* Face prevalence < 60% → document as "inspected, excluded due to low domain relevance" in thesis
* Image quality too low (< 128×128 most images) → exclude regardless

Run `audit_gravex.py` (see Trial 0) before deciding.

---

### Dataset Summary Table

| Dataset       | Images | Real       | Fake | Generators             | Resolution | Role                        |
| ------------- | ------ | ---------- | ---- | ---------------------- | ---------- | --------------------------- |
| 130K faces    | ~130K  | 70K Flickr | 60K  | FLUX, SDXL             | varies     | Primary train/val           |
| SFHQ-T2I      | 122K   | 0          | 122K | FLUX×3, SDXL, DALL-E3 | 1024px     | Fake diversity (40K sample) |
| 9.6K existing | 9.6K   | 5K         | 4.6K | GAN (unknown)          | varies     | OOD test only               |
| GRAVEX-200K   | 200K   | ?          | ?    | unknown                | ~small     | Pending audit               |

 **Training corpus target** : ~70K real + ~100K fake = 170K total (before GRAVEX decision)

---

## 6. Results Database — SQLite Schema

Create this database at `/data/results/experiments.db` on `avatar-results-vol`.
Every experiment writes one row to `experiments` and N rows to `training_history`.

```python
# src/utils/db.py

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "/results/experiments.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_name       TEXT NOT NULL,      -- e.g. "trial1_dct_baseline"
    model_arch       TEXT NOT NULL,      -- e.g. "efficientnet_b1_dct4ch"
    backbone         TEXT,               -- e.g. "convnext_tiny"
  
    -- Dataset config
    train_datasets   TEXT,               -- JSON list: ["130k", "sfhq_sample"]
    test_dataset     TEXT,               -- e.g. "130k_test" or "9.6k_ood"
    n_train          INTEGER,
    n_val            INTEGER,
    n_test           INTEGER,
  
    -- Ablation flags (1=used, 0=ablated away)
    use_srm_branch       INTEGER DEFAULT 1,
    use_supcon           INTEGER DEFAULT 1,
    use_freq_augmentation INTEGER DEFAULT 1,
    use_attention_gate   INTEGER DEFAULT 1,
  
    -- Final test metrics
    test_auc         REAL,    -- primary metric, report to 4 decimal places
    test_f1          REAL,    -- weighted F1
    test_accuracy    REAL,
    test_precision   REAL,
    test_recall      REAL,    -- sensitivity (TPR)
    test_specificity REAL,    -- TNR
    test_eer         REAL,    -- Equal Error Rate (lower is better)
    test_tpr_at_fpr1 REAL,   -- TPR at FPR=1% (critical for deployment)
    val_best_auc     REAL,    -- best val AUC during training (sanity check)
  
    -- Training config
    epochs           INTEGER,
    batch_size       INTEGER,
    learning_rate    REAL,
    weight_decay     REAL,
    supcon_weight    REAL,    -- lambda for L_total = L_BCE + lambda*L_SupCon
  
    -- Artifacts
    checkpoint_path  TEXT,    -- path on results volume
    plots_dir        TEXT,    -- path on results volume
    config_json      TEXT,    -- full hyperparameter dict as JSON string
  
    -- Metadata
    gpu_type         TEXT,    -- "T4" or "A10G"
    training_time_s  INTEGER, -- wall-clock seconds
    timestamp        TEXT,    -- ISO format
    notes            TEXT     -- free text observations
);

CREATE TABLE IF NOT EXISTS training_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    epoch         INTEGER NOT NULL,
    train_loss    REAL,
    val_loss      REAL,
    val_auc       REAL,
    val_f1        REAL,
    lr            REAL,      -- learning rate at this epoch
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS dataset_audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name    TEXT,
    n_total         INTEGER,
    n_real          INTEGER,
    n_fake          INTEGER,
    face_rate       REAL,          -- from MediaPipe audit
    avg_resolution  TEXT,          -- e.g. "256x256"
    generators      TEXT,          -- JSON list
    notes           TEXT,
    audited_at      TEXT
);
"""
```

**Query pattern for comparison reports** (run after all trials complete):

```python
# Produces the thesis cross-dataset matrix
SELECT trial_name, train_datasets, test_dataset, 
       test_auc, use_srm_branch, use_supcon
FROM experiments
ORDER BY trial_name, test_dataset;
```

---

## 7. Trial Specifications

### Trial 0: Dataset Audit and Volume Setup

 **Script** : `modal/audit_gravex.py` and `modal/setup_volume.py`
 **GPU** : CPU only
 **Cost** : ~$0.10 total

#### 7.0a — GRAVEX Content Audit

Write `audit_gravex.py` as a Modal CPU function that:

1. Downloads 1,000 random GRAVEX images via Kaggle API
2. Runs `mediapipe.solutions.face_detection` on each
3. Counts face prevalence rate
4. Checks average image resolution
5. Saves `dataset_audit` row to SQLite DB
6. Prints full decision: include / exclude + reasoning

 **Decision threshold** : face_rate ≥ 0.60 → include. Below that → exclude.

Log to `dataset_audit` table. Print clearly: `"GRAVEX DECISION: INCLUDE"` or `"GRAVEX DECISION: EXCLUDE"`.

#### 7.0b — Dataset Structure Inspection

Before writing setup_volume.py, write a separate inspection function that:

1. Downloads just 20 files from each dataset
2. Prints full directory tree (`os.walk` to depth 4)
3. Prints one sample image shape, format, mode
4. Prints any CSV headers found

**Only after reading this output** should you write the actual download + manifest pipeline.
The manifest approach (single `manifest.csv` with columns `path,label,dataset,split`) is the
target output. How you get there depends on what the directory structure actually looks like.

#### 7.0c — Full Volume Setup

Write `setup_volume.py` to:

1. Download all datasets via Kaggle API
2. Resize all images to 256×256 LANCZOS (saves ~8× storage vs 1024px originals)
3. For SFHQ-T2I: sample 40K total, ~8K per generator, read from metadata CSV
4. Build `manifest.csv` with columns: `path, label, dataset, split, generator`
5. Save manifest to `/data/manifest.csv` on volume
6. Print summary: total counts per (dataset × label)
7. Call `vol.commit()`

The `split` column should be: `train`, `val`, `test`, `ood_test`.

* `ood_test` → all of the 9.6K existing dataset (never in train/val)
* `test` → held-out 15% from Dataset A
* `val` → 5% from Dataset A
* `train` → remaining 80% from Dataset A + all SFHQ-T2I sample

---

### Trial 1: DCT Dual-Branch Baseline

 **Purpose** : Prove that feeding DCT as an actual model input beats RGB-only.
This closes the gap from v11 where DCT was analyzed but never used.
 **Script** : `modal/train_nb1_dct.py` (can also run as Kaggle T4 notebook)
 **GPU** : T4 (16GB VRAM — fits comfortably)
 **Expected training time** : ~45 minutes on T4
 **Dataset** : 9.6K existing ONLY (keeps comparison fair with v11)

#### Architecture: `EfficientNet-B1 + DCT 4-channel`

* Compute 2D DCT of the grayscale image at full 224×224 resolution
  → take log1p(abs(DCT)) → normalize → use as a 4th channel
* Modify EfficientNet-B1 stem: first Conv2d kernel goes from 3 → 4 channels
  → initialize the 4th channel weights as mean of the 3 pretrained channels
  → this is a standard "weight surgery" trick for extra channels
* Rest of the model is unchanged: EfficientNet-B1 → 1280 → 256 → 2
* Loss: CrossEntropy only (no SupCon — save that for Trial 2 comparison)

#### Ablations within Trial 1

Run 3 configs in sequence:

1. RGB only (v11 reproduction — sanity check that v11 numbers reproduce)
2. RGB + DCT 4th channel (the upgrade)
3. RGB + DCT + label smoothing 0.1

Save each as a separate `experiments` row with descriptive `trial_name`.

#### What to log

* Full training curves (loss + AUC per epoch) → `training_history` table
* Test metrics for all 3 configs → `experiments` table
* Confusion matrix PNG
* ROC curve PNG
* DCT frequency heatmap (real vs fake mean |DCT| spectrum — reuse logic from v11)
* Grad-CAM on 4 samples (2 real, 2 fake)
* Score distribution histogram (real scores vs fake scores)

#### What good looks like here

* Config 1 (RGB only) reproduces v11 AUC ± 0.01 → confirms reproducibility
* Config 2 (RGB + DCT) should improve AUC by 1–3% vs Config 1
  → If improvement is < 0.5%: DCT as a 4th channel adds little; the model may not be
  learning the DCT branch. Check that the 4th channel is normalized correctly.
  → If improvement is > 5%: unusual, check for data leakage or normalization bug
* Config 3 (+ label smoothing) usually adds 0.5–1% on small datasets by reducing overconfidence

---

### Trial 2: SRM + ConvNeXt with SupCon (Main Contribution)

 **Purpose** : The primary thesis model. Dual-branch SRM+RGB with cross-modal attention and
supervised contrastive regularization trained on modern diffusion-model fakes.
 **Script** : `modal/train_nb2_srm.py`
 **GPU** : A10G (24GB VRAM — needed for ConvNeXt + SRM branch at batch 64)
 **Expected training time** : ~3–4 hours on A10G
 **Dataset** : Full training corpus (130K + SFHQ-T2I 40K sample)

#### Architecture: `DualBranchSRM`

##### SRM Branch (`src/models/srm_filters.py`)

Implement 3 SRM high-pass filter kernels as a fixed (non-trainable) `nn.Conv2d`:

```python
# The 3 kernels below are from Fridrich & Kodovsky (2012) Table 1
# They suppress low-frequency image content and amplify residuals
SRM_KERNEL_1 = [
    [ 0,  0, 0,  0,  0],
    [ 0, -1, 2, -1,  0],
    [ 0,  2,-4,  2,  0],
    [ 0, -1, 2, -1,  0],
    [ 0,  0, 0,  0,  0],
]   # Laplacian-style, catches interpolation residuals

SRM_KERNEL_2 = [
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1],
]   # Second-order, catches upsampling grid artifacts

SRM_KERNEL_3 = [
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  1, -2,  1,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
]   # Simple edge residual, catches blending boundaries
```

Apply to grayscale image: `nn.Conv2d(1, 3, kernel_size=5, padding=2, bias=False)` with
`requires_grad=False` and kernel weights initialized from the above.
Output: [B, 3, H, W] residual maps — clamp values to [-3, 3] after division by std.

Follow with a small CNN encoder:
`Conv(3→32, 3×3, BN, ReLU) → Conv(32→64, 3×3, stride=2, BN, ReLU) → Conv(64→128, 3×3, stride=2, BN, ReLU) → AdaptiveAvgPool → Linear(128×?×? → 256)`

##### RGB Branch

Use `convnext_tiny` from `timm` (verify exact model string with Context7).
Keep pretrained ImageNet weights. Output: 768-d feature vector via global average pool.

##### Cross-Modal Attention Gate

The SRM feature vector (256-d) computes attention weights over spatial positions of the
RGB feature map before pooling:

```
SRM_feat (256) → FC → 7×7 attention map (sigmoid)
RGB_spatial (B,768,7,7) × attention_map → weighted spatial features → pool → 768-d
```

This forces RGB features to attend to regions the SRM branch flagged as anomalous.

##### Fusion + Classifier

Concatenate: RGB_attended (768) + SRM_feat (256) = 1024-d
→ FC(1024→256, ReLU, Dropout 0.4) → FC(256→2)

##### Loss

```python
L_total = L_BCE + lambda_supcon * L_SupCon(embeddings, labels)
# lambda_supcon = 0.3  (tune in range [0.1, 0.5])
# L_SupCon uses the 1024-d concatenated features (after projection head)
# Temperature tau = 0.07
```

Use `pytorch-metric-learning` for SupCon — verify version with Context7.

#### Ablation Matrix for Trial 2

Run these 6 configs, saving each to `experiments` table:

| Config | SRM branch | Attention gate | SupCon | Notes                       |
| ------ | ---------- | -------------- | ------ | --------------------------- |
| 2a     | ✗         | ✗             | ✗     | RGB-only ConvNeXt baseline  |
| 2b     | ✓         | ✗             | ✗     | + SRM naive concat          |
| 2c     | ✓         | ✓             | ✗     | + attention gate            |
| 2d     | ✓         | ✗             | ✓     | + SupCon only               |
| 2e     | ✓         | ✓             | ✓     | **Full model (main)** |
| 2f     | ✓         | ✓             | ✓     | + freq augmentation (extra) |

Config 2a is the ablation baseline. Each config tests one design decision.
The difference between 2b and 2c isolates the attention gate.
The difference between 2c and 2e isolates SupCon.

#### Frequency-Aware Augmentation (for Config 2f)

During training, with probability p=0.3:

1. Compute DCT of image
2. Zero out one random frequency band (low, mid, or high, chosen randomly)
3. Reconstruct via IDCT
4. This forces the model to not rely on any single frequency band

This is inspired by *FreqBlender* (arXiv 2304.07193) which shows that frequency-domain
augmentation improves cross-generator generalization.

#### Hyperparameters

```python
config = {
    "backbone": "convnext_tiny",
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
    "augmentations": ["RandomHorizontalFlip", "ColorJitter", 
                      "JpegCompression(40,90)",   # critical for deployment realism
                      "GaussianBlur", "RandomResizeCrop"],
}
```

The JPEG compression augmentation (quality 40–90) is  **critical** . Real interview frames
go through H.264 encoding then JPEG extraction. Without this augmentation the model will
learn frequency artifacts from uncompressed training images that disappear after platform
transcoding. Reference: DeepSpeak dataset paper explicitly warns about this codec leakage.

#### What to log

Everything from Trial 1, plus:

* Feature embedding UMAP (2D projection of test embeddings, colored by label and by dataset)
  → should show real/fake separation improving from 2a → 2e
* Ablation comparison bar chart: AUC per config (2a through 2f)
* Frequency heatmap: mean |SRM residual| for real vs fake images
  → real images should show near-uniform residual; fake images should show structured peaks
* Attention gate visualization: heatmap of attention weights overlaid on sample face crops
  → attention should concentrate on face boundaries and eye/mouth regions for fake images

#### What good looks like here

* 2a AUC vs 2e AUC: main model should improve by 2–5% (if improvement is 0–1%,
  the architecture isn't working — debug SRM normalization and attention gate first)
* Test AUC 2e on Dataset A test split: target ≥ 0.93
  → Below 0.88: underfit — increase epochs or check data loading
  → Above 0.98 with simple data: check for data leakage (test images in train set)
* SupCon effect (2c→2e): expect +1–3% AUC on within-dataset, but the real gain shows
  in cross-dataset (Trial 3). Within-dataset AUC may actually decrease slightly with SupCon.
  That is correct and expected — SupCon sacrifices some in-distribution accuracy for generalization.

---

### Trial 3: Cross-Dataset Generalization Study

 **Purpose** : The central thesis scientific claim. Measure how performance degrades when
training and testing distributions differ in their fake generator family.
 **Script** : `modal/train_nb3_cross.py`
 **GPU** : A10G
 **Expected training time** : ~5 hours total (4 × training runs)

#### Protocol

Run 4 training experiments and evaluate each on all test sets:

| Run | Train on        | Val on  | Test on A (130K) | Test on C (9.6K OOD) |
| --- | --------------- | ------- | ---------------- | -------------------- |
| 3a  | Dataset A only  | A val   | A test           | C (OOD)              |
| 3b  | Dataset C only  | C val   | A test (OOD)     | C test               |
| 3c  | A + C combined  | A+C val | A test           | C test               |
| 3d  | A + SFHQ sample | A val   | A test           | C (OOD)              |

 **The key numbers the thesis cares about** :

* `3a: AUC on A test` vs `3a: AUC on C OOD` → the GAN-drift gap
* `3b: AUC on C test` vs `3b: AUC on A OOD` → the diffusion-to-GAN gap
* Does combining datasets (3c) close the gap?
* Does SFHQ diversity (3d) help more than just using more data of the same type?

#### For each run, test EVERY config from Trial 2 ablations

This means running 4 runs × 6 configs = 24 total experiment rows in the database.
The generalization benefit of SupCon should appear specifically in the cross-dataset columns,
not in the within-dataset columns.

#### Generate the thesis table

After all 24 runs, query the DB and generate a CSV:

```sql
SELECT trial_name, config, train_datasets, test_dataset,
       test_auc, use_srm_branch, use_supcon, use_attention_gate
FROM experiments
WHERE trial_name LIKE 'trial3%'
ORDER BY config, test_dataset;
```

Pivot this into a markdown table (rows = configs, columns = test datasets).
Save as `results/cross_dataset_matrix.csv` and `results/cross_dataset_matrix.md`.

#### Visualization: AUC Heatmap

Create a 6×4 heatmap (rows=configs, columns=test sets) colored by AUC value.
Green = high AUC, red = low AUC. This is the primary thesis figure.

#### What good looks like here

* Without SRM/SupCon (config 2a): expect AUC to drop by 10–20% from within- to cross-dataset.
  This IS the expected finding — it's the problem the thesis addresses, not a failure.
* With full model (config 2e): expect the cross-dataset drop to be 5–10% smaller.
  If the gap is not reduced: the SRM branch may be learning dataset-specific artifacts too.
  Debug by checking if the SRM residuals look different for the two dataset's real images.
* SupCon should reduce the cross-dataset gap specifically (test 2c→2e on the OOD column).
  If SupCon doesn't help on cross-dataset: check that the batch construction includes
  samples from both datasets so SupCon is pushing representations across distribution.

---

### Trial 4: Video-Level Aggregation Simulation

 **Purpose** : Connect frame-level classifiers to the actual EraMatch use case.
Show that temporal consistency signals add value beyond per-frame scores.
 **Script** : `modal/train_nb4_video.py`
 **GPU** : T4 (no training here, just inference on existing checkpoints)

#### Setup

Load the best checkpoint from Trial 2 (config 2e). Run inference over simulated "video clips"
constructed by sampling N consecutive frames from the dataset images.

Since we have static images (not video), simulate temporal consistency by:

1. Taking real face images → apply slight augmentation per "frame" (small jitter, mild blur)
   → this creates a sequence that should have HIGH variance in scores (natural variation)
2. Taking AI face images → apply no augmentation or only brightness shift
   → this creates a sequence that should have LOW variance in scores (avatar is consistent)

#### Three aggregation strategies

Implement and compare:

 **Strategy A — Mean score** : `P(ai) = mean(frame_scores)`
Baseline. Simple. Works when most frames are clearly fake.

 **Strategy B — Variance-gated mean** :

```python
if std(frame_scores) < CONSISTENCY_THRESHOLD:
    # Too consistent — flag as AI avatar even if mean is moderate
    P(ai) = max(mean(frame_scores), 0.7)
else:
    P(ai) = mean(frame_scores)
```

The insight: real people have natural micro-variation. AI avatars are eerily consistent.

 **Strategy C — Temporal drift detection** :

```python
# Flag if scores are high AND non-decreasing (avatar quality doesn't degrade over time)
drift = np.polyfit(range(len(scores)), scores, 1)[0]  # slope
P(ai) = sigmoid(mean(scores) + 3 * max(0, drift))
```

#### Evaluate all three on simulated clips

Report: precision, recall, F1 at threshold 0.5 for all three strategies.
Generate ROC curves for each strategy.
Plot: score trajectories over simulated clip length for 5 real and 5 fake examples.

#### What good looks like

* Strategy B should improve recall (catches more avatars) at the cost of some precision
  (flags some real people who happen to be low-motion in a clip)
* The consistency threshold needs calibration. Report AUC over a range of thresholds (0.05–0.30).
* Real person variance baseline: compute std of scores on the real-face test clips.
  This tells you where to set the consistency threshold in deployment.

---

## 8. Logging and Artifact Standards

### Per-experiment, always save these artifacts

All paths are relative to `/results/` on `avatar-results-vol`.

```
results/
├── experiments.db                      # SQLite — every experiment
├── trial1/
│   ├── config_1a.json                  # full hyperparameter dict
│   ├── config_1a_training_curves.png   # loss + AUC per epoch (2 subplots)
│   ├── config_1a_confusion_matrix.png
│   ├── config_1a_roc_curve.png         # with AUC annotation
│   ├── config_1a_score_dist.png        # score histogram: real (blue) vs fake (red)
│   ├── config_1a_freq_heatmap.png      # DCT real vs fake vs ratio
│   ├── config_1a_gradcam.png           # 2×4 grid: 2 real, 2 fake, orig+cam each
│   └── checkpoints/
│       └── config_1a_best.pt
├── trial2/
│   ├── ... (same per config 2a–2f)
│   ├── ablation_comparison.png         # bar chart: AUC per config
│   ├── embedding_umap.png              # UMAP of test embeddings
│   └── attention_maps.png             # attention heatmaps on sample faces
├── trial3/
│   ├── cross_dataset_matrix.csv        # the main thesis table
│   ├── cross_dataset_heatmap.png       # AUC heatmap
│   └── generalization_gap.png          # bar chart: within vs cross-dataset AUC per config
└── trial4/
    ├── score_trajectories.png
    ├── aggregation_roc_comparison.png
    └── consistency_threshold_sweep.png
```

### Training curve requirements

Every training curve plot must show:

* X axis: epoch number
* Y axis: metric value (use dual y-axis if combining loss + AUC)
* Both train and val curves on same plot
* Marker for best val AUC epoch (vertical dotted line)
* Annotation: best val AUC value in legend

Bad training curves to watch for and report in notes:

* Val loss goes down but val AUC stays flat → model is predicting confidently but wrong class
* Val AUC > train AUC after epoch 5 → data leakage (test images in train set)
* Val loss increases while train loss decreases after epoch 10 → overfitting
  → add to experiment notes: "overfit after epoch N, consider early stopping at N"

### Score distribution requirements

Plot histogram of P(ai) scores separately for real and fake test images.
The distribution should show clear bimodal separation (real near 0, fake near 1).
Report the overlap area as an informal calibration metric.
A well-calibrated model has real scores clustering at 0.1–0.2 and fake scores at 0.8–0.9.
A poorly calibrated model has both distributions centered near 0.5 (model is uncertain).

### Commit to Volume after every run

```python
results_vol.commit()  # after saving all artifacts for this experiment
```

---

## 9. Interpretation Guide — What Is Good Research vs. What Is a Problem

### Understanding AUC

**AUC (Area Under ROC Curve)** is your primary metric for this binary classification task
because it is threshold-independent. The test set has roughly equal classes so accuracy
is also meaningful, but AUC is more informative.

| AUC        | Interpretation                                                                       |
| ---------- | ------------------------------------------------------------------------------------ |
| > 0.97     | Excellent — but verify no data leakage                                              |
| 0.93–0.97 | Good — publishable range for within-dataset                                         |
| 0.87–0.93 | Acceptable — within-dataset baseline territory                                      |
| 0.80–0.87 | Weak — model is learning something but not enough                                   |
| < 0.80     | Poor — likely a data loading bug, normalization issue, or the model is not training |

 **Cross-dataset AUC** : expect 5–15% lower than within-dataset. A drop of > 20% indicates
the model learned dataset-specific shortcuts. A drop of < 5% is suspiciously good — check
that the two datasets don't share images or identical generators.

### Understanding EER (Equal Error Rate)

EER is where FPR = FNR. Lower is better. For interview use case, you want EER < 15%.
This means: at the operating threshold, you flag 15% of real candidates as AI (false alarms)
and miss 15% of AI candidates (misses). This is the deployment trade-off.

### Understanding TPR@FPR=1%

In a real interview platform, you cannot flag 30% of genuine candidates. The operational
constraint is low false positive rate. **TPR@FPR=1% is the metric that matters for deployment.**
For a model to be deployable: TPR@FPR=1% should be > 50%.

### Reading ablation results

The ablation table (Trial 2, configs 2a–2f) should show monotonically improving AUC
as you add components: 2a < 2b < 2c < 2e (approximately).

If adding a component *reduces* AUC:

* SRM branch hurts: your SRM residuals are too noisy or poorly normalized — check clamping
* Attention gate hurts: the gate may be collapsing (outputting ~0.5 everywhere) — add
  gradient monitoring, check if attention weights have meaningful variance
* SupCon hurts within-dataset: this is actually EXPECTED and acceptable — SupCon trades
  in-domain accuracy for cross-domain generalization

### Reading cross-dataset results

The generalization gap (within AUC − cross-dataset AUC) is the core thesis number.
Plot this gap for each config from 2a to 2e.

Expected story:

* Config 2a (RGB only): large gap (model learned generator-specific cues)
* Config 2e (full model): smaller gap (SRM + SupCon generalize better)
* Gap reduction of 5+ percentage points = strong thesis contribution

If the gap does NOT reduce: this is also a valid finding — you can argue that
frequency residuals alone are insufficient and more diverse training data (Trial 3d) is needed.
A negative result explained rigorously is still a valid academic contribution.

---

## 10. What the Agent Should NOT Do

* Do not skip the smoke test before full training. Read the Modal skill gate requirements.
* Do not assume dataset directory structure — always inspect before writing code.
* Do not use the 9.6K dataset for training. It is OOD test only.
* Do not run full A10G training until the smoke test (8-sample overfit) succeeds.
* Do not merge all datasets into one pool without recording `dataset` origin in manifest.
* Do not commit API keys to git.
* Do not compute AUC on the validation set during hyperparameter search — that leaks.
  Use a separate "development test" set or rely purely on val AUC.
* Do not run 24 experiment configs sequentially without pausing to check first 2-3 look sane.

---

## 11. Final Report Integration

After all trials complete, generate `results/THESIS_SUMMARY.md`:

```markdown
# Avatar Detection Module — Results Summary

## Dataset Summary
[query dataset_audit table, print as table]

## Trial 1: DCT Baseline
Best config: [config name]
Test AUC (within): [value]
Improvement over v11: +[X]%

## Trial 2: SRM+ConvNeXt Ablations
[paste ablation comparison table]

## Trial 3: Cross-Dataset Generalization Matrix
[paste cross_dataset_matrix.csv as markdown table]
Generalization gap (best model): [within AUC] - [cross AUC] = [gap]%
Gap reduction vs baseline: [X]%

## Trial 4: Video Aggregation
Best strategy: [A/B/C]
TPR@FPR=1%: [value]

## References Used
[list all papers cited above]
```

This file is the direct input for Section 6.2 of the EraMatch Final Report thesis template.

---

## 12. Key References (cite these in thesis)

1. Fridrich & Kodovsky (2012).  *Rich Models for Steganalysis of Digital Images* .
   IEEE TIFS. — SRM filter kernels source.
2. Frank et al. (2020).  *Leveraging Frequency Analysis for Deep Fake Image Recognition* .
   ICML. arXiv:2003.08685 — foundational frequency-domain deepfake detection.
3. Khosla et al. (2020).  *Supervised Contrastive Learning* .
   NeurIPS. arXiv:2004.11362 — SupCon loss theory.
4. Yan et al. (2023).  *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection* .
   NeurIPS. — evaluation protocols and reproducibility baselines.
5. Nguyen et al. (2024).  *LAA-Net: Localized Artifact Attention Network for
   Quality-Agnostic and Generalizable Deepfake Detection* . CVPR 2024. — attention gate design.
6. Liu & Tan (2023).  *Contrastive learning-based general Deepfake detection with
   multi-scale RGB frequency clues* . Pattern Recognition 2023. — CLGD, SupCon + SRM fusion.
7. arXiv:2504.04827 (2025).  *From Specificity to Generality: Revisiting Generalizable
   Artifacts in Detecting Face Deepfakes* . — cross-generator artifact analysis.
8. arXiv:2304.07193 (2023).  *FreqBlender* . — frequency-domain augmentation for generalization.
9. Zheng et al. (2021).  *Exploring Temporal Coherence for More General Video Face Forgery
   Detection (FTCN)* . ICCV 2021. — temporal video detection baseline.

---

*Generated for EraMatch graduation project — Zewail City 2025/2026*
*Module: Avatar Detection (AI-generated interview video detection)*
*Author: Anas Ahmed | ID: 202202029*
