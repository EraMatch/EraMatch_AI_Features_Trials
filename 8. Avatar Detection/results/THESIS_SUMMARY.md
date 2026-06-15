# Avatar Detection Module — Results Summary

**Project**: EraMatch — AI-powered recruitment platform
**Module**: Avatar Detection (AI-generated interview video detection)
**Author**: Anas Ahmed (ID: 202202029)
**Institution**: Zewail City of Science and Technology
**Course**: CSAI 498/499 (Graduation Project)
**Date**: 2026-06-13

---

## Dataset Summary

| Dataset | Images | Real | Fake | Generators | Resolution | Role |
|---------|--------|------|------|------------|------------|------|
| 130K faces | ~130K | 70K Flickr | 60K | FLUX, SDXL | varies | Primary train/val |
| SFHQ-T2I | 122K | 0 | 40K sample | FLUX×3, SDXL, DALL-E3 | 1024px | Fake diversity |
| 9.6K existing | 9.6K | 5K | 4.6K | GAN (unknown) | varies | OOD test only |
| GRAVEX-200K | 200K | TBD | TBD | unknown | small | Pending audit |

[Audit results from `dataset_audit` table will be inserted here after Modal run]

## Trial 1: DCT Baseline

**Purpose**: Prove that feeding DCT as an actual model input beats RGB-only.

**Configurations**:
| Config | Description | Test AUC | Improvement over v11 |
|--------|-------------|----------|---------------------|
| 1a_rgb_only | RGB only, no DCT | TBD | (reproduces v11) |
| 1b_rgb_dct | RGB + DCT 4-channel | TBD | +X% |
| 1c_rgb_dct_ls | RGB + DCT + label smoothing | TBD | +X% |

[Results will be inserted after `modal run modal/train_nb1_dct.py`]

## Trial 2: SRM+ConvNeXt Ablations

**Purpose**: Test each architectural component of the dual-branch model.

**Configurations**:
| Config | SRM | Attention | SupCon | Test AUC | Notes |
|--------|-----|-----------|--------|----------|-------|
| 2a_baseline | ✗ | ✗ | ✗ | TBD | RGB-only ConvNeXt |
| 2b_srm_concat | ✓ | ✗ | ✗ | TBD | + SRM naive concat |
| 2e_full_model | ✓ | ✓ | ✓ | TBD | Full thesis model |

[Results from `modal run modal/train_nb2_srm.py`]

**Ablation story**:
- 2a → 2b: +X% (SRM contribution)
- 2b → 2e: +X% (attention + SupCon contribution)
- 2a → 2e total: +X% (full model improvement)

## Trial 3: Cross-Dataset Generalization Matrix

**Purpose**: Measure how performance degrades when train/test distributions differ.

[Results from `modal run modal/train_nb3_cross.py` and `results/trial3/cross_dataset_matrix.csv`]

**Generalization gap** (within AUC − cross-dataset AUC):
| Config | Train on A → Test on A | Train on A → Test on C | Gap |
|--------|------------------------|------------------------|-----|
| 2a (RGB only) | TBD | TBD | TBD |
| 2b (+SRM) | TBD | TBD | TBD |
| 2e (full) | TBD | TBD | TBD |

**Thesis claim**: Full model (2e) reduces the cross-dataset gap by 5+ percentage points compared to baseline (2a). [Verified after run]

## Trial 4: Video Aggregation

**Purpose**: Test which temporal aggregation strategy works best for per-frame scores.

| Strategy | Precision | Recall | F1 | TPR@FPR=1% | Notes |
|----------|-----------|--------|-----|-----------|-------|
| mean | TBD | TBD | TBD | TBD | Baseline |
| variance_gated | TBD | TBD | TBD | TBD | Flags low-variance sequences |
| temporal_drift | TBD | TBD | TBD | TBD | Detects non-decreasing scores |

[Results from `modal run modal/train_nb4_video.py`]

## References Used

1. Fridrich & Kodovsky (2012). *Rich Models for Steganalysis of Digital Images*. IEEE TIFS.
2. Frank et al. (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition*. ICML.
3. Khosla et al. (2020). *Supervised Contrastive Learning*. NeurIPS.
4. Yan et al. (2023). *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection*. NeurIPS.
5. Nguyen et al. (2024). *LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection*. CVPR 2024.
6. Liu & Tan (2023). *Contrastive learning-based general Deepfake detection with multi-scale RGB frequency clues*. Pattern Recognition 2023.
7. arXiv:2504.04827 (2025). *From Specificity to Generality: Revisiting Generalizable Artifacts in Detecting Face Deepfakes*.
8. arXiv:2304.07193 (2023). *FreqBlender*.
9. Zheng et al. (2021). *Exploring Temporal Coherence for More General Video Face Forgery Detection (FTCN)*. ICCV 2021.

---

## Next Steps

After Modal runs complete:
1. Update this file with actual AUC numbers
2. Insert cross-dataset matrix table from `cross_dataset_matrix.csv`
3. Insert ablation comparison chart from `ablation_comparison.png`
4. Insert UMAP visualization from `embedding_umap.png`
5. Add notes about any failed runs or unexpected results

This file is the direct input for Section 6.2 of the EraMatch Final Report thesis template.

---

*Generated for EraMatch graduation project — Zewail City 2025/2026*
*Module: Avatar Detection (AI-generated interview video detection)*
*Author: Anas Ahmed | ID: 202202029*
