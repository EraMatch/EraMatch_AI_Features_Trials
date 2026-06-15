# Avatar Detection

AI-powered avatar (AI-generated face) detection for interview candidate verification.

## Project Structure

### `src/`
Core source code for the dual-branch avatar detection model:
- **`src/data/`** — Data loading (`dataset.py`) and augmentations (`augmentations.py`)
- **`src/models/`** — DCT branch (`dct_branch.py`), SRM filters (`srm_filters.py`), dual-branch fusion (`dual_branch.py`), and custom losses (`losses.py`)
- **`src/utils/`** — Frequency analysis (`freq_analysis.py`), metrics (`metrics.py`), video aggregation (`video_aggregation.py`), visualization (`viz.py`), and database helpers (`db.py`)
- **`src/config.py`** — Central configuration

### `notebooks/`
Progressive complexity detection notebooks:
1. **`01_avatar_detection_efficientnet_baseline.ipynb`** — EfficientNet-B0 baseline with modified regression head → 6 behavioral scores
2. **`02_avatar_detection_cnn_lstm_temporal.ipynb`** — Temporal Transformer Encoder (8-frame sequence) → 5 scores + trend
3. **`03_avatar_detection_multimodal_audiovisual.ipynb`** — Cross-modal fusion (visual + audio via wav2vec2-base) → 5 scores + trend + mismatch
4. **`notebook_avatar_detection_dct_v11.ipynb`** — DCT-based frequency domain detection (latest version)
5. **`notebook_avatar_detection_dct.ipynb`** — DCT-based detection (base version)

### `modal/`
Modal Labs serverless training scripts:
- `train_nb1_dct.py` — DCT branch training
- `train_nb2_srm.py` — SRM branch training
- `train_nb3_cross.py` — Cross-modal fusion training
- `train_nb4_video.py` — Video-level aggregation training
- `setup_volume.py`, `setup_staged.py`, `quick_setup_C.py`, `quick_test_train.py` — Infrastructure
- `audit_gravex.py` — Model audit utility

### `kaggle/`
Kaggle deployment scripts:
- `trial_srm_convnext.py` — SRM + ConvNext trial
- `trial_srm_convnext_v2.py` — SRM + ConvNext v2 trial
- `push.py` — Kaggle push script
- `test_env.py` — Environment validation

### `tests/`
Pytest test suite (`conftest.py`, `pytest.ini`)

### `results/`
- `THESIS_SUMMARY.md` — Research thesis summary
- `checkpoints/` — Model checkpoint storage
- `plots/` — Visualization output

## Integration Target

FastAPI microservice on port 8001:
```python
# POST /api/v1/avatar-detection
# Input:  { "video_url": "https://storage.supabase.../interview_answer.mp4" }
# Output: avatar_detection_result JSON → candidate_interview_results table
```

## Dependencies

See `requirements.txt` for Python dependencies.

> **Full project documentation is available in [`AGENTS_PUBLIC.md`](AGENTS_PUBLIC.md).**

## About

This is the avatar detection module for **EraMatch**, an AI-powered recruitment platform. It classifies per-frame face crops as real or AI-generated during live video interviews.

**Research module** — not production-ready code.

Graduation project (CSAI 498/499) at Zewail City of Science and Technology.

**Author:** Anas Ahmed (ID 202202029)