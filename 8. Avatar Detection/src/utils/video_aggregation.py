"""Video-level aggregation strategies for Trial 4.

Three strategies for converting per-frame P(ai) scores into a
video-level decision:

1. mean_score       — Baseline arithmetic mean.
2. variance_gated   — Flags eerily consistent (low variance) sequences as AI.
3. temporal_drift   — Flags non-decreasing score trajectories (avatar stability).

The core insight: real people have natural micro-variation in frame-level
scores; AI avatars are eerily consistent → low std signals AI.

Reference: Zheng et al. (2021) "Exploring Temporal Coherence for More
General Video Face Forgery Detection" (FTCN), ICCV 2021.
"""

from typing import List, Sequence

import numpy as np


def mean_score(scores: Sequence[float]) -> float:
    """Baseline: arithmetic mean of per-frame scores.

    Args:
        scores: Per-frame P(ai) probabilities.

    Returns:
        Mean probability. Returns 0.0 for empty input.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def variance_gated(scores: Sequence[float], threshold: float = 0.1) -> float:
    """Variance-gated mean: flags eerily consistent sequences.

    Real people have natural micro-variation in per-frame scores.
    AI avatars are eerily consistent (low std). If the score sequence
    is too consistent (std < threshold), flag as AI even if mean is moderate.

    Args:
        scores: Per-frame P(ai) probabilities.
        threshold: Standard deviation below which the sequence is considered
            suspiciously consistent (default 0.1).

    Returns:
        max(mean, 0.7) if std < threshold, else mean.
        Returns 0.0 for empty input.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    m = float(np.mean(arr))
    if len(arr) < 2:
        # Single frame: can't compute variance; return raw mean
        return m
    std = float(np.std(arr, ddof=0))
    if std < threshold:
        return max(m, 0.7)
    return m


def temporal_drift(scores: Sequence[float]) -> float:
    """Temporal drift detection: flags non-decreasing score trajectories.

    AI avatars maintain consistent quality over time — their scores don't
    degrade. Real people may have moments where the model is less certain,
    causing score dips. A non-decreasing (positive slope) trajectory is
    a signal of AI-generated content.

    Args:
        scores: Per-frame P(ai) probabilities.

    Returns:
        sigmoid(mean + 3 * max(0, drift_slope)). Bounded to [0, 1].
        Returns 0.0 for empty input.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    if len(arr) < 2:
        # Single frame: no drift possible
        return float(arr[0])

    m = float(np.mean(arr))
    # Linear regression slope: positive = scores increasing over time
    x = np.arange(len(arr), dtype=np.float64)
    slope = float(np.polyfit(x, arr, 1)[0])

    # Only reward positive drift (non-decreasing = AI signal)
    positive_drift = max(0.0, slope)

    # Sigmoid to bound output in [0, 1]
    z = m + 3.0 * positive_drift
    return float(1.0 / (1.0 + np.exp(-z)))


# ---------------------------------------------------------------------------
# Strategy registry for Trial 4
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY = {
    "mean": mean_score,
    "variance_gated": variance_gated,
    "temporal_drift": temporal_drift,
}
