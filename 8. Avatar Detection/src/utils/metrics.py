"""Evaluation metrics for avatar detection module.

Primary metric: AUC (threshold-independent).
Deployment metric: TPR@FPR=1% (critical per project spec).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under ROC Curve.

    Returns 0.5 for degenerate cases (single class, single sample).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) <= 1:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5

    return float(roc_auc_score(y_true, y_score))


def compute_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> float:
    """Compute F1 score."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0.0))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Equal Error Rate — threshold where FPR ≈ FNR.

    Returns 0.5 for degenerate cases.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) <= 1 or len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer = float(fpr[np.argmin(np.abs(fpr - fnr))])
    return eer


def compute_tpr_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = 0.01
) -> float:
    """Compute TPR at a given FPR target via linear interpolation.

    Critical deployment metric: TPR@FPR=1% means detecting 61%+ of fakes
    while only flagging 1% of real candidates.
    Returns 0.5 for degenerate cases.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) <= 1 or len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, _ = roc_curve(y_true, y_score)

    if fpr_target <= fpr[1]:
        return float(tpr[1])

    for i in range(1, len(fpr)):
        if fpr[i] >= fpr_target:
            slope = (tpr[i] - tpr[i - 1]) / (fpr[i] - fpr[i - 1])
            tpr_at_target = tpr[i - 1] + slope * (fpr_target - fpr[i - 1])
            return float(np.clip(tpr_at_target, 0.0, 1.0))

    return float(tpr[-1])


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute all evaluation metrics as a dict.

    Binary predictions are derived from y_score >= threshold.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tn = int(np.sum(tn_mask))
    fp = int(np.sum(fp_mask))

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": compute_auc(y_true, y_score),
        "f1": compute_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "specificity": specificity,
        "eer": compute_eer(y_true, y_score),
        "tpr_at_fpr1": compute_tpr_at_fpr(y_true, y_score, fpr_target=0.01),
    }


def calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions, computes |avg_confidence - avg_accuracy| per bin,
    weighted by bin size.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)

        count = int(np.sum(mask))
        if count == 0:
            continue

        avg_confidence = float(np.mean(y_score[mask]))
        avg_accuracy = float(np.mean(y_true[mask]))
        ece += (count / total) * abs(avg_confidence - avg_accuracy)

    return ece
