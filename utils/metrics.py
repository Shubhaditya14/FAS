"""Metrics calculation utilities for FAS system."""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    classification_report,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate comprehensive metrics for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities (optional, for AUC)
        threshold: Classification threshold

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positive"] = int(tp)
    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)

    # Rates
    metrics["tpr"] = (
        tp / (tp + fn) if (tp + fn) > 0 else 0
    )  # True Positive Rate (Sensitivity)
    metrics["tnr"] = (
        tn / (tn + fp) if (tn + fp) > 0 else 0
    )  # True Negative Rate (Specificity)
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    # AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0

    # Attack Presentation Classification Error Rate (APCER)
    # Proportion of spoof images incorrectly classified as real
    spoof_mask = y_true == 1
    if spoof_mask.sum() > 0:
        metrics["apcer"] = (y_pred[spoof_mask] == 0).sum() / spoof_mask.sum()
    else:
        metrics["apcer"] = 0.0

    # Bona Fide Presentation Classification Error Rate (BPCER)
    # Proportion of real images incorrectly classified as spoof
    real_mask = y_true == 0
    if real_mask.sum() > 0:
        metrics["bpcer"] = (y_pred[real_mask] == 1).sum() / real_mask.sum()
    else:
        metrics["bpcer"] = 0.0

    # Average Classification Error Rate (ACER)
    # Average of APCER and BPCER
    metrics["acer"] = (metrics["apcer"] + metrics["bpcer"]) / 2.0

    return metrics


def calculate_eer(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Calculate Equal Error Rate (EER).

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities

    Returns:
        Tuple of (EER value, threshold at EER)
    """
    # Calculate FPR and TPR for different thresholds
    thresholds = np.linspace(0, 1, 100)
    fprs = []
    fnrs = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        fprs.append(fpr)
        fnrs.append(fnr)

    fprs = np.array(fprs)
    fnrs = np.array(fnrs)

    # Find threshold where FPR and FNR are closest
    diff = np.abs(fprs - fnrs)
    eer_idx = np.argmin(diff)

    eer = (fprs[eer_idx] + fnrs[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]

    return eer, eer_threshold


class MetricsTracker:
    """Track metrics across multiple epochs/batches."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics_history = []
        self.best_metrics = {}
        self.best_epoch = 0

    def update(self, metrics: Dict[str, float], epoch: int):
        """Update metrics for current epoch.

        Args:
            metrics: Dictionary of metric values
            epoch: Current epoch number
        """
        metrics["epoch"] = epoch
        self.metrics_history.append(metrics)

        # Update best metrics
        if not self.best_metrics or metrics.get("f1", 0) > self.best_metrics.get(
            "f1", 0
        ):
            self.best_metrics = metrics.copy()
            self.best_epoch = epoch

    def get_summary(self) -> str:
        """Get formatted summary of metrics.

        Returns:
            Formatted string with metrics summary
        """
        if not self.metrics_history:
            return "No metrics tracked yet."

        latest = self.metrics_history[-1]

        summary = f"\n{'=' * 60}\n"
        summary += f"Latest Metrics (Epoch {latest['epoch']}):\n"
        summary += f"{'-' * 60}\n"
        summary += f"Accuracy:  {latest.get('accuracy', 0):.4f}\n"
        summary += f"Precision: {latest.get('precision', 0):.4f}\n"
        summary += f"Recall:    {latest.get('recall', 0):.4f}\n"
        summary += f"F1 Score:  {latest.get('f1', 0):.4f}\n"
        if "auc" in latest:
            summary += f"AUC:       {latest.get('auc', 0):.4f}\n"
        summary += f"ACER:      {latest.get('acer', 0):.4f}\n"
        summary += f"{'-' * 60}\n"
        summary += f"Best F1 Score: {self.best_metrics.get('f1', 0):.4f} (Epoch {self.best_epoch})\n"
        summary += f"{'=' * 60}\n"
        summary += f"{'='*60}\n"

        return summary
