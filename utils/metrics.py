"""Metrics calculation utilities for FAS system."""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
        summary += f"{'=' * 60}\n"

        return summary


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class FASEvaluator:
    """Comprehensive FAS evaluation class."""
    
    def __init__(self, device='cpu'):
        """Initialize evaluator.
        
        Args:
            device: Device to use for evaluation
        """
        self.device = device
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Run full evaluation on dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            threshold: Classification threshold
            
        Returns:
            Dictionary with all metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # Run inference
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Evaluating'):
                images = images.to(self.device)
                
                # Get predictions
                outputs = model(images)
                probs = outputs.cpu().numpy().flatten()
                
                all_probabilities.extend(probs)
                all_labels.extend(labels.numpy())
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_prob = np.array(all_probabilities)
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate all metrics
        metrics = calculate_metrics(y_true, y_pred, y_prob, threshold)
        
        # Calculate EER
        eer, eer_threshold = calculate_eer(y_true, y_prob)
        metrics['eer'] = eer
        metrics['eer_threshold'] = eer_threshold
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(y_true, y_prob)
        metrics['optimal_threshold'] = optimal_threshold
        
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = 'f1'
    ) -> float:
        """Find optimal classification threshold.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1', 'accuracy', 'acer')
            
        Returns:
            Optimal threshold value
        """
        thresholds = np.linspace(0, 1, 100)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'acer':
                metrics = calculate_metrics(y_true, y_pred)
                score = 1.0 - metrics['acer']  # Lower ACER is better
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: str = None
    ):
        """Plot ROC curve.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        save_path: str = None
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save report
            
        Returns:
            Formatted report string
        """
        report = "\n" + "="*70 + "\n"
        report += "FACE ANTI-SPOOFING EVALUATION REPORT\n"
        report += "="*70 + "\n\n"
        
        report += "Classification Metrics:\n"
        report += "-"*70 + "\n"
        report += f"  Accuracy:       {metrics.get('accuracy', 0):.4f}\n"
        report += f"  Precision:      {metrics.get('precision', 0):.4f}\n"
        report += f"  Recall:         {metrics.get('recall', 0):.4f}\n"
        report += f"  F1 Score:       {metrics.get('f1', 0):.4f}\n"
        if 'auc' in metrics:
            report += f"  AUC-ROC:        {metrics.get('auc', 0):.4f}\n"
        report += "\n"
        
        report += "FAS-Specific Metrics:\n"
        report += "-"*70 + "\n"
        report += f"  APCER:          {metrics.get('apcer', 0):.4f} (Attack pass rate)\n"
        report += f"  BPCER:          {metrics.get('bpcer', 0):.4f} (Real rejection rate)\n"
        report += f"  ACER:           {metrics.get('acer', 0):.4f} (Average error rate)\n"
        if 'eer' in metrics:
            report += f"  EER:            {metrics.get('eer', 0):.4f}\n"
            report += f"  EER Threshold:  {metrics.get('eer_threshold', 0.5):.4f}\n"
        if 'optimal_threshold' in metrics:
            report += f"  Optimal Thresh: {metrics.get('optimal_threshold', 0.5):.4f}\n"
        report += "\n"
        
        report += "Confusion Matrix:\n"
        report += "-"*70 + "\n"
        report += f"  True Positives:  {metrics.get('true_positive', 0)}\n"
        report += f"  True Negatives:  {metrics.get('true_negative', 0)}\n"
        report += f"  False Positives: {metrics.get('false_positive', 0)}\n"
        report += f"  False Negatives: {metrics.get('false_negative', 0)}\n"
        report += "\n"
        
        report += "Error Rates:\n"
        report += "-"*70 + "\n"
        report += f"  TPR (Sensitivity): {metrics.get('tpr', 0):.4f}\n"
        report += f"  TNR (Specificity): {metrics.get('tnr', 0):.4f}\n"
        report += f"  FPR:               {metrics.get('fpr', 0):.4f}\n"
        report += f"  FNR:               {metrics.get('fnr', 0):.4f}\n"
        
        report += "="*70 + "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
