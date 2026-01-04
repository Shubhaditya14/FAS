"""Visualization utilities for FAS system."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "Accuracy",
    save_path: Optional[str] = None,
):
    """Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: List of training metrics per epoch (optional)
        val_metrics: List of validation metrics per epoch (optional)
        metric_name: Name of the metric being plotted
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(train_losses) + 1)

    if train_metrics is not None and val_metrics is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Plot losses
    ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot metrics if provided
    if train_metrics is not None and val_metrics is not None:
        ax2.plot(
            epochs, train_metrics, "b-", label=f"Training {metric_name}", linewidth=2
        )
        ax2.plot(
            epochs, val_metrics, "r-", label=f"Validation {metric_name}", linewidth=2
        )
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.set_title(
            f"Training and Validation {metric_name}", fontsize=14, fontweight="bold"
        )
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def visualize_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    num_images: int = 8,
    save_path: Optional[str] = None,
):
    """Visualize model predictions on a batch of images.

    Args:
        images: Batch of images (B, C, H, W)
        labels: Ground truth labels (B,)
        predictions: Predicted labels (B,)
        probabilities: Prediction probabilities (B, num_classes) (optional)
        class_names: List of class names (default: ['Real', 'Spoof'])
        num_images: Number of images to display
        save_path: Path to save the plot (optional)
    """
    if class_names is None:
        class_names = ["Real", "Spoof"]

    num_images = min(num_images, len(images))
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_images):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Convert image to numpy and denormalize
        img = images[idx].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))

        # Denormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        # Create title with prediction info
        true_label = class_names[labels[idx].item()]
        pred_label = class_names[predictions[idx].item()]

        if probabilities is not None:
            prob = probabilities[idx, predictions[idx]].item()
            title = f"True: {true_label}\nPred: {pred_label} ({prob:.2%})"
        else:
            title = f"True: {true_label}\nPred: {pred_label}"

        # Color based on correctness
        color = "green" if labels[idx] == predictions[idx] else "red"
        ax.set_title(title, color=color, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
):
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: List of class names (default: ['Real', 'Spoof'])
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot (optional)
    """
    if class_names is None:
        class_names = ["Real", "Spoof"]

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def visualize_attention_maps(
    images: torch.Tensor,
    attention_maps: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    num_images: int = 4,
    save_path: Optional[str] = None,
):
    """Visualize attention maps overlaid on original images.

    Args:
        images: Batch of images (B, C, H, W)
        attention_maps: Attention maps (B, H, W) or (B, 1, H, W)
        labels: Ground truth labels (optional)
        num_images: Number of images to display
        save_path: Path to save the plot (optional)
    """
    num_images = min(num_images, len(images))

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_images):
        # Original image
        img = images[idx].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))

        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Attention map
        if attention_maps.dim() == 4:
            attn = attention_maps[idx, 0].cpu().numpy()
        else:
            attn = attention_maps[idx].cpu().numpy()

        # Plot original image
        axes[idx, 0].imshow(img)
        if labels is not None:
            axes[idx, 0].set_title(
                f"Original (Label: {labels[idx].item()})", fontsize=12
            )
        else:
            axes[idx, 0].set_title("Original", fontsize=12)
        axes[idx, 0].axis("off")

        # Plot attention overlay
        axes[idx, 1].imshow(img)
        axes[idx, 1].imshow(attn, cmap="jet", alpha=0.5)
        axes[idx, 1].set_title("Attention Map", fontsize=12)
        axes[idx, 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()
