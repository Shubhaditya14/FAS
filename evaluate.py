"""Evaluation script for FAS system."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from models.backbones import get_backbone
from utils.augmentations import get_val_transforms
from utils.data_loader import create_data_loaders
from utils.metrics import calculate_eer, calculate_metrics
from utils.visualization import plot_confusion_matrix, visualize_predictions


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint.get("metrics", {})


def evaluate(
    model: nn.Module, test_loader, device: torch.device, save_dir: Path
) -> dict:
    """Evaluate model on test set."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []

    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            # Store some images for visualization
            if len(all_images) < 16:
                all_images.append(
                    (images.cpu(), labels.cpu(), preds.cpu(), probs.cpu())
                )

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    # Calculate EER
    eer, eer_threshold = calculate_eer(all_labels, all_probs)
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold

    # Print metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1 Score:       {metrics['f1']:.4f}")
    print(f"AUC:            {metrics['auc']:.4f}")
    print(
        f"EER:            {metrics['eer']:.4f} (threshold: {metrics['eer_threshold']:.4f})"
    )
    print("-" * 60)
    print(f"APCER:          {metrics['apcer']:.4f}")
    print(f"BPCER:          {metrics['bpcer']:.4f}")
    print(f"ACER:           {metrics['acer']:.4f}")
    print("-" * 60)
    print(f"True Positives:  {metrics['true_positive']}")
    print(f"True Negatives:  {metrics['true_negative']}")
    print(f"False Positives: {metrics['false_positive']}")
    print(f"False Negatives: {metrics['false_negative']}")
    print("=" * 60)

    # Save confusion matrix
    print("\nGenerating visualizations...")
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        cm,
        class_names=["Real", "Spoof"],
        normalize=True,
        save_path=save_dir / "confusion_matrix.png",
    )

    # Visualize predictions
    if all_images:
        # Combine first batch of images
        vis_images = torch.cat([img[0] for img in all_images[:4]], dim=0)[:16]
        vis_labels = torch.cat([img[1] for img in all_images[:4]], dim=0)[:16]
        vis_preds = torch.cat([img[2] for img in all_images[:4]], dim=0)[:16]
        vis_probs = torch.cat([img[3] for img in all_images[:4]], dim=0)[:16]

        visualize_predictions(
            vis_images,
            vis_labels,
            vis_preds,
            vis_probs,
            class_names=["Real", "Spoof"],
            num_images=min(16, len(vis_images)),
            save_path=save_dir / "predictions.png",
        )

    # Save metrics to file
    metrics_file = save_dir / "evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")

    print(f"\nMetrics saved to {metrics_file}")

    return metrics


def main(args):
    """Main evaluation function."""
    # Load configurations
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create data loader
    print("Creating data loader...")
    val_transforms = get_val_transforms(
        image_size=tuple(train_config["augmentation"]["resize"]),
        mean=tuple(train_config["augmentation"]["normalize"]["mean"]),
        std=tuple(train_config["augmentation"]["normalize"]["std"]),
    )

    test_path = args.test_path or train_config["data"]["test_path"]
    _, _, test_loader = create_data_loaders(
        train_path=train_config["data"]["train_path"],
        val_path=train_config["data"]["val_path"],
        test_path=test_path,
        val_transform=val_transforms,
        batch_size=args.batch_size,
        num_workers=train_config["data"]["num_workers"],
        shuffle=False,
        pin_memory=False,
    )

    if test_loader is None:
        print("Error: Test data path not found!")
        return

    # Create model
    print("Creating model...")
    model = get_backbone(
        name=model_config["backbone"]["name"],
        pretrained=False,
        num_classes=model_config["classifier"]["num_classes"],
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, _ = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    metrics = evaluate(model, test_loader, device, save_dir)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FAS model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test data (overrides config)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to use (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results",
        help='Directory to save evaluation results'
    )

    args = parser.parse_args()
    main(args)
