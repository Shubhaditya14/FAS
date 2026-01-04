"""Training script for FAS system."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.backbones import get_backbone
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.data_loader import create_data_loaders
from utils.metrics import MetricsTracker, calculate_metrics
from utils.visualization import plot_training_curves


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_device(device_type: str = "mps") -> torch.device:
    """Setup compute device."""
    if device_type == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    log_frequency: int = 10,
) -> tuple:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())

        # Update progress bar
        if (batch_idx + 1) % log_frequency == 0:
            pbar.set_postfix({"loss": running_loss / (batch_idx + 1)})

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )

    return epoch_loss, metrics


def validate_epoch(
    model: nn.Module, val_loader, criterion, device: torch.device, epoch: int
) -> tuple:
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            pbar.set_postfix(
                {"loss": running_loss / (len(all_labels) / val_loader.batch_size)}
            )

    # Calculate epoch metrics
    epoch_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )

    return epoch_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    save_path: str,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = str(Path(save_path).parent / "best_model.pth")
        torch.save(checkpoint, best_path)


def main(args):
    """Main training function."""
    # Load configurations
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)

    # Set random seed for reproducibility
    if train_config.get("seed"):
        torch.manual_seed(train_config["seed"])
        np.random.seed(train_config["seed"])

    # Setup device
    device = setup_device(train_config["device"]["type"])
    print(f"Using device: {device}")

    # Create data loaders
    print("Creating data loaders...")
    train_transforms = get_train_transforms(
        image_size=tuple(train_config["augmentation"]["resize"]),
        mean=tuple(train_config["augmentation"]["normalize"]["mean"]),
        std=tuple(train_config["augmentation"]["normalize"]["std"]),
        horizontal_flip=train_config["augmentation"]["horizontal_flip"],
        rotation=train_config["augmentation"]["rotation"],
        brightness=train_config["augmentation"]["color_jitter"]["brightness"],
        contrast=train_config["augmentation"]["color_jitter"]["contrast"],
        saturation=train_config["augmentation"]["color_jitter"]["saturation"],
        hue=train_config["augmentation"]["color_jitter"]["hue"],
    )

    val_transforms = get_val_transforms(
        image_size=tuple(train_config["augmentation"]["resize"]),
        mean=tuple(train_config["augmentation"]["normalize"]["mean"]),
        std=tuple(train_config["augmentation"]["normalize"]["std"]),
    )

    train_loader, val_loader, _ = create_data_loaders(
        train_path=train_config["data"]["train_path"],
        val_path=train_config["data"]["val_path"],
        train_transform=train_transforms,
        val_transform=val_transforms,
        batch_size=train_config["data"]["batch_size"],
        num_workers=train_config["data"]["num_workers"],
        shuffle=train_config["data"]["shuffle"],
        pin_memory=train_config["data"]["pin_memory"],
    )

    # Create model
    print("Creating model...")
    model = get_backbone(
        name=model_config["backbone"]["name"],
        pretrained=model_config["backbone"]["pretrained"],
        num_classes=model_config["classifier"]["num_classes"],
        freeze_layers=model_config["backbone"]["freeze_layers"],
    )
    model = model.to(device)

    # Setup loss function
    if train_config["loss"]["label_smoothing"] > 0:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=train_config["loss"]["label_smoothing"]
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # Setup optimizer
    if train_config["training"]["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config["training"]["learning_rate"],
            weight_decay=train_config["training"]["weight_decay"],
        )
    elif train_config["training"]["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config["training"]["learning_rate"],
            weight_decay=train_config["training"]["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config["training"]["learning_rate"],
            momentum=train_config["training"]["momentum"],
            weight_decay=train_config["training"]["weight_decay"],
        )

    # Setup learning rate scheduler
    if train_config["scheduler"]["type"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config["training"]["epochs"],
            eta_min=train_config["scheduler"]["min_lr"],
        )
    elif train_config["scheduler"]["type"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config["scheduler"]["step_size"],
            gamma=train_config["scheduler"]["gamma"],
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=train_config["scheduler"]["patience"],
            factor=train_config["scheduler"]["gamma"],
        )

    # Setup logging
    if train_config["logging"]["use_tensorboard"]:
        log_dir = Path(train_config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # Create checkpoint directory
    checkpoint_dir = Path(train_config["checkpoint"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {train_config['training']['epochs']} epochs...")
    metrics_tracker = MetricsTracker()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, train_config["training"]["epochs"] + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            train_config["logging"]["log_frequency"],
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Update learning rate
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])
        metrics_tracker.update(val_metrics, epoch)

        # Log to tensorboard
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("F1/val", val_metrics["f1"], epoch)
            writer.add_scalar("ACER/val", val_metrics["acer"], epoch)

        # Print metrics
        print(f"\nEpoch {epoch}/{train_config['training']['epochs']}")
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}"
        )
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val ACER: {val_metrics['acer']:.4f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % train_config["checkpoint"]["save_frequency"] == 0 or is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                is_best=is_best,
            )

        # Early stopping
        if train_config["regularization"]["early_stopping"]["enabled"]:
            if (
                patience_counter
                >= train_config["regularization"]["early_stopping"]["patience"]
            ):
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    # Save final plots
    print("\nGenerating training curves...")
    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        metric_name="Accuracy",
        save_path=checkpoint_dir / "training_curves.png",
    )

    # Print final summary
    print(metrics_tracker.get_summary())

    if writer:
        writer.close()

    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FAS model")
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
        help='Path to training configuration file'
    )

    args = parser.parse_args()
    main(args)
