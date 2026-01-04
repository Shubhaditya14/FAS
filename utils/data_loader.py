"""Data loading utilities for FAS system."""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class FASDataset(Dataset):
    """Face Anti-Spoofing Dataset.

    Expected directory structure:
        data_path/
            real/
                img1.jpg
                img2.jpg
            spoof/
                img1.jpg
                img2.jpg
    """

    def __init__(
        self,
        data_path: str,
        transform=None,
        label_map: Optional[dict] = None
    ):
        """Initialize FAS Dataset.

        Args:
            data_path: Path to dataset directory
            transform: Albumentations or torchvision transforms
            label_map: Custom label mapping (default: {'real': 0, 'spoof': 1})
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.label_map = label_map or {'real': 0, 'spoof': 1}

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all samples with their labels."""
        samples = []

        for class_name, label in self.label_map.items():
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist")
                continue

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append((img_path, label))

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Apply transforms
        if self.transform is not None:
            if hasattr(self.transform, '__call__'):
                # Albumentations transform
                if isinstance(self.transform, object) and hasattr(self.transform, 'transform'):
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    # torchvision transform
                    image = self.transform(image)

        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    train_transform=None,
    val_transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create data loaders for training, validation, and optionally testing.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data (optional)
        train_transform: Transforms for training data
        val_transform: Transforms for validation/test data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        shuffle: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = FASDataset(train_path, transform=train_transform)
    val_dataset = FASDataset(val_path, transform=val_transform)
    test_dataset = FASDataset(test_path, transform=val_transform) if test_path else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    return train_loader, val_loader, test_loader
