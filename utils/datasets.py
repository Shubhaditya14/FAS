"""Dataset classes for FAS system."""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OULUDataset(Dataset):
    """OULU-NPU dataset loader."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """Initialize OULU dataset.

        Args:
            root_dir: Path to Oulu-NPU directory
            split: 'train', 'val', or 'test'
            transform: Transform pipeline
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            seed: Random seed for splitting
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Load samples
        all_samples = self._load_all_samples()

        # Split data
        random.seed(seed)
        random.shuffle(all_samples)

        total = len(all_samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if split == "train":
            self.samples = all_samples[:train_end]
        elif split == "val":
            self.samples = all_samples[train_end:val_end]
        else:  # test
            self.samples = all_samples[val_end:]

        print(f"OULU {split}: {len(self.samples)} samples")

    def _load_all_samples(self) -> List[Tuple[Path, int]]:
        """Load all samples with labels."""
        samples = []

        # Real faces (label 0)
        true_dir = self.root_dir / "true"
        if true_dir.exists():
            for img_path in true_dir.glob("*.jpg"):
                samples.append((img_path, 0))

        # Spoof faces (label 1)
        false_dir = self.root_dir / "false"
        if false_dir.exists():
            for img_path in false_dir.glob("*.jpg"):
                samples.append((img_path, 1))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform
        if self.transform:
            if hasattr(self.transform, "__call__"):
                image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """Get class distribution."""
        labels = [label for _, label in self.samples]
        return {"real": labels.count(0), "spoof": labels.count(1)}


class SIWDataset(Dataset):
    """SIW dataset loader (already split into train/val/test)."""

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        """Initialize SIW dataset.

        Args:
            root_dir: Path to siw directory
            split: 'train', 'val', or 'test'
            transform: Transform pipeline
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Load samples
        self.samples = self._load_samples()

        print(f"SIW {split}: {len(self.samples)} samples")

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load samples for the split."""
        samples = []
        split_dir = self.root_dir / self.split

        # Real faces (label 0)
        real_dir = split_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*.jpg"):
                samples.append((img_path, 0))

        # Spoof faces (label 1)
        spoof_dir = split_dir / "spoof"
        if spoof_dir.exists():
            for img_path in spoof_dir.glob("*.jpg"):
                samples.append((img_path, 1))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform
        if self.transform:
            if hasattr(self.transform, "__call__"):
                image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """Get class distribution."""
        labels = [label for _, label in self.samples]
        return {"real": labels.count(0), "spoof": labels.count(1)}


class CombinedDataset(Dataset):
    """Combined OULU + SIW dataset for cross-dataset training."""

    def __init__(
        self,
        oulu_root: str,
        siw_root: str,
        split: str = "train",
        transform=None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """Initialize combined dataset.

        Args:
            oulu_root: Path to OULU-NPU directory
            siw_root: Path to SIW directory
            split: 'train', 'val', or 'test'
            transform: Transform pipeline
            train_ratio: Training ratio for OULU split
            val_ratio: Validation ratio for OULU split
            seed: Random seed
        """
        self.transform = transform

        # Load OULU dataset
        oulu_dataset = OULUDataset(
            oulu_root,
            split=split,
            transform=None,  # Apply transform in this class
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        # Load SIW dataset
        siw_dataset = SIWDataset(
            siw_root,
            split=split,
            transform=None,  # Apply transform in this class
        )

        # Combine samples
        self.samples = oulu_dataset.samples + siw_dataset.samples

        # Shuffle combined samples
        random.seed(seed)
        random.shuffle(self.samples)

        print(
            f"Combined {split}: {len(self.samples)} samples "
            f"(OULU: {len(oulu_dataset.samples)}, SIW: {len(siw_dataset.samples)})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform
        if self.transform:
            if hasattr(self.transform, "__call__"):
                image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """Get class distribution."""
        labels = [label for _, label in self.samples]
        return {"real": labels.count(0), "spoof": labels.count(1)}


def create_dataloader(
    dataset_name: str,
    root_dir: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    transform=None,
    **kwargs,
):
    """Factory function to create dataset and dataloader.

    Args:
        dataset_name: 'oulu', 'siw', or 'combined'
        root_dir: Root directory (or dict for combined)
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Shuffle data
        transform: Transform pipeline
        **kwargs: Additional dataset arguments

    Returns:
        DataLoader instance
    """
    if dataset_name.lower() == "oulu":
        dataset = OULUDataset(root_dir, split, transform, **kwargs)
    elif dataset_name.lower() == "siw":
        dataset = SIWDataset(root_dir, split, transform)
    elif dataset_name.lower() == "combined":
        if not isinstance(root_dir, dict):
            raise ValueError(
                "Combined dataset requires dict with 'oulu' and 'siw' keys"
            )
        dataset = CombinedDataset(
            root_dir["oulu"], root_dir["siw"], split, transform, **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
