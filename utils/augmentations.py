"""Data augmentation utilities for FAS system."""

from typing import Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    horizontal_flip: float = 0.5,
    rotation: int = 15,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> A.Compose:
    """Get training data augmentation pipeline.

    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean for RGB channels
        std: Normalization std for RGB channels
        horizontal_flip: Probability of horizontal flip
        rotation: Maximum rotation angle in degrees
        brightness: Brightness jitter factor
        contrast: Contrast jitter factor
        saturation: Saturation jitter factor
        hue: Hue jitter factor

    Returns:
        Albumentations composition of transforms
    """
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=horizontal_flip),
            A.Rotate(limit=rotation, p=0.5),
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                ],
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get validation/test data augmentation pipeline.

    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean for RGB channels
        std: Normalization std for RGB channels

    Returns:
        Albumentations composition of transforms
    """
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_test_time_augmentation(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    num_augmentations: int = 5,
) -> list:
    """Get test-time augmentation transforms for ensemble prediction.

    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean for RGB channels
        std: Normalization std for RGB channels
        num_augmentations: Number of different augmentations to generate

    Returns:
        List of transform compositions
    """
    base_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    augmentations = [base_transform]

    # Add horizontal flip
    if num_augmentations > 1:
        augmentations.append(
            A.Compose(
                [
                    A.Resize(height=image_size[0], width=image_size[1]),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        )

    # Add slight rotations
    if num_augmentations > 2:
        for angle in [-5, 5][: num_augmentations - 2]:
            augmentations.append(
                A.Compose(
                    [
                        A.Resize(height=image_size[0], width=image_size[1]),
                        A.Rotate(limit=(angle, angle), p=1.0),
                        A.Normalize(mean=mean, std=std),
                        ToTensorV2(),
                    ]
                )
            )

    return augmentations[:num_augmentations]
