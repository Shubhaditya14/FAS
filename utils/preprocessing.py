"""Preprocessing pipeline for FAS system."""

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms


class Preprocessor:
    """Basic preprocessing for inference."""

    def __init__(
        self,
        image_size=128,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        to_rgb=True,
    ):
        """Initialize preprocessor.

        Args:
            image_size: Target image size
            mean: Normalization mean (ImageNet stats)
            std: Normalization std (ImageNet stats)
            to_rgb: Convert BGR to RGB if using OpenCV
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb

        self.transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        """Preprocess image.

        Args:
            image: PIL Image, numpy array, or path to image

        Returns:
            Preprocessed tensor (C, H, W)
        """
        # Track if we loaded from path (and need BGR->RGB conversion)
        loaded_from_path = False

        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
            loaded_from_path = True

        # Convert PIL to numpy (PIL is RGB format)
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert BGR to RGB only if loaded from cv2.imread (which uses BGR)
        if (
            loaded_from_path
            and self.to_rgb
            and len(image.shape) == 3
            and image.shape[2] == 3
        ):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        return transformed["image"]

    def preprocess_batch(self, images):
        """Preprocess batch of images.

        Args:
            images: List of images (PIL, numpy, or paths)

        Returns:
            Batch tensor (B, C, H, W)
        """
        processed = [self(img) for img in images]
        return torch.stack(processed)


class Augmentor:
    """Training augmentations for FAS."""

    def __init__(
        self,
        image_size=128,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        horizontal_flip_prob=0.5,
        rotation_limit=15,
        brightness_limit=0.2,
        contrast_limit=0.2,
        use_strong_aug=False,
    ):
        """Initialize augmentor.

        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
            horizontal_flip_prob: Probability of horizontal flip
            rotation_limit: Maximum rotation angle
            brightness_limit: Brightness jitter limit
            contrast_limit: Contrast jitter limit
            use_strong_aug: Use stronger augmentations
        """
        self.image_size = image_size

        if use_strong_aug:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=horizontal_flip_prob),
                    A.Rotate(limit=rotation_limit, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=contrast_limit,
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=(10.0, 50.0)),
                            A.GaussianBlur(blur_limit=(3, 7)),
                            A.MotionBlur(blur_limit=5),
                        ],
                        p=0.3,
                    ),
                    A.OneOf(
                        [
                            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05),
                            A.GridDistortion(num_steps=5, distort_limit=0.05),
                        ],
                        p=0.2,
                    ),
                    A.CLAHE(clip_limit=2.0, p=0.2),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=image_size // 8,
                        max_width=image_size // 8,
                        p=0.2,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=horizontal_flip_prob),
                    A.Rotate(limit=rotation_limit, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=contrast_limit,
                        p=0.5,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

    def __call__(self, image):
        """Apply augmentations.

        Args:
            image: PIL Image or numpy array

        Returns:
            Augmented tensor (C, H, W)
        """
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Apply transforms
        transformed = self.transform(image=image)
        return transformed["image"]


class TestTimeAugmentor:
    """Test-time augmentation for improved inference."""

    def __init__(
        self,
        image_size=128,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_augmentations=5,
    ):
        """Initialize TTA.

        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
            num_augmentations: Number of augmented versions
        """
        self.image_size = image_size
        self.num_augmentations = num_augmentations

        # Base transform (no augmentation)
        self.transforms = [
            A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        ]

        # Horizontal flip
        if num_augmentations > 1:
            self.transforms.append(
                A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.HorizontalFlip(p=1.0),
                        A.Normalize(mean=mean, std=std),
                        ToTensorV2(),
                    ]
                )
            )

        # Rotations
        if num_augmentations > 2:
            for angle in [-5, 5, -10, 10][: num_augmentations - 2]:
                self.transforms.append(
                    A.Compose(
                        [
                            A.Resize(image_size, image_size),
                            A.Rotate(limit=(angle, angle), p=1.0),
                            A.Normalize(mean=mean, std=std),
                            ToTensorV2(),
                        ]
                    )
                )

    def __call__(self, image):
        """Apply TTA.

        Args:
            image: PIL Image or numpy array

        Returns:
            List of augmented tensors
        """
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Apply all transforms
        augmented = []
        for transform in self.transforms[: self.num_augmentations]:
            transformed = transform(image=image)
            augmented.append(transformed["image"])

        return augmented

    def predict_with_tta(self, image, model, device="cpu"):
        """Predict with TTA ensemble.

        Args:
            image: Input image
            model: Model for inference
            device: Device to use

        Returns:
            Averaged prediction
        """
        augmented = self(image)
        batch = torch.stack(augmented).to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(batch)

        # Average predictions
        return predictions.mean()


def create_preprocessor(config=None, augment=False, test_time_aug=False):
    """Factory function to create preprocessor.

    Args:
        config: Configuration dict (optional)
        augment: Use augmentation
        test_time_aug: Use test-time augmentation

    Returns:
        Preprocessor instance
    """
    if config is None:
        config = {
            "image_size": 128,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }

    if test_time_aug:
        return TestTimeAugmentor(
            image_size=config.get("image_size", 128),
            mean=config.get("mean", (0.485, 0.456, 0.406)),
            std=config.get("std", (0.229, 0.224, 0.225)),
            num_augmentations=config.get("num_augmentations", 5),
        )
    elif augment:
        return Augmentor(
            image_size=config.get("image_size", 128),
            mean=config.get("mean", (0.485, 0.456, 0.406)),
            std=config.get("std", (0.229, 0.224, 0.225)),
            horizontal_flip_prob=config.get("horizontal_flip_prob", 0.5),
            rotation_limit=config.get("rotation_limit", 15),
            use_strong_aug=config.get("use_strong_aug", False),
        )
    else:
        return Preprocessor(
            image_size=config.get("image_size", 128),
            mean=config.get("mean", (0.485, 0.456, 0.406)),
            std=config.get("std", (0.229, 0.224, 0.225)),
        )
