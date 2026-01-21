"""Multi-model ensemble predictor for face anti-spoofing."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from models.feathernet import create_feathernet


class MultiModelPredictor:
    """Predictor that uses multiple models and averages their predictions."""

    def __init__(
        self,
        checkpoint_paths: List[str],
        device: str = "cpu",
        fusion_type: str = "average",
        weights: Optional[List[float]] = None,
        probability_mode: str = "spoof",
    ):
        """Initialize multi-model predictor.

        Args:
            checkpoint_paths: List of paths to model checkpoints
            device: Device to run inference on
            fusion_type: Type of fusion ('average', 'weighted', 'max', 'voting')
            weights: Optional weights for weighted fusion (must sum to 1.0)
            probability_mode: What the raw model output represents ('spoof' or 'real')
        """
        self.device = torch.device(device)
        self.fusion_type = fusion_type
        self.weights = weights
        if probability_mode not in {"spoof", "real"}:
            raise ValueError("probability_mode must be 'spoof' or 'real'")
        self.probability_mode = probability_mode

        if fusion_type == "weighted" and weights is None:
            raise ValueError("Weights must be provided for weighted fusion")

        if weights is not None:
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            if len(weights) != len(checkpoint_paths):
                raise ValueError("Number of weights must match number of models")

        self.models = []
        self.model_names = []

        for ckpt_path in checkpoint_paths:
            path = Path(ckpt_path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            model_name = path.stem
            self.model_names.append(model_name)

            model = create_feathernet(
                num_classes=2,
                pretrained_path=str(path),
                device=str(self.device)
            )
            model.eval()
            self.models.append(model)

        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        print(f"Loaded {len(self.models)} models on {self.device}")
        for name in self.model_names:
            print(f"  - {name}")

    def _get_single_prediction(
        self,
        model: nn.Module,
        image_tensor: torch.Tensor,
    ) -> float:
        """Get single model prediction (spoof probability)."""
        with torch.no_grad():
            prob = model(image_tensor).item()
            if self.probability_mode == "spoof":
                return prob
            return 1.0 - prob

    def predict(
        self,
        image: Image.Image,
        return_individual: bool = False,
    ) -> Dict:
        """Make prediction on image using ensemble.

        Args:
            image: PIL Image to classify
            return_individual: If True, return individual model predictions

        Returns:
            Dictionary with ensemble results
        """
        # Preprocess
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Get predictions from all models
        individual_predictions = []
        for model in self.models:
            prob = self._get_single_prediction(model, image_tensor)
            individual_predictions.append(prob)

        # Apply fusion strategy
        if self.fusion_type == "average":
            ensemble_prob = sum(individual_predictions) / len(individual_predictions)

        elif self.fusion_type == "weighted":
            if self.weights is not None:
                ensemble_prob = sum(
                    w * p for w, p in zip(self.weights, individual_predictions)
                )
            else:
                ensemble_prob = sum(individual_predictions) / len(individual_predictions)

        elif self.fusion_type == "max":
            ensemble_prob = max(individual_predictions)

        elif self.fusion_type == "voting":
            # Majority voting: count how many predict spoof (prob > 0.5)
            spoof_votes = sum(1 for p in individual_predictions if p > 0.5)
            ensemble_prob = spoof_votes / len(individual_predictions)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Determine final classification
        is_real = ensemble_prob < 0.5
        label = "Real" if is_real else "Fake"
        confidence = ensemble_prob if not is_real else (1.0 - ensemble_prob)

        result = {
            "label": label,
            "confidence": confidence,
            "is_real": is_real,
            "probabilities": {
                "real": 1.0 - ensemble_prob,
                "spoof": ensemble_prob,
            },
            "fusion_type": self.fusion_type,
        }

        if return_individual:
            result["individual_predictions"] = {
                name: prob
                for name, prob in zip(self.model_names, individual_predictions)
            }

        return result

    def predict_batch(
        self,
        images: List[Image.Image],
        return_individual: bool = False,
    ) -> List[Dict]:
        """Make predictions on a batch of images.

        Args:
            images: List of PIL Images
            return_individual: If True, return individual model predictions

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image, return_individual=return_individual)
            results.append(result)
        return results

    def predict_directory(
        self,
        directory: Path,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        return_individual: bool = False,
    ) -> List[Dict]:
        """Make predictions on all images in a directory.

        Args:
            directory: Path to directory
            extensions: Allowed image extensions
            return_individual: If True, return individual model predictions

        Returns:
            List of prediction dictionaries
        """
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))

        results = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                result = self.predict(image, return_individual=return_individual)
                result["image_path"] = str(img_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        return results


def create_simple_ensemble(
    checkpoint_dir: str = "pth",
    device: str = "cpu",
) -> MultiModelPredictor:
    """Create a simple averaging ensemble from all models in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        device: Device to run on

    Returns:
        MultiModelPredictor with averaging fusion
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    checkpoint_paths = sorted(list(checkpoint_path.glob("*.pth")) + list(checkpoint_path.glob("*.pth.tar")))
    if not checkpoint_paths:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    print(f"Found {len(checkpoint_paths)} checkpoints")

    return MultiModelPredictor(
        checkpoint_paths=[str(p) for p in checkpoint_paths],
        device=device,
        fusion_type="average",
    )


def create_weighted_ensemble(
    checkpoint_dir: str = "pth",
    weights: Optional[List[float]] = None,
    device: str = "cpu",
) -> MultiModelPredictor:
    """Create a weighted averaging ensemble.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        weights: Weights for each model (must sum to 1.0)
        device: Device to run on

    Returns:
        MultiModelPredictor with weighted fusion
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_paths = sorted(list(checkpoint_path.glob("*.pth")) + list(checkpoint_path.glob("*.pth.tar")))

    if weights is None:
        # Equal weights by default
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)

    print(f"Found {len(checkpoint_paths)} checkpoints")
    print(f"Using weights: {weights}")

    return MultiModelPredictor(
        checkpoint_paths=[str(p) for p in checkpoint_paths],
        device=device,
        fusion_type="weighted",
        weights=weights,
    )


if __name__ == "__main__":
    # Test the ensemble
    print("Testing multi-model ensemble...")

    # Create ensemble
    ensemble = create_simple_ensemble(device="mps" if torch.backends.mps.is_available() else "cpu")

    # Test with dummy image
    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))

    result = ensemble.predict(dummy_image, return_individual=True)

    print("\nPrediction result:")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Real prob: {result['probabilities']['real']:.3f}")
    print(f"  Spoof prob: {result['probabilities']['spoof']:.3f}")

    if "individual_predictions" in result:
        print("\nIndividual predictions:")
        for name, prob in result["individual_predictions"].items():
            print(f"  {name}: {prob:.3f}")
