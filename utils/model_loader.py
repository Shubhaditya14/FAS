"""Model loading and inference utilities."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from models.feathernet import FeatherNetB, create_feathernet
from utils.preprocessing import Preprocessor, create_preprocessor


def load_pretrained_model(
    model_path: str,
    model_type: str = "binary",
    device: str = "cpu",
    num_classes: int = 2,
) -> nn.Module:
    """Load pretrained FeatherNet model.

    Args:
        model_path: Path to .pth checkpoint
        model_type: 'binary' or 'multiclass'
        device: Device to load on ('cpu', 'cuda', 'mps')
        num_classes: Number of output classes

    Returns:
        Loaded model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix from DataParallel (only from START, preserve 'se_module')
    new_state_dict = {}
    for k, v in state_dict.items():
        # Use slicing to only remove prefix, not replace all occurrences
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Auto-detect num_classes from checkpoint if not specified
    detected_num_classes = num_classes
    if num_classes == 2:  # Only auto-detect if using default
        if "model.prob.weight" in new_state_dict:
            detected_num_classes = new_state_dict["model.prob.weight"].shape[0]
        elif "prob.weight" in new_state_dict:
            detected_num_classes = new_state_dict["prob.weight"].shape[0]
        elif "model.linear.weight" in new_state_dict:
            # Fallback: check if this is a binary model (128 -> 2) or multiclass
            detected_num_classes = 2  # Default for binary

        if detected_num_classes != num_classes:
            print(f"Auto-detected {detected_num_classes} classes from checkpoint")
            num_classes = detected_num_classes

    # Create model with correct num_classes
    model = FeatherNetB(num_classes=num_classes)

    # Load weights (allow missing keys for flexibility)
    model.load_state_dict(new_state_dict, strict=False)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Model type: {model_type}, Device: {device}")

    return model


class FASPredictor:
    """Face anti-spoofing predictor with easy interface.

    The model returns a 'spoof probability' - the probability that the image
    is a spoof/attack. Higher values indicate more likely spoof.

    Interpretation:
        - spoof_prob < threshold → predicted as REAL (0)
        - spoof_prob > threshold → predicted as SPOOF (1)

    Note: The pretrained model was trained on CelebA-Spoof with a low threshold
    (around 0.01-0.05) because it uses sigmoid on logits. For this model, a
    threshold of 0.1-0.3 may work better than the default 0.5.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        threshold: float = 0.1,
        image_size: int = 128,
    ):
        """Initialize FAS predictor.

        Args:
            model_path: Path to model checkpoint
            device: Device to use
            threshold: Classification threshold (default 0.1, adjust as needed)
            image_size: Input image size
        """
        self.device = device
        self.threshold = threshold

        # Load model
        self.model = load_pretrained_model(model_path, device=device)

        # Create preprocessor
        self.preprocessor = Preprocessor(image_size=image_size)

    def predict_image(
        self, image: Union[str, np.ndarray], return_confidence: bool = True
    ) -> Union[Tuple[int, float], int]:
        """Predict on single image.

        Args:
            image: Image path or numpy array
            return_confidence: Return confidence score

        Returns:
            (prediction, confidence) if return_confidence else prediction
            prediction: 0=real, 1=spoof
        """
        # Preprocess
        img_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)

        # Get prediction
        # Model returns probability of being SPOOF (after sigmoid)
        # output > threshold → predicted as SPOOF
        # output < threshold → predicted as REAL
        confidence = output.item()
        prediction = 1 if confidence > self.threshold else 0  # 1=spoof, 0=real

        if return_confidence:
            return prediction, confidence
        return prediction

    def predict_batch(
        self, images: List[Union[str, np.ndarray]], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on batch of images.

        Args:
            images: List of image paths or arrays
            batch_size: Batch size for processing

        Returns:
            (predictions, confidences) arrays
        """
        all_predictions = []
        all_confidences = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # Preprocess batch
            batch_tensors = [self.preprocessor(img) for img in batch]
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(batch_tensor)

            # Get predictions
            confidences = outputs.cpu().numpy().flatten()
            predictions = (confidences > self.threshold).astype(int)

            all_predictions.extend(predictions)
            all_confidences.extend(confidences)

        return np.array(all_predictions), np.array(all_confidences)

    def extract_features(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Extract feature embeddings from image.

        Args:
            image: Image path or numpy array

        Returns:
            Feature vector
        """
        # Preprocess
        img_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model.extract_features(img_tensor)

        return features.cpu().numpy().flatten()

    def predict_video(
        self, video_path: str, skip_frames: int = 1, aggregate: str = "mean"
    ) -> Tuple[int, float]:
        """Predict on video by aggregating frame predictions.

        Args:
            video_path: Path to video file
            skip_frames: Process every Nth frame
            aggregate: 'mean', 'max', or 'majority'

        Returns:
            (final_prediction, confidence)
        """
        import cv2

        cap = cv2.VideoCapture(video_path)

        predictions = []
        confidences = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Predict
                pred, conf = self.predict_image(frame_rgb)
                predictions.append(pred)
                confidences.append(conf)

            frame_idx += 1

        cap.release()

        # Aggregate predictions
        if len(predictions) == 0:
            return 0, 0.0

        if aggregate == "mean":
            avg_confidence = np.mean(confidences)
            final_pred = 1 if avg_confidence > self.threshold else 0
            return final_pred, avg_confidence
        elif aggregate == "max":
            max_conf = np.max(confidences)
            final_pred = 1 if max_conf > self.threshold else 0
            return final_pred, max_conf
        else:  # majority
            final_pred = int(np.round(np.mean(predictions)))
            avg_confidence = np.mean(confidences)
            return final_pred, avg_confidence

    def set_threshold(self, threshold: float):
        """Update classification threshold.

        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
        print(f"Threshold updated to {threshold}")


def create_predictor(
    model_path: str, device: str = "cpu", threshold: float = 0.5, config: dict = None
) -> FASPredictor:
    """Factory function to create FAS predictor.

    Args:
        model_path: Path to model checkpoint
        device: Device to use
        threshold: Classification threshold
        config: Configuration dict

    Returns:
        FASPredictor instance
    """
    if config is None:
        config = {}

    return FASPredictor(
        model_path=model_path,
        device=device,
        threshold=threshold,
        image_size=config.get("image_size", 128),
    )
