"""Inference script for FAS system with ensemble support."""

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from models.feathernet import create_feathernet
from multi_model_predictor import MultiModelPredictor, create_simple_ensemble


class TemporalSmoothing:
    """Temporal smoothing for video predictions using moving average."""

    def __init__(self, window_size: int = 5, alpha: float = 0.7):
        """Initialize temporal smoothing.

        Args:
            window_size: Number of frames to smooth over
            alpha: Exponential moving average weight (0-1)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
        self.ema = None

    def update(self, prediction: float) -> float:
        """Update with new prediction and return smoothed value.

        Args:
            prediction: Current frame prediction (spoof probability)

        Returns:
            Smoothed prediction
        """
        self.history.append(prediction)

        # Simple moving average
        smoothed = sum(self.history) / len(self.history)

        # Exponential moving average
        if self.ema is None:
            self.ema = prediction
        else:
            self.ema = self.alpha * prediction + (1 - self.alpha) * self.ema

        # Combine both
        final = 0.6 * smoothed + 0.4 * self.ema

        return final

    def reset(self):
        """Reset smoothing history."""
        self.history.clear()
        self.ema = None


class FASInference:
    """Production-ready inference class with ensemble support."""

    def __init__(
        self,
        checkpoint_paths: Union[str, List[str]],
        device: str = "cpu",
        fusion_type: str = "average",
        weights: Optional[List[float]] = None,
        temporal_smoothing: bool = True,
        smoothing_window: int = 5,
        probability_mode: str = "spoof",
    ):
        """Initialize FAS inference.

        Args:
            checkpoint_paths: Single checkpoint or list of checkpoints
            device: Device to run on
            fusion_type: Fusion strategy for ensemble
            weights: Optional weights for weighted fusion
            temporal_smoothing: Whether to use temporal smoothing for video
            smoothing_window: Frames to smooth over
            probability_mode: What the model output represents ('spoof' or 'real')
        """
        self.device = torch.device(device)
        self.temporal_smoothing = temporal_smoothing
        self.smoother = TemporalSmoothing(window_size=smoothing_window) if temporal_smoothing else None
        if probability_mode not in {"spoof", "real"}:
            raise ValueError("probability_mode must be 'spoof' or 'real'")
        self.probability_mode = probability_mode

        # Setup transform
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Load model(s)
        if isinstance(checkpoint_paths, str):
            # Single model
            self.model = create_feathernet(
                num_classes=2,
                pretrained_path=checkpoint_paths,
                device=str(self.device)
            )
            self.model.eval()
            self.is_ensemble = False
        else:
            # Ensemble
            self.model = MultiModelPredictor(
                checkpoint_paths=checkpoint_paths,
                device=str(self.device),
                fusion_type=fusion_type,
                weights=weights,
                probability_mode=probability_mode,
            )
            self.is_ensemble = True

        print(f"FAS Inference initialized on {self.device}")
        if self.is_ensemble:
            print(f"  Ensemble with {len(checkpoint_paths)} models ({fusion_type} fusion)")
        print(f"  Temporal smoothing: {'enabled' if temporal_smoothing else 'disabled'}")
        print(f"  Probability mode: {probability_mode}")

    def predict_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        threshold: float = 0.5,
        return_features: bool = False,
    ) -> Dict:
        """Predict on a single image.

        Args:
            image: Image path, PIL Image, or numpy array
            threshold: Classification threshold
            return_features: Whether to return feature embeddings

        Returns:
            Dictionary with prediction results
        """
        # Load and convert image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        start_time = time.time()

        if self.is_ensemble:
            result = self.model.predict(image, return_individual=False)
            inference_time = time.time() - start_time
            result["inference_time"] = inference_time
            spoof_prob = result["probabilities"]["spoof"]
            real_prob = result["probabilities"]["real"]
            if self.probability_mode == "spoof":
                is_real = spoof_prob < threshold
            else:
                is_real = real_prob >= threshold
            result["is_real"] = is_real
            result["label"] = "Real" if is_real else "Fake"
            result["confidence"] = real_prob if is_real else spoof_prob
        else:
            # Single model inference
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                raw_prob = self.model(img_tensor).item()

            if self.probability_mode == "spoof":
                spoof_prob = raw_prob
                real_prob = 1.0 - raw_prob
            else:
                real_prob = raw_prob
                spoof_prob = 1.0 - raw_prob

            inference_time = time.time() - start_time

            if self.probability_mode == "spoof":
                is_real = spoof_prob < threshold
            else:
                is_real = real_prob >= threshold

            label = "Real" if is_real else "Fake"

            confidence = real_prob if is_real else spoof_prob
            probabilities = {
                "real": real_prob,
                "spoof": spoof_prob,
            }

            # Normalize any potential drift outside [0,1]
            probabilities = {k: max(0.0, min(1.0, v)) for k, v in probabilities.items()}

            result = {
                "label": label,
                "confidence": confidence,
                "is_real": is_real,
                "probabilities": probabilities,
                "inference_time": inference_time,
            }

        return result

    def predict_video(
        self,
        video_path: Union[str, int],
        threshold: float = 0.5,
        output_path: Optional[str] = None,
        display: bool = True,
        skip_frames: int = 0,
    ) -> List[Dict]:
        """Predict on video frames with temporal smoothing.

        Args:
            video_path: Path to video file or camera index
            threshold: Classification threshold
            output_path: Path to save output video
            display: Whether to display results
            skip_frames: Process every N frames (0 = all frames)

        Returns:
            List of predictions for processed frames
        """
        # Open video
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path)
            is_webcam = True
        elif str(video_path).isdigit():
            cap = cv2.VideoCapture(int(video_path))
            is_webcam = True
        else:
            cap = cv2.VideoCapture(str(video_path))
            is_webcam = False

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        predictions = []
        frame_count = 0
        processed_count = 0

        # Reset smoother
        if self.smoother:
            self.smoother.reset()

        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        display_fps = 0

        print(f"Processing {'webcam' if is_webcam else 'video'}...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames if requested
                if skip_frames > 0 and (frame_count % (skip_frames + 1)) != 1:
                    continue

                processed_count += 1

                # Convert to PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Predict
                result = self.predict_image(pil_image, threshold)

                # Apply temporal smoothing
                if self.smoother:
                    raw_prob = result["probabilities"]["spoof"]
                    smoothed_prob = self.smoother.update(raw_prob)

                    # Update result with smoothed values
                    is_real = smoothed_prob < threshold
                    result["label"] = "Real" if is_real else "Spoof"
                    result["is_real"] = is_real
                    result["probabilities"]["spoof"] = smoothed_prob
                    result["probabilities"]["real"] = 1.0 - smoothed_prob
                    result["confidence"] = smoothed_prob if not is_real else (1.0 - smoothed_prob)

                predictions.append({
                    "frame": frame_count,
                    **result
                })

                # Draw results on frame
                color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
                text = f"{result['label']}: {result['confidence']:.1%}"
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                # Draw FPS
                fps_frame_count += 1
                if fps_frame_count % 10 == 0:
                    elapsed = time.time() - fps_start_time
                    display_fps = fps_frame_count / elapsed if elapsed > 0 else 0

                cv2.putText(
                    frame,
                    f"FPS: {display_fps:.1f}",
                    (width - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Draw frame counter
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

                # Write frame
                if writer:
                    writer.write(frame)

                # Display frame
                if display:
                    cv2.imshow("FAS Inference", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord(" "):
                        # Pause on spacebar
                        cv2.waitKey(0)

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        total_time = time.time() - fps_start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0

        print(f"Processed {processed_count}/{frame_count} frames")
        print(f"Average FPS: {avg_fps:.1f}")
        if output_path:
            print(f"Output saved to {output_path}")

        return predictions

    def predict_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        threshold: float = 0.5,
    ) -> List[Dict]:
        """Predict on a batch of images.

        Args:
            images: List of images
            threshold: Classification threshold

        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            result = self.predict_image(image, threshold)
            results.append(result)
        return results


def main(args):
    """Main inference function."""
    # Setup device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load model(s)
    if args.ensemble:
        # Load ensemble
        checkpoint_dir = Path(args.checkpoint_dir)
        if args.checkpoints:
            checkpoint_paths = args.checkpoints
        else:
            checkpoint_paths = sorted(
                list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pth.tar"))
            )
            checkpoint_paths = [str(p) for p in checkpoint_paths]

        if not checkpoint_paths:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

        print(f"Loading ensemble with {len(checkpoint_paths)} models...")

        inference = FASInference(
            checkpoint_paths=checkpoint_paths,
            device=device,
            fusion_type=args.fusion_type,
            weights=args.weights,
            temporal_smoothing=args.temporal_smoothing,
            smoothing_window=args.smoothing_window,
        )
    else:
        # Single model
        checkpoint = args.checkpoint or "pth/AntiSpoofing_bin_128.pth"
        print(f"Loading model from {checkpoint}...")

        inference = FASInference(
            checkpoint_paths=checkpoint,
            device=device,
            temporal_smoothing=args.temporal_smoothing,
            smoothing_window=args.smoothing_window,
        )

    # Run inference
    if args.image:
        # Single image inference
        print(f"\nPredicting on image: {args.image}")
        result = inference.predict_image(args.image, threshold=args.threshold)

        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Label:           {result['label']}")
        print(f"Confidence:      {result['confidence']:.2%}")
        print(f"Real prob:       {result['probabilities']['real']:.2%}")
        print(f"Spoof prob:      {result['probabilities']['spoof']:.2%}")
        print(f"Inference time:  {result['inference_time']*1000:.1f}ms")
        print("=" * 60)

    elif args.video or args.camera is not None:
        # Video or camera inference
        video_source = args.video if args.video else args.camera
        print(f"\nPredicting on video: {video_source}")

        predictions = inference.predict_video(
            video_source,
            threshold=args.threshold,
            output_path=args.output,
            display=args.display,
            skip_frames=args.skip_frames,
        )

        # Print summary
        if predictions:
            total_frames = len(predictions)
            real_frames = sum(1 for p in predictions if p["is_real"])
            spoof_frames = total_frames - real_frames
            avg_confidence = sum(p["confidence"] for p in predictions) / total_frames

            print("\n" + "=" * 60)
            print("VIDEO PREDICTION SUMMARY")
            print("=" * 60)
            print(f"Total frames:      {total_frames}")
            print(f"Real frames:       {real_frames} ({real_frames / total_frames:.1%})")
            print(f"Spoof frames:      {spoof_frames} ({spoof_frames / total_frames:.1%})")
            print(f"Avg confidence:    {avg_confidence:.2%}")
            print("=" * 60)

    else:
        print("Error: Please specify --image, --video, or --camera")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAS Inference with Ensemble Support")

    # Input options
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--camera", type=int, help="Camera index (e.g., 0)")

    # Model options
    parser.add_argument("--checkpoint", type=str, help="Path to single model checkpoint")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble of models")
    parser.add_argument("--checkpoint-dir", type=str, default="pth", help="Directory containing checkpoints for ensemble")
    parser.add_argument("--checkpoints", type=str, nargs="+", help="Specific checkpoints to use in ensemble")
    parser.add_argument("--fusion-type", type=str, default="average", choices=["average", "weighted", "max", "voting"], help="Ensemble fusion strategy")
    parser.add_argument("--weights", type=float, nargs="+", help="Weights for weighted fusion")

    # Inference options
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--temporal-smoothing", action="store_true", default=True, help="Use temporal smoothing for video")
    parser.add_argument("--no-temporal-smoothing", dest="temporal_smoothing", action="store_false", help="Disable temporal smoothing")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Temporal smoothing window size")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process every N frames (0 = all)")

    # Output options
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--display", action="store_true", default=True, help="Display results in real-time")
    parser.add_argument("--no-display", dest="display", action="store_false", help="Disable display")

    args = parser.parse_args()
    main(args)
