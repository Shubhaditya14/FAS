"""Inference script for FAS system."""

import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image

from models.backbones import get_backbone


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(
    model_config: dict, checkpoint_path: str, device: torch.device
) -> nn.Module:
    """Load model from checkpoint."""
    model = get_backbone(
        name=model_config["backbone"]["name"],
        pretrained=False,
        num_classes=model_config["classifier"]["num_classes"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def get_inference_transform(config: dict) -> A.Compose:
    """Get inference transform pipeline."""
    return A.Compose(
        [
            A.Resize(
                height=config["input"]["image_size"][0],
                width=config["input"]["image_size"][1],
            ),
            A.Normalize(
                mean=config["input"]["normalize"]["mean"],
                std=config["input"]["normalize"]["std"],
            ),
            ToTensorV2(),
        ]
    )


def predict_image(
    image_path: str,
    model: nn.Module,
    transform: A.Compose,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Predict on a single image.

    Args:
        image_path: Path to image file
        model: Trained model
        transform: Image transform pipeline
        device: Compute device
        threshold: Classification threshold

    Returns:
        Dictionary with prediction results
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Apply transforms
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    # Determine if real or spoof
    is_real = pred_class == 0
    label = "Real" if is_real else "Spoof"

    return {
        "label": label,
        "confidence": confidence,
        "is_real": is_real,
        "probabilities": {"real": probs[0, 0].item(), "spoof": probs[0, 1].item()},
    }


def predict_video(
    video_path: str,
    model: nn.Module,
    transform: A.Compose,
    device: torch.device,
    threshold: float = 0.5,
    output_path: str = None,
    display: bool = True,
) -> list:
    """Predict on video frames.

    Args:
        video_path: Path to video file or camera index
        model: Trained model
        transform: Image transform pipeline
        device: Compute device
        threshold: Classification threshold
        output_path: Path to save output video (optional)
        display: Whether to display results in real-time

    Returns:
        List of predictions for each frame
    """
    # Open video
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    predictions = []
    frame_count = 0

    print("Processing video...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply transforms
            transformed = transform(image=rgb_frame)
            image_tensor = transformed["image"].unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            is_real = pred_class == 0
            label = "Real" if is_real else "Spoof"

            predictions.append(
                {
                    "frame": frame_count,
                    "label": label,
                    "confidence": confidence,
                    "is_real": is_real,
                }
            )

            # Draw results on frame
            color = (0, 255, 0) if is_real else (0, 0, 255)
            text = f"{label}: {confidence:.2%}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Draw frame number
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Write frame
            if writer:
                writer.write(frame)

            # Display frame
            if display:
                cv2.imshow("FAS Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")
    if output_path:
        print(f"Output saved to {output_path}")

    return predictions


def main(args):
    """Main inference function."""
    # Load configurations
    model_config = load_config(args.model_config)
    inference_config = load_config(args.inference_config)

    # Setup device
    device_type = args.device or inference_config["model"]["device"]
    if device_type == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model
    checkpoint_path = args.checkpoint or inference_config["model"]["checkpoint_path"]
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(model_config, checkpoint_path, device)

    # Get transform
    transform = get_inference_transform(inference_config)
    threshold = inference_config["inference"]["confidence_threshold"]

    # Run inference
    if args.image:
        # Single image inference
        print(f"\nPredicting on image: {args.image}")
        result = predict_image(args.image, model, transform, device, threshold)

        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        print(f"Label:      {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Real prob:  {result['probabilities']['real']:.2%}")
        print(f"Spoof prob: {result['probabilities']['spoof']:.2%}")
        print("=" * 50)

    elif args.video or args.camera:
        # Video or camera inference
        video_source = args.video if args.video else str(args.camera)
        print(f"\nPredicting on video: {video_source}")

        predictions = predict_video(
            video_source,
            model,
            transform,
            device,
            threshold,
            output_path=args.output,
            display=args.display,
        )

        # Print summary
        total_frames = len(predictions)
        real_frames = sum(1 for p in predictions if p["is_real"])
        spoof_frames = total_frames - real_frames

        print("\n" + "=" * 50)
        print("VIDEO PREDICTION SUMMARY")
        print("=" * 50)
        print(f"Total frames:  {total_frames}")
        print(f"Real frames:   {real_frames} ({real_frames / total_frames:.1%})")
        print(f"Spoof frames:  {spoof_frames} ({spoof_frames / total_frames:.1%})")
        print("=" * 50)

    else:
        print("Error: Please specify --image, --video, or --camera")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAS Inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--inference-config",
        type=str,
        default="configs/inference_config.yaml",
        help="Path to inference configuration file",
    )
    parser.add_argument("--device", type=str, help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument(
        "--display", action="store_true", help="Display results in real-time"
        help='Display results in real-time'
    )

    args = parser.parse_args()
    main(args)
