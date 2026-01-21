"""Deepfake detection script using HF ViT model plus local FAS anti-spoofing."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from inference import FASInference


HF_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"


def load_hf_model(device: str) -> Tuple[AutoImageProcessor, AutoModelForImageClassification, torch.device]:
    """Load Hugging Face deepfake detector."""
    torch_device = torch.device(device)
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
    model.to(torch_device)
    model.eval()
    return processor, model, torch_device


def predict_deepfake(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModelForImageClassification,
    device: torch.device,
) -> Dict:
    """Run the HF deepfake detector and return fake/real probabilities."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    id2label = model.config.id2label
    # Normalize keys
    label_probs = {id2label[i].lower(): float(probs[i].item()) for i in range(len(probs))}
    # Try common keys
    fake_prob = label_probs.get("fake") or label_probs.get("deepfake") or max(label_probs.values())
    real_prob = label_probs.get("real") or (1.0 - fake_prob)

    fake_prob = min(max(fake_prob, 0.0), 1.0)
    real_prob = min(max(real_prob, 0.0), 1.0)

    return {
        "probabilities": {"fake": fake_prob, "real": real_prob},
        "label": "Fake" if fake_prob >= real_prob else "Real",
    }


def run(args: argparse.Namespace) -> None:
    """Execute ensemble deepfake detection."""
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Load models
    fas = FASInference(
        checkpoint_paths=args.checkpoint,
        device=args.device,
        temporal_smoothing=False,
        probability_mode="spoof",
    )
    processor, hf_model, torch_device = load_hf_model(args.device)

    # Inference
    fas_result = fas.predict_image(image, threshold=args.threshold)
    hf_result = predict_deepfake(image, processor, hf_model, torch_device)

    fas_spoof = fas_result["probabilities"]["spoof"]
    hf_fake = hf_result["probabilities"]["fake"]

    # Ensemble: weighted average of spoof/fake probs
    weight = args.fas_weight
    ensemble_fake = weight * fas_spoof + (1.0 - weight) * hf_fake
    ensemble_fake = min(max(ensemble_fake, 0.0), 1.0)
    ensemble_real = 1.0 - ensemble_fake
    ensemble_label = "Fake" if ensemble_fake >= args.threshold else "Real"

    output = {
        "ensemble": {
            "label": ensemble_label,
            "probabilities": {"real": ensemble_real, "fake": ensemble_fake},
            "weight_fas": weight,
            "weight_hf": 1.0 - weight,
        },
        "fas": fas_result,
        "huggingface": hf_result,
    }

    print(json.dumps(output, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake detection ensemble")
    parser.add_argument("--image", required=True, help="Path to image to score")
    parser.add_argument(
        "--checkpoint",
        default="pth/AntiSpoofing_bin_128.pth",
        help="Local FAS checkpoint (spoof-prob model)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on fake/spoof probability",
    )
    parser.add_argument(
        "--fas-weight",
        type=float,
        default=0.6,
        help="Weight for FAS spoof probability in the ensemble (0-1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
