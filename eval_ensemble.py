"""Evaluate ensemble model on test datasets."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from multi_model_predictor import MultiModelPredictor, create_simple_ensemble, create_weighted_ensemble
from utils.metrics import calculate_apcer_bpcer, calculate_eer


def evaluate_on_dataset(
    predictor: MultiModelPredictor,
    real_dir: Path,
    spoof_dir: Path,
    return_individual: bool = False,
) -> Dict:
    """Evaluate ensemble on a dataset with real and spoof directories.

    Args:
        predictor: MultiModelPredictor instance
        real_dir: Directory containing real face images
        spoof_dir: Directory containing spoof face images
        return_individual: If True, return individual model predictions

    Returns:
        Dictionary with evaluation metrics
    """
    # Get image paths
    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    spoof_images = list(spoof_dir.glob("*.jpg")) + list(spoof_dir.glob("*.png"))

    if len(real_images) == 0 or len(spoof_images) == 0:
        raise ValueError(f"No images found in real ({len(real_images)}) or spoof ({len(spoof_images)}) directories")

    print(f"Found {len(real_images)} real images and {len(spoof_images)} spoof images")

    # Collect predictions
    real_probs = []
    spoof_probs = []
    real_labels = []  # 0 for real
    spoof_labels = []  # 1 for spoof

    # Predict on real images
    print("Evaluating real images...")
    for img_path in real_images:
        try:
            image = Image.open(img_path).convert("RGB")
            result = predictor.predict(image, return_individual=return_individual)
            prob = result["probabilities"]["spoof"]
            real_probs.append(prob)
            real_labels.append(0)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Predict on spoof images
    print("Evaluating spoof images...")
    for img_path in spoof_images:
        try:
            image = Image.open(img_path).convert("RGB")
            result = predictor.predict(image, return_individual=return_individual)
            prob = result["probabilities"]["spoof"]
            spoof_probs.append(prob)
            spoof_labels.append(1)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Combine all predictions and labels
    all_probs = np.array(real_probs + spoof_probs)
    all_labels = np.array(real_labels + spoof_labels)

    # Calculate metrics
    apcer, bpcer = calculate_apcer_bpcer(all_probs, all_labels, threshold=0.5)
    acer = (apcer + bpcer) / 2
    eer, eer_threshold = calculate_eer(all_probs, all_labels)

    # Calculate accuracy
    predictions = (all_probs >= 0.5).astype(int)
    accuracy = np.mean(predictions == all_labels)

    results = {
        "num_real": len(real_probs),
        "num_spoof": len(spoof_probs),
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }

    if return_individual:
        results["individual_predictions"] = {
            "real": real_probs,
            "spoof": spoof_probs,
        }

    return results


def print_results(results: Dict, fusion_type: str):
    """Print evaluation results in a formatted way.

    Args:
        results: Dictionary with evaluation metrics
        fusion_type: Type of fusion used
    """
    print("\n" + "="*60)
    print(f"Evaluation Results ({fusion_type})")
    print("="*60)
    print(f"Real images:    {results['num_real']}")
    print(f"Spoof images:   {results['num_spoof']}")
    print("-"*60)
    print(f"Accuracy:       {results['accuracy']:.4f}")
    print(f"APCER:          {results['apcer']:.4f}")
    print(f"BPCER:          {results['bpcer']:.4f}")
    print(f"ACER:           {results['acer']:.4f}")
    print(f"EER:            {results['eer']:.4f} (threshold={results['eer_threshold']:.4f})")
    print("="*60 + "\n")


def compare_fusion_strategies(
    checkpoint_paths: List[str],
    real_dir: Path,
    spoof_dir: Path,
    device: str = "cpu",
) -> Dict:
    """Compare different fusion strategies on the same dataset.

    Args:
        checkpoint_paths: List of checkpoint paths
        real_dir: Directory with real images
        spoof_dir: Directory with spoof images
        device: Device to run on

    Returns:
        Dictionary with results for each fusion strategy
    """
    fusion_types = ["average", "weighted", "max", "voting"]
    all_results = {}

    for fusion_type in fusion_types:
        print(f"\nTesting fusion strategy: {fusion_type}")

        # Create predictor with this fusion type
        if fusion_type == "weighted":
            # Use equal weights by default
            weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
            predictor = MultiModelPredictor(
                checkpoint_paths=checkpoint_paths,
                device=device,
                fusion_type=fusion_type,
                weights=weights,
            )
        else:
            predictor = MultiModelPredictor(
                checkpoint_paths=checkpoint_paths,
                device=device,
                fusion_type=fusion_type,
            )

        # Evaluate
        results = evaluate_on_dataset(predictor, real_dir, spoof_dir)
        all_results[fusion_type] = results

        # Print results
        print_results(results, fusion_type)

    return all_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ensemble FAS model")

    # Dataset arguments
    parser.add_argument(
        "--real-dir",
        type=str,
        required=True,
        help="Directory containing real face images",
    )
    parser.add_argument(
        "--spoof-dir",
        type=str,
        required=True,
        help="Directory containing spoof face images",
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="pth",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        help="Specific checkpoint files to use (default: all in dir)",
    )

    # Fusion arguments
    parser.add_argument(
        "--fusion-type",
        type=str,
        default="average",
        choices=["average", "weighted", "max", "voting", "all"],
        help="Fusion strategy to use",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Weights for weighted fusion (must sum to 1.0)",
    )

    # Device argument
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
    )

    # Output argument
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Get checkpoint paths
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.checkpoint:
        checkpoint_paths = [str(p) for p in args.checkpoint]
    else:
        checkpoint_paths = [str(p) for p in sorted(list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pth.tar")))]

    if not checkpoint_paths:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    print(f"Using {len(checkpoint_paths)} models")

    # Validate weights
    if args.fusion_type == "weighted" and args.weights:
        if abs(sum(args.weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0 (got {sum(args.weights)})")
        if len(args.weights) != len(checkpoint_paths):
            raise ValueError(
                f"Number of weights ({len(args.weights)}) must match number of models ({len(checkpoint_paths)})"
            )

    # Get dataset directories
    real_dir = Path(args.real_dir)
    spoof_dir = Path(args.spoof_dir)

    if not real_dir.exists():
        raise FileNotFoundError(f"Real directory not found: {real_dir}")
    if not spoof_dir.exists():
        raise FileNotFoundError(f"Spoof directory not found: {spoof_dir}")

    # Run evaluation
    if args.fusion_type == "all":
        # Compare all fusion strategies
        results = compare_fusion_strategies(
            checkpoint_paths=checkpoint_paths,
            real_dir=real_dir,
            spoof_dir=spoof_dir,
            device=args.device,
        )

        # Save all results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        import json
        for fusion_type, fusion_results in results.items():
            output_file = output_dir / f"results_{fusion_type}.json"
            with open(output_file, "w") as f:
                json.dump(fusion_results, f, indent=2)
            print(f"Results saved to {output_file}")

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for fusion_type, fusion_results in results.items():
            print(f"{fusion_type:10s}: ACER={fusion_results['acer']:.4f}, EER={fusion_results['eer']:.4f}")
        print("="*60 + "\n")

    else:
        # Evaluate single fusion strategy
        print(f"\nEvaluating with {args.fusion_type} fusion...")

        if args.fusion_type == "weighted":
            predictor = MultiModelPredictor(
                checkpoint_paths=checkpoint_paths,
                device=args.device,
                fusion_type=args.fusion_type,
                weights=args.weights if args.weights else None,
            )
        else:
            predictor = MultiModelPredictor(
                checkpoint_paths=checkpoint_paths,
                device=args.device,
                fusion_type=args.fusion_type,
            )

        results = evaluate_on_dataset(predictor, real_dir, spoof_dir)
        print_results(results, args.fusion_type)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"results_{args.fusion_type}.json"

        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
