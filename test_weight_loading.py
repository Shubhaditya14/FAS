"""Test 2: Pretrained Weight Loading Test"""

import sys

import torch

sys.path.append(".")

from models.feathernet import FeatherNetB
from utils.model_loader import load_pretrained_model

# Device configuration
device_str = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device_str}")

print("=" * 50)
print("TEST 2: Pretrained Weight Loading")
print("=" * 50)

model_files = {
    "binary": ("pth/AntiSpoofing_bin_128.pth", 2),
    "multiclass": ("pth/AntiSpoofing_print-replay_128.pth", 3),
}

for model_name, (model_path, num_classes) in model_files.items():
    print(f"\nLoading {model_name} model (num_classes={num_classes}):")

    try:
        model = load_pretrained_model(
            model_path, device=device_str, num_classes=num_classes
        )
        print(f"  ✓ Loaded successfully")

        # Check model is in eval mode
        print(f"  ✓ Training mode: {model.training}")

        # Test inference
        device = torch.device(device_str)
        dummy_input = torch.randn(1, 3, 128, 128, device=device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  ✓ Inference works, output: {output.item():.4f}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
