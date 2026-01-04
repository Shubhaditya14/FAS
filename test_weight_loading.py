"""Test 2: Pretrained Weight Loading Test"""

import sys


sys.path.append(".")

from models.feathernet import FeatherNetB
from utils.model_loader import load_pretrained_model

print("=" * 50)
print("TEST 2: Pretrained Weight Loading")
print("=" * 50)

model_files = {
    "binary": "pth/AntiSpoofing_bin_128.pth",
    "multiclass": "pth/AntiSpoofing_print-replay_128.pth",
}

for model_name, model_path in model_files.items():
    print(f"\nLoading {model_name} model:")

    try:
        model = load_pretrained_model(model_path, device="cpu")
        print(f"  ✓ Loaded successfully")

        # Check model is in eval mode
        print(f"  ✓ Training mode: {model.training}")

        # Test inference
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  ✓ Inference works, output: {output.item():.4f}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
