"""Test 1: Model Architecture Test"""

import sys
import torch

sys.path.append(".")

from models.feathernet import FeatherNetB

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 50)
print("TEST 1: Model Architecture")
print("=" * 50)

# Initialize model and move to device
model = FeatherNetB(num_classes=2).to(device)
print("✓ Model initialized")

# Check parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Total parameters: {total_params:,}")

# Test forward pass with dummy input (using batch size 2 to avoid BatchNorm issues)
dummy_input = torch.randn(2, 3, 128, 128, device=device)
model.eval()
try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output[0].item():.4f}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Test feature extraction
try:
    with torch.no_grad():
        features = model.extract_features(dummy_input)
    print(f"✓ Feature extraction works")
    print(f"  Feature shape: {features.shape}")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
