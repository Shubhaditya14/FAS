"""Test 1: Model Architecture Test"""

import sys


sys.path.append(".")

sys.path.append('.')

from models.feathernet import FeatherNetB

print("=" * 50)
print("TEST 1: Model Architecture")
print("=" * 50)

# Initialize model
model = FeatherNetB(num_classes=2)
print("✓ Model initialized")

# Check parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Total parameters: {total_params:,}")

# Test forward pass with dummy input
dummy_input = torch.randn(1, 3, 128, 128)
try:
    output = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.item():.4f}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")

# Test feature extraction
try:
    features = model.extract_features(dummy_input)
    print(f"✓ Feature extraction works")
    print(f"  Feature shape: {features.shape}")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
