"""Test 3: Preprocessing Test"""

import sys

import cv2

sys.path.append(".")

from utils.preprocessing import Preprocessor

print("=" * 50)
print("TEST 3: Preprocessing Pipeline")
print("=" * 50)

# Load a sample image from OULU
image_path = "Oulu-NPU/true/1_2_47_1_1.jpg"

print(f"\nTesting with image: {image_path}")

# Initialize preprocessor
preprocessor = Preprocessor()

try:
    # Test with file path
    tensor = preprocessor(image_path)
    print(f"✓ Preprocessing from path works")
    print(f"  Output shape: {tensor.shape}")
    print(f"  Output dtype: {tensor.dtype}")
    print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    # Test with numpy array
    img_np = cv2.imread(image_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    tensor2 = preprocessor(img_np)
    print(f"✓ Preprocessing from numpy works")
    print(f"  Output shape: {tensor2.shape}")

    # Test batch preprocessing
    tensors = preprocessor.preprocess_batch([image_path, image_path])
    print(f"✓ Batch preprocessing works")
    print(f"  Batch shape: {tensors.shape}")

except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    import traceback

    print(f"✗ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
