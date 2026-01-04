"""Test 5: Single Image Inference Test"""

import sys

sys.path.append(".")

from utils.model_loader import FASPredictor

print("=" * 50)
print("TEST 5: Single Image Inference")
print("=" * 50)

# Initialize predictor
predictor = FASPredictor(model_path="pth/AntiSpoofing_bin_128.pth", device="cpu")
print("✓ Predictor initialized")

# Test on real image
print("\n[Real Image Test]")
real_img = "Oulu-NPU/true/1_2_47_1_1.jpg"
try:
    pred, conf = predictor.predict_image(real_img)
    print(f"✓ Prediction successful")
    print(f"  Image: {real_img}")
    print(f"  Prediction: {'REAL' if pred == 0 else 'SPOOF'}")
    print(f"  Confidence: {conf:.4f}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback

    traceback.print_exc()

# Test on spoof image
print("\n[Spoof Image Test]")
spoof_img = "Oulu-NPU/false/1_3_48_3_1.jpg"
try:
    pred, conf = predictor.predict_image(spoof_img)
    print(f"✓ Prediction successful")
    print(f"  Image: {spoof_img}")
    print(f"  Prediction: {'REAL' if pred == 0 else 'SPOOF'}")
    print(f"  Confidence: {conf:.4f}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback

    traceback.print_exc()

# Test batch prediction
print("\n[Batch Prediction Test]")
images = [real_img, spoof_img, real_img]
try:
    preds, confs = predictor.predict_batch(images)
    print(f"✓ Batch prediction successful")
    for i, (p, c) in enumerate(zip(preds, confs)):
        print(f"  Image {i + 1}: {'REAL' if p == 0 else 'SPOOF'} (conf: {c:.4f})")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback

    import traceback
    traceback.print_exc()
