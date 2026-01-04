"""Test 6: Full Dataset Evaluation Test"""

import sys

sys.path.append(".")

from torch.utils.data import DataLoader

from utils.datasets import OULUDataset, SIWDataset
from utils.metrics import FASEvaluator
from utils.model_loader import load_pretrained_model
from utils.preprocessing import Preprocessor

print("=" * 50)
print("TEST 6: Full Dataset Evaluation")
print("=" * 50)

# Load model
model = load_pretrained_model("pth/AntiSpoofing_bin_128.pth", device="cpu")
print("✓ Model loaded")

# Initialize evaluator
evaluator = FASEvaluator()

# Test on OULU test set
print("\n[OULU Test Set Evaluation]")
try:
    preprocessor = Preprocessor()
    oulu_test = OULUDataset(root_dir="Oulu-NPU", split="test", transform=preprocessor)
    test_loader = DataLoader(oulu_test, batch_size=32, shuffle=False)

    print(f"  Test samples: {len(oulu_test)}")

    # Run evaluation
    results = evaluator.evaluate(model, test_loader)

    print(f"\n  Results:")
    print(f"    Accuracy:  {results['accuracy']:.4f}")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall:    {results['recall']:.4f}")
    print(f"    F1:        {results['f1']:.4f}")
    print(f"    AUC:       {results['auc']:.4f}")
    print(f"    APCER:     {results['apcer']:.4f}")
    print(f"    BPCER:     {results['bpcer']:.4f}")
    print(f"    ACER:      {results['acer']:.4f}")
    print(f"    EER:       {results['eer']:.4f}")

except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    import traceback

    traceback.print_exc()

# Test on SIW test set
print("\n[SIW Test Set Evaluation]")
try:
    siw_test = SIWDataset(root_dir="siw", split="test", transform=preprocessor)
    test_loader = DataLoader(siw_test, batch_size=32, shuffle=False)

    print(f"  Test samples: {len(siw_test)}")

    # Run evaluation
    results = evaluator.evaluate(model, test_loader)

    print(f"\n  Results:")
    print(f"    Accuracy:  {results['accuracy']:.4f}")
    print(f"    ACER:      {results['acer']:.4f}")
    print(f"    AUC:       {results['auc']:.4f}")

except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    import traceback

    traceback.print_exc()
