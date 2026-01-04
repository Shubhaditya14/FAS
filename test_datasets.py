"""Test 4: Dataset Loading Test"""

import sys

sys.path.append(".")

from utils.datasets import OULUDataset, SIWDataset
from utils.preprocessing import Preprocessor
from torch.utils.data import DataLoader

print("=" * 50)
print("TEST 4: Dataset Loading")
print("=" * 50)

preprocessor = Preprocessor()

# Test OULU Dataset
print("\n[OULU Dataset]")
try:
    oulu_train = OULUDataset(root_dir="Oulu-NPU", split="train", transform=preprocessor)
    print(f"✓ OULU train dataset created")
    print(f"  Total samples: {len(oulu_train)}")
    print(f"  Class distribution: {oulu_train.get_class_distribution()}")

    # Load one sample
    img, label = oulu_train[0]
    print(f"✓ Sample loaded")
    print(f"  Image shape: {img.shape}")
    print(f"  Label: {label} ({'real' if label == 0 else 'spoof'})")

    # Test dataloader
    dataloader = DataLoader(oulu_train, batch_size=16, shuffle=True)
    batch_img, batch_label = next(iter(dataloader))
    print(f"✓ DataLoader works")
    print(f"  Batch shape: {batch_img.shape}")
    print(f"  Batch labels: {batch_label}")

except Exception as e:
    print(f"✗ OULU dataset failed: {e}")
    import traceback

    traceback.print_exc()

# Test SIW Dataset
print("\n[SIW Dataset]")
try:
    siw_train = SIWDataset(root_dir="siw", split="train", transform=preprocessor)
    print(f"✓ SIW train dataset created")
    print(f"  Total samples: {len(siw_train)}")
    print(f"  Class distribution: {siw_train.get_class_distribution()}")

    # Load one sample
    img, label = siw_train[0]
    print(f"✓ Sample loaded")
    print(f"  Image shape: {img.shape}")
    print(f"  Label: {label}")

except Exception as e:
    print(f"✗ SIW dataset failed: {e}")
    import traceback

    print(f"✗ SIW dataset failed: {e}")
    import traceback
    traceback.print_exc()
