"""
Run this once to create the project folder structure.
Usage: python setup_folders.py
"""
import os

folders = [
    "data/raw/real",
    "data/raw/fake",
    "data/train/real",
    "data/train/fake",
    "data/val/real",
    "data/val/fake",
    "data/test/real",
    "data/test/fake",
    "checkpoints",
    "logs",
    "src",
]

for f in folders:
    os.makedirs(f, exist_ok=True)
    print(f"Created: {f}")

print("\nFolder structure ready.")
print("Next step: put your REAL face images in data/raw/real/")
print("and FAKE (deepfake) face images in data/raw/fake/")
print("Then run split_dataset.py to auto-split into train/val/test.")
