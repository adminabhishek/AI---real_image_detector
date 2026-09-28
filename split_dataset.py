"""
Splits data/raw/real and data/raw/fake into train/val/test (70/15/15).
Run AFTER you've put images into data/raw/real/ and data/raw/fake/

Usage: python split_dataset.py
"""
import os
import shutil
import random

random.seed(42)

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASSES = ["real", "fake"]

def split_class(cls_name):
    src_dir = f"data/raw/{cls_name}"
    if not os.path.isdir(src_dir):
        print(f"WARNING: {src_dir} does not exist, skipping.")
        return

    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    split_map = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    for split_name, split_files in split_map.items():
        dest_dir = f"data/{split_name}/{cls_name}"
        os.makedirs(dest_dir, exist_ok=True)
        for fname in split_files:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))

    print(f"[{cls_name}] total={n}  train={len(split_map['train'])}  "
          f"val={len(split_map['val'])}  test={len(split_map['test'])}")


if __name__ == "__main__":
    for c in CLASSES:
        split_class(c)
    print("\nDone. Check data/train, data/val, data/test folders.")
