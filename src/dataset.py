"""
PyTorch Dataset + DataLoaders for the deepfake image classifier.

Expects folder structure:
  data/train/real/*.jpg
  data/train/fake/*.jpg
  data/val/real/*.jpg
  data/val/fake/*.jpg
  data/test/real/*.jpg
  data/test/fake/*.jpg

Label convention: real = 0, fake = 1
"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 224

# Training augmentation: helps the model generalize instead of memorizing
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),  # mimics compression blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),  # ImageNet stats (backbone pretrained on this)
])

# No augmentation for val/test -- we want to measure real performance
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


class DeepfakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: e.g. 'data/train'  (must contain real/ and fake/ subfolders)
        """
        self.samples = []  # list of (filepath, label)
        self.transform = transform

        for label_name, label_val in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_val))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. "
                f"Did you run split_dataset.py and put images in data/raw/real and data/raw/fake?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, float(label)


def get_dataloaders(data_root="data", batch_size=32, num_workers=4):
    train_ds = DeepfakeImageDataset(os.path.join(data_root, "train"), transform=train_transform)
    val_ds = DeepfakeImageDataset(os.path.join(data_root, "val"), transform=eval_transform)
    test_ds = DeepfakeImageDataset(os.path.join(data_root, "test"), transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # quick sanity check
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)  # [8, 3, 224, 224]
    print("Batch labels:", labels)
