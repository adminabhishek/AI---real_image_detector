"""
Trains the deepfake image classifier.

Usage:
    python src/train.py

Checkpoints get saved to checkpoints/best_model.pth (best val accuracy so far).
Run this from the project root folder (so data/, checkpoints/ paths resolve correctly).
"""
import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model
from src.dataset import get_dataloaders

# ---------------- Config ----------------
EPOCHS = 15
BATCH_SIZE = 8       # RTX 2050 has 4GB VRAM — keep this small
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 4          # early stopping: stop if val loss doesn't improve for N epochs
CHECKPOINT_DIR = "checkpoints"
NUM_WORKERS = 0       # 0 avoids the shared-memory fork issue on Windows + saves RAM
ACCUMULATION_STEPS = 4  # simulate effective batch_size of 32 (8 * 4)
# -----------------------------------------


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # happens if val set has only one class present

    return {
        "loss": avg_loss, "accuracy": acc,
        "precision": precision, "recall": recall,
        "f1": f1, "auc": auc,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. Training will be slow. Check your CUDA/torch install.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    model = build_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            # Scale loss so gradients average over ACCUMULATION_STEPS micro-batches
            (loss / ACCUMULATION_STEPS).backward()

            # Step the optimizer only every ACCUMULATION_STEPS micro-batches
            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - start_time
        print(
            f"\nEpoch {epoch}: train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['auc']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        # Save best model + early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # Final test evaluation using best checkpoint
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\n===== FINAL TEST RESULTS =====")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
