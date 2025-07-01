"""
utils.py
========
Utility helpers for training:
• trainModel(...)   – full epoch/validation loop with CutMix-or-MixUp
• saveModel(...)    – checkpoint saver
• loadModel(...)    – checkpoint loader
• CustomCrossEntropyLoss – label-smoothed CE that accepts hard or soft labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from pathlib import Path

# Default checkpoint path (will be created if absent)
SAVE_PATH = Path("models/weights/model.pth")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------
# Training loop
# --------------------------------------------------------
def trainModel(
    epochs,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    model,
    device,
    num_classes,
):
    """
    Full training + validation loop with CutMix / MixUp augmentation.

    Resumes automatically if a checkpoint exists.
    """

    # 1️⃣ Resume training if checkpoint exists
    try:
        start_epoch, best_loss = loadModel(model, optimizer, scheduler)
        print(f"[Resume] Training from epoch {start_epoch}")
    except FileNotFoundError:
        print("[Start] No checkpoint found – training from scratch.")
        start_epoch = 0
        best_loss = float("inf")

    # 2️⃣ Epoch loop
    for epoch in range(start_epoch, epochs):
        # --- Training phase ----------------------------------------
        model.train()
        running_loss = 0.0
        correct = total = 0

        # Albumentations opts are done; here we mix samples
        cutmix = v2.CutMix(num_classes=num_classes)
        mixup  = v2.MixUp(num_classes=num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images, labels = cutmix_or_mixup(images, labels)  # On-the-fly mixing

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Accuracy calculation – handle soft labels
            _, predicted = outputs.max(1)
            total += labels.size(0)
            if labels.ndim == 2:              # Soft targets from MixUp / CutMix
                labels = labels.argmax(dim=1)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = 100.0 * correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%"
        )

        # --- Validation phase -------------------------------------
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc  = 100.0 * val_correct / val_total
        print(
            f"   └─ Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%"
        )

        # Step LR scheduler and display LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"      LR @ epoch-end: {current_lr:.6f}")

        # --- Checkpointing ----------------------------------------
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            saveModel(epoch, model, optimizer, scheduler, best_loss)

        # Log to plaintext for quick inspection / plotting
        with open(SAVE_PATH.parent / "log.txt", "a") as f:
            f.write(
                f"{epoch+1},{epoch_loss:.4f},{epoch_acc:.2f},"
                f"{val_epoch_loss:.4f},{val_epoch_acc:.2f}\n"
            )


# --------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------
def saveModel(epoch, model, optimizer, scheduler, best_loss):
    """Save model, scheduler & optimizer state."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": best_loss,
        },
        SAVE_PATH,
    )
    print(f"[Checkpoint] Saved best model to {SAVE_PATH}")


def loadModel(model, optimizer, scheduler):
    """Load latest checkpoint and restore training objects."""
    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Continue after the saved epoch
    best_loss   = checkpoint["loss"]
    return start_epoch, best_loss


# --------------------------------------------------------
# Custom loss with label smoothing & soft-label support
# --------------------------------------------------------
class CustomCrossEntropyLoss(nn.Module):
    """
    Cross-entropy that:
    • Accepts either hard labels (int) or soft labels (probabilities)  
    • Applies label smoothing when hard labels are given
    """

    def __init__(self, label_smoothing: float = 0.0, num_classes: int = 1_000):
        super().__init__()
        self.epsilon = label_smoothing
        self.num_classes = num_classes

    def forward(self, outputs, labels):
        """
        Parameters
        ----------
        outputs : Tensor [batch, C] – raw logits
        labels  : Tensor – shape [batch] (hard) or [batch, C] (soft)
        """
        # Hard labels → one-hot → optional smoothing
        if labels.dim() == 1:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
            labels = (1 - self.epsilon) * labels + self.epsilon / self.num_classes
        # Soft labels: optionally smooth again (rarely needed)
        elif self.epsilon > 0:
            labels = (1 - self.epsilon) * labels + self.epsilon / self.num_classes

        log_probs = F.log_softmax(outputs, dim=1)
        loss = -(labels * log_probs).sum(dim=1).mean()
        return loss