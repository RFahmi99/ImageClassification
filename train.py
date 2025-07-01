"""
train.py
========
Entry-point script for training a Tiny-ImageNet classifier.

Key features
------------
1. Heavy data augmentation via Albumentations  
2. CutMix + MixUp inside the training loop  
3. Cosine-annealing LR schedule  
4. Custom label-smoothed cross-entropy loss
"""

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# Local modules
from models.cnn import CNN
from utils import trainModel, CustomCrossEntropyLoss
from dataset.dataset import Dataset

# ---------------------------- #
# Configuration / Hyperparams  #
# ---------------------------- #
NUM_CLASSES = 200             # Tiny-ImageNet has 200 label IDs
IMG_SIZE    = 64              # Images are 64 × 64 pixels
BATCH_SIZE  = 128
EPOCHS      = 200
SEED        = 42

# Reproducibility
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- #
# Albumentations Pipelines     #
# ---------------------------- #
train_transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(p=0.3),
        A.ColorJitter(p=0.4),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.2,
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),   # Standard ImageNet mean / std
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ],
    additional_targets={"image": "label"},  # Needed for CutMix / MixUp targets
)

val_transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ],
    additional_targets={"image": "label"},
)

# ---------------------------- #
# Dataset & DataLoader         #
# ---------------------------- #
train_dataset = Dataset("./tiny-imagenet/train", transform=train_transform)
val_dataset   = Dataset("./tiny-imagenet/val",   transform=val_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------- #
# Model, Loss, Optimizer, LR   #
# ---------------------------- #
model     = CNN(num_classes=NUM_CLASSES).to(device)

# Label smoothing is applied inside the custom loss
loss_fn   = CustomCrossEntropyLoss(label_smoothing=0.1, num_classes=NUM_CLASSES)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

# Cosine annealing drops LR to 1 e-5 by epoch 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5,
)

# ---------------------------- #
# Kick off training            #
# ---------------------------- #
if __name__ == "__main__":
    trainModel(
        epochs       = EPOCHS,
        loss_fn      = loss_fn,
        optimizer    = optimizer,
        scheduler    = scheduler,
        train_loader = train_loader,
        val_loader   = val_loader,
        model        = model,
        device       = device,
        num_classes  = NUM_CLASSES,
    )
