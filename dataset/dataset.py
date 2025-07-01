"""
dataset.py
==========
Tiny-ImageNet Dataset wrapper compatible with Albumentations.

Folder layout expected
----------------------
train/
    n01443537/
        images/ILSVRC2012_val_00000293.JPEG
        images/...
val/
    images/...
    val_annotations.txt   (ignored here; using subfolders like train)

Each class folder is assigned an integer ID based on alphabetical order.
"""

import os
from pathlib import Path
from typing import Tuple, Any

import torch
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    """
    Args
    ----
    root_dir : str | Path
        Path to the *train* or *val* directory containing class subfolders.
    transform : callable, optional
        Albumentations Compose that will receive a NumPy image.

    Yields
    ------
    image : Tensor[C, H, W] – normalized by transform
    label : int             – class index
    """

    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Alphabetical class mapping e.g. {'n01443537':0, ...}
        self.classes = sorted(
            d for d in os.listdir(self.root_dir)
            if (self.root_dir / d).is_dir()
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Build (filepath, class_id) list
        self.images: list[Tuple[Path, int]] = []
        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            # Tiny-ImageNet has an extra 'images' subfolder
            images_subdir = class_dir / "images"
            if images_subdir.is_dir():
                class_dir = images_subdir

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(
                        (class_dir / img_name, self.class_to_idx[cls_name])
                    )

    # Required by Dataset interface ---------------------------------
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.images[idx]

        # PIL → NumPy array for Albumentations
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply augmentation / normalization
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label