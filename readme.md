# Tiny-ImageNet CNN Classifier 🔥

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Top-1%20Accuracy-65%25-brightgreen)](#-performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏆 Performance](#-performance)
- [🚀 Quick Start](#-quick-start)
- [📊 Model Architecture](#-model-architecture)
- [🛠️ Installation](#️-installation)
- [💻 Usage](#-usage)
- [📈 Training Details](#-training-details)
- [📁 Project Structure](#-project-structure)
- [🔧 Configuration](#-configuration)
- [📝 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview
A **ResNet-inspired Convolutional Neural Network** trained from scratch on Tiny-ImageNet (200 classes, 64 × 64 px).  
Through heavy data augmentation and modern regularization (CutMix, MixUp, label smoothing), it reaches **65% top-1 accuracy**.

## ✨ Key Features

| ✔️  | Description |
|-----|-------------|
| **Custom Architecture** | Residual CNN optimized for 64 × 64 images |
| **Advanced Augmentation** | Elastic transform, ColorJitter, CoarseDropout & more via Albumentations |
| **Modern Regularization** | CutMix, MixUp, label smoothing |
| **Cosine LR Schedule** | Smooth 1e-3 → 1e-5 over 200 epochs |
| **Checkpointing & Resume** | Auto-save best model and resume training |
| **Lightweight** | ~2.8M parameters, fast to train |

## 🏆 Performance

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | **65%** |
| Parameters | ~2.8M |
| Training Epochs | 200 |
| Dataset | Tiny-ImageNet-200 |

> Typical from-scratch baselines score 50–60%. This model beats them by a healthy margin.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/RFahmi99/ImageClassification.git
cd ImageClassification

# Install dependencies
pip install -r requirements.txt

# Download dataset (≈ 244 MB)
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip && mv tiny-imagenet-200 tiny-imagenet

# Train
python train.py
```

## 📊 Model Architecture

```text
Input (3×64×64)
  │
  ├── Conv-BN-ReLU (64)
  ├── Stage 1: 2× ResidualBlock (64)
  ├── Stage 2: 2× ResidualBlock (128) ↓2×
  ├── Stage 3: 2× ResidualBlock (256) ↓2×
  ├── Stage 4: 2× ResidualBlock (512) ↓2×
  ├── GlobalAvgPool
  ├── Dropout 0.5
  └── FC → 200
```

Key design choices: residual connections for deep supervision, batch norm everywhere, global average pooling to reduce overfitting.

## 🛠️ Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+, Albumentations 1.3+, Pillow, NumPy

```bash
pip install torch torchvision albumentations pillow numpy
```

Or:

```bash
pip install -r requirements.txt
```

### Dataset Layout
```text
tiny-imagenet/
├── train/
│   └── <class_id>/images/*.JPEG
└── val/
    ├── images/*.JPEG
    └── val_annotations.txt
```

## 💻 Usage

### Train from scratch
```bash
python train.py
```

Training automatically resumes if a checkpoint exists in `models/weights/model.pth`.

### Inference
```python
from models.cnn import CNN
import torch, albumentations as A

model = CNN(num_classes=200)
model.load_state_dict(torch.load('models/weights/model.pth')['model_state_dict'])
model.eval()

# preprocess your image to 3×64×64 Tensor and run
with torch.no_grad():
    logits = model(image_tensor)
    prediction = logits.argmax(1)
```

## 📈 Training Details

- **Augmentations**: HorizontalFlip, RandomRotate90, ElasticTransform, ColorJitter, GaussNoise, CoarseDropout, RandomBrightnessContrast, ImageNet-style normalization.  
- **Optimizer**: AdamW, LR 1e-3, weight decay 1e-4.  
- **Regularization**: CutMix / MixUp with dynamic selection, label smoothing 0.1, dropout 0.5, gradient clipping 1.0.  
- **Scheduler**: CosineAnnealingLR for 200 epochs.

## 📁 Project Structure
```text
tiny-imagenet-cnn/
├── models/
│   ├── cnn.py
│   └── weights/
│       ├── model.pth
│       └── log.txt
├── dataset/
│   └── dataset.py
├── train.py
├── utils.py
├── tiny-imagenet/   # dataset here
└── README.md
```

## 🔧 Configuration

| Hyperparameter | Value | Reason |
|----------------|-------|--------|
| Batch size | 128 | Fits 8GB GPU |
| LR | 1e-3 → 1e-5 | Cosine decay |
| Weight decay | 1e-4 | L2 regularization |
| Label smoothing | 0.1 | Better calibration |
| Dropout | 0.5 | Combat overfitting |

### Hardware
- GPU: NVIDIA GTX 1080 (8GB) or better
- RAM: ≥ 16GB
- Storage: 2GB (dataset + checkpoints)

## 📝 Results

Training curves (loss ↓, accuracy ↑) show smooth convergence without over-fitting thanks to strong regularization.

| Model | Top-1 Acc | Params | Notes |
|-------|-----------|--------|-------|
| **This work** | **65%** | 2.8M | Custom CNN |
| ResNet-18 | 50–60% | 11.7M | From scratch |
| VGG-16 (TL) | ~49% | 138M | Transfer learning |

## 🤝 Contributing

1. Fork → Create branch → Commit → Pull request.  
2. Run `black . && flake8 .` before submitting.  
3. Open issues for bugs or ideas.

## 📄 License

Released under the **MIT License** — see `LICENSE` for details.

<div align="center">

**Star this repo if you found it helpful!**

Made with ❤️ for the deep-learning community.

</div>