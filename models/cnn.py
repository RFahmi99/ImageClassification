"""
cnn.py
======
A lightweight ResNet-like CNN tailored for 64×64 Tiny-ImageNet images.

Architecture
------------
Conv-BN-ReLU → 4 residual stages (64-128-256-512) → GAP → FC

Total params ≈ 2.8 M – small enough for quick experiments yet deep enough
to benefit from residual learning.
"""

import torch.nn as nn


# --------------------------------------------------------
# Basic residual block (pre-activation style)
# --------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        stride = 2 if downsample else 1

        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection (identity or projection)
        self.projection = nn.Identity()
        if downsample or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.projection(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# --------------------------------------------------------
# Full network
# --------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes: int = 200):
        super().__init__()

        # Stem – preserves 64×64 resolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual stages (numbers follow output channels)
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, downsample=False),
            ResidualBlock(64, 64, downsample=False),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128, downsample=False),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 256, downsample=False),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, downsample=True),
            ResidualBlock(512, 512, downsample=False),
        )

        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

        # He initialization for conv / linear layers
        self.init_weights()

    # ----------------------------------------------------
    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avg_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    # ----------------------------------------------------
    def init_weights(self):
        """Custom weight init: He-normal for Conv, Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)