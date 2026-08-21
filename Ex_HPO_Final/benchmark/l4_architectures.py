"""
L4 Architectures: 5 мікро-нейромереж із різними топологіями з'єднань.

Кожна архітектура < 100K параметрів, навчається за 2-5 секунд (5 epochs, CPU).
Ключова відмінність — формат з'єднань (connectivity pattern).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Arch-1: Sequential MLP (baseline — лінійний ланцюг)
# ============================================================
class SequentialMLP(nn.Module):
    def __init__(self, input_dim, n_classes, h1=128, h2=64, dropout=0.2, act='relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            _get_act(act),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            _get_act(act),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Arch-2: Residual CNN (skip connections через 2 conv-блоки)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, act='relu', skip_scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act1 = _get_act(act)
        self.act2 = _get_act(act)
        self.skip_scale = skip_scale

    def forward(self, x):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip_scale * identity
        return self.act2(out)


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, n_classes, c1=16, c2=32, act='relu', skip_scale=1.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            _get_act(act),
        )
        self.block1 = ResidualBlock(c1, act, skip_scale)
        self.pool1 = nn.MaxPool2d(2)
        self.transition = nn.Sequential(
            nn.Conv2d(c1, c2, 1),
            nn.BatchNorm2d(c2),
            _get_act(act),
        )
        self.block2 = ResidualBlock(c2, act, skip_scale)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c2, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.transition(x)
        x = self.pool2(self.block2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# Arch-3: DenseBlock CNN (кожен шар конкатенований з попередніми)
# ============================================================
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth, act='relu'):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, growth, 3, padding=1)
        self.act = _get_act(act)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class DenseCNN(nn.Module):
    def __init__(self, in_channels, n_classes, c0=16, growth=8, n_dense=3, dropout=0.1, act='relu'):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, 3, padding=1),
            nn.BatchNorm2d(c0),
            _get_act(act),
        )
        layers = []
        ch = c0
        for _ in range(n_dense):
            layers.append(DenseLayer(ch, growth, act))
            ch += growth
        self.dense_layers = nn.ModuleList(layers)
        self.growth = growth
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.dense_layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(self.drop(x))


# ============================================================
# Arch-4: Bottleneck CNN (звуження → розширення)
# ============================================================
class BottleneckBlock(nn.Module):
    def __init__(self, c_in, bottleneck_ratio=0.25, expand_ratio=1.0, act='relu'):
        super().__init__()
        c_bn = max(4, int(c_in * bottleneck_ratio))
        c_out = max(4, int(c_in * expand_ratio))
        self.pw1 = nn.Conv2d(c_in, c_bn, 1)
        self.bn1 = nn.BatchNorm2d(c_bn)
        self.dw = nn.Conv2d(c_bn, c_bn, 3, padding=1, groups=max(1, c_bn))
        self.bn2 = nn.BatchNorm2d(c_bn)
        self.pw2 = nn.Conv2d(c_bn, c_out, 1)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.act = _get_act(act)
        self.use_residual = (c_in == c_out)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.pw1(x)))
        out = self.act(self.bn2(self.dw(out)))
        out = self.bn3(self.pw2(out))
        if self.use_residual:
            out = out + identity
        return self.act(out)


class BottleneckCNN(nn.Module):
    def __init__(self, in_channels, n_classes, c_in=24, bottleneck_ratio=0.25,
                 n_blocks=2, expand_ratio=1.0, act='relu'):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_in, 3, padding=1),
            nn.BatchNorm2d(c_in),
            _get_act(act),
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(BottleneckBlock(c_in, bottleneck_ratio, expand_ratio, act))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# Arch-5: Multi-Branch / Inception-micro (паралельні гілки)
# ============================================================
class MultiBranchCNN(nn.Module):
    def __init__(self, in_channels, n_classes, c0=16, c_1x1=8, c_3x3=8, c_5x5=4, act='relu'):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, 3, padding=1),
            nn.BatchNorm2d(c0),
            _get_act(act),
        )
        # Branch 1: 1×1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(c0, c_1x1, 1), nn.BatchNorm2d(c_1x1), _get_act(act))
        # Branch 2: 3×3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(c0, c_3x3, 3, padding=1), nn.BatchNorm2d(c_3x3), _get_act(act))
        # Branch 3: 5×5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(c0, c_5x5, 5, padding=2), nn.BatchNorm2d(c_5x5), _get_act(act))
        # Branch 4: pool → 1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(c0, c_1x1, 1), nn.BatchNorm2d(c_1x1), _get_act(act))

        cat_ch = c_1x1 + c_3x3 + c_5x5 + c_1x1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cat_ch, n_classes)

    def forward(self, x):
        x = self.stem(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# Utilities
# ============================================================
def _get_act(name):
    """Factory: returns a NEW activation instance each call (avoids nn.Sequential dedup)."""
    return {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}.get(name, nn.ReLU)()


# Registry
ARCHITECTURES = {
    'sequential': SequentialMLP,
    'residual': ResidualCNN,
    'dense': DenseCNN,
    'bottleneck': BottleneckCNN,
    'multibranch': MultiBranchCNN,
}
