import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv  # Ultralytics標準Conv利用（=CBS）

class LargeKernelAttention(nn.Module):
    """Lightweight Large-Kernel Attention (LSKA)."""
    def __init__(self, c, k=9):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c, c, k, 1, p, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return x * self.act(self.bn(self.conv(x)))


class SPPFL(nn.Module):
    """SPPF with Large-Kernel Attention (used in LSOD-YOLO)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # ★ Pooling kernels拡張
        self.m5 = nn.MaxPool2d(5, 1, 2)
        self.m7 = nn.MaxPool2d(7, 1, 3)
        self.m9 = nn.MaxPool2d(9, 1, 4)
        # ★ Attention追加
        self.attn = LargeKernelAttention(c_)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        p5 = self.m5(y)
        p7 = self.m7(y)
        p9 = self.m9(y)
        att = self.attn(y)
        return self.cv2(torch.cat([y, p5, p7, p9], 1) + att)



# -----------------------------
# Normalized Attention Module (NAM)
# -----------------------------
class NAM(nn.Module):
    """
    Normalized Attention Module (NAM)
    Ref: LSOD-YOLO (Expert Systems 2025)
    Combines channel-wise normalization and spatial attention.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(1, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        y = self.pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        x = x * y

        # Spatial attention
        spatial = x.mean(1, keepdim=True)
        spatial = self.sigmoid(self.conv_spatial(spatial))
        return x * spatial


# -----------------------------
# C2f-N: C2f with NAM
# -----------------------------
class C2fN(nn.Module):
    """
    C2f-N: Modified C2f block with Normalized Attention Module.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                Conv(self.c, self.c, 3, 1, g=g),
                nn.BatchNorm2d(self.c),
                nn.SiLU(),
            )
            for _ in range(n)
        )
        self.nam = NAM(c2, reduction=reduction)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        return self.nam(out)

