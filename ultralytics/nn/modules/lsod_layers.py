import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv  # Ultralytics標準Conv利用（=CBS）
from .block import Bottleneck
from .block import C2f, C3k

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
    """SPPF with Lightweight Large-Kernel Attention (LSKA) — per LSOD-YOLO."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.attn = LargeKernelAttention(c_)  # only addition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        # original SPPF pooling stack (same tensor repeatedly pooled)
        y1 = self.m(y)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # concatenate + apply LKA to the reduced feature
        att = self.attn(y)
        return self.cv2(torch.cat([y, y1, y2, y3], 1) + att)


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


class C2fN(nn.Module):
    """C2f-N: Modified C2f block with Normalized Attention Module (NAM)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 🔸 Use Bottleneck as in original C2f
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        self.nam = NAM(c2, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))  # split input
        y.extend(m(y[-1]) for m in self.m) # process through Bottlenecks
        out = self.cv2(torch.cat(y, 1))    # fuse partial + processed
        return self.nam(out)               # apply attention after fusion
    

class C3k2N(C2f):
    """C3k2 with Normalization-based Attention (NAM)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
        self.nam = NAM(self.c)  # ← 追加（C2f-N と同じ原理）

    def forward(self, x):
        y = super().forward(x)
        return self.nam(y)  # NAM によるチャネル＋空間リウェイト

