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
