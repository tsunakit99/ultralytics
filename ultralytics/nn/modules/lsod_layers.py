import torch
import torch.nn as nn
import math
from .conv import Conv  # Ultralytics標準Conv利用（=CBS）
from .block import Bottleneck, C2f, C3k, SPPF

class LargeSeparableKernelAttention(nn.Module):
    def __init__(self, c, k=9, d=3):
        super().__init__()
        # 系列①: (2d-1) の 1D DWConv（サイズ保持）
        k1 = 2 * d - 1
        self.dw_h1 = nn.Conv2d(c, c, (k1, 1), 1, (k1 // 2, 0), groups=c, bias=False)
        self.dw_w1 = nn.Conv2d(c, c, (1, k1), 1, (0, k1 // 2), groups=c, bias=False)

        # 系列②: (ceil(k/d)) の dilated 1D DWConv（サイズ保持になるようにパディングを計算）
        k2 = max(1, math.ceil(k / d))

        pad_h2 = (d * (k2 - 1)) // 2  # 高さ方向 1D conv の有効パディング
        pad_w2 = (d * (k2 - 1)) // 2  # 幅方向  1D conv の有効パディング

        self.dw_h2 = nn.Conv2d(
            c, c, (k2, 1), stride=1, padding=(pad_h2, 0),
            dilation=(d, 1), groups=c, bias=False
        )
        self.dw_w2 = nn.Conv2d(
            c, c, (1, k2), stride=1, padding=(0, pad_w2),
            dilation=(1, d), groups=c, bias=False
        )

        # 1×1 で注意マップ A を生成
        self.attn = nn.Conv2d(c, c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 系列①（サイズ保持）
        z1 = self.dw_w1(self.dw_h1(x))
        # 系列②（ここもサイズ保持）
        z2 = self.dw_w2(self.dw_h2(x))
        # 形状が一致するので安全に加算できる
        z = z1 + z2
        A = self.sigmoid(self.attn(z))
        return x * A



class SPPFL(SPPF):
    """SPPF with Lightweight Large-Kernel Attention (LSKA) — per LSOD-YOLO."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__(c1, c2, k)
        self.attn = LargeSeparableKernelAttention((c1 // 2) * 4)  # only addition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.cv1(x)
        # original SPPF pooling stack (same tensor repeatedly pooled)
        y1 = self.m(y0)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # concatenate + apply LKA to the reduced feature
        ycat = torch.cat([y0, y1, y2, y3], 1)   # [B, c_*4, H, W]
        ycat = ycat + self.attn(ycat)           # 加算は同次元で
        return self.cv2(ycat)


class NAM(nn.Module):
    """
    論文の要旨に沿った軽量実装：
    - Channel: BNのγ（scale）からチャネル重みを作る
    - Spatial: BNのλ（spatial scale）に相当する正規化量から空間重み
    """
    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.eps = eps
        # γ, λ から線形正規化するための learnable 射影（Wγ, Wλ の代替）
        self.proj_c = nn.Conv2d(c, c, 1, bias=False)  # 軽量1x1
        self.proj_s = nn.Conv2d(1, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention: BNのスケールに対応する「チャネルごとの大きさ」を近似
        # ここでは x をチャネル方向に標準化して、そのスケールを重み化の素にする簡易版
        mean_c = x.mean(dim=(2,3), keepdim=True)
        var_c  = x.var(dim=(2,3), keepdim=True, unbiased=False)
        x_bn   = (x - mean_c) / (var_c + self.eps).sqrt()      # BN(F1) に相当
        Mc     = self.sigmoid(self.proj_c(x_bn))               # sigmoid(Wγ(BN(F1)))（式(6)の近似）

        # Spatial attention: ピクセル正規化（空間方向のスケール）
        mean_s = x.mean(dim=1, keepdim=True)
        var_s  = x.var(dim=1, keepdim=True, unbiased=False)
        xs_bn  = (mean_s - mean_s.mean([2,3], keepdim=True)) / (var_s.mean([2,3], keepdim=True) + self.eps).sqrt()
        Ms     = self.sigmoid(self.proj_s(xs_bn))              # sigmoid(Wλ(BNs(F2)))（式(7)の近似）

        return x * Mc * Ms


class C2fN(C2f):
    """C2f-N: Modified C2f block with Normalized Attention Module (NAM)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.nam = NAM(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)           # use official C2f forward
        return self.nam(out)               # apply attention after fusion
    

class C3k2N(C2f):
    """C3k2 with Normalization-based Attention (NAM)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
        self.nam = NAM(c2)  # apply NAM on the fused output

    def forward(self, x):
        y = super().forward(x)
        return self.nam(y)  # NAM によるチャネル＋空間リウェイト
