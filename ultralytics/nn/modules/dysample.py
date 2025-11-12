# ultralytics/nn/modules/dysample.py
# DySample (paper-faithful): lightweight dynamic upsampler
# LSOD-YOLO: linear layer -> pixel shuffle -> offsets (2, sH, sW) -> grid_sample
# Ref: "linear layer (out=2*s^2) + pixel shuffle" and sampling grid + offsets. 
#      (No lp/pl, no groups). See Fig.7 and text. 

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DySample"]

def _normal_init(m, std=1e-3, bias=0.0):
    if hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight, mean=0.0, std=std)
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias, bias)

class DySample(nn.Module):
    """
    Paper-faithful DySample:
      - offset: 1x1 conv, out_channels = 2 * (scale**2)
      - reshape offsets by pixel_shuffle -> (B, 2, sH, sW)
      - coords = (G + O) normalized to [-1, 1]
      - grid_sample(x, coords) to get upsampled features (C x sH x sW)
    Notes:
      * channels are preserved (in == out)
      * no groups, no 'lp'/'pl' modes
    """

    def __init__(self, c1: int, scale: int = 2, use_scope: bool = False, alpha: float = 0.25):
        """
        Args:
            c1: input channels (from Ultralytics parser as first arg)
            scale: upsample factor (typically 2)
            use_scope: optional gating conv for offsets (sigmoid), zeros init
            alpha: scale factor for offset magnitude before adding base grid
        """
        super().__init__()
        assert isinstance(scale, int) and scale >= 2, "scale must be integer >= 2"

        self.scale = scale
        self.alpha = float(alpha)

        # 1x1 conv -> 2*s^2 channels (x/y offsets for each sub-pixel)
        self.offset = nn.Conv2d(c1, 2 * (scale ** 2), kernel_size=1, bias=True)
        _normal_init(self.offset, std=1e-3, bias=0.0)

        self.scope = None
        if use_scope:
            # gating branch (sigmoid) as optional "dynamic factor" in the paper text
            self.scope = nn.Conv2d(c1, 2 * (scale ** 2), kernel_size=1, bias=False)
            with torch.no_grad():
                # start as pass-through (all zeros -> sigmoid(0)=0.5, effectively halves offsets initially)
                nn.init.constant_(self.scope.weight, 0.0)

        # precompute base grid offset (init_pos) of shape (1, 2*s^2, 1, 1) *after* shuffle becomes (1,2,1,1)
        self.register_buffer("init_pos", self._make_init_pos(), persistent=False)

    def _make_init_pos(self) -> torch.Tensor:
        # static base grid offsets in range centered around 0 as in paper’s description
        # Produce a (2, s, s) small grid, then reshape to (1, 2*s*s, 1, 1) to match conv output prior to shuffle.
        s = self.scale
        # coordinate centers in [-(s-1)/2, ..., +(s-1)/2] normalized by s
        h = torch.arange((1 - s) / 2, (s - 1) / 2 + 1, dtype=torch.float32) / s
        # meshgrid with ij indexing to avoid future warnings
        yy, xx = torch.meshgrid(h, h, indexing="ij")  # (s, s)
        base = torch.stack((xx, yy), dim=0)          # (2, s, s) with order (x, y)
        base = base.reshape(1, 2 * s * s, 1, 1)      # (1, 2*s*s, 1, 1)
        return base

    def _build_coords(self, H: int, W: int, device, dtype):
        # Build normalized base grid in [-1,1] for size (sH, sW)
        s = self.scale
        # target spatial size
        sH, sW = s * H, s * W

        # pixel centers: [0.5, 1.5, ..., sH-0.5] / sW, sH (x,y)
        y = torch.arange(sH, device=device, dtype=dtype) + 0.5
        x = torch.arange(sW, device=device, dtype=dtype) + 0.5
        # meshgrid with ij indexing: y first (rows), x second (cols)
        yy, xx = torch.meshgrid(y, x, indexing="ij")   # (sH, sW)

        # normalize to [-1, 1]
        # grid_sample expects last dim order (x, y) in [-1,1]
        gx = 2.0 * (xx / sW) - 1.0
        gy = 2.0 * (yy / sH) - 1.0
        grid = torch.stack((gx, gy), dim=-1)  # (sH, sW, 2)
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            y: (B, C, sH, sW)
        """
        B, C, H, W = x.shape
        s = self.scale

        # raw offsets: (B, 2*s*s, H, W)
        off = self.offset(x)
        if self.scope is not None:
            off = off * self.scope(x).sigmoid()  # gate

        # add small static prior then magnitude scaling
        # (B, 2*s*s, H, W) + (1, 2*s*s, 1, 1)
        off = self.alpha * off + self.init_pos

        # reshape offsets to (B, 2, sH, sW) via pixel shuffle
        off = F.pixel_shuffle(off, upscale_factor=s)  # (B, 2, sH, sW)
        # build base coordinates (normalized) and add normalized offsets
        grid = self._build_coords(H, W, device=x.device, dtype=x.dtype)  # (sH, sW, 2)

        # offsets are in absolute pixel units of sub-grid; normalize them to [-1,1]
        # scale by (1/sW, 1/sH) and map to [-1,1] space
        dx = (off[:, 0]) * (2.0 / (s * W))  # (B, sH, sW)
        dy = (off[:, 1]) * (2.0 / (s * H))

        grid = grid.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, sH, sW, 2)
        grid[..., 0] = torch.clamp(grid[..., 0] + dx, -1.0001, 1.0001)
        grid[..., 1] = torch.clamp(grid[..., 1] + dy, -1.0001, 1.0001)

        # sample
        y = F.grid_sample(
            x, grid, mode="bilinear", align_corners=False, padding_mode="border"
        )  # (B, C, sH, sW)
        return y
