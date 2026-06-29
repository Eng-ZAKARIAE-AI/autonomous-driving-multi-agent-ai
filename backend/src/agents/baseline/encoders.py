"""CNN feature encoders for the TransFuser baseline.

RGBTokenEncoder   : ResNet-lite CNN → (B, T, C) spatial tokens from RGB frame
LidarBEVTokenEncoder : Same structure for Bird-Eye-View LiDAR grid
"""

import torch
import torch.nn as nn
from typing import Tuple


class _ResBlock(nn.Module):
    """Single residual block (no downsampling)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class _DownBlock(nn.Module):
    """Conv + BN + ReLU with stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RGBTokenEncoder(nn.Module):
    """Encodes an RGB image into a sequence of spatial feature tokens.

    Input  : (B, 3, H, W)   — expects 224×224 or 84×84
    Output : (B, T, embed_dim)   where T = (H//32)*(W//32)
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _DownBlock(3, 32),    # /2
            _ResBlock(32),
            _DownBlock(32, 64),   # /4
            _ResBlock(64),
            _DownBlock(64, 128),  # /8
            _ResBlock(128),
            _DownBlock(128, 256), # /16
            _ResBlock(256),
            _DownBlock(256, embed_dim),  # /32
            _ResBlock(embed_dim),
        )
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)          # (B, embed_dim, H/32, W/32)
        feat = self.proj(feat)
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2)  # (B, H*W, C)


class LidarBEVTokenEncoder(nn.Module):
    """Encodes a LiDAR Bird-Eye-View occupancy grid into spatial tokens.

    Input  : (B, bev_channels, bev_size, bev_size)
    Output : (B, T, embed_dim)
    """

    def __init__(self, in_channels: int = 2, embed_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _DownBlock(in_channels, 32),
            _ResBlock(32),
            _DownBlock(32, 64),
            _ResBlock(64),
            _DownBlock(64, 128),
            _ResBlock(128),
            _DownBlock(128, embed_dim),
            _ResBlock(embed_dim),
        )
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2)
