"""TransFuser-style multi-modal fusion network.

Architecture (simplified from Prakash et al. 2021):
  1. RGB tokens  ← RGBTokenEncoder
  2. LiDAR tokens ← LidarBEVTokenEncoder  (optional, skipped when lidar not available)
  3. Cross-modal Transformer fusion
  4. ControlHead → [steer, throttle, brake]

When model_type='interfuser' a SafetyHead is also attached (same output, just a
secondary prediction used for uncertainty-based safety gating).
"""

import torch
import torch.nn as nn
from typing import Optional

from .encoders import RGBTokenEncoder, LidarBEVTokenEncoder


class ControlHead(nn.Module):
    """Maps fused features to [steer, throttle, brake]."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineNet(nn.Module):
    """TransFuser / InterFuser fusion network.

    Parameters
    ----------
    embed_dim      : token embedding dimension
    n_heads        : number of attention heads in the Transformer
    n_layers       : number of Transformer encoder layers
    lidar_channels : number of BEV channels; set 0 to disable LiDAR branch
    model_type     : 'transfuser' | 'interfuser'
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        lidar_channels: int = 2,
        model_type: str = "transfuser",
    ) -> None:
        super().__init__()

        self.use_lidar = lidar_channels > 0
        self.model_type = model_type

        self.rgb_encoder = RGBTokenEncoder(embed_dim=embed_dim)

        if self.use_lidar:
            self.lidar_encoder = LidarBEVTokenEncoder(
                in_channels=lidar_channels, embed_dim=embed_dim
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, T, C) → (B, C, 1) → (B, C)

        self.control_head = ControlHead(embed_dim)

        if model_type == "interfuser":
            self.safety_head = ControlHead(embed_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        lidar_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        rgb       : (B, 3, H, W)
        lidar_bev : (B, C_lidar, bev_size, bev_size) or None

        Returns
        -------
        control : (B, 3)  values in [-1, 1]
        """
        rgb_tokens = self.rgb_encoder(rgb)  # (B, T_rgb, embed_dim)

        if self.use_lidar and lidar_bev is not None:
            lidar_tokens = self.lidar_encoder(lidar_bev)  # (B, T_lidar, embed_dim)
            tokens = torch.cat([rgb_tokens, lidar_tokens], dim=1)
        else:
            tokens = rgb_tokens

        fused = self.transformer(tokens)  # (B, T, embed_dim)

        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)  # (B, embed_dim)

        control = self.control_head(pooled)  # (B, 3)
        return control
