from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# COPY POINT: Replace these base classes with your first private network file.
# Keep project-specific tensor conversion out of this module; entrypoints.py owns it.


def _require_bchw(tensor: Tensor, *, channels: int, label: str) -> None:
    if tensor.ndim != 4 or tensor.shape[1] != channels:
        raise ValueError(
            f"{label} must have shape [B,{channels},H,W], got {tuple(tensor.shape)}"
        )


class ConvEncoderBase(nn.Module):
    """Small RGB encoder used only as a runnable integration example."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        feature_channels: int = 32,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or feature_channels <= 0:
            raise ValueError("encoder channel counts must be positive")
        self.latent_channels = latent_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, feature_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                feature_channels,
                feature_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                feature_channels * 2,
                latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, rgb: Tensor) -> Tensor:
        _require_bchw(rgb, channels=3, label="encoder RGB input")
        if rgb.shape[-2] % 8 or rgb.shape[-1] % 8:
            raise ValueError(
                "encoder RGB height and width must be divisible by 8, "
                f"got {tuple(rgb.shape)}"
            )
        return self.net(rgb)


class ConvConditionalDecoderBase(nn.Module):
    """Fuses a latent tensor with LQ RGB and returns RGB at the LQ size."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        feature_channels: int = 32,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or feature_channels <= 0:
            raise ValueError("decoder channel counts must be positive")
        self.latent_channels = latent_channels
        self.latent_projection = nn.Sequential(
            nn.Conv2d(latent_channels, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.lq_projection = nn.Sequential(
            nn.Conv2d(3, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, dit_latent: Tensor, lq_rgb: Tensor) -> Tensor:
        _require_bchw(lq_rgb, channels=3, label="decoder LQ RGB input")
        if dit_latent.ndim != 4 or dit_latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"decoder latent channels={self.latent_channels} requires BCHW input, "
                f"got {tuple(dit_latent.shape)}"
            )
        if dit_latent.shape[0] != lq_rgb.shape[0]:
            raise ValueError(
                "decoder latent and LQ RGB batch size must match, "
                f"got {tuple(dit_latent.shape)} and {tuple(lq_rgb.shape)}"
            )

        latent_features = self.latent_projection(dit_latent)
        latent_features = F.interpolate(
            latent_features,
            size=lq_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        lq_features = self.lq_projection(lq_rgb)
        return self.fusion(
            F.silu(torch.cat((latent_features, lq_features), dim=1))
        )


class ConvUnconditionalDecoderBase(nn.Module):
    """Legacy latent-only decoder retained for the Wan example recipes."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        feature_channels: int = 32,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or feature_channels <= 0:
            raise ValueError("decoder channel counts must be positive")
        self.latent_channels = latent_channels
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, feature_channels * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, dit_latent: Tensor) -> Tensor:
        if dit_latent.ndim != 4 or dit_latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"decoder latent channels={self.latent_channels} requires BCHW input, "
                f"got {tuple(dit_latent.shape)}"
            )
        features = F.interpolate(
            dit_latent,
            scale_factor=8,
            mode="bilinear",
            align_corners=False,
        )
        return self.net(features)
