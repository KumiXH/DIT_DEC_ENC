from __future__ import annotations

from torch import Tensor, nn


# COPY POINT: Replace this file with the base class from your private network.
# The public project input has already been converted to [B,3,T,H,W].


class ConvConditionEncoderBase(nn.Module):
    """Small video-to-feature backbone used as a runnable LQ_proj_in example."""

    def __init__(
        self,
        *,
        feature_channels: int = 32,
        output_channels: int = 16,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or output_channels <= 0:
            raise ValueError("condition encoder channel counts must be positive")
        self.output_channels = output_channels
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
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, rgb_video: Tensor) -> Tensor:
        if rgb_video.ndim != 5 or rgb_video.shape[1] != 3:
            raise ValueError(
                "LQ_proj_in input must have shape [B,3,T,H,W], "
                f"got {tuple(rgb_video.shape)}"
            )
        if rgb_video.shape[2] <= 0:
            raise ValueError("LQ_proj_in input must contain at least one frame")
        if rgb_video.shape[-2] % 16 or rgb_video.shape[-1] % 16:
            raise ValueError(
                "LQ_proj_in RGB height and width must be divisible by 16, "
                f"got {tuple(rgb_video.shape)}"
            )

        # COPY POINT: This example merges the repeated frames by averaging them.
        # Replace this line if your private backbone handles time internally.
        rgb = rgb_video.mean(dim=2)
        return self.net(rgb)

