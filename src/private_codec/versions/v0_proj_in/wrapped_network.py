from __future__ import annotations

from torch import Tensor, nn

from .base_network import ConvConditionEncoderBase


# COPY POINT: Replace this inherited wrapper with the second file from your
# private network. Keep the final output contract [B,N,1536].


class V0ProjInConditionEncoder(ConvConditionEncoderBase):
    """Converts the example backbone feature map to FlashVSR DiT conditions."""

    def __init__(
        self,
        *,
        feature_channels: int = 32,
        output_channels: int = 16,
        unshuffle_factor: int = 2,
        condition_dim: int = 1536,
    ) -> None:
        if unshuffle_factor <= 0:
            raise ValueError("unshuffle_factor must be positive")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        super().__init__(
            feature_channels=feature_channels,
            output_channels=output_channels,
        )
        self.unshuffle = nn.PixelUnshuffle(unshuffle_factor)
        self.proj = nn.Conv2d(
            output_channels * unshuffle_factor * unshuffle_factor,
            condition_dim,
            kernel_size=1,
        )

    def forward(self, private_rgb_video: Tensor) -> Tensor:
        features = super().forward(private_rgb_video)
        features = self.unshuffle(features)
        features = self.proj(features)

        # [B,1536,H/16,W/16] -> [B,(H/16)*(W/16),1536]
        return features.flatten(2).transpose(1, 2).contiguous()

