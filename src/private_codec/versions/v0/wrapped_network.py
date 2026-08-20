from __future__ import annotations

from torch import Tensor

from .base_network import (
    ConvConditionalDecoderBase,
    ConvEncoderBase,
    ConvUnconditionalDecoderBase,
)


# COPY POINT: Replace this file with the second private network file that inherits
# the base implementation. Keep each version's defaults and private forward shape
# here; the project never imports these classes directly for bridge-backed models.


class V0Encoder(ConvEncoderBase):
    def __init__(
        self,
        *,
        latent_channels: int = 16,
        feature_channels: int = 32,
    ) -> None:
        super().__init__(
            latent_channels=latent_channels,
            feature_channels=feature_channels,
        )

    def forward(self, private_rgb: Tensor) -> Tensor:
        return super().forward(private_rgb)


class V0ConditionalDecoder(ConvConditionalDecoderBase):
    def __init__(
        self,
        *,
        latent_channels: int = 16,
        feature_channels: int = 32,
    ) -> None:
        super().__init__(
            latent_channels=latent_channels,
            feature_channels=feature_channels,
        )

    def forward(self, private_lq_rgb: Tensor, private_latent: Tensor) -> Tensor:
        # This private order intentionally differs from the project's public order.
        return super().forward(private_latent, private_lq_rgb)


class V0UnconditionalDecoder(ConvUnconditionalDecoderBase):
    """Runnable class-import example for the legacy unconditional decoder config."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        feature_channels: int = 32,
    ) -> None:
        super().__init__(
            latent_channels=latent_channels,
            feature_channels=feature_channels,
        )
