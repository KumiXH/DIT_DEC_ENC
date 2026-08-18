from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _init_module(module: nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
            else:
                parameter.zero_()


class _ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int, strides: int, seed: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        width = 24
        for _ in range(strides):
            layers.extend((nn.Conv2d(channels, width, 3, stride=2, padding=1), nn.SiLU()))
            channels = width
            width = min(width * 2, 64)
        layers.append(nn.Conv2d(channels, latent_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
        _init_module(self, seed)

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)


class MockStudentEncoder(_ImageEncoder):
    def __init__(self, latent_channels: int = 16, seed: int = 101) -> None:
        super().__init__(6, latent_channels, strides=2, seed=seed)


class MockWanEncoder(_ImageEncoder):
    def __init__(self, latent_channels: int = 16, seed: int = 201) -> None:
        super().__init__(3, latent_channels, strides=3, seed=seed)

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim == 5:
            batch, channels, frames, height, width = video.shape
            images = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
            latent = super().forward(images)
            latent = latent.reshape(batch, frames, *latent.shape[1:]).permute(0, 2, 1, 3, 4)
            return latent
        return super().forward(video)


class _LatentDecoder(nn.Module):
    def __init__(self, in_channels: int, output_channels: int, seed: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 24, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(24, output_channels, 3, padding=1),
        )
        _init_module(self, seed)

    def forward(self, latent: Tensor) -> Tensor:
        return torch.sigmoid(self.net(latent))


class MockStudentDecoder(_LatentDecoder):
    def __init__(self, latent_channels: int = 16, seed: int = 301) -> None:
        super().__init__(latent_channels, 3, seed)


class MockWanDecoder(_LatentDecoder):
    def __init__(self, latent_channels: int = 16, seed: int = 401) -> None:
        super().__init__(latent_channels, 3, seed)


class MockConditionalStudentDecoder(_LatentDecoder):
    def __init__(self, latent_channels: int = 16, condition_channels: int = 3, seed: int = 501) -> None:
        super().__init__(latent_channels + condition_channels, 3, seed)

    def forward(self, latent: Tensor, condition_rgb: Tensor) -> Tensor:
        condition = F.interpolate(condition_rgb, size=latent.shape[-2:], mode="bilinear", align_corners=False)
        return super().forward(torch.cat((latent, condition), dim=1))


class MockTCDecoder(MockConditionalStudentDecoder):
    def __init__(self, latent_channels: int = 16, seed: int = 601) -> None:
        super().__init__(latent_channels=latent_channels, seed=seed)


class MockLQProjIn(nn.Module):
    def __init__(self, feature_dim: int = 64, seed: int = 701) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, feature_dim, 3, padding=1)
        _init_module(self, seed)

    def forward(self, video: Tensor) -> Tensor:
        image = video.mean(dim=2) if video.ndim == 5 else video
        features = self.projection(F.adaptive_avg_pool2d(image, (8, 8)))
        return features.flatten(2).transpose(1, 2)

