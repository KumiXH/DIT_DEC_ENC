from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .contracts import ColorSpec, ContractError


_LUMA = {
    "bt601": (0.2990, 0.1140),
    "bt709": (0.2126, 0.0722),
}


def _validate_rgb_like(tensor: Tensor, name: str) -> None:
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ContractError(f"{name} must have shape [B,3,H,W], got {tuple(tensor.shape)}")


def rgb_to_yuv(rgb: Tensor, spec: ColorSpec | None = None) -> Tensor:
    spec = spec or ColorSpec()
    _validate_rgb_like(rgb, "rgb")
    kr, kb = _LUMA[spec.matrix]
    kg = 1.0 - kr - kb
    red, green, blue = rgb.unbind(dim=1)
    y = kr * red + kg * green + kb * blue
    u = (blue - y) / (2.0 * (1.0 - kb)) + 0.5
    v = (red - y) / (2.0 * (1.0 - kr)) + 0.5
    if spec.range == "limited":
        y = y * (219.0 / 255.0) + (16.0 / 255.0)
        u = (u - 0.5) * (224.0 / 255.0) + (128.0 / 255.0)
        v = (v - 0.5) * (224.0 / 255.0) + (128.0 / 255.0)
    return torch.stack((y, u, v), dim=1)


def yuv_to_rgb(yuv: Tensor, spec: ColorSpec | None = None) -> Tensor:
    spec = spec or ColorSpec()
    _validate_rgb_like(yuv, "yuv")
    y, u, v = yuv.unbind(dim=1)
    if spec.range == "limited":
        y = (y - (16.0 / 255.0)) * (255.0 / 219.0)
        u = (u - (128.0 / 255.0)) * (255.0 / 224.0) + 0.5
        v = (v - (128.0 / 255.0)) * (255.0 / 224.0) + 0.5
    kr, kb = _LUMA[spec.matrix]
    kg = 1.0 - kr - kb
    u_centered = u - 0.5
    v_centered = v - 0.5
    red = y + 2.0 * (1.0 - kr) * v_centered
    blue = y + 2.0 * (1.0 - kb) * u_centered
    green = (y - kr * red - kb * blue) / kg
    return torch.stack((red, green, blue), dim=1)


def rgb_to_packed_6ch(rgb: Tensor, spec: ColorSpec | None = None) -> Tensor:
    spec = spec or ColorSpec()
    _validate_rgb_like(rgb, "rgb")
    height, width = rgb.shape[-2:]
    if height % 2 or width % 2:
        raise ContractError(f"packed_6ch requires even height and width, got {(height, width)}")
    yuv = rgb_to_yuv(rgb, spec)
    y = yuv[:, :1]
    y_samples = torch.cat(
        (y[:, :, 0::2, 0::2], y[:, :, 0::2, 1::2], y[:, :, 1::2, 0::2], y[:, :, 1::2, 1::2]),
        dim=1,
    )
    chroma = F.avg_pool2d(yuv[:, 1:], kernel_size=2, stride=2)
    return torch.cat((y_samples, chroma), dim=1)


def packed_6ch_to_rgb(
    packed: Tensor,
    spec: ColorSpec | None = None,
    output_size: tuple[int, int] | None = None,
) -> Tensor:
    spec = spec or ColorSpec()
    if packed.ndim != 4 or packed.shape[1] != 6:
        raise ContractError(f"packed tensor must have shape [B,6,H,W], got {tuple(packed.shape)}")
    y = F.pixel_shuffle(packed[:, :4], upscale_factor=2)
    target_size = output_size or tuple(y.shape[-2:])
    if tuple(y.shape[-2:]) != tuple(target_size):
        raise ContractError(f"packed luma resolves to {tuple(y.shape[-2:])}, requested {target_size}")
    mode = spec.chroma_upsample
    interpolate_kwargs = {"size": target_size, "mode": mode}
    if mode == "bilinear":
        interpolate_kwargs["align_corners"] = False
    chroma = F.interpolate(packed[:, 4:], **interpolate_kwargs)
    return yuv_to_rgb(torch.cat((y, chroma), dim=1), spec)


def sparse_yuv420_to_rgb(sparse_yuv: Tensor, spec: ColorSpec | None = None) -> Tensor:
    spec = spec or ColorSpec()
    _validate_rgb_like(sparse_yuv, "sparse_yuv")
    height, width = sparse_yuv.shape[-2:]
    if height % 2 or width % 2:
        raise ContractError(f"sparse YUV requires even height and width, got {(height, width)}")
    chroma_samples = sparse_yuv[:, 1:, 0::2, 0::2]
    kwargs = {"size": (height, width), "mode": spec.chroma_upsample}
    if spec.chroma_upsample == "bilinear":
        kwargs["align_corners"] = False
    chroma = F.interpolate(chroma_samples, **kwargs)
    return yuv_to_rgb(torch.cat((sparse_yuv[:, :1], chroma), dim=1), spec)

