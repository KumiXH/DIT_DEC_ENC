from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..contracts import ContractError


def load_snapshot_module(source_file: str | Path, module_name: str) -> ModuleType:
    path = Path(source_file).resolve()
    if not path.is_file():
        raise ContractError(f"snapshot source file does not exist: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot create import spec for snapshot file: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise ContractError(f"snapshot {path} is missing a dependency: {error}") from error
    return module


def _unwrap_state_dict(payload: Any) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise ContractError(f"checkpoint payload must be a mapping, got {type(payload).__name__}")
    for key in ("state_dict", "model_state", "model", "module"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return nested
    return payload


def _load_weights(module: nn.Module, checkpoint: str | Path, strict: bool) -> None:
    path = Path(checkpoint)
    if not path.is_file():
        raise ContractError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(payload)
    try:
        incompatible = module.load_state_dict(state_dict, strict=strict)
    except RuntimeError as error:
        raise ContractError(f"cannot load checkpoint {path} into {module.__class__.__name__}: {error}") from error
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise ContractError(
            f"checkpoint {path} mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


class WanEncoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.vae = vae
        self.device_name = str(device)

    def forward(self, rgb_video: Tensor) -> Tensor:
        if rgb_video.ndim != 5:
            raise ContractError(f"Wan encoder expects [B,C,T,H,W], got {tuple(rgb_video.shape)}")
        video = rgb_video.mul(2.0).sub(1.0)
        if hasattr(self.vae, "single_encode"):
            return self.vae.single_encode(video, self.device_name)
        return self.vae.encode(video)


class WanDecoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.vae = vae
        self.device_name = str(device)

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        if hasattr(self.vae, "single_decode"):
            video = self.vae.single_decode(latent, self.device_name)
        else:
            video = self.vae.decode(latent)
        return video.add(1.0).mul(0.5)


class FlashVSRTCDecoderWrapper(nn.Module):
    def __init__(self, decoder: nn.Module, condition_frames: int = 4) -> None:
        super().__init__()
        self.decoder = decoder
        self.condition_frames = condition_frames

    def forward(self, latent: Tensor, lq_rgb: Tensor) -> Tensor:
        latent_video = latent.unsqueeze(1) if latent.ndim == 4 else latent.transpose(1, 2)
        condition = lq_rgb.unsqueeze(1).expand(-1, self.condition_frames, -1, -1, -1)
        if hasattr(self.decoder, "clean_mem"):
            self.decoder.clean_mem()
        output = self.decoder.decode_video(latent_video, parallel=False, cond=condition)
        if output.ndim != 5:
            raise ContractError(f"TCDecoder returned expected NTCHW video, got {tuple(output.shape)}")
        return output[:, output.shape[1] // 2]


def _build_wan(
    *,
    source_file: str | Path,
    checkpoint: str | Path | None,
    device: str,
    strict: bool,
    z_dim: int,
) -> nn.Module:
    snapshot = load_snapshot_module(source_file, "distill_codec_wan_snapshot")
    vae = snapshot.WanVideoVAE(z_dim=z_dim)
    if checkpoint is not None:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        state_dict = snapshot.WanVideoVAEStateDictConverter().from_civitai(payload)
        vae.load_state_dict(state_dict, strict=strict)
    return vae.to(device)


def create_wan_encoder(
    source_file: str = "third_party/wan/wan_video_vae.py",
    checkpoint: str | None = None,
    device: str = "cpu",
    strict: bool = True,
    z_dim: int = 16,
) -> WanEncoderWrapper:
    return WanEncoderWrapper(
        _build_wan(source_file=source_file, checkpoint=checkpoint, device=device, strict=strict, z_dim=z_dim),
        device,
    )


def create_wan_decoder(
    source_file: str = "third_party/wan/wan_video_vae.py",
    checkpoint: str | None = None,
    device: str = "cpu",
    strict: bool = True,
    z_dim: int = 16,
) -> WanDecoderWrapper:
    return WanDecoderWrapper(
        _build_wan(source_file=source_file, checkpoint=checkpoint, device=device, strict=strict, z_dim=z_dim),
        device,
    )


def create_lq_proj_in(
    source_file: str = "third_party/flashvsr/utils.py",
    checkpoint: str | None = None,
    class_name: str = "Causal_LQ4x_Proj",
    strict: bool = True,
    **kwargs: Any,
) -> nn.Module:
    snapshot = load_snapshot_module(source_file, "distill_codec_flashvsr_utils_snapshot")
    if not hasattr(snapshot, class_name):
        raise ContractError(f"FlashVSR snapshot has no class {class_name!r}")
    module = getattr(snapshot, class_name)(**kwargs)
    if checkpoint is not None:
        _load_weights(module, checkpoint, strict)
    return module


def create_tc_decoder(
    source_file: str = "third_party/flashvsr/TCDecoder.py",
    checkpoint: str | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    channels: list[int] | None = None,
    latent_channels: int = 784,
    condition_frames: int = 4,
    strict: bool = False,
) -> FlashVSRTCDecoderWrapper:
    snapshot = load_snapshot_module(source_file, "distill_codec_flashvsr_tcdecoder_snapshot")
    torch_dtype = getattr(torch, dtype)
    decoder = snapshot.build_tcdecoder(
        new_channels=channels or [512, 256, 128, 128],
        device=device,
        dtype=torch_dtype,
        new_latent_channels=latent_channels,
    )
    if checkpoint is not None:
        _load_weights(decoder, checkpoint, strict)
    return FlashVSRTCDecoderWrapper(decoder, condition_frames=condition_frames)

