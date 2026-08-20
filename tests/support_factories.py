from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from distill_codec.models.mock import MockStudentEncoder


def create_encoder(channels=16):
    return MockStudentEncoder(latent_channels=channels)


private_bridge_calls: list[dict[str, Any]] = []


class PrivateBridgeNetwork(nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = offset


def clear_private_bridge_calls() -> None:
    private_bridge_calls.clear()


def build_private_bridge_network(offset: float = 0.0) -> nn.Module:
    return PrivateBridgeNetwork(offset=offset)


def build_private_bridge_non_module() -> object:
    return object()


def build_private_bridge_should_not_run() -> nn.Module:
    raise AssertionError("builder should not run before runner validation")


def _record_private_bridge_call(
    *,
    network: nn.Module,
    teacher_reference: Mapping[str, Any],
    **tensors: Tensor,
) -> None:
    private_bridge_calls.append(
        {
            "network": network,
            "teacher_reference": copy.deepcopy(dict(teacher_reference)),
            **tensors,
        }
    )


def run_private_encoder(
    *,
    network: PrivateBridgeNetwork,
    rgb: Tensor,
    teacher_reference: dict[str, Any],
    gain: float = 1.0,
) -> Tensor:
    _record_private_bridge_call(
        network=network,
        teacher_reference=teacher_reference,
        rgb=rgb,
    )
    teacher_reference["runner_mutated"] = True
    return (rgb + network.offset) * gain


def run_private_video_encoder(
    *,
    network: PrivateBridgeNetwork,
    rgb: Tensor,
    teacher_reference: dict[str, Any],
) -> Tensor:
    _record_private_bridge_call(
        network=network,
        teacher_reference=teacher_reference,
        rgb=rgb,
    )
    shape = teacher_reference["outputs"]["latent"]["shape"]
    return torch.zeros(
        rgb.shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        dtype=rgb.dtype,
        device=rgb.device,
    ) + network.offset


def run_private_two_frame_encoder(
    *,
    network: PrivateBridgeNetwork,
    rgb: Tensor,
    teacher_reference: dict[str, Any],
) -> Tensor:
    _record_private_bridge_call(
        network=network,
        teacher_reference=teacher_reference,
        rgb=rgb,
    )
    return torch.zeros(
        rgb.shape[0],
        16,
        2,
        rgb.shape[-2] // 8,
        rgb.shape[-1] // 8,
        dtype=rgb.dtype,
        device=rgb.device,
    ) + network.offset


def run_private_decoder(
    *,
    network: PrivateBridgeNetwork,
    lq_rgb: Tensor,
    dit_latent: Tensor,
    teacher_reference: dict[str, Any],
    gain: float = 1.0,
) -> Tensor:
    _record_private_bridge_call(
        network=network,
        teacher_reference=teacher_reference,
        lq_rgb=lq_rgb,
        dit_latent=dit_latent,
    )
    teacher_reference["runner_mutated"] = True
    return (lq_rgb + network.offset) * gain


def run_private_non_tensor(
    *,
    network: nn.Module,
    rgb: Tensor,
    teacher_reference: Mapping[str, Any],
) -> object:
    _record_private_bridge_call(
        network=network,
        teacher_reference=teacher_reference,
        rgb=rgb,
    )
    return {"latent": rgb}

