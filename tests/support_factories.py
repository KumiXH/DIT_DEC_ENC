from __future__ import annotations

import copy
from typing import Any, Mapping

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

