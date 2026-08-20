from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .wrapped_network import V0ConditionalDecoder, V0Encoder


# COPY POINT: This is the only project-to-private integration layer. When copying
# v0 to v1, keep these four public signatures and replace the code inside them.


def build_encoder(**kwargs: Any) -> nn.Module:
    # COPY POINT: Import and construct the real encoder wrapper here. All values in
    # YAML builder_kwargs arrive in kwargs, including config or weight paths.
    return V0Encoder(**kwargs)


def run_encoder(
    *,
    network: nn.Module,
    rgb: Tensor,
    teacher_reference: Mapping[str, Any],
    **kwargs: Any,
) -> Tensor:
    # COPY POINT: Convert project RGB to the private input here. Normalization,
    # colorspace conversion, resize, padding, nesting, and device-specific logic
    # all belong here or inside the private network.
    private_rgb = rgb

    # teacher_reference is theoretical teacher shape metadata for logging/asserts.
    # It must not resize or otherwise mutate the actual project tensor.
    _ = teacher_reference, kwargs
    return network(private_rgb)


def build_decoder(**kwargs: Any) -> nn.Module:
    # COPY POINT: Import and construct the real conditional decoder wrapper here.
    return V0ConditionalDecoder(**kwargs)


def run_decoder(
    *,
    network: nn.Module,
    lq_rgb: Tensor,
    dit_latent: Tensor,
    teacher_reference: Mapping[str, Any],
    **kwargs: Any,
) -> Tensor:
    # COPY POINT: Adapt both project tensors to the private forward convention here.
    # This v0 private class deliberately wants (lq_rgb, latent), even though the
    # project's public call is conditional_student_decoder(dit_latent, lq_rgb).
    private_lq_rgb = lq_rgb
    private_latent = dit_latent

    # Keep the training graph intact: do not detach and do not use torch.no_grad().
    _ = teacher_reference, kwargs
    return network(private_lq_rgb, private_latent)
