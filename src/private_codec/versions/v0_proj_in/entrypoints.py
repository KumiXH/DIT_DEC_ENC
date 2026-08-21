from __future__ import annotations

from typing import Any

from torch import nn

from .wrapped_network import V0ProjInConditionEncoder


# COPY POINT: The YAML imports only this function. When adding v1_proj_in, copy
# this whole version directory and change construction details inside it.


def create_condition_encoder(**kwargs: Any) -> nn.Module:
    """Build the private LQ_proj_in student used by ConditionEncoderAdapter."""

    return V0ProjInConditionEncoder(**kwargs)

