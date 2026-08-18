from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import ContractError


def import_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ContractError(f"factory must use 'module:object' syntax, got {import_path!r}")
    module_name, object_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, object_name)
    except (ImportError, AttributeError) as error:
        raise ContractError(f"cannot import factory {import_path!r}: {error}") from error


def build_from_factory(
    factory: str,
    kwargs: Mapping[str, Any] | None = None,
    *,
    checkpoint: str | Path | None = None,
    strict: bool = True,
) -> nn.Module:
    constructor = import_object(factory)
    module = constructor(**dict(kwargs or {}))
    if not isinstance(module, nn.Module):
        raise ContractError(f"factory {factory!r} returned {type(module).__name__}, expected nn.Module")
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise ContractError(f"checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        module.load_state_dict(state_dict, strict=strict)
    return module

