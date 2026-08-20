from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from torch import nn

from .bridge import PrivateConditionalDecoderBridge, PrivateEncoderBridge


def _create_model(
    *,
    module_path: str,
    class_name: str,
    init_kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    try:
        module = importlib.import_module(module_path)
    except ImportError as error:
        raise ValueError(
            f"cannot import private codec module {module_path!r}: {error}"
        ) from error

    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ValueError(f"cannot find class {class_name!r} in module {module_path!r}")

    model = model_class(**dict(init_kwargs or {}))
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"{module_path}:{class_name} returned {type(model).__name__}, expected torch.nn.Module"
        )
    return model


def create_encoder(**kwargs: Any) -> nn.Module:
    if "builder" in kwargs or "runner" in kwargs:
        return PrivateEncoderBridge(**kwargs)
    return _create_model(**kwargs)


def create_decoder(**kwargs: Any) -> nn.Module:
    return _create_model(**kwargs)


def create_conditional_decoder(**kwargs: Any) -> nn.Module:
    return PrivateConditionalDecoderBridge(**kwargs)
