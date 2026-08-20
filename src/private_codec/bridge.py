from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor, nn


def _import_callable(import_path: str, *, role: str) -> Callable[..., Any]:
    if not isinstance(import_path, str) or import_path.count(":") != 1:
        raise ValueError(
            f"private codec {role} must use 'module:symbol' syntax, got {import_path!r}"
        )
    module_name, symbol_name = import_path.split(":", 1)
    if (
        not module_name
        or not symbol_name
        or module_name.startswith(".")
        or module_name != module_name.strip()
        or symbol_name != symbol_name.strip()
    ):
        raise ValueError(
            f"private codec {role} must use 'module:symbol' syntax with a non-empty "
            f"absolute module and symbol, got {import_path!r}"
        )
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, symbol_name)
    except (ImportError, AttributeError) as error:
        raise ValueError(
            f"cannot import private codec {role} {import_path!r}: {error}"
        ) from error
    if not callable(value):
        raise TypeError(
            f"private codec {role} {import_path!r} is {type(value).__name__}, expected callable"
        )
    return value


class _PrivateBridge(nn.Module):
    def __init__(
        self,
        *,
        builder: str,
        runner: str,
        teacher_reference: Mapping[str, Any],
        builder_kwargs: Mapping[str, Any] | None = None,
        runner_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        constructor = _import_callable(builder, role="builder")
        runner_callable = _import_callable(runner, role="runner")
        network = constructor(**dict(builder_kwargs or {}))
        if not isinstance(network, nn.Module):
            raise TypeError(
                f"private codec builder {builder!r} returned {type(network).__name__}, "
                "expected torch.nn.Module"
            )
        self.network = network
        self._runner = runner_callable
        self._runner_path = runner
        self._runner_kwargs = dict(runner_kwargs or {})
        self.teacher_reference = copy.deepcopy(dict(teacher_reference))

    def _run(self, **inputs: Tensor) -> Tensor:
        call_kwargs = dict(self._runner_kwargs)
        call_kwargs.update(
            network=self.network,
            teacher_reference=copy.deepcopy(self.teacher_reference),
        )
        call_kwargs.update(inputs)
        output = self._runner(**call_kwargs)
        if not isinstance(output, Tensor):
            raise TypeError(
                f"private codec runner {self._runner_path!r} returned "
                f"{type(output).__name__}, expected torch.Tensor"
            )
        return output


class PrivateEncoderBridge(_PrivateBridge):
    def forward(self, rgb: Tensor) -> Tensor:
        return self._run(rgb=rgb)


class PrivateConditionalDecoderBridge(_PrivateBridge):
    def forward(self, dit_latent: Tensor, lq_rgb: Tensor) -> Tensor:
        return self._run(lq_rgb=lq_rgb, dit_latent=dit_latent)
