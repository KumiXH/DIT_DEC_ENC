from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml
from torch import nn

from .adapters import ConditionEncoderAdapter, DecoderAdapter, EncoderAdapter, freeze_module
from .contracts import ColorSpec, ConditionSpec, ContractError, LatentSpec
from .factories import build_from_factory
from .models.mock import (
    MockConditionalStudentDecoder,
    MockLQProjIn,
    MockStudentDecoder,
    MockStudentEncoder,
    MockTCDecoder,
    MockWanDecoder,
    MockWanEncoder,
)


PATH_KEYS = {
    "lq_root",
    "gt_root",
    "output_dir",
    "checkpoint",
    "repository",
    "cache_root",
    "source_file",
}


MOCK_MODELS = {
    "student_encoder": MockStudentEncoder,
    "student_decoder": MockStudentDecoder,
    "conditional_student_decoder": MockConditionalStudentDecoder,
    "wan_encoder": MockWanEncoder,
    "wan_decoder": MockWanDecoder,
    "lq_proj_in": MockLQProjIn,
    "tc_decoder": MockTCDecoder,
}


def _resolve_paths(value: Any, base_dir: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {child_key: _resolve_paths(child, base_dir, child_key) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(child, base_dir, key) for child in value]
    if key in PATH_KEYS and isinstance(value, str) and value:
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (base_dir / path).resolve())
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ContractError(f"config file does not exist: {path}")
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ContractError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(values, dict):
        raise ContractError(f"config root must be a mapping: {path}")
    config = _resolve_paths(values, path.parent)
    config["_config_path"] = str(path)
    return config


def apply_overrides(config: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ContractError(f"override must be key=value, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        target: MutableMapping[str, Any] = result
        for key in keys[:-1]:
            child = target.setdefault(key, {})
            if not isinstance(child, MutableMapping):
                raise ContractError(f"cannot assign override below non-mapping key {key!r}")
            target = child
        target[keys[-1]] = yaml.safe_load(raw_value)
    return result


def _build_module(name: str, values: Mapping[str, Any]) -> nn.Module:
    backend = values.get("backend", "mock")
    if backend not in {"mock", "external", "snapshot"}:
        raise ContractError(
            f"component {name!r} has unknown backend {backend!r}; expected mock, external, or snapshot"
        )
    kwargs = values.get("kwargs", {})
    if backend == "mock":
        model_name = values.get("model")
        if model_name not in MOCK_MODELS:
            raise ContractError(
                f"component {name!r} mock model must be one of {sorted(MOCK_MODELS)}, got {model_name!r}"
            )
        module = MOCK_MODELS[model_name](**kwargs)
    else:
        factory = values.get("factory")
        if not factory:
            raise ContractError(f"component {name!r} backend={backend} requires factory")
        module = build_from_factory(
            factory,
            kwargs,
            checkpoint=values.get("checkpoint"),
            strict=values.get("strict", True),
        )
    if values.get("freeze", False):
        freeze_module(module)
    return module


def build_components(config: Mapping[str, Any]) -> dict[str, nn.Module]:
    if "latent_spec" not in config:
        raise ContractError("config requires latent_spec")
    latent_spec = LatentSpec.from_dict(config["latent_spec"])
    color_spec = ColorSpec.from_dict(config.get("color", {}))
    result: dict[str, nn.Module] = {}
    for name, values in config.get("components", {}).items():
        module = _build_module(name, values)
        adapter = values.get("adapter", {})
        kind = adapter.get("kind")
        if kind == "encoder":
            result[name] = EncoderAdapter(
                module,
                latent_spec=latent_spec,
                input_mode=adapter["input_mode"],
                color_spec=color_spec,
                temporal_frames=adapter.get("temporal_frames", 1),
            )
        elif kind == "decoder":
            result[name] = DecoderAdapter(
                module,
                output_mode=adapter["output_mode"],
                color_spec=color_spec,
                accepts_condition=adapter.get("accepts_condition", False),
            )
        elif kind == "condition_encoder":
            condition_values = adapter.get("condition_spec")
            condition_spec = ConditionSpec.from_dict(condition_values) if condition_values else None
            result[name] = ConditionEncoderAdapter(
                module,
                temporal_frames=adapter.get("temporal_frames", 1),
                condition_spec=condition_spec,
            )
        else:
            raise ContractError(f"component {name!r} has unsupported adapter kind {kind!r}")
    return result
