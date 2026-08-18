from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml
from torch import nn

from .adapters import ConditionEncoderAdapter, DecoderAdapter, EncoderAdapter, freeze_module
from .contracts import ColorSpec, ConditionSpec, ContractError, LatentSpec
from .factories import build_from_factory
from .latents import CachedLatentProvider, DatasetLatentProvider, TeacherEncoderLatentProvider
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
    "root",
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


RECIPE_COMPONENTS = {
    "wan_encoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder"},
    "flashvsr_vae_encoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder"},
    "wan_decoder_distill": {"teacher_decoder", "student_decoder"},
    "wan_autoencoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder", "student_decoder"},
    "flashvsr_lq_proj_distill": {"student_condition_encoder", "teacher_condition_encoder"},
    "flashvsr_decoder_unconditional_student": {"tc_decoder", "student_decoder"},
    "flashvsr_decoder_conditional_student": {"tc_decoder", "conditional_student_decoder"},
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


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ContractError(f"config file does not exist: {path}")
    if path in _stack:
        cycle = " -> ".join(str(item) for item in (*_stack, path))
        raise ContractError(f"config include cycle: {cycle}")
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ContractError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(values, dict):
        raise ContractError(f"config root must be a mapping: {path}")
    merged: dict[str, Any] = {}
    includes = values.pop("includes", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ContractError(f"config includes must be a string or list: {path}")
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        included = load_config(include_path, (*_stack, path))
        included.pop("_config_path", None)
        merged = _deep_merge(merged, included)
    config = _deep_merge(merged, _resolve_paths(values, path.parent))
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
    expected_sha = values.get("sha256")
    checkpoint_value = values.get("checkpoint") or kwargs.get("checkpoint")
    if expected_sha:
        if not checkpoint_value:
            raise ContractError(f"component {name!r} declares sha256 without a checkpoint")
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_file():
            raise ContractError(f"component {name!r} checkpoint does not exist: {checkpoint_path}")
        actual_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if actual_sha.lower() != str(expected_sha).lower():
            raise ContractError(
                f"SHA256 mismatch for component {name!r}: expected={expected_sha}, actual={actual_sha}"
            )
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
    recipe_name = config.get("recipe", {}).get("name")
    required = set(RECIPE_COMPONENTS.get(recipe_name, config.get("components", {}).keys()))
    provider_values = config.get("latent_provider")
    if recipe_name in {
        "wan_decoder_distill",
        "flashvsr_decoder_unconditional_student",
        "flashvsr_decoder_conditional_student",
    } and (provider_values is None or provider_values.get("type", "teacher_encoder") == "teacher_encoder"):
        required.add("teacher_encoder")
    result: dict[str, nn.Module] = {}
    for name, values in config.get("components", {}).items():
        if name not in required:
            continue
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
    if provider_values:
        provider_type = provider_values.get("type", "teacher_encoder")
        if provider_type == "teacher_encoder":
            if "teacher_encoder" not in result:
                raise ContractError("teacher_encoder latent provider requires a teacher_encoder component")
            result["latent_provider"] = TeacherEncoderLatentProvider(
                result["teacher_encoder"],
                source=provider_values.get("source", "gt"),
                latent_spec=latent_spec,
            )
        elif provider_type == "cached":
            result["latent_provider"] = CachedLatentProvider(provider_values["root"], latent_spec=latent_spec)
        elif provider_type == "dataset":
            result["latent_provider"] = DatasetLatentProvider(latent_spec=latent_spec)
        elif provider_type == "frozen_dit":
            raise ContractError("latent_provider type='frozen_dit' is reserved but not implemented in v1")
        else:
            raise ContractError(
                f"unknown latent_provider type {provider_type!r}; expected teacher_encoder, cached, dataset, or frozen_dit"
            )
    return result
