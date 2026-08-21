from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml
from torch import nn

from .augmentation import paired_augmentation_from_config
from .adapters import ConditionEncoderAdapter, DecoderAdapter, EncoderAdapter, freeze_module
from .contracts import ColorSpec, ConditionSpec, ContractError, LatentSpec
from .factories import build_from_factory
from .latents import (
    CachedLatentProvider,
    DatasetGTTeacherTargetProvider,
    DatasetLatentProvider,
    TeacherEncoderLatentProvider,
)
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


def _teacher_target_type(config: Mapping[str, Any]) -> str:
    provider = config.get("teacher_target_provider")
    if provider is None:
        return "online"
    if not isinstance(provider, Mapping):
        raise ContractError("config field teacher_target_provider must be a mapping")
    provider_type = provider.get("type", "online")
    if provider_type not in {"online", "dataset_gt"}:
        raise ContractError(
            f"unknown teacher_target_provider type {provider_type!r}; expected online or dataset_gt"
        )
    return str(provider_type)


def _required_components(config: Mapping[str, Any], recipe_name: str) -> set[str]:
    required = set(RECIPE_COMPONENTS[recipe_name])
    if _teacher_target_type(config) == "dataset_gt":
        if recipe_name not in {
            "flashvsr_decoder_unconditional_student",
            "flashvsr_decoder_conditional_student",
        }:
            raise ContractError(
                "teacher_target_provider.type='dataset_gt' is only supported by FlashVSR decoder recipes"
            )
        required.discard("tc_decoder")
    return required


def _validate_teacher_target_provider(
    config: Mapping[str, Any],
    latent_provider: Mapping[str, Any] | None,
    *,
    location: str,
) -> None:
    if _teacher_target_type(config) == "dataset_gt" and (
        latent_provider is None or latent_provider.get("type") != "cached"
    ):
        raise ContractError(
            "teacher_target_provider.type='dataset_gt' requires latent_provider.type='cached'; "
            f"config={location}"
        )


def _config_location(config: Mapping[str, Any]) -> str:
    return str(config.get("_config_path", "<in-memory config>"))


def _require_mapping(
    values: Mapping[str, Any],
    key: str,
    *,
    path: str,
    location: str,
) -> Mapping[str, Any]:
    if key not in values:
        raise ContractError(f"config requires {path}; config={location}")
    child = values[key]
    if not isinstance(child, Mapping):
        raise ContractError(f"config field {path} must be a mapping; config={location}")
    return child


def preflight_config(config: Mapping[str, Any]) -> None:
    location = _config_location(config)
    recipe = _require_mapping(config, "recipe", path="recipe.name", location=location)
    recipe_name = recipe.get("name")
    if not isinstance(recipe_name, str) or not recipe_name:
        raise ContractError(f"config requires recipe.name; config={location}")
    if recipe_name not in RECIPE_COMPONENTS:
        raise ContractError(
            f"config recipe.name={recipe_name!r} is unknown; "
            f"available={sorted(RECIPE_COMPONENTS)}; config={location}"
        )
    if "latent_spec" not in config:
        raise ContractError(f"config requires latent_spec; config={location}")
    if "color" not in config:
        raise ContractError(f"config requires color; config={location}")
    try:
        LatentSpec.from_dict(_require_mapping(config, "latent_spec", path="latent_spec", location=location))
        ColorSpec.from_dict(_require_mapping(config, "color", path="color", location=location))
    except (ContractError, TypeError) as error:
        raise ContractError(f"invalid tensor/color contract: {error}; config={location}") from error
    data = _require_mapping(config, "data", path="data", location=location)
    for name in ("lq_root", "gt_root"):
        if not data.get(name):
            raise ContractError(f"config requires data.{name}; config={location}")
    components = _require_mapping(config, "components", path="components", location=location)
    required = _required_components(config, recipe_name)
    provider = config.get("latent_provider")
    if provider is not None and not isinstance(provider, Mapping):
        raise ContractError(f"config field latent_provider must be a mapping; config={location}")
    if recipe_name in {
        "wan_decoder_distill",
        "flashvsr_decoder_unconditional_student",
        "flashvsr_decoder_conditional_student",
    } and (provider is None or provider.get("type", "teacher_encoder") == "teacher_encoder"):
        required.add("teacher_encoder")
    for name in sorted(required):
        if name not in components:
            raise ContractError(f"config requires components.{name}; config={location}")
        if not isinstance(components[name], Mapping):
            raise ContractError(
                f"config field components.{name} must be a mapping; config={location}"
            )
    if isinstance(provider, Mapping) and provider.get("type") == "cached" and not provider.get("root"):
        raise ContractError(f"config requires latent_provider.root; config={location}")
    _validate_teacher_target_provider(
        config,
        provider if isinstance(provider, Mapping) else None,
        location=location,
    )
    try:
        paired_augmentation_from_config(config)
    except ContractError as error:
        raise ContractError(
            f"invalid paired augmentation: {error}; config={location}"
        ) from error


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


def _reference_image_size(config: Mapping[str, Any], source: str) -> tuple[int, int]:
    key = f"{source}_size"
    data = config.get("data", {})
    value = data.get(key) if isinstance(data, Mapping) else None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(size) is not int or size <= 0 for size in value)
    ):
        raise ContractError(
            f"data.{key} must contain positive integer height and width for teacher_reference"
        )
    return int(value[0]), int(value[1])


def _teacher_temporal_frames(config: Mapping[str, Any]) -> int:
    components = config.get("components", {})
    if not isinstance(components, Mapping):
        return 1
    teacher_encoder = components.get("teacher_encoder", {})
    if not isinstance(teacher_encoder, Mapping):
        return 1
    adapter = teacher_encoder.get("adapter", {})
    if not isinstance(adapter, Mapping):
        return 1
    value = adapter.get("temporal_frames", 1)
    if type(value) is not int or value <= 0:
        raise ContractError(
            "components.teacher_encoder.adapter.temporal_frames must be a positive integer "
            "for teacher_reference"
        )
    return value


def _reference_latent_shape(
    latent_spec: LatentSpec,
    *,
    image_size: tuple[int, int],
    temporal_frames: int,
) -> list[int | None]:
    height, width = image_size
    spatial = [
        height // latent_spec.spatial_downsample,
        width // latent_spec.spatial_downsample,
    ]
    if latent_spec.layout == "BCHW":
        return [None, latent_spec.channels, *spatial]
    temporal = (
        temporal_frames + latent_spec.temporal_downsample - 1
    ) // latent_spec.temporal_downsample
    return [None, latent_spec.channels, temporal, *spatial]


def _teacher_reference(
    config: Mapping[str, Any],
    *,
    name: str,
    values: Mapping[str, Any],
    latent_spec: LatentSpec,
) -> dict[str, Any]:
    adapter = values.get("adapter", {})
    if not isinstance(adapter, Mapping):
        raise ContractError(
            f"component {name!r} adapter must be a mapping for teacher_reference"
        )
    kind = adapter.get("kind")
    temporal_frames = _teacher_temporal_frames(config)
    if kind == "encoder":
        recipe = config.get("recipe", {})
        source = recipe.get("source", "gt") if isinstance(recipe, Mapping) else "gt"
        if source not in {"lq", "gt"}:
            raise ContractError(
                f"component {name!r} teacher_reference encoder source must be lq or gt"
            )
        image_size = _reference_image_size(config, source)
        return {
            "role": "encoder",
            "inputs": {
                "rgb": {
                    "layout": "BCHW",
                    "shape": [None, 3, *image_size],
                    "source": source,
                }
            },
            "outputs": {
                "latent": {
                    "layout": latent_spec.layout,
                    "shape": _reference_latent_shape(
                        latent_spec,
                        image_size=image_size,
                        temporal_frames=temporal_frames,
                    ),
                }
            },
        }
    if kind == "decoder" and adapter.get("accepts_condition", False):
        provider = config.get("latent_provider", {})
        latent_source = (
            provider.get("source", "gt") if isinstance(provider, Mapping) else "gt"
        )
        if latent_source not in {"lq", "gt"}:
            raise ContractError(
                f"component {name!r} teacher_reference latent source must be lq or gt"
            )
        teacher_size = _reference_image_size(config, "gt")
        latent_size = _reference_image_size(config, latent_source)
        return {
            "role": "conditional_decoder",
            "inputs": {
                "lq_rgb": {
                    "layout": "BCHW",
                    "shape": [None, 3, *teacher_size],
                    "source": "teacher_condition",
                },
                "dit_latent": {
                    "layout": latent_spec.layout,
                    "shape": _reference_latent_shape(
                        latent_spec,
                        image_size=latent_size,
                        temporal_frames=temporal_frames,
                    ),
                },
            },
            "outputs": {
                "rgb": {
                    "layout": "BCHW",
                    "shape": [None, 3, *teacher_size],
                }
            },
        }
    raise ContractError(
        f"component {name!r} teacher_reference=auto supports encoder or conditional decoder adapters"
    )


def _component_build_values(
    config: Mapping[str, Any],
    *,
    name: str,
    values: Mapping[str, Any],
    latent_spec: LatentSpec,
) -> Mapping[str, Any]:
    reference_mode = values.get("teacher_reference")
    if reference_mode is None:
        return values
    if reference_mode != "auto":
        raise ContractError(
            f"component {name!r} teacher_reference must be 'auto', got {reference_mode!r}"
        )
    raw_kwargs = values.get("kwargs", {})
    if not isinstance(raw_kwargs, Mapping):
        raise ContractError(f"component {name!r} kwargs must be a mapping")
    if "teacher_reference" in raw_kwargs:
        raise ContractError(
            f"component {name!r} cannot define teacher_reference both automatically and in kwargs"
        )
    build_values = dict(values)
    kwargs = dict(raw_kwargs)
    kwargs["teacher_reference"] = _teacher_reference(
        config,
        name=name,
        values=values,
        latent_spec=latent_spec,
    )
    build_values["kwargs"] = kwargs
    return build_values


def build_components(config: Mapping[str, Any]) -> dict[str, nn.Module]:
    if "latent_spec" not in config:
        raise ContractError("config requires latent_spec")
    if "color" not in config:
        raise ContractError("config requires color")
    latent_spec = LatentSpec.from_dict(config["latent_spec"])
    color_spec = ColorSpec.from_dict(config["color"])
    recipe_name = config.get("recipe", {}).get("name")
    required = (
        _required_components(config, recipe_name)
        if recipe_name in RECIPE_COMPONENTS
        else set(config.get("components", {}).keys())
    )
    provider_values = config.get("latent_provider")
    if provider_values is not None and not isinstance(provider_values, Mapping):
        raise ContractError("config field latent_provider must be a mapping")
    _validate_teacher_target_provider(
        config,
        provider_values,
        location=_config_location(config),
    )
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
        module = _build_module(
            name,
            _component_build_values(
                config,
                name=name,
                values=values,
                latent_spec=latent_spec,
            ),
        )
        adapter = values.get("adapter", {})
        kind = adapter.get("kind")
        if kind == "encoder":
            raw_latent_temporal_frames = adapter.get("latent_temporal_frames")
            if raw_latent_temporal_frames == "teacher":
                latent_temporal_frames = _teacher_temporal_frames(config)
            elif raw_latent_temporal_frames is None:
                latent_temporal_frames = None
            elif (
                type(raw_latent_temporal_frames) is int
                and raw_latent_temporal_frames > 0
            ):
                latent_temporal_frames = raw_latent_temporal_frames
            else:
                raise ContractError(
                    f"component {name!r} adapter.latent_temporal_frames must be "
                    "'teacher' or a positive integer"
                )
            result[name] = EncoderAdapter(
                module,
                latent_spec=latent_spec,
                input_mode=adapter["input_mode"],
                color_spec=color_spec,
                temporal_frames=adapter.get("temporal_frames", 1),
                latent_temporal_frames=latent_temporal_frames,
                frame_selection=adapter.get("frame_selection", "center"),
            )
        elif kind == "decoder":
            result[name] = DecoderAdapter(
                module,
                output_mode=adapter["output_mode"],
                color_spec=color_spec,
                accepts_condition=adapter.get("accepts_condition", False),
                frame_selection=adapter.get("frame_selection", "center"),
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
    if _teacher_target_type(config) == "dataset_gt":
        result["teacher_target_provider"] = DatasetGTTeacherTargetProvider()
    return result
