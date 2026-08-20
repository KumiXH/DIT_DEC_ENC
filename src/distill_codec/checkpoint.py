from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import ColorSpec, ConditionSpec, LatentSpec
from .augmentation import paired_augmentation_from_config


def training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    trainer = config.get("trainer", {})
    run = config.get("run", {})
    data = config.get("data", {})
    recipe = config.get("recipe", {})
    scheduler = str(trainer.get("scheduler", "none"))
    return {
        "optimizer": str(trainer.get("optimizer", "adamw")),
        "learning_rate": float(trainer.get("learning_rate", 1e-4)),
        "weight_decay": float(trainer.get("weight_decay", 0.0)),
        "scheduler": scheduler,
        "scheduler_max_steps": (
            int(trainer.get("scheduler_max_steps", trainer.get("max_steps", 1)))
            if scheduler == "cosine"
            else None
        ),
        "batch_size": int(trainer.get("batch_size", 1)),
        "gradient_accumulation": int(trainer.get("gradient_accumulation", 1)),
        "clip_grad_norm": float(trainer.get("clip_grad_norm", 0.0)),
        "amp": bool(trainer.get("amp", True)),
        "seed": int(run.get("seed", 0)),
        "lq_root": str(data.get("lq_root", "")),
        "gt_root": str(data.get("gt_root", "")),
        "lq_size": list(data["lq_size"]) if data.get("lq_size") is not None else None,
        "gt_size": list(data["gt_size"]) if data.get("gt_size") is not None else None,
        "recipe_source": str(recipe.get("source", "gt")),
        "recipe_weights": dict(recipe.get("weights", {})),
        "compatibility_every": int(recipe.get("compatibility_every", 1)),
        "augmentation": paired_augmentation_from_config(config).to_dict(),
    }


def _latent_shape_contract(spec: LatentSpec) -> dict[str, Any]:
    return {
        "channels": spec.channels,
        "layout": spec.layout,
        "spatial_downsample": spec.spatial_downsample,
        "temporal_downsample": spec.temporal_downsample,
    }


def _condition_shape_contract(spec: ConditionSpec) -> dict[str, Any]:
    contract = {
        "layout": spec.layout,
        "feature_dim": spec.feature_dim,
    }
    for name in ("spatial_downsample", "temporal_downsample"):
        if hasattr(spec, name):
            contract[name] = getattr(spec, name)
    return contract


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def trainable_component_state(components: nn.ModuleDict) -> dict[str, dict[str, Any]]:
    return {
        name: component.state_dict()
        for name, component in components.items()
        if any(parameter.requires_grad for parameter in component.parameters())
    }


def optimizer_parameter_names(
    components: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
) -> list[list[str]]:
    names_by_id = {id(parameter): name for name, parameter in components.named_parameters()}
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter outside recipe components")
            names.append(name)
        groups.append(names)
    return groups


def save_checkpoint(
    path: str | Path,
    *,
    recipe_name: str,
    components: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    global_step: int,
    epoch: int,
    data_batches_consumed: int,
    config: Mapping[str, Any],
    best_metrics: Mapping[str, float],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recipe_name": recipe_name,
        "student_state": trainable_component_state(components),
        "optimizer": optimizer.state_dict(),
        "optimizer_parameter_names": optimizer_parameter_names(components, optimizer),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "data_batches_consumed": data_batches_consumed,
        "config": dict(config),
        "training_contract": training_contract(config),
        "contracts": {
            "latent_spec": dict(config["latent_spec"]),
            "color_spec": dict(config.get("color", {})),
            "condition_specs": {
                name: dict(values["adapter"]["condition_spec"])
                for name, values in config.get("components", {}).items()
                if values.get("adapter", {}).get("condition_spec")
            },
        },
        "best_metrics": dict(best_metrics),
        "rng_state": capture_rng_state(),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    recipe_name: str,
    components: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload["recipe_name"] != recipe_name:
        raise ValueError(
            f"checkpoint recipe={payload['recipe_name']!r} does not match current recipe={recipe_name!r}"
        )
    expected_training = training_contract(config)
    saved_training = payload.get("training_contract", training_contract(payload["config"]))
    if expected_training != saved_training:
        mismatches = [
            f"{name}: expected={expected_training[name]!r}, saved={saved_training.get(name)!r}"
            for name in expected_training
            if expected_training[name] != saved_training.get(name)
        ]
        raise ValueError("incompatible training contract: " + "; ".join(mismatches))
    allow_contract_override = bool(config.get("trainer", {}).get("allow_contract_override", False))
    expected_latent = LatentSpec.from_dict(config["latent_spec"])
    saved_latent = LatentSpec.from_dict(payload["contracts"]["latent_spec"])
    if allow_contract_override:
        expected_shape = _latent_shape_contract(expected_latent)
        saved_shape = _latent_shape_contract(saved_latent)
        if expected_shape != saved_shape:
            raise ValueError(
                f"incompatible latent shape contract: expected={expected_shape}, saved={saved_shape}"
            )
    else:
        expected_latent.assert_compatible(saved_latent)
    expected_color = ColorSpec.from_dict(config.get("color", {}))
    saved_color = ColorSpec.from_dict(payload["contracts"].get("color_spec", {}))
    if not allow_contract_override and expected_color != saved_color:
        raise ValueError(f"incompatible color contract: expected={expected_color}, saved={saved_color}")
    expected_conditions = {
        name: values["adapter"]["condition_spec"]
        for name, values in config.get("components", {}).items()
        if values.get("adapter", {}).get("condition_spec")
    }
    saved_conditions = payload["contracts"].get("condition_specs", {})
    if allow_contract_override:
        if set(expected_conditions) != set(saved_conditions):
            raise ValueError(
                f"incompatible condition shape contracts: expected components={sorted(expected_conditions)}, "
                f"saved components={sorted(saved_conditions)}"
            )
        for name in expected_conditions:
            expected_condition = ConditionSpec.from_dict(expected_conditions[name])
            saved_condition = ConditionSpec.from_dict(saved_conditions[name])
            expected_shape = _condition_shape_contract(expected_condition)
            saved_shape = _condition_shape_contract(saved_condition)
            if expected_shape != saved_shape:
                raise ValueError(
                    f"incompatible condition shape contract for {name!r}: "
                    f"expected={expected_shape}, saved={saved_shape}"
                )
    elif expected_conditions != saved_conditions:
        raise ValueError(
            f"incompatible condition contracts: expected={expected_conditions}, saved={saved_conditions}"
        )
    current_trainable = trainable_component_state(components)
    if set(current_trainable) != set(payload["student_state"]):
        raise ValueError(
            f"checkpoint trainable components={sorted(payload['student_state'])} do not match "
            f"current components={sorted(current_trainable)}"
        )
    current_parameter_names = optimizer_parameter_names(components, optimizer)
    saved_parameter_names = payload.get("optimizer_parameter_names")
    if saved_parameter_names is None:
        if len(current_trainable) > 1:
            raise ValueError(
                "legacy checkpoint has multiple trainable components but no optimizer parameter order; "
                "resume from a checkpoint created by the current version"
            )
    elif saved_parameter_names != current_parameter_names:
        raise ValueError(
            "incompatible optimizer parameter order: "
            f"expected={current_parameter_names}, saved={saved_parameter_names}"
        )
    for name, state_dict in payload["student_state"].items():
        components[name].load_state_dict(state_dict)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng_state"])
    return payload
