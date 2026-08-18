from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import ColorSpec, LatentSpec


def training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    trainer = config.get("trainer", {})
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
    }


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
    expected_latent = LatentSpec.from_dict(config["latent_spec"])
    saved_latent = LatentSpec.from_dict(payload["contracts"]["latent_spec"])
    expected_latent.assert_compatible(saved_latent)
    expected_color = ColorSpec.from_dict(config.get("color", {}))
    saved_color = ColorSpec.from_dict(payload["contracts"].get("color_spec", {}))
    if expected_color != saved_color:
        raise ValueError(f"incompatible color contract: expected={expected_color}, saved={saved_color}")
    expected_conditions = {
        name: values["adapter"]["condition_spec"]
        for name, values in config.get("components", {}).items()
        if values.get("adapter", {}).get("condition_spec")
    }
    if expected_conditions != payload["contracts"].get("condition_specs", {}):
        raise ValueError(
            f"incompatible condition contracts: expected={expected_conditions}, "
            f"saved={payload['contracts'].get('condition_specs', {})}"
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
