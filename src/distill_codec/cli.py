from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .config import apply_overrides, build_components, load_config
from .contracts import ContractError
from .data import PairedImageDataset, collate_distill_batch, create_mock_dataset
from .recipes import build_recipe
from .trainer import Trainer


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="dotted.key=value")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distill-codec")
    subparsers = parser.add_subparsers(dest="command", required=True)
    make_data = subparsers.add_parser("make-mock-data", help="create paired mock LQ/GT images")
    make_data.add_argument("--output", required=True)
    make_data.add_argument("--count", type=int, default=8)
    make_data.add_argument("--size", type=int, default=64)
    make_data.add_argument("--seed", type=int, default=0)
    probe = subparsers.add_parser("probe", help="construct and run one forward pass")
    _add_config_arguments(probe)
    train = subparsers.add_parser("train", help="train a configured distillation recipe")
    _add_config_arguments(train)
    train.add_argument("--resume")
    return parser


def _load_recipe(config_path: str, overrides: list[str]):
    config = apply_overrides(load_config(config_path), overrides)
    components = build_components(config)
    recipe = build_recipe(
        config["recipe"]["name"],
        components,
        config["recipe"].get("weights"),
        source=config["recipe"].get("source", "gt"),
    )
    return config, recipe


def _probe(config_path: str, overrides: list[str]) -> int:
    config, recipe = _load_recipe(config_path, overrides)
    if config.get("latent_provider", {}).get("type") == "dataset":
        raise ContractError(
            "probe cannot use the dataset latent provider with the standard paired-image dataset; "
            "use type='cached' or a custom probe that populates DistillBatch.latent"
        )
    trainer_config = config.get("trainer", {})
    requested_device = trainer_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("probe requested CUDA but torch.cuda.is_available() is false")
    recipe.to(device)
    data = config["data"]
    dataset = PairedImageDataset(
        data["lq_root"],
        data["gt_root"],
        lq_size=tuple(data["lq_size"]) if data.get("lq_size") else None,
        gt_size=tuple(data["gt_size"]) if data.get("gt_size") else None,
    )
    batch_size = min(int(config.get("trainer", {}).get("batch_size", 1)), len(dataset))
    batch = collate_distill_batch([dataset[index] for index in range(batch_size)]).to(device)
    output = recipe(batch)
    result = {
        "recipe": recipe.name,
        "trainable_parameters": sum(parameter.numel() for parameter in recipe.trainable_parameters()),
        "total_loss": float(output.total_loss.detach()),
        "losses": {name: float(value.detach()) for name, value in output.losses.items()},
        "images": {name: list(value.shape) for name, value in output.images.items()},
        "metadata": output.metadata,
        "device": str(device),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "make-mock-data":
        paths = create_mock_dataset(
            arguments.output,
            count=arguments.count,
            size=(arguments.size, arguments.size),
            seed=arguments.seed,
        )
        print(json.dumps({"lq_root": str(paths.lq_root), "gt_root": str(paths.gt_root)}))
        return 0
    if arguments.command == "probe":
        return _probe(arguments.config, arguments.overrides)
    config, recipe = _load_recipe(arguments.config, arguments.overrides)
    result = Trainer(config, recipe).fit(resume=arguments.resume)
    print(
        json.dumps(
            {
                "global_step": result.global_step,
                "start_step": result.start_step,
                "checkpoint": str(result.latest_checkpoint),
                "output_dir": str(result.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
