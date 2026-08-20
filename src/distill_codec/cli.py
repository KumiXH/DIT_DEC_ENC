from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

import torch

from .augmentation import collate_augmented_batch, paired_augmentation_from_config
from .config import apply_overrides, build_components, load_config, preflight_config
from .contracts import ContractError
from .data import PairedImageDataset, collate_raw_paired_batch, create_mock_dataset
from .latents import CachedLatentProvider
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
    preflight_config(config)
    components = build_components(config)
    recipe = build_recipe(
        config["recipe"]["name"],
        components,
        config["recipe"].get("weights"),
        source=config["recipe"].get("source", "gt"),
        compatibility_every=int(config["recipe"].get("compatibility_every", 1)),
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
    augmentation = paired_augmentation_from_config(config)
    lq_size = augmentation.target_size or (
        tuple(data["lq_size"]) if data.get("lq_size") else None
    )
    gt_size = augmentation.target_size or (
        tuple(data["gt_size"]) if data.get("gt_size") else None
    )
    dataset = PairedImageDataset(
        data["lq_root"],
        data["gt_root"],
        lq_size=lq_size,
        gt_size=gt_size,
        augmentation=augmentation,
    )
    latent_provider = (
        recipe.components["latent_provider"]
        if "latent_provider" in recipe.components
        else None
    )
    if isinstance(latent_provider, CachedLatentProvider):
        latent_provider.preflight(dataset.gt_sizes_by_relative)
    batch_size = min(int(config.get("trainer", {}).get("batch_size", 1)), len(dataset))
    raw_batch = collate_raw_paired_batch([dataset[index] for index in range(batch_size)])
    batch = collate_augmented_batch(
        raw_batch,
        augmentation,
        phase="probe",
        seed=int(config.get("run", {}).get("seed", 0)),
        device=device,
    )
    output = recipe(batch)
    result = {
        "recipe": recipe.name,
        "preflight": asdict(dataset.preflight_report),
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
