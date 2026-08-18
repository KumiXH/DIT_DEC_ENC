from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .contracts import ContractError, DistillBatch
from .data import PairedImageDataset, collate_distill_batch
from .metrics import save_validation_grid
from .recipes import DistillationRecipe, RecipeOutput


@dataclass(frozen=True)
class TrainResult:
    global_step: int
    start_step: int
    latest_checkpoint: Path
    output_dir: Path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: Mapping[str, Any]) -> None:
        serializable = {
            key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
            for key, value in event.items()
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, sort_keys=True) + "\n")


class Trainer:
    def __init__(self, config: Mapping[str, Any], recipe: DistillationRecipe) -> None:
        self.config = dict(config)
        self.recipe = recipe
        run_config = config.get("run", {})
        trainer_config = config.get("trainer", {})
        self.output_dir = Path(run_config.get("output_dir", "runs/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(run_config.get("seed", 0))
        _seed_everything(self.seed)
        requested_device = trainer_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ContractError("trainer requested CUDA but torch.cuda.is_available() is false")
        self.recipe.to(self.device)
        parameters = list(self.recipe.trainable_parameters())
        if not parameters:
            raise ContractError(f"recipe {self.recipe.name!r} has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(trainer_config.get("learning_rate", 1e-4)),
            weight_decay=float(trainer_config.get("weight_decay", 0.0)),
        )
        scheduler_name = trainer_config.get("scheduler", "none")
        self.scheduler = None
        if scheduler_name == "cosine":
            max_steps = max(1, int(trainer_config.get("max_steps", 1)))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_steps)
        elif scheduler_name != "none":
            raise ContractError(f"unsupported scheduler {scheduler_name!r}")
        self.amp_enabled = bool(trainer_config.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.logger = JsonlLogger(self.output_dir / "metrics.jsonl")
        self.tensorboard = None
        if trainer_config.get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise ContractError("tensorboard logging requested; install distill-codec[train]") from error
            self.tensorboard = SummaryWriter(self.output_dir / "tensorboard")
        self.train_loader, self.validation_loader = self._build_loaders(config)

    @staticmethod
    def _build_loaders(config: Mapping[str, Any]) -> tuple[DataLoader, DataLoader]:
        data_config = config["data"]
        dataset = PairedImageDataset(
            data_config["lq_root"],
            data_config["gt_root"],
            lq_size=tuple(data_config["lq_size"]) if data_config.get("lq_size") else None,
            gt_size=tuple(data_config["gt_size"]) if data_config.get("gt_size") else None,
        )
        trainer_config = config.get("trainer", {})
        common = {
            "batch_size": int(trainer_config.get("batch_size", 1)),
            "num_workers": int(trainer_config.get("num_workers", 0)),
            "collate_fn": collate_distill_batch,
        }
        generator = torch.Generator().manual_seed(int(config.get("run", {}).get("seed", 0)))
        train = DataLoader(dataset, shuffle=True, generator=generator, **common)
        validation = DataLoader(dataset, shuffle=False, **common)
        return train, validation

    def _set_training_modes(self) -> None:
        self.recipe.train()
        for component in self.recipe.components.values():
            if not any(parameter.requires_grad for parameter in component.parameters()):
                component.eval()

    def _infinite_batches(self) -> Iterable[tuple[int, DistillBatch]]:
        epoch = 0
        while True:
            for batch in self.train_loader:
                yield epoch, batch
            epoch += 1

    def fit(self, *, resume: str | Path | None = None) -> TrainResult:
        trainer_config = self.config.get("trainer", {})
        max_steps = int(trainer_config.get("max_steps", 1))
        accumulation = max(1, int(trainer_config.get("gradient_accumulation", 1)))
        validate_every = max(1, int(trainer_config.get("validate_every", max_steps)))
        checkpoint_every = max(1, int(trainer_config.get("checkpoint_every", max_steps)))
        clip_grad = float(trainer_config.get("clip_grad_norm", 0.0))
        global_step = 0
        epoch = 0
        best_metrics: dict[str, float] = {}
        if resume is not None:
            payload = load_checkpoint(
                resume,
                recipe_name=self.recipe.name,
                components=self.recipe.components,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                config=self.config,
            )
            global_step = int(payload["global_step"])
            epoch = int(payload["epoch"])
            best_metrics = dict(payload.get("best_metrics", {}))
        start_step = global_step
        latest_checkpoint = Path(resume) if resume is not None else self.output_dir / "checkpoints" / "initial.pt"
        batches = self._infinite_batches()
        self.optimizer.zero_grad(set_to_none=True)
        while global_step < max_steps:
            self._set_training_modes()
            accumulated_output: RecipeOutput | None = None
            for micro_step in range(accumulation):
                epoch, batch = next(batches)
                batch = batch.to(self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.amp_enabled,
                    dtype=torch.float16,
                ):
                    output = self.recipe(batch)
                    loss = output.total_loss / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at global_step={global_step}, samples={batch.relative_path}, "
                        f"losses={{{', '.join(f'{key}: {float(value.detach())}' for key, value in output.losses.items())}}}"
                    )
                self.scaler.scale(loss).backward()
                accumulated_output = output
            if clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(list(self.recipe.trainable_parameters()), clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            global_step += 1
            assert accumulated_output is not None
            self._log_output("train", global_step, accumulated_output)
            if global_step % validate_every == 0 or global_step == max_steps:
                metrics = self.validate(global_step)
                best_metrics.update(metrics)
            if global_step % checkpoint_every == 0 or global_step == max_steps:
                latest_checkpoint = save_checkpoint(
                    self.output_dir / "checkpoints" / f"step_{global_step:08d}.pt",
                    recipe_name=self.recipe.name,
                    components=self.recipe.components,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    global_step=global_step,
                    epoch=epoch,
                    config=self.config,
                    best_metrics=best_metrics,
                )
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
        return TrainResult(global_step, start_step, latest_checkpoint, self.output_dir)

    @torch.no_grad()
    def validate(self, global_step: int) -> dict[str, float]:
        self.recipe.eval()
        batch = next(iter(self.validation_loader)).to(self.device)
        output = self.recipe(batch)
        self._log_output("validation", global_step, output)
        if output.images:
            save_validation_grid(
                self.output_dir / "validation" / f"step_{global_step:08d}.png",
                output.images,
            )
        return {name: float(value.detach().cpu()) for name, value in output.metrics.items()}

    def _log_output(self, phase: str, global_step: int, output: RecipeOutput) -> None:
        event: dict[str, Any] = {
            "phase": phase,
            "global_step": global_step,
            "total_loss": output.total_loss,
            **{f"loss/{name}": value for name, value in output.losses.items()},
            **{f"metric/{name}": value for name, value in output.metrics.items()},
        }
        self.logger.write(event)
        if self.tensorboard is not None:
            for name, value in event.items():
                if isinstance(value, torch.Tensor):
                    self.tensorboard.add_scalar(f"{phase}/{name}", float(value.detach().cpu()), global_step)

