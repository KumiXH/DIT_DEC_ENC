from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from .contracts import ContractError, DistillBatch
from .data import RawPairedBatch


@dataclass(frozen=True)
class CropAugmentation:
    enabled: bool = False
    mode: str = "random"


@dataclass(frozen=True)
class RotationAugmentation:
    enabled: bool = False
    mode: str = "continuous"
    probability: float = 0.0
    degrees: tuple[float, float] = (-5.0, 5.0)
    interpolation: str = "bilinear"
    padding_mode: str = "reflection"


@dataclass(frozen=True)
class TranslationAugmentation:
    enabled: bool = False
    probability: float = 0.0
    max_fraction: tuple[float, float] = (0.0, 0.0)
    padding_mode: str = "reflection"


@dataclass(frozen=True)
class PairedAugmentation:
    enabled: bool
    shared_across_batch: bool
    target_size: tuple[int, int] | None
    crop: CropAugmentation
    rotation: RotationAugmentation
    translation: TranslationAugmentation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a boolean")
    return value


def _pair(
    values: object,
    *,
    name: str,
    cast_type: Callable[[Any], T],
) -> tuple[T, T]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ContractError(f"{name} must contain exactly two values")
    try:
        return cast_type(values[0]), cast_type(values[1])
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} must contain numeric values") from error


def _probability(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float, str)):
        raise ContractError(f"{name} must be numeric")
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} must be numeric") from error
    if not 0.0 <= probability <= 1.0:
        raise ContractError(f"{name} must be in [0, 1], got {probability}")
    return probability


def paired_augmentation_from_config(config: Mapping[str, Any]) -> PairedAugmentation:
    data = config.get("data", {})
    values = data.get("augmentation", {})
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ContractError("data.augmentation must be a mapping")
    enabled = _boolean(
        values.get("enabled", False),
        name="data.augmentation.enabled",
    )
    shared = _boolean(
        values.get("shared_across_batch", True),
        name="data.augmentation.shared_across_batch",
    )

    crop_values = values.get("crop", {})
    rotation_values = values.get("rotation", {})
    translation_values = values.get("translation", {})
    for name, section in (
        ("crop", crop_values),
        ("rotation", rotation_values),
        ("translation", translation_values),
    ):
        if not isinstance(section, Mapping):
            raise ContractError(f"data.augmentation.{name} must be a mapping")

    crop = CropAugmentation(
        enabled=_boolean(
            crop_values.get("enabled", False),
            name="data.augmentation.crop.enabled",
        ),
        mode=str(crop_values.get("mode", "random")),
    )
    degrees = _pair(
        rotation_values.get("degrees", (-5.0, 5.0)),
        name="data.augmentation.rotation.degrees",
        cast_type=float,
    )
    rotation = RotationAugmentation(
        enabled=_boolean(
            rotation_values.get("enabled", False),
            name="data.augmentation.rotation.enabled",
        ),
        mode=str(rotation_values.get("mode", "continuous")),
        probability=_probability(
            rotation_values.get("probability", 0.0),
            name="data.augmentation.rotation.probability",
        ),
        degrees=degrees,
        interpolation=str(rotation_values.get("interpolation", "bilinear")),
        padding_mode=str(rotation_values.get("padding_mode", "reflection")),
    )
    max_fraction = _pair(
        translation_values.get("max_fraction", (0.0, 0.0)),
        name="data.augmentation.translation.max_fraction",
        cast_type=float,
    )
    translation = TranslationAugmentation(
        enabled=_boolean(
            translation_values.get("enabled", False),
            name="data.augmentation.translation.enabled",
        ),
        probability=_probability(
            translation_values.get("probability", 0.0),
            name="data.augmentation.translation.probability",
        ),
        max_fraction=max_fraction,
        padding_mode=str(translation_values.get("padding_mode", rotation.padding_mode)),
    )

    lq_size = data.get("lq_size")
    gt_size = data.get("gt_size")
    target_size = None
    if lq_size is not None and gt_size is not None:
        lq_target = _pair(lq_size, name="data.lq_size", cast_type=int)
        gt_target = _pair(gt_size, name="data.gt_size", cast_type=int)
        if lq_target != gt_target and enabled:
            raise ContractError(
                "paired augmentation requires data.lq_size and data.gt_size to be equal"
            )
        target_size = lq_target if lq_target == gt_target else None

    if enabled:
        if not shared:
            raise ContractError(
                "data.augmentation.shared_across_batch=false is not supported in v1"
            )
        if target_size is None:
            raise ContractError(
                "paired augmentation requires equal data.lq_size and data.gt_size"
            )
        if any(size <= 0 for size in target_size):
            raise ContractError("data.lq_size and data.gt_size must be positive")
        if crop.mode != "random":
            raise ContractError("data.augmentation.crop.mode must be random")
        if rotation.mode not in {"continuous", "right_angle"}:
            raise ContractError(
                "data.augmentation.rotation.mode must be continuous or right_angle"
            )
        if rotation.degrees[0] > rotation.degrees[1]:
            raise ContractError("data.augmentation.rotation.degrees must be ascending")
        if rotation.interpolation not in {"bilinear", "nearest"}:
            raise ContractError(
                "data.augmentation.rotation.interpolation must be bilinear or nearest"
            )
        if rotation.padding_mode not in {"reflection", "border", "zeros"}:
            raise ContractError(
                "data.augmentation.rotation.padding_mode must be reflection, border, or zeros"
            )
        if translation.padding_mode not in {"reflection", "border", "zeros"}:
            raise ContractError(
                "data.augmentation.translation.padding_mode must be reflection, border, or zeros"
            )
        if (
            rotation.enabled
            and translation.enabled
            and rotation.padding_mode != translation.padding_mode
        ):
            raise ContractError(
                "rotation and translation padding_mode values must match"
            )
        if any(value < 0.0 or value >= 1.0 for value in translation.max_fraction):
            raise ContractError(
                "data.augmentation.translation.max_fraction values must be in [0, 1)"
            )
        if (
            rotation.enabled
            and rotation.mode == "right_angle"
            and target_size[0] != target_size[1]
        ):
            raise ContractError("right_angle rotation requires a square target size")
        provider = config.get("latent_provider")
        provider_type = provider.get("type") if isinstance(provider, Mapping) else None
        if provider_type in {"cached", "dataset"}:
            raise ContractError(
                f"paired augmentation cannot be used with latent_provider.type={provider_type!r}"
            )

    return PairedAugmentation(
        enabled=enabled,
        shared_across_batch=shared,
        target_size=target_size,
        crop=crop,
        rotation=rotation,
        translation=translation,
    )


def _generator(
    seed: int, phase: str, global_step: int, micro_step: int
) -> torch.Generator:
    phase_value = {"train": 11, "validation": 23, "probe": 37}.get(phase, 53)
    mixed = (
        int(seed) * 1_000_003
        + phase_value * 97_409
        + int(global_step) * 65_537
        + int(micro_step) * 4_099
    ) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(mixed)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    if low == high:
        return low
    return low + (high - low) * float(torch.rand((), generator=generator))


def _crop_pair(
    lq: Tensor,
    gt: Tensor,
    target_size: tuple[int, int],
    *,
    vertical: float,
    horizontal: float,
) -> tuple[Tensor, Tensor]:
    height, width = int(lq.shape[-2]), int(lq.shape[-1])
    target_height, target_width = target_size
    top = round(vertical * (height - target_height))
    left = round(horizontal * (width - target_width))
    slices = (..., slice(top, top + target_height), slice(left, left + target_width))
    return lq[slices], gt[slices]


def _apply_translation(
    images: Tensor,
    *,
    vertical: float,
    horizontal: float,
    interpolation: str,
    padding_mode: str,
) -> Tensor:
    theta = (
        images.new_tensor([[1.0, 0.0, -2.0 * horizontal], [0.0, 1.0, -2.0 * vertical]])
        .unsqueeze(0)
        .expand(images.shape[0], -1, -1)
    )
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=False,
    )


def _apply_continuous_affine(
    images: Tensor,
    *,
    angle: float,
    vertical: float,
    horizontal: float,
    interpolation: str,
    padding_mode: str,
) -> Tensor:
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    theta = (
        images.new_tensor(
            [
                [cosine, -sine, -2.0 * horizontal],
                [sine, cosine, -2.0 * vertical],
            ]
        )
        .unsqueeze(0)
        .expand(images.shape[0], -1, -1)
    )
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=False,
    )


def collate_augmented_batch(
    raw: RawPairedBatch,
    augmentation: PairedAugmentation,
    *,
    phase: str,
    seed: int,
    global_step: int = 0,
    micro_step: int = 0,
    device: torch.device | str | None = None,
) -> DistillBatch:
    generator = _generator(seed, phase, global_step, micro_step)
    random_training = phase == "train"
    if augmentation.enabled and augmentation.crop.enabled:
        position = (
            (
                float(torch.rand((), generator=generator)),
                float(torch.rand((), generator=generator)),
            )
            if random_training
            else (0.5, 0.5)
        )
        assert augmentation.target_size is not None
        pairs = [
            _crop_pair(
                lq,
                gt,
                augmentation.target_size,
                vertical=position[0],
                horizontal=position[1],
            )
            for lq, gt in zip(raw.lq_rgb, raw.gt_rgb, strict=True)
        ]
    else:
        pairs = list(zip(raw.lq_rgb, raw.gt_rgb, strict=True))

    lq_batch = torch.stack([pair[0] for pair in pairs])
    gt_batch = torch.stack([pair[1] for pair in pairs])
    if device is not None:
        lq_batch = lq_batch.to(device)
        gt_batch = gt_batch.to(device)

    if augmentation.enabled and random_training:
        angle = 0.0
        rotate = (
            augmentation.rotation.enabled
            and float(torch.rand((), generator=generator))
            < augmentation.rotation.probability
        )
        if rotate:
            if augmentation.rotation.mode == "right_angle":
                quarter_turns = int(torch.randint(0, 4, (), generator=generator))
                lq_batch = torch.rot90(lq_batch, quarter_turns, dims=(-2, -1))
                gt_batch = torch.rot90(gt_batch, quarter_turns, dims=(-2, -1))
            else:
                angle = _uniform(
                    generator,
                    augmentation.rotation.degrees[0],
                    augmentation.rotation.degrees[1],
                )
        vertical = horizontal = 0.0
        translate = (
            augmentation.translation.enabled
            and float(torch.rand((), generator=generator))
            < augmentation.translation.probability
        )
        if translate:
            vertical = _uniform(
                generator,
                -augmentation.translation.max_fraction[0],
                augmentation.translation.max_fraction[0],
            )
            horizontal = _uniform(
                generator,
                -augmentation.translation.max_fraction[1],
                augmentation.translation.max_fraction[1],
            )
        combined = torch.cat((lq_batch, gt_batch), dim=0)
        padding_mode = (
            augmentation.rotation.padding_mode
            if augmentation.rotation.enabled
            else augmentation.translation.padding_mode
        )
        if rotate and augmentation.rotation.mode == "continuous":
            combined = _apply_continuous_affine(
                combined,
                angle=angle,
                vertical=vertical,
                horizontal=horizontal,
                interpolation=augmentation.rotation.interpolation,
                padding_mode=padding_mode,
            )
        elif translate:
            combined = _apply_translation(
                combined,
                vertical=vertical,
                horizontal=horizontal,
                interpolation=augmentation.rotation.interpolation,
                padding_mode=padding_mode,
            )
        lq_batch, gt_batch = combined.chunk(2, dim=0)

    return DistillBatch(
        lq_rgb=lq_batch,
        gt_rgb=gt_batch,
        relative_path=raw.relative_path,
    )
