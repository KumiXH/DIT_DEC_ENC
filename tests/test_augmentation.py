import re
from pathlib import Path

import pytest
import torch
from PIL import Image

from distill_codec.augmentation import (
    collate_augmented_batch,
    paired_augmentation_from_config,
)
from distill_codec.config import load_config, preflight_config
from distill_codec.contracts import ContractError
from distill_codec.data import PairedImageDataset, collate_raw_paired_batch


def _write_pattern(path: Path, *, height: int, width: int, offset: int = 0) -> None:
    yy = torch.arange(height, dtype=torch.uint8).view(height, 1).expand(height, width)
    xx = torch.arange(width, dtype=torch.uint8).view(1, width).expand(height, width)
    array = torch.stack(
        ((xx + offset) % 255, (yy + offset) % 255, (xx + yy + offset) % 255), dim=-1
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.numpy(), mode="RGB").save(path)


def _augmentation_config(**overrides):
    augmentation = {
        "enabled": True,
        "shared_across_batch": True,
        "crop": {"enabled": True, "mode": "random"},
        "rotation": {
            "enabled": True,
            "mode": "continuous",
            "probability": 1.0,
            "degrees": [-5.0, 5.0],
            "interpolation": "bilinear",
            "padding_mode": "reflection",
        },
        "translation": {
            "enabled": True,
            "probability": 1.0,
            "max_fraction": [0.05, 0.05],
            "padding_mode": "reflection",
        },
    }
    augmentation.update(overrides)
    return {
        "data": {
            "lq_size": [8, 8],
            "gt_size": [8, 8],
            "augmentation": augmentation,
        },
        "run": {"seed": 7},
    }


def test_augmentation_config_parses_documented_defaults():
    spec = paired_augmentation_from_config(_augmentation_config())

    assert spec.enabled
    assert spec.crop.enabled
    assert spec.rotation.degrees == (-5.0, 5.0)
    assert spec.translation.max_fraction == (0.05, 0.05)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda values: values["data"].update({"gt_size": [16, 16]}),
            "lq_size.*gt_size",
        ),
        (
            lambda values: values["data"]["augmentation"].update(
                {"shared_across_batch": False}
            ),
            "shared_across_batch",
        ),
        (
            lambda values: values["data"]["augmentation"]["rotation"].update(
                {"probability": 1.1}
            ),
            "rotation.probability",
        ),
        (
            lambda values: values["data"]["augmentation"]["translation"].update(
                {"max_fraction": [0.05, 1.0]}
            ),
            "translation.max_fraction",
        ),
    ),
)
def test_augmentation_config_rejects_invalid_contracts(mutation, message):
    config = _augmentation_config()
    mutation(config)

    with pytest.raises(ContractError, match=message):
        paired_augmentation_from_config(config)


@pytest.mark.parametrize(
    "path",
    (
        ("data", "augmentation", "enabled"),
        ("data", "augmentation", "shared_across_batch"),
        ("data", "augmentation", "crop", "enabled"),
        ("data", "augmentation", "rotation", "enabled"),
        ("data", "augmentation", "translation", "enabled"),
    ),
)
def test_augmentation_config_rejects_non_boolean_flags(path):
    config = _augmentation_config()
    section = config
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = "false"

    with pytest.raises(ContractError, match=re.escape(".".join(path))):
        paired_augmentation_from_config(config)


@pytest.mark.parametrize("section_name", ("crop", "rotation", "translation"))
def test_augmentation_config_rejects_non_mapping_sections(section_name):
    config = _augmentation_config()
    config["data"]["augmentation"][section_name] = False

    with pytest.raises(
        ContractError,
        match=re.escape(f"data.augmentation.{section_name}"),
    ):
        paired_augmentation_from_config(config)


def test_augmentation_config_accepts_null_latent_provider():
    config = _augmentation_config()
    config["latent_provider"] = None

    assert paired_augmentation_from_config(config).enabled


def test_dataset_rejects_images_smaller_than_crop_target(tmp_path):
    _write_pattern(tmp_path / "lq" / "a.png", height=7, width=8)
    _write_pattern(tmp_path / "gt" / "a.png", height=7, width=8)
    spec = paired_augmentation_from_config(_augmentation_config())

    with pytest.raises(ContractError, match="a.png.*at least.*8.*8"):
        PairedImageDataset(
            tmp_path / "lq",
            tmp_path / "gt",
            lq_size=(8, 8),
            gt_size=(8, 8),
            augmentation=spec,
        )


def test_random_crop_keeps_lq_gt_aligned_and_batch_shared(tmp_path):
    for index, shape in enumerate(((12, 14), (16, 18))):
        height, width = shape
        relative = f"{index}.png"
        _write_pattern(
            tmp_path / "lq" / relative, height=height, width=width, offset=index
        )
        _write_pattern(
            tmp_path / "gt" / relative, height=height, width=width, offset=index
        )
    spec = paired_augmentation_from_config(
        _augmentation_config(
            rotation={"enabled": False},
            translation={"enabled": False},
        )
    )
    dataset = PairedImageDataset(
        tmp_path / "lq",
        tmp_path / "gt",
        lq_size=(8, 8),
        gt_size=(8, 8),
        augmentation=spec,
    )
    raw = collate_raw_paired_batch([dataset[0], dataset[1]])

    batch = collate_augmented_batch(
        raw, spec, phase="train", seed=7, global_step=3, micro_step=0
    )

    assert batch.lq_rgb.shape == (2, 3, 8, 8)
    assert torch.equal(batch.lq_rgb, batch.gt_rgb)


def test_continuous_affine_is_reproducible_shared_and_shape_preserving(tmp_path):
    for index in range(2):
        relative = f"{index}.png"
        _write_pattern(tmp_path / "lq" / relative, height=12, width=12, offset=index)
        _write_pattern(tmp_path / "gt" / relative, height=12, width=12, offset=index)
    spec = paired_augmentation_from_config(_augmentation_config())
    dataset = PairedImageDataset(
        tmp_path / "lq",
        tmp_path / "gt",
        lq_size=(8, 8),
        gt_size=(8, 8),
        augmentation=spec,
    )
    raw = collate_raw_paired_batch([dataset[0], dataset[1]])

    first = collate_augmented_batch(
        raw, spec, phase="train", seed=11, global_step=2, micro_step=0
    )
    repeated = collate_augmented_batch(
        raw, spec, phase="train", seed=11, global_step=2, micro_step=0
    )
    next_step = collate_augmented_batch(
        raw, spec, phase="train", seed=11, global_step=3, micro_step=0
    )

    assert first.lq_rgb.shape == (2, 3, 8, 8)
    assert torch.equal(first.lq_rgb, first.gt_rgb)
    assert torch.equal(first.lq_rgb, repeated.lq_rgb)
    assert not torch.equal(first.lq_rgb, next_step.lq_rgb)


def test_validation_uses_center_crop_without_rotation_or_translation(tmp_path):
    _write_pattern(tmp_path / "lq" / "a.png", height=12, width=12)
    _write_pattern(tmp_path / "gt" / "a.png", height=12, width=12)
    spec = paired_augmentation_from_config(_augmentation_config())
    dataset = PairedImageDataset(
        tmp_path / "lq",
        tmp_path / "gt",
        lq_size=(8, 8),
        gt_size=(8, 8),
        augmentation=spec,
    )
    raw = collate_raw_paired_batch([dataset[0]])

    validation = collate_augmented_batch(
        raw, spec, phase="validation", seed=1, global_step=1
    )
    expected = dataset[0]["lq_rgb"][:, 2:10, 2:10].unsqueeze(0)

    assert torch.equal(validation.lq_rgb, expected)
    assert torch.equal(validation.lq_rgb, validation.gt_rgb)


def test_config_preflight_rejects_augmentation_with_cached_latents(tmp_path):
    config = load_config("configs/smoke/wan_decoder.yaml")
    config["data"]["augmentation"] = _augmentation_config()["data"]["augmentation"]
    config["latent_provider"] = {"type": "cached", "root": str(tmp_path / "latents")}

    with pytest.raises(ContractError, match="augmentation.*cached"):
        preflight_config(config)
