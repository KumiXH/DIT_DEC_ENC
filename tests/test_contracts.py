import pytest
import torch

from distill_codec.contracts import (
    ColorSpec,
    ConditionSpec,
    ContractError,
    DistillBatch,
    LatentSpec,
)


def test_latent_spec_round_trip_and_tensor_validation():
    spec = LatentSpec(
        family="wan_vae_v2",
        channels=16,
        layout="BCHW",
        spatial_downsample=8,
        temporal_downsample=1,
        normalization="wan_vae",
    )

    restored = LatentSpec.from_dict(spec.to_dict())
    restored.validate_tensor(torch.zeros(2, 16, 32, 32), image_size=(256, 256))

    assert restored == spec


def test_latent_spec_rejects_semantically_incompatible_latent():
    expected = LatentSpec("wan", 16, "BCHW", 8, 1, "wan_norm")
    actual = LatentSpec("custom", 16, "BCHW", 8, 1, "identity")

    with pytest.raises(ContractError, match="family.*normalization"):
        expected.assert_compatible(actual)


def test_latent_spec_reports_shape_mismatch():
    spec = LatentSpec("wan", 16, "BCHW", 8, 1, "wan_norm")

    with pytest.raises(ContractError, match="expected channels=16.*actual shape"):
        spec.validate_tensor(torch.zeros(1, 8, 32, 32), image_size=(256, 256))


def test_condition_and_color_specs_validate_enums():
    condition = ConditionSpec(
        family="flashvsr_lq_proj",
        layout="BNC",
        feature_dim=1536,
        source="lq",
        consumer="dit",
    )
    color = ColorSpec(
        matrix="bt709",
        range="full",
        packed_order="Y00Y01Y10Y11UV",
        chroma_location="top_left",
        chroma_upsample="nearest",
    )

    assert ConditionSpec.from_dict(condition.to_dict()) == condition
    assert ColorSpec.from_dict(color.to_dict()) == color

    with pytest.raises(ContractError, match="consumer"):
        ConditionSpec("x", "BNC", 4, "lq", "vae")
    with pytest.raises(ContractError, match="matrix"):
        ColorSpec("unknown", "full", "Y00Y01Y10Y11UV", "top_left", "nearest")


def test_distill_batch_requires_matching_rgb_batches_and_paths():
    batch = DistillBatch(
        lq_rgb=torch.zeros(2, 3, 32, 32),
        gt_rgb=torch.ones(2, 3, 64, 64),
        relative_path=("a.png", "nested/b.png"),
    )

    assert batch.batch_size == 2
    moved = batch.to("cpu")
    assert moved.relative_path == batch.relative_path

    with pytest.raises(ContractError, match="relative_path"):
        DistillBatch(
            lq_rgb=torch.zeros(2, 3, 32, 32),
            gt_rgb=torch.zeros(2, 3, 64, 64),
            relative_path=("a.png",),
        )

