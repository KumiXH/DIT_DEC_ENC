import pytest
import torch
from torch import nn

from distill_codec.adapters import ConditionEncoderAdapter, DecoderAdapter, EncoderAdapter, freeze_module
from distill_codec.contracts import ColorSpec, DistillBatch, LatentSpec
from distill_codec.models.mock import (
    MockConditionalStudentDecoder,
    MockLQProjIn,
    MockStudentDecoder,
    MockStudentEncoder,
    MockTCDecoder,
    MockWanDecoder,
    MockWanEncoder,
)
from distill_codec.recipes import RECIPE_NAMES, build_recipe


LATENT_SPEC = LatentSpec("mock_wan", 16, "BCHW", 8, 1, "mock_wan")


def _batch():
    return DistillBatch(
        lq_rgb=torch.rand(2, 3, 64, 64),
        gt_rgb=torch.rand(2, 3, 64, 64),
        relative_path=("a.png", "b.png"),
    )


def _components():
    return {
        "student_encoder": EncoderAdapter(
            MockStudentEncoder(), latent_spec=LATENT_SPEC, input_mode="packed_6ch", color_spec=ColorSpec()
        ),
        "teacher_encoder": EncoderAdapter(
            freeze_module(MockWanEncoder()), latent_spec=LATENT_SPEC, input_mode="rgb_video", temporal_frames=3
        ),
        "student_decoder": DecoderAdapter(
            MockStudentDecoder(), output_mode="sparse_yuv", color_spec=ColorSpec()
        ),
        "conditional_student_decoder": DecoderAdapter(
            MockConditionalStudentDecoder(),
            output_mode="sparse_yuv",
            color_spec=ColorSpec(),
            accepts_condition=True,
        ),
        "teacher_decoder": DecoderAdapter(freeze_module(MockWanDecoder()), output_mode="rgb"),
        "tc_decoder": DecoderAdapter(
            freeze_module(MockTCDecoder()), output_mode="rgb", accepts_condition=True
        ),
        "student_condition_encoder": ConditionEncoderAdapter(MockLQProjIn(feature_dim=32, seed=702), temporal_frames=3),
        "teacher_condition_encoder": ConditionEncoderAdapter(
            freeze_module(MockLQProjIn(feature_dim=32, seed=701)), temporal_frames=3
        ),
    }


@pytest.mark.parametrize("recipe_name", sorted(RECIPE_NAMES))
def test_each_recipe_runs_one_backward_step(recipe_name):
    recipe = build_recipe(recipe_name, _components())

    output = recipe(_batch())
    output.total_loss.backward()

    gradients = [parameter.grad for parameter in recipe.trainable_parameters()]
    assert torch.isfinite(output.total_loss)
    assert output.losses
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)


def test_unconditional_flashvsr_recipe_does_not_pass_condition_to_student():
    components = _components()
    recipe = build_recipe("flashvsr_decoder_unconditional_student", components)
    batch = DistillBatch(
        lq_rgb=torch.rand(2, 3, 32, 32),
        gt_rgb=torch.rand(2, 3, 64, 64),
        relative_path=("a.png", "b.png"),
    )

    output = recipe(batch)

    assert output.metadata["student_accepts_condition"] is False
    assert set(output.images) >= {"lq", "gt", "teacher", "student"}
    assert set(output.metrics) >= {"y_mae_vs_gt", "u_mae_vs_gt", "v_mae_vs_gt"}
    assert output.metadata["condition_shape"] == [2, 3, 64, 64]


def test_encoder_compatibility_loss_reaches_student_but_not_teacher_decoder():
    components = _components()
    recipe = build_recipe("wan_encoder_distill", components, weights={"compat": 1.0})

    output = recipe(_batch())
    output.total_loss.backward()

    assert any(parameter.grad is not None for parameter in components["student_encoder"].parameters())
    assert all(parameter.grad is None for parameter in components["teacher_decoder"].parameters())
    assert set(output.metrics) >= {
        "latent_mae",
        "latent_rmse",
        "latent_cosine",
        "channel_mean_mae",
        "channel_std_mae",
        "compat_psnr",
        "compat_ssim",
    }


def test_condition_recipe_reports_element_cosine_and_statistics_losses():
    recipe = build_recipe("flashvsr_lq_proj_distill", _components())

    output = recipe(_batch())

    assert set(output.losses) == {"condition", "condition_cos", "condition_stat"}


def test_lpips_disabled_does_not_construct_optional_dependency(monkeypatch):
    def fail_if_constructed():
        raise AssertionError("LPIPS must stay lazy while its weight is zero")

    monkeypatch.setattr("distill_codec.recipes.LPIPSLoss", fail_if_constructed)

    recipe = build_recipe("wan_decoder_distill", _components(), weights={"lpips": 0.0})

    assert "perceptual_loss" not in recipe.components


def test_injected_perceptual_loss_is_frozen_but_backpropagates_to_student():
    class FakePerceptualLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, student, target):
            return self.scale * (student - target).abs().mean()

    components = _components()
    components["perceptual_loss"] = FakePerceptualLoss()
    recipe = build_recipe("wan_decoder_distill", components, weights={"lpips": 1.0})

    output = recipe(_batch())
    output.total_loss.backward()

    assert "lpips" in output.losses
    assert all(not parameter.requires_grad for parameter in recipe.components["perceptual_loss"].parameters())
    assert all(parameter.grad is None for parameter in recipe.components["perceptual_loss"].parameters())
    assert any(parameter.grad is not None for parameter in components["student_decoder"].parameters())


def test_unknown_recipe_lists_available_names():
    with pytest.raises(ValueError, match="available recipes"):
        build_recipe("not_a_recipe", _components())
