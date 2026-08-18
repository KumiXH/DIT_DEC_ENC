import pytest
import torch

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

    output = recipe(_batch())

    assert output.metadata["student_accepts_condition"] is False
    assert set(output.images) >= {"lq", "gt", "teacher", "student"}


def test_encoder_compatibility_loss_reaches_student_but_not_teacher_decoder():
    components = _components()
    recipe = build_recipe("wan_encoder_distill", components, weights={"compat": 1.0})

    recipe(_batch()).total_loss.backward()

    assert any(parameter.grad is not None for parameter in components["student_encoder"].parameters())
    assert all(parameter.grad is None for parameter in components["teacher_decoder"].parameters())


def test_unknown_recipe_lists_available_names():
    with pytest.raises(ValueError, match="available recipes"):
        build_recipe("not_a_recipe", _components())

