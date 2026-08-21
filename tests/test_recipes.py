import pytest
import torch
from torch import nn

from distill_codec.adapters import ConditionEncoderAdapter, DecoderAdapter, EncoderAdapter, freeze_module
from distill_codec.contracts import ColorSpec, DistillBatch, LatentSpec
from distill_codec.color import rgb_to_yuv
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


def test_conditional_flashvsr_teacher_gets_aligned_lq_while_student_gets_raw_lq():
    class RecordingTeacherDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.received_condition = None

        def forward(self, latent, condition):
            self.received_condition = condition
            return torch.nn.functional.interpolate(
                condition,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )

    class RecordingStudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.received_condition = None

        def forward(self, latent, condition):
            self.received_condition = condition
            return torch.nn.functional.interpolate(
                condition,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            ) * self.scale

    batch = DistillBatch(
        lq_rgb=torch.rand(2, 3, 32, 32),
        gt_rgb=torch.rand(2, 3, 64, 64),
        relative_path=("a.png", "b.png"),
    )
    components = _components()
    teacher = RecordingTeacherDecoder()
    student = RecordingStudentDecoder()
    components["tc_decoder"] = DecoderAdapter(
        freeze_module(teacher),
        output_mode="rgb",
        accepts_condition=True,
    )
    components["conditional_student_decoder"] = DecoderAdapter(
        student,
        output_mode="rgb",
        accepts_condition=True,
    )
    recipe = build_recipe("flashvsr_decoder_conditional_student", components)

    output = recipe(batch)

    assert teacher.received_condition is not batch.lq_rgb
    assert teacher.received_condition.shape[-2:] == (64, 64)
    assert student.received_condition is batch.lq_rgb
    assert student.received_condition.shape[-2:] == (32, 32)
    assert output.metadata["condition_shape"] == [2, 3, 64, 64]


def test_conditional_flashvsr_can_use_dataset_gt_as_offline_teacher_target():
    class FixedLatentProvider(nn.Module):
        def forward(self, batch):
            return torch.zeros(batch.batch_size, 16, 8, 8, device=batch.lq_rgb.device)

    class DatasetGTTeacherTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            return batch.gt_rgb

    batch = _batch()
    target_provider = DatasetGTTeacherTarget()
    components = {
        "conditional_student_decoder": DecoderAdapter(
            MockConditionalStudentDecoder(),
            output_mode="sparse_yuv",
            color_spec=ColorSpec(),
            accepts_condition=True,
        ),
        "latent_provider": FixedLatentProvider(),
        "teacher_target_provider": target_provider,
    }
    recipe = build_recipe("flashvsr_decoder_conditional_student", components)

    output = recipe(batch)

    assert target_provider.calls == 1
    assert output.images["teacher"] is batch.gt_rgb
    assert output.metadata["teacher_target_source"] == "dataset_gt"


def test_offline_teacher_targets_are_rejected_for_non_flashvsr_recipes():
    components = _components()
    components["teacher_target_provider"] = nn.Identity()

    with pytest.raises(
        ValueError,
        match="teacher_target_provider.*FlashVSR decoder",
    ):
        build_recipe("wan_decoder_distill", components)


def test_private_v0_conditional_decoder_uses_latent_target_size_with_raw_lq():
    from private_codec.factories import create_conditional_decoder

    batch = DistillBatch(
        lq_rgb=torch.rand(2, 3, 32, 32),
        gt_rgb=torch.rand(2, 3, 64, 64),
        relative_path=("a.png", "b.png"),
    )
    components = _components()
    components["conditional_student_decoder"] = DecoderAdapter(
        create_conditional_decoder(
            builder="private_codec.versions.v0.entrypoints:build_decoder",
            runner="private_codec.versions.v0.entrypoints:run_decoder",
            teacher_reference={"role": "conditional_decoder"},
        ),
        output_mode="rgb",
        accepts_condition=True,
    )
    recipe = build_recipe("flashvsr_decoder_conditional_student", components)

    output = recipe(batch)
    output.total_loss.backward()

    assert output.images["student"].shape == batch.gt_rgb.shape
    assert any(
        parameter.grad is not None
        for parameter in components["conditional_student_decoder"].parameters()
    )


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


def test_encoder_compatibility_loss_can_be_disabled_without_running_teacher_decoder():
    class FailIfCalled(nn.Module):
        def forward(self, latent):
            raise AssertionError("teacher decoder should not run when compatibility loss is disabled")

    components = _components()
    components["teacher_decoder"] = freeze_module(FailIfCalled())
    recipe = build_recipe("wan_encoder_distill", components, weights={"compat": 0.0})

    output = recipe(_batch(), global_step=1)
    recipe.eval()
    validation_output = recipe(_batch(), global_step=2)

    assert "compat" not in output.losses
    assert "compat_psnr" not in output.metrics
    assert "compat" not in validation_output.losses
    assert "compat_psnr" not in validation_output.metrics


def test_encoder_compatibility_loss_respects_training_interval_but_runs_for_validation():
    components = _components()
    recipe = build_recipe(
        "wan_encoder_distill",
        components,
        weights={"compat": 1.0},
        compatibility_every=3,
    )

    recipe.train()
    skipped = recipe(_batch(), global_step=2)
    computed = recipe(_batch(), global_step=3)
    recipe.eval()
    validated = recipe(_batch(), global_step=4)

    assert "compat" not in skipped.losses
    assert "compat" in computed.losses
    assert "compat" in validated.losses
    assert set(validated.images) >= {"lq", "gt", "teacher", "student"}


def test_condition_recipe_reports_element_cosine_and_statistics_losses():
    recipe = build_recipe("flashvsr_lq_proj_distill", _components())

    output = recipe(_batch())

    assert set(output.losses) == {"condition", "condition_cos", "condition_stat"}


@pytest.mark.parametrize(
    "recipe_name",
    (
        "wan_decoder_distill",
        "flashvsr_decoder_unconditional_student",
        "flashvsr_decoder_conditional_student",
    ),
)
def test_decoder_recipes_report_explicit_rgb_mae(recipe_name):
    recipe = build_recipe(recipe_name, _components())

    output = recipe(_batch())

    assert torch.isfinite(output.metrics["rgb_mae_vs_teacher"])
    assert torch.isfinite(output.metrics["rgb_mae_vs_gt"])


def test_decoder_yuv_metrics_use_the_student_decoder_color_contract():
    color_spec = ColorSpec(matrix="bt601", range="limited", chroma_upsample="bilinear")
    components = _components()
    components["student_decoder"] = DecoderAdapter(
        MockStudentDecoder(),
        output_mode="sparse_yuv",
        color_spec=color_spec,
    )
    recipe = build_recipe("wan_decoder_distill", components)

    output = recipe(_batch())
    student_yuv = rgb_to_yuv(output.images["student"].detach().clamp(0, 1), color_spec)
    gt_yuv = rgb_to_yuv(output.images["gt"], color_spec)

    assert torch.allclose(
        output.metrics["y_mae_vs_gt"],
        torch.nn.functional.l1_loss(student_yuv[:, 0], gt_yuv[:, 0]),
    )
    assert torch.allclose(
        output.metrics["u_mae_vs_gt"],
        torch.nn.functional.l1_loss(student_yuv[:, 1], gt_yuv[:, 1]),
    )
    assert torch.allclose(
        output.metrics["v_mae_vs_gt"],
        torch.nn.functional.l1_loss(student_yuv[:, 2], gt_yuv[:, 2]),
    )


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
