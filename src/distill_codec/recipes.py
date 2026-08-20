from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import freeze_module
from .contracts import ColorSpec, ContractError, DistillBatch, LatentSpec
from .color import rgb_to_yuv
from .losses import LPIPSLoss, channel_stat_loss, cosine_loss, edge_loss, latent_smooth_l1
from .metrics import psnr, ssim
from .latents import TeacherEncoderLatentProvider


RECIPE_NAMES = frozenset(
    {
        "wan_encoder_distill",
        "wan_decoder_distill",
        "wan_autoencoder_distill",
        "flashvsr_vae_encoder_distill",
        "flashvsr_lq_proj_distill",
        "flashvsr_decoder_unconditional_student",
        "flashvsr_decoder_conditional_student",
    }
)


@dataclass
class RecipeOutput:
    total_loss: Tensor
    losses: dict[str, Tensor]
    images: dict[str, Tensor] = field(default_factory=dict)
    metrics: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


DEFAULT_WEIGHTS = {
    "latent": 1.0,
    "cos": 0.1,
    "stat": 0.1,
    "compat": 0.1,
    "teacher": 1.0,
    "gt": 0.5,
    "edge": 0.1,
    "condition": 1.0,
    "condition_cos": 0.1,
    "condition_stat": 0.1,
    "lpips": 0.0,
}


REQUIRED_COMPONENTS = {
    "wan_encoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder"},
    "flashvsr_vae_encoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder"},
    "wan_decoder_distill": {"teacher_decoder", "student_decoder"},
    "wan_autoencoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder", "student_decoder"},
    "flashvsr_lq_proj_distill": {"student_condition_encoder", "teacher_condition_encoder"},
    "flashvsr_decoder_unconditional_student": {"tc_decoder", "student_decoder"},
    "flashvsr_decoder_conditional_student": {
        "tc_decoder",
        "conditional_student_decoder",
    },
}


TEACHER_COMPONENTS = {
    "teacher_encoder",
    "teacher_decoder",
    "teacher_condition_encoder",
    "tc_decoder",
    "latent_provider",
}


class DistillationRecipe(nn.Module):
    def __init__(
        self,
        name: str,
        components: Mapping[str, nn.Module],
        weights: Mapping[str, float] | None = None,
        *,
        source: str = "gt",
        compatibility_every: int = 1,
    ) -> None:
        super().__init__()
        if name not in RECIPE_NAMES:
            raise ValueError(f"unknown recipe {name!r}; available recipes={sorted(RECIPE_NAMES)}")
        if source not in {"lq", "gt"}:
            raise ContractError(f"recipe source must be lq or gt, got {source!r}")
        missing = REQUIRED_COMPONENTS[name] - set(components)
        if missing:
            raise ContractError(f"recipe {name!r} is missing components: {sorted(missing)}")
        self.name = name
        self.weights = {**DEFAULT_WEIGHTS, **dict(weights or {})}
        if compatibility_every <= 0:
            raise ContractError("compatibility_every must be positive")
        self.compatibility_every = compatibility_every
        selected = {key: components[key] for key in sorted(REQUIRED_COMPONENTS[name])}
        if name in {
            "wan_decoder_distill",
            "flashvsr_decoder_unconditional_student",
            "flashvsr_decoder_conditional_student",
        }:
            if "latent_provider" in components:
                selected["latent_provider"] = components["latent_provider"]
            elif "teacher_encoder" in components:
                teacher_encoder = components["teacher_encoder"]
                latent_spec = getattr(teacher_encoder, "latent_spec", None)
                if not isinstance(latent_spec, LatentSpec):
                    raise ContractError("teacher_encoder must expose a LatentSpec as latent_spec")
                selected["latent_provider"] = TeacherEncoderLatentProvider(
                    teacher_encoder,
                    source="gt",
                    latent_spec=latent_spec,
                )
            else:
                raise ContractError(f"recipe {name!r} requires latent_provider or teacher_encoder")
        if self.weights.get("lpips", 0.0) > 0 and name in {
            "wan_decoder_distill",
            "wan_autoencoder_distill",
            "flashvsr_decoder_unconditional_student",
            "flashvsr_decoder_conditional_student",
        }:
            perceptual_loss = components.get("perceptual_loss")
            selected["perceptual_loss"] = freeze_module(
                perceptual_loss if perceptual_loss is not None else LPIPSLoss()
            )
        self.components = nn.ModuleDict(selected)
        self.source = source
        self._freeze_teachers()

    def _freeze_teachers(self) -> None:
        for name in TEACHER_COMPONENTS & set(self.components):
            freeze_module(self.components[name])

    def train(self, mode: bool = True) -> "DistillationRecipe":
        super().train(mode)
        self._freeze_teachers()
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, batch: DistillBatch, *, global_step: int | None = None) -> RecipeOutput:
        if self.name in {"wan_encoder_distill", "flashvsr_vae_encoder_distill"}:
            return self._encoder_distill(batch, global_step=global_step)
        if self.name == "wan_decoder_distill":
            return self._decoder_distill(batch, flashvsr=False, conditional=False)
        if self.name == "wan_autoencoder_distill":
            return self._autoencoder_distill(batch, global_step=global_step)
        if self.name == "flashvsr_lq_proj_distill":
            return self._condition_distill(batch)
        if self.name == "flashvsr_decoder_unconditional_student":
            return self._decoder_distill(batch, flashvsr=True, conditional=False)
        return self._decoder_distill(batch, flashvsr=True, conditional=True)

    def _weighted(self, losses: Mapping[str, Tensor]) -> Tensor:
        total = next(iter(losses.values())).new_zeros(())
        for name, loss in losses.items():
            total = total + self.weights.get(name, 1.0) * loss
        return total

    def _should_compute_compatibility(self, global_step: int | None) -> bool:
        if self.weights.get("compat", 0.0) <= 0:
            return False
        if not self.training:
            return True
        return global_step is None or global_step % self.compatibility_every == 0

    def _encoder_distill(
        self,
        batch: DistillBatch,
        *,
        global_step: int | None,
    ) -> RecipeOutput:
        source_rgb = batch.gt_rgb if self.source == "gt" else batch.lq_rgb
        with torch.no_grad():
            teacher_latent = self.components["teacher_encoder"](source_rgb)
        student_latent = self.components["student_encoder"](source_rgb)
        losses = {
            "latent": latent_smooth_l1(student_latent, teacher_latent),
            "cos": cosine_loss(student_latent, teacher_latent),
            "stat": channel_stat_loss(student_latent, teacher_latent),
        }
        reduce_dims = (0, *range(2, student_latent.ndim))
        student_mean = student_latent.detach().mean(dim=reduce_dims)
        teacher_mean = teacher_latent.mean(dim=reduce_dims)
        student_std = student_latent.detach().std(dim=reduce_dims, unbiased=False)
        teacher_std = teacher_latent.std(dim=reduce_dims, unbiased=False)
        images: dict[str, Tensor] = {}
        metrics = {
            "latent_mae": F.l1_loss(student_latent.detach(), teacher_latent),
            "latent_rmse": F.mse_loss(student_latent.detach(), teacher_latent).sqrt(),
            "latent_cosine": 1.0 - cosine_loss(student_latent.detach(), teacher_latent),
            "channel_mean_mae": F.l1_loss(student_mean, teacher_mean),
            "channel_std_mae": F.l1_loss(student_std, teacher_std),
        }
        if self._should_compute_compatibility(global_step):
            with torch.no_grad():
                teacher_rgb = self.components["teacher_decoder"](teacher_latent)
            compatibility_rgb = self.components["teacher_decoder"](student_latent)
            losses["compat"] = F.l1_loss(compatibility_rgb, source_rgb)
            metrics.update(
                {
                    "compat_psnr": psnr(compatibility_rgb.detach(), source_rgb),
                    "compat_ssim": ssim(compatibility_rgb.detach(), source_rgb),
                }
            )
            images = {
                "lq": batch.lq_rgb,
                "gt": batch.gt_rgb,
                "teacher": teacher_rgb,
                "student": compatibility_rgb,
            }
        return RecipeOutput(total_loss=self._weighted(losses), losses=losses, images=images, metrics=metrics)

    def _decoder_distill(
        self,
        batch: DistillBatch,
        *,
        flashvsr: bool,
        conditional: bool,
    ) -> RecipeOutput:
        student_condition_rgb = batch.lq_rgb
        teacher_condition_rgb = student_condition_rgb
        if flashvsr and teacher_condition_rgb.shape[-2:] != batch.gt_rgb.shape[-2:]:
            teacher_condition_rgb = F.interpolate(
                teacher_condition_rgb,
                size=batch.gt_rgb.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
        with torch.no_grad():
            latent = self.components["latent_provider"](batch)
            if flashvsr:
                teacher_rgb = self.components["tc_decoder"](latent, teacher_condition_rgb)
            else:
                teacher_rgb = self.components["teacher_decoder"](latent)
        if conditional:
            student_rgb = self.components["conditional_student_decoder"](
                latent,
                student_condition_rgb,
            )
        else:
            student_rgb = self.components["student_decoder"](latent)
        losses = {
            "teacher": F.l1_loss(student_rgb, teacher_rgb),
            "gt": F.l1_loss(student_rgb, batch.gt_rgb),
            "edge": edge_loss(student_rgb, batch.gt_rgb),
        }
        if "perceptual_loss" in self.components:
            losses["lpips"] = self.components["perceptual_loss"](student_rgb, batch.gt_rgb)
        student_component = self.components[
            "conditional_student_decoder" if conditional else "student_decoder"
        ]
        color_spec = getattr(student_component, "color_spec", None)
        if not isinstance(color_spec, ColorSpec):
            raise ContractError("student decoder must expose its ColorSpec as color_spec")
        student_yuv = rgb_to_yuv(student_rgb.detach().clamp(0, 1), color_spec)
        gt_yuv = rgb_to_yuv(batch.gt_rgb, color_spec)
        return RecipeOutput(
            total_loss=self._weighted(losses),
            losses=losses,
            images={"lq": batch.lq_rgb, "gt": batch.gt_rgb, "teacher": teacher_rgb, "student": student_rgb},
            metrics={
                "psnr_vs_teacher": psnr(student_rgb.detach(), teacher_rgb),
                "psnr_vs_gt": psnr(student_rgb.detach(), batch.gt_rgb),
                "ssim_vs_teacher": ssim(student_rgb.detach(), teacher_rgb),
                "ssim_vs_gt": ssim(student_rgb.detach(), batch.gt_rgb),
                "rgb_mae_vs_teacher": F.l1_loss(student_rgb.detach(), teacher_rgb),
                "rgb_mae_vs_gt": F.l1_loss(student_rgb.detach(), batch.gt_rgb),
                "y_mae_vs_gt": F.l1_loss(student_yuv[:, 0], gt_yuv[:, 0]),
                "u_mae_vs_gt": F.l1_loss(student_yuv[:, 1], gt_yuv[:, 1]),
                "v_mae_vs_gt": F.l1_loss(student_yuv[:, 2], gt_yuv[:, 2]),
            },
            metadata={
                "student_accepts_condition": conditional,
                "condition_shape": list(teacher_condition_rgb.shape),
            },
        )

    def _autoencoder_distill(
        self,
        batch: DistillBatch,
        *,
        global_step: int | None,
    ) -> RecipeOutput:
        with torch.no_grad():
            teacher_latent = self.components["teacher_encoder"](batch.gt_rgb)
            teacher_rgb = self.components["teacher_decoder"](teacher_latent)
        student_latent = self.components["student_encoder"](batch.gt_rgb)
        student_rgb = self.components["student_decoder"](student_latent)
        losses = {
            "latent": latent_smooth_l1(student_latent, teacher_latent),
            "cos": cosine_loss(student_latent, teacher_latent),
            "stat": channel_stat_loss(student_latent, teacher_latent),
            "teacher": F.l1_loss(student_rgb, teacher_rgb),
            "gt": F.l1_loss(student_rgb, batch.gt_rgb),
            "edge": edge_loss(student_rgb, batch.gt_rgb),
        }
        if self._should_compute_compatibility(global_step):
            compatibility_rgb = self.components["teacher_decoder"](student_latent)
            losses["compat"] = F.l1_loss(compatibility_rgb, batch.gt_rgb)
        if "perceptual_loss" in self.components:
            losses["lpips"] = self.components["perceptual_loss"](student_rgb, batch.gt_rgb)
        return RecipeOutput(
            total_loss=self._weighted(losses),
            losses=losses,
            images={"lq": batch.lq_rgb, "gt": batch.gt_rgb, "teacher": teacher_rgb, "student": student_rgb},
            metrics={"psnr_vs_gt": psnr(student_rgb.detach(), batch.gt_rgb)},
        )

    def _condition_distill(self, batch: DistillBatch) -> RecipeOutput:
        condition_rgb = batch.lq_rgb
        if condition_rgb.shape[-2:] != batch.gt_rgb.shape[-2:]:
            condition_rgb = F.interpolate(
                condition_rgb,
                size=batch.gt_rgb.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
        with torch.no_grad():
            teacher = self.components["teacher_condition_encoder"](condition_rgb)
        student = self.components["student_condition_encoder"](condition_rgb)
        if set(student) != set(teacher):
            raise ContractError(
                f"condition keys differ: student={sorted(student)}, teacher={sorted(teacher)}"
            )
        element_losses = []
        cosine_losses = []
        stat_losses = []
        for name in sorted(teacher):
            if student[name].shape != teacher[name].shape:
                raise ContractError(
                    f"condition {name!r} shape mismatch: student={tuple(student[name].shape)}, "
                    f"teacher={tuple(teacher[name].shape)}"
                )
            element_losses.append(F.smooth_l1_loss(student[name], teacher[name]))
            cosine_losses.append(cosine_loss(student[name].transpose(1, 2), teacher[name].transpose(1, 2)))
            stat_losses.append(channel_stat_loss(student[name].transpose(1, 2), teacher[name].transpose(1, 2)))
        losses = {
            "condition": torch.stack(element_losses).sum(),
            "condition_cos": torch.stack(cosine_losses).sum(),
            "condition_stat": torch.stack(stat_losses).sum(),
        }
        return RecipeOutput(
            total_loss=self._weighted(losses),
            losses=losses,
            images={"lq": batch.lq_rgb, "gt": batch.gt_rgb, "teacher": batch.lq_rgb, "student": batch.lq_rgb},
            metadata={"condition_keys": sorted(student)},
        )


def build_recipe(
    name: str,
    components: Mapping[str, nn.Module],
    weights: Mapping[str, float] | None = None,
    *,
    source: str = "gt",
    compatibility_every: int = 1,
) -> DistillationRecipe:
    return DistillationRecipe(
        name,
        components,
        weights,
        source=source,
        compatibility_every=compatibility_every,
    )
