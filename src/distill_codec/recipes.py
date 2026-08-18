from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import ContractError, DistillBatch
from .losses import channel_stat_loss, cosine_loss, edge_loss, latent_smooth_l1
from .metrics import psnr, ssim


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
}


REQUIRED_COMPONENTS = {
    "wan_encoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder"},
    "flashvsr_vae_encoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder"},
    "wan_decoder_distill": {"teacher_encoder", "teacher_decoder", "student_decoder"},
    "wan_autoencoder_distill": {"student_encoder", "teacher_encoder", "teacher_decoder", "student_decoder"},
    "flashvsr_lq_proj_distill": {"student_condition_encoder", "teacher_condition_encoder"},
    "flashvsr_decoder_unconditional_student": {"teacher_encoder", "tc_decoder", "student_decoder"},
    "flashvsr_decoder_conditional_student": {
        "teacher_encoder",
        "tc_decoder",
        "conditional_student_decoder",
    },
}


class DistillationRecipe(nn.Module):
    def __init__(
        self,
        name: str,
        components: Mapping[str, nn.Module],
        weights: Mapping[str, float] | None = None,
        *,
        source: str = "gt",
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
        self.components = nn.ModuleDict({key: components[key] for key in REQUIRED_COMPONENTS[name]})
        self.weights = {**DEFAULT_WEIGHTS, **dict(weights or {})}
        self.source = source

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, batch: DistillBatch) -> RecipeOutput:
        if self.name in {"wan_encoder_distill", "flashvsr_vae_encoder_distill"}:
            return self._encoder_distill(batch)
        if self.name == "wan_decoder_distill":
            return self._decoder_distill(batch, flashvsr=False, conditional=False)
        if self.name == "wan_autoencoder_distill":
            return self._autoencoder_distill(batch)
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

    def _encoder_distill(self, batch: DistillBatch) -> RecipeOutput:
        source_rgb = batch.gt_rgb if self.source == "gt" else batch.lq_rgb
        with torch.no_grad():
            teacher_latent = self.components["teacher_encoder"](source_rgb)
            teacher_rgb = self.components["teacher_decoder"](teacher_latent)
        student_latent = self.components["student_encoder"](source_rgb)
        compatibility_rgb = self.components["teacher_decoder"](student_latent)
        losses = {
            "latent": latent_smooth_l1(student_latent, teacher_latent),
            "cos": cosine_loss(student_latent, teacher_latent),
            "stat": channel_stat_loss(student_latent, teacher_latent),
            "compat": F.l1_loss(compatibility_rgb, source_rgb),
        }
        return RecipeOutput(
            total_loss=self._weighted(losses),
            losses=losses,
            images={
                "lq": batch.lq_rgb,
                "gt": batch.gt_rgb,
                "teacher": teacher_rgb,
                "student": compatibility_rgb,
            },
            metrics={
                "latent_mae": F.l1_loss(student_latent.detach(), teacher_latent),
                "compat_psnr": psnr(compatibility_rgb.detach(), source_rgb),
            },
        )

    def _decoder_distill(
        self,
        batch: DistillBatch,
        *,
        flashvsr: bool,
        conditional: bool,
    ) -> RecipeOutput:
        with torch.no_grad():
            latent = self.components["teacher_encoder"](batch.gt_rgb)
            if flashvsr:
                teacher_rgb = self.components["tc_decoder"](latent, batch.lq_rgb)
            else:
                teacher_rgb = self.components["teacher_decoder"](latent)
        if conditional:
            student_rgb = self.components["conditional_student_decoder"](latent, batch.lq_rgb)
        else:
            student_rgb = self.components["student_decoder"](latent)
        losses = {
            "teacher": F.l1_loss(student_rgb, teacher_rgb),
            "gt": F.l1_loss(student_rgb, batch.gt_rgb),
            "edge": edge_loss(student_rgb, batch.gt_rgb),
        }
        return RecipeOutput(
            total_loss=self._weighted(losses),
            losses=losses,
            images={"lq": batch.lq_rgb, "gt": batch.gt_rgb, "teacher": teacher_rgb, "student": student_rgb},
            metrics={
                "psnr_vs_teacher": psnr(student_rgb.detach(), teacher_rgb),
                "psnr_vs_gt": psnr(student_rgb.detach(), batch.gt_rgb),
                "ssim_vs_teacher": ssim(student_rgb.detach(), teacher_rgb),
                "ssim_vs_gt": ssim(student_rgb.detach(), batch.gt_rgb),
            },
            metadata={"student_accepts_condition": conditional},
        )

    def _autoencoder_distill(self, batch: DistillBatch) -> RecipeOutput:
        with torch.no_grad():
            teacher_latent = self.components["teacher_encoder"](batch.gt_rgb)
            teacher_rgb = self.components["teacher_decoder"](teacher_latent)
        student_latent = self.components["student_encoder"](batch.gt_rgb)
        student_rgb = self.components["student_decoder"](student_latent)
        compatibility_rgb = self.components["teacher_decoder"](student_latent)
        losses = {
            "latent": latent_smooth_l1(student_latent, teacher_latent),
            "cos": cosine_loss(student_latent, teacher_latent),
            "stat": channel_stat_loss(student_latent, teacher_latent),
            "compat": F.l1_loss(compatibility_rgb, batch.gt_rgb),
            "teacher": F.l1_loss(student_rgb, teacher_rgb),
            "gt": F.l1_loss(student_rgb, batch.gt_rgb),
            "edge": edge_loss(student_rgb, batch.gt_rgb),
        }
        return RecipeOutput(
            total_loss=self._weighted(losses),
            losses=losses,
            images={"lq": batch.lq_rgb, "gt": batch.gt_rgb, "teacher": teacher_rgb, "student": student_rgb},
            metrics={"psnr_vs_gt": psnr(student_rgb.detach(), batch.gt_rgb)},
        )

    def _condition_distill(self, batch: DistillBatch) -> RecipeOutput:
        with torch.no_grad():
            teacher = self.components["teacher_condition_encoder"](batch.lq_rgb)
        student = self.components["student_condition_encoder"](batch.lq_rgb)
        if set(student) != set(teacher):
            raise ContractError(
                f"condition keys differ: student={sorted(student)}, teacher={sorted(teacher)}"
            )
        per_tensor = []
        for name in sorted(teacher):
            if student[name].shape != teacher[name].shape:
                raise ContractError(
                    f"condition {name!r} shape mismatch: student={tuple(student[name].shape)}, "
                    f"teacher={tuple(teacher[name].shape)}"
                )
            per_tensor.append(F.smooth_l1_loss(student[name], teacher[name]))
            per_tensor.append(0.1 * cosine_loss(student[name].transpose(1, 2), teacher[name].transpose(1, 2)))
        condition_loss = torch.stack(per_tensor).sum()
        losses = {"condition": condition_loss}
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
) -> DistillationRecipe:
    return DistillationRecipe(name, components, weights, source=source)

