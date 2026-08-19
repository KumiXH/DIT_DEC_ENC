from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from PIL import Image
from torch import Tensor


def psnr(prediction: Tensor, target: Tensor, max_value: float = 1.0) -> Tensor:
    mse = torch.mean((prediction - target) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"), device=prediction.device)
    return 10.0 * torch.log10(torch.tensor(max_value**2, device=prediction.device) / mse)


def ssim(prediction: Tensor, target: Tensor, max_value: float = 1.0) -> Tensor:
    dims = tuple(range(1, prediction.ndim))
    mean_x = prediction.mean(dim=dims)
    mean_y = target.mean(dim=dims)
    var_x = prediction.var(dim=dims, unbiased=False)
    var_y = target.var(dim=dims, unbiased=False)
    covariance = ((prediction - mean_x.view(-1, 1, 1, 1)) * (target - mean_y.view(-1, 1, 1, 1))).mean(
        dim=dims
    )
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    )
    return score.mean().clamp(-1.0, 1.0)


def _as_chw(tensor: Tensor) -> Tensor:
    tensor = tensor.detach().cpu()
    return tensor[0] if tensor.ndim == 4 else tensor


def _to_pil(tensor: Tensor) -> Image.Image:
    array = _as_chw(tensor).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _absolute_error_heatmap(student: Tensor, teacher: Tensor) -> Tensor:
    error = (student - teacher).abs().mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    position = 4.0 * error
    red = (1.5 - (position - 3.0).abs()).clamp(0.0, 1.0)
    green = (1.5 - (position - 2.0).abs()).clamp(0.0, 1.0)
    blue = (1.5 - (position - 1.0).abs()).clamp(0.0, 1.0)
    return torch.cat((red, green, blue), dim=0)


def save_validation_grid(path: str | Path, images: Mapping[str, Tensor]) -> None:
    required = ("lq", "gt", "teacher", "student")
    missing = [name for name in required if name not in images]
    if missing:
        raise ValueError(f"validation grid is missing images: {missing}")
    lq, gt, teacher, student = (_as_chw(images[name]) for name in required)
    if lq.shape[-2:] != gt.shape[-2:]:
        lq = torch.nn.functional.interpolate(
            lq.unsqueeze(0), size=gt.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(0)
    error_heatmap = _absolute_error_heatmap(student, teacher)
    panels = [_to_pil(panel) for panel in (lq, gt, teacher, student, error_heatmap)]
    width, height = panels[0].size
    grid = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        if panel.size != (width, height):
            panel = panel.resize((width, height), Image.Resampling.BILINEAR)
        grid.paste(panel, (index * width, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
