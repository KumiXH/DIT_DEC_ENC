from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from .color import rgb_to_packed_6ch, sparse_yuv420_to_rgb
from .contracts import ColorSpec, ConditionSpec, ContractError, LatentSpec


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def repeat_video_frames(image: Tensor, frames: int) -> Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ContractError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
    if frames <= 0:
        raise ContractError("frames must be positive")
    return image.unsqueeze(2).expand(-1, -1, frames, -1, -1)


def _unwrap_tensor(output: object, component: str) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("latent", "sample", "images", "rgb", "output"):
            if key in output and isinstance(output[key], Tensor):
                return output[key]
    raise ContractError(f"{component} returned unsupported output type {type(output).__name__}")


def _validate_frame_selection(frame_selection: str) -> None:
    if frame_selection not in {"first", "center", "last"}:
        raise ContractError(
            f"frame_selection must be first, center, or last, got {frame_selection!r}"
        )


def _select_frame(tensor: Tensor, frame_selection: str) -> Tensor:
    if frame_selection == "first":
        index = 0
    elif frame_selection == "last":
        index = tensor.shape[2] - 1
    else:
        index = tensor.shape[2] // 2
    return tensor[:, :, index]


class EncoderAdapter(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        *,
        latent_spec: LatentSpec,
        input_mode: str,
        color_spec: ColorSpec | None = None,
        temporal_frames: int = 1,
        frame_selection: str = "center",
    ) -> None:
        super().__init__()
        if input_mode not in {"rgb", "rgb_video", "packed_6ch"}:
            raise ContractError(f"unsupported encoder input_mode {input_mode!r}")
        _validate_frame_selection(frame_selection)
        self.module = module
        self.latent_spec = latent_spec
        self.input_mode = input_mode
        self.color_spec = color_spec or ColorSpec()
        self.temporal_frames = temporal_frames
        self.frame_selection = frame_selection

    def forward(self, rgb: Tensor) -> Tensor:
        if self.input_mode == "packed_6ch":
            model_input = rgb_to_packed_6ch(rgb, self.color_spec)
        elif self.input_mode == "rgb_video":
            model_input = repeat_video_frames(rgb, self.temporal_frames)
        else:
            model_input = rgb
        latent = _unwrap_tensor(self.module(model_input), "encoder")
        if latent.ndim == 5 and self.latent_spec.layout == "BCHW":
            latent = _select_frame(latent, self.frame_selection)
        image_size = (int(rgb.shape[-2]), int(rgb.shape[-1]))
        temporal_size = self.temporal_frames if self.input_mode == "rgb_video" else 1
        self.latent_spec.validate_tensor(
            latent,
            image_size=image_size,
            temporal_size=temporal_size,
        )
        return latent


class DecoderAdapter(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        *,
        output_mode: str,
        color_spec: ColorSpec | None = None,
        accepts_condition: bool = False,
        frame_selection: str = "center",
    ) -> None:
        super().__init__()
        if output_mode not in {"rgb", "sparse_yuv"}:
            raise ContractError(f"unsupported decoder output_mode {output_mode!r}")
        _validate_frame_selection(frame_selection)
        self.module = module
        self.output_mode = output_mode
        self.color_spec = color_spec or ColorSpec()
        self.accepts_condition = accepts_condition
        self.frame_selection = frame_selection

    def forward(self, latent: Tensor, condition: Tensor | None = None) -> Tensor:
        if self.accepts_condition:
            if condition is None:
                raise ContractError("decoder requires a condition tensor")
            output = self.module(latent, condition)
        else:
            output = self.module(latent)
        image = _unwrap_tensor(output, "decoder")
        if image.ndim == 5:
            image = _select_frame(image, self.frame_selection)
        if self.output_mode == "sparse_yuv":
            image = sparse_yuv420_to_rgb(image, self.color_spec)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ContractError(f"decoder RGB output must be [B,3,H,W], got {tuple(image.shape)}")
        return image


class ConditionEncoderAdapter(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        *,
        temporal_frames: int = 1,
        condition_spec: ConditionSpec | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.temporal_frames = temporal_frames
        self.condition_spec = condition_spec

    def forward(self, rgb: Tensor) -> dict[str, Tensor]:
        output = self.module(repeat_video_frames(rgb, self.temporal_frames))
        if isinstance(output, Tensor):
            result = {"features": output}
        elif isinstance(output, Mapping) and all(isinstance(value, Tensor) for value in output.values()):
            result = dict(output)
        elif isinstance(output, (tuple, list)) and all(isinstance(value, Tensor) for value in output):
            result = (
                {"features": output[0]}
                if len(output) == 1
                else {f"features_{index}": value for index, value in enumerate(output)}
            )
        else:
            raise ContractError(f"condition encoder returned unsupported output type {type(output).__name__}")
        if self.condition_spec is not None:
            for name, value in result.items():
                try:
                    self.condition_spec.validate_tensor(
                        value,
                        batch_size=int(rgb.shape[0]),
                    )
                except ContractError as error:
                    raise ContractError(f"condition {name!r} violates its contract: {error}") from error
        return result
