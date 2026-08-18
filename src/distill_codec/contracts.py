from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


class ContractError(ValueError):
    """Raised when a tensor or component violates an explicit contract."""


@dataclass(frozen=True)
class LatentSpec:
    family: str
    channels: int
    layout: str
    spatial_downsample: int
    temporal_downsample: int
    normalization: str
    value_range: str = "unbounded"

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ContractError("channels must be positive")
        if self.layout not in {"BCHW", "BCTHW"}:
            raise ContractError(f"layout must be BCHW or BCTHW, got {self.layout!r}")
        if self.spatial_downsample <= 0 or self.temporal_downsample <= 0:
            raise ContractError("downsample factors must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "LatentSpec":
        return cls(**dict(values))

    def assert_compatible(self, actual: "LatentSpec") -> None:
        fields = (
            "family",
            "channels",
            "layout",
            "spatial_downsample",
            "temporal_downsample",
            "normalization",
            "value_range",
        )
        mismatches = [
            f"{name}: expected {getattr(self, name)!r}, got {getattr(actual, name)!r}"
            for name in fields
            if getattr(self, name) != getattr(actual, name)
        ]
        if mismatches:
            raise ContractError("incompatible latent contract: " + "; ".join(mismatches))

    def validate_tensor(
        self,
        tensor: Tensor,
        image_size: tuple[int, int] | None = None,
        temporal_size: int | None = None,
    ) -> None:
        expected_ndim = 4 if self.layout == "BCHW" else 5
        if tensor.ndim != expected_ndim:
            raise ContractError(
                f"expected layout={self.layout} ({expected_ndim} dims), actual shape={tuple(tensor.shape)}"
            )
        if tensor.shape[1] != self.channels:
            raise ContractError(
                f"expected channels={self.channels}, actual shape={tuple(tensor.shape)}"
            )
        if self.layout == "BCTHW" and temporal_size is not None:
            expected_temporal = (temporal_size + self.temporal_downsample - 1) // self.temporal_downsample
            if tensor.shape[2] != expected_temporal:
                raise ContractError(
                    f"expected temporal size={expected_temporal} for temporal_size={temporal_size}, "
                    f"actual shape={tuple(tensor.shape)}"
                )
        if image_size is not None:
            expected_hw = tuple(size // self.spatial_downsample for size in image_size)
            if tuple(tensor.shape[-2:]) != expected_hw:
                raise ContractError(
                    f"expected spatial size={expected_hw} for image_size={image_size}, "
                    f"actual shape={tuple(tensor.shape)}"
                )


@dataclass(frozen=True)
class ConditionSpec:
    family: str
    layout: str
    feature_dim: int
    source: str
    consumer: str
    spatial_downsample: int = 1
    temporal_downsample: int = 1

    def __post_init__(self) -> None:
        if self.layout not in {"BNC"}:
            raise ContractError(f"layout must be BNC, got {self.layout!r}")
        if self.feature_dim <= 0:
            raise ContractError("feature_dim must be positive")
        if self.source not in {"lq", "gt", "cached"}:
            raise ContractError(f"source must be lq, gt, or cached, got {self.source!r}")
        if self.consumer not in {"dit", "decoder"}:
            raise ContractError(f"consumer must be dit or decoder, got {self.consumer!r}")
        if self.spatial_downsample <= 0 or self.temporal_downsample <= 0:
            raise ContractError("downsample factors must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ConditionSpec":
        return cls(**dict(values))

    def validate_tensor(
        self,
        tensor: Tensor,
        image_size: tuple[int, int] | None = None,
        temporal_size: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        if tensor.ndim != 3:
            raise ContractError(
                f"expected layout={self.layout} (3 dims), actual shape={tuple(tensor.shape)}"
            )
        if tensor.shape[-1] != self.feature_dim:
            raise ContractError(
                f"expected feature_dim={self.feature_dim}, actual shape={tuple(tensor.shape)}"
            )
        if batch_size is not None and tensor.shape[0] != batch_size:
            raise ContractError(
                f"expected batch size={batch_size}, actual shape={tuple(tensor.shape)}"
            )
        if image_size is not None and temporal_size is not None:
            spatial_tokens = 1
            for size in image_size:
                spatial_tokens *= size // self.spatial_downsample
            temporal_tokens = (
                temporal_size + self.temporal_downsample - 1
            ) // self.temporal_downsample
            expected_tokens = temporal_tokens * spatial_tokens
            if tensor.shape[1] != expected_tokens:
                raise ContractError(
                    f"expected tokens={expected_tokens} for image_size={image_size} and "
                    f"temporal_size={temporal_size}, actual shape={tuple(tensor.shape)}"
                )


@dataclass(frozen=True)
class ColorSpec:
    matrix: str = "bt709"
    range: str = "full"
    packed_order: str = "Y00Y01Y10Y11UV"
    chroma_location: str = "top_left"
    chroma_upsample: str = "nearest"

    def __post_init__(self) -> None:
        if self.matrix not in {"bt601", "bt709"}:
            raise ContractError(f"matrix must be bt601 or bt709, got {self.matrix!r}")
        if self.range not in {"full", "limited"}:
            raise ContractError(f"range must be full or limited, got {self.range!r}")
        if self.packed_order != "Y00Y01Y10Y11UV":
            raise ContractError(f"unsupported packed_order {self.packed_order!r}")
        if self.chroma_location != "top_left":
            raise ContractError(f"unsupported chroma_location {self.chroma_location!r}")
        if self.chroma_upsample not in {"nearest", "bilinear"}:
            raise ContractError(
                f"chroma_upsample must be nearest or bilinear, got {self.chroma_upsample!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ColorSpec":
        return cls(**dict(values))


@dataclass(frozen=True)
class DistillBatch:
    lq_rgb: Tensor
    gt_rgb: Tensor
    relative_path: tuple[str, ...]
    latent: Tensor | None = None

    def __post_init__(self) -> None:
        for name, tensor in (("lq_rgb", self.lq_rgb), ("gt_rgb", self.gt_rgb)):
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ContractError(f"{name} must have shape [B,3,H,W], got {tuple(tensor.shape)}")
        if self.lq_rgb.shape[0] != self.gt_rgb.shape[0]:
            raise ContractError("lq_rgb and gt_rgb batch sizes must match")
        if len(self.relative_path) != self.lq_rgb.shape[0]:
            raise ContractError(
                f"relative_path length={len(self.relative_path)} does not match "
                f"batch size={self.lq_rgb.shape[0]}"
            )
        if self.latent is not None and self.latent.shape[0] != self.lq_rgb.shape[0]:
            raise ContractError(
                f"latent batch size={self.latent.shape[0]} does not match RGB batch size={self.lq_rgb.shape[0]}"
            )

    @property
    def batch_size(self) -> int:
        return self.lq_rgb.shape[0]

    def to(self, device: torch.device | str) -> "DistillBatch":
        return DistillBatch(
            lq_rgb=self.lq_rgb.to(device),
            gt_rgb=self.gt_rgb.to(device),
            relative_path=self.relative_path,
            latent=self.latent.to(device) if self.latent is not None else None,
        )
