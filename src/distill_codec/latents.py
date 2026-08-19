from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from .contracts import ContractError, DistillBatch, LatentSpec


@dataclass(frozen=True)
class CachedLatentPreflightReport:
    sample_count: int
    relative_paths: tuple[str, ...]


class LatentProvider(nn.Module):
    def __init__(self, latent_spec: LatentSpec) -> None:
        super().__init__()
        self.latent_spec = latent_spec

    def _validate(self, latent: Tensor, batch: DistillBatch) -> Tensor:
        image_size = (int(batch.gt_rgb.shape[-2]), int(batch.gt_rgb.shape[-1]))
        self.latent_spec.validate_tensor(latent, image_size=image_size)
        return latent


class TeacherEncoderLatentProvider(LatentProvider):
    def __init__(self, encoder: nn.Module, *, source: str, latent_spec: LatentSpec) -> None:
        super().__init__(latent_spec)
        if source not in {"lq", "gt"}:
            raise ContractError(f"teacher encoder latent source must be lq or gt, got {source!r}")
        self.encoder = encoder
        self.source = source

    def forward(self, batch: DistillBatch) -> Tensor:
        image = batch.gt_rgb if self.source == "gt" else batch.lq_rgb
        latent = self.encoder(image)
        image_size = (int(image.shape[-2]), int(image.shape[-1]))
        self.latent_spec.validate_tensor(latent, image_size=image_size)
        return latent


class CachedLatentProvider(LatentProvider):
    def __init__(self, root: str | Path, *, latent_spec: LatentSpec) -> None:
        super().__init__(latent_spec)
        self.root = Path(root)
        manifest_path = self.root / "manifest.pt"
        if not manifest_path.is_file():
            raise ContractError(f"cached latent manifest does not exist: {manifest_path}")
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        saved_spec = LatentSpec.from_dict(manifest["latent_spec"])
        latent_spec.assert_compatible(saved_spec)
        self.preflight_report: CachedLatentPreflightReport | None = None

    @staticmethod
    def _latent_path(relative: str) -> Path:
        return Path(relative).with_suffix(".pt")

    @staticmethod
    def _unwrap_latent(payload: object, path: Path) -> Tensor:
        latent = payload["latent"] if isinstance(payload, dict) and "latent" in payload else payload
        if not isinstance(latent, Tensor):
            raise ContractError(f"cached latent {path} must contain a tensor")
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent[0]
        if latent.ndim != 3:
            raise ContractError(
                f"cached latent {path} must have shape [C,H,W] or [1,C,H,W], "
                f"got {tuple(latent.shape)}"
            )
        return latent

    def _load_latent(self, path: Path, *, map_location: torch.device | str) -> Tensor:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        return self._unwrap_latent(payload, path)

    def preflight(
        self,
        image_sizes_by_relative: Mapping[str, tuple[int, int]],
    ) -> CachedLatentPreflightReport:
        expected = {
            self._latent_path(relative).as_posix(): relative
            for relative in image_sizes_by_relative
        }
        actual = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*.pt")
            if path.name != "manifest.pt"
        }
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        if missing or extra:
            raise ContractError(
                "cached latent sample set mismatch; "
                f"missing={missing or '[]'}; extra={extra or '[]'}"
            )
        for latent_relative, image_relative in sorted(expected.items()):
            latent = self._load_latent(self.root / latent_relative, map_location="cpu")
            try:
                self.latent_spec.validate_tensor(
                    latent.unsqueeze(0),
                    image_size=image_sizes_by_relative[image_relative],
                )
            except ContractError as error:
                raise ContractError(
                    f"cached latent for sample {image_relative!r} violates its contract: {error}"
                ) from error
        report = CachedLatentPreflightReport(
            sample_count=len(expected),
            relative_paths=tuple(sorted(image_sizes_by_relative)),
        )
        self.preflight_report = report
        return report

    def forward(self, batch: DistillBatch) -> Tensor:
        latents = []
        for relative in batch.relative_path:
            path = self.root / self._latent_path(relative)
            if not path.is_file():
                raise ContractError(f"cached latent does not exist for sample {relative!r}: {path}")
            latents.append(self._load_latent(path, map_location=batch.gt_rgb.device))
        return self._validate(torch.stack(latents), batch)


class DatasetLatentProvider(LatentProvider):
    def forward(self, batch: DistillBatch) -> Tensor:
        if batch.latent is None:
            raise ContractError("DistillBatch does not contain latent for dataset provider")
        return self._validate(batch.latent, batch)
