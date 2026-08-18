from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .contracts import ContractError, DistillBatch, LatentSpec


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

    def forward(self, batch: DistillBatch) -> Tensor:
        latents = []
        for relative in batch.relative_path:
            path = self.root / Path(relative).with_suffix(".pt")
            if not path.is_file():
                raise ContractError(f"cached latent does not exist for sample {relative!r}: {path}")
            payload = torch.load(path, map_location=batch.gt_rgb.device, weights_only=True)
            latent = payload["latent"] if isinstance(payload, dict) and "latent" in payload else payload
            if latent.ndim == 4 and latent.shape[0] == 1:
                latent = latent[0]
            latents.append(latent)
        return self._validate(torch.stack(latents), batch)


class DatasetLatentProvider(LatentProvider):
    def forward(self, batch: DistillBatch) -> Tensor:
        if batch.latent is None:
            raise ContractError("DistillBatch does not contain latent for dataset provider")
        return self._validate(batch.latent, batch)
