from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .contracts import ContractError, DistillBatch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetPreflightReport:
    pair_count: int
    relative_paths: tuple[str, ...]
    lq_sizes: tuple[tuple[int, int], ...]
    gt_sizes: tuple[tuple[int, int], ...]


def _normalized_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _scan_images(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ContractError(f"image root does not exist or is not a directory: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative = _normalized_relative(path, root)
            if relative in result:
                raise ContractError(f"duplicate relative image path {relative!r} under {root}")
            result[relative] = path
    if not result:
        raise ContractError(f"no supported images found under {root}")
    return result


def _load_rgb(path: Path) -> Tensor:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor = data.reshape(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot decode image {path}: {error}") from error
    return tensor


class PairedImageDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        lq_root: str | Path,
        gt_root: str | Path,
        *,
        lq_size: tuple[int, int] | None = None,
        gt_size: tuple[int, int] | None = None,
    ) -> None:
        self.lq_root = Path(lq_root)
        self.gt_root = Path(gt_root)
        self.lq_size = lq_size
        self.gt_size = gt_size
        lq_files = _scan_images(self.lq_root)
        gt_files = _scan_images(self.gt_root)
        only_lq = sorted(set(lq_files) - set(gt_files))
        only_gt = sorted(set(gt_files) - set(lq_files))
        if only_lq or only_gt:
            raise ContractError(
                "paired dataset mismatch; "
                f"only in LQ={only_lq or '[]'}; only in GT={only_gt or '[]'}"
            )
        self._pairs = tuple((relative, lq_files[relative], gt_files[relative]) for relative in sorted(lq_files))
        self.preflight_report = self._preflight()

    @property
    def relative_paths(self) -> tuple[str, ...]:
        return tuple(pair[0] for pair in self._pairs)

    @property
    def gt_sizes_by_relative(self) -> dict[str, tuple[int, int]]:
        return dict(self._gt_sizes_by_relative)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        relative, lq_path, gt_path = self._pairs[index]
        lq = _load_rgb(lq_path)
        gt = _load_rgb(gt_path)
        self._validate_size(relative, "LQ", lq, self.lq_size)
        self._validate_size(relative, "GT", gt, self.gt_size)
        return {"lq_rgb": lq, "gt_rgb": gt, "relative_path": relative}

    def _preflight(self) -> DatasetPreflightReport:
        lq_sizes: set[tuple[int, int]] = set()
        gt_sizes: set[tuple[int, int]] = set()
        self._gt_sizes_by_relative: dict[str, tuple[int, int]] = {}
        for relative, lq_path, gt_path in self._pairs:
            lq = _load_rgb(lq_path)
            gt = _load_rgb(gt_path)
            self._validate_size(relative, "LQ", lq, self.lq_size)
            self._validate_size(relative, "GT", gt, self.gt_size)
            lq_sizes.add((int(lq.shape[-2]), int(lq.shape[-1])))
            gt_size = (int(gt.shape[-2]), int(gt.shape[-1]))
            gt_sizes.add(gt_size)
            self._gt_sizes_by_relative[relative] = gt_size
        return DatasetPreflightReport(
            pair_count=len(self._pairs),
            relative_paths=self.relative_paths,
            lq_sizes=tuple(sorted(lq_sizes)),
            gt_sizes=tuple(sorted(gt_sizes)),
        )

    @staticmethod
    def _validate_size(
        relative: str,
        kind: str,
        tensor: Tensor,
        expected: tuple[int, int] | None,
    ) -> None:
        if expected is not None and tuple(tensor.shape[-2:]) != tuple(expected):
            raise ContractError(
                f"{relative}: expected {kind} size={tuple(expected)}, got {tuple(tensor.shape[-2:])}"
            )


def collate_distill_batch(samples: Sequence[Mapping[str, object]]) -> DistillBatch:
    if not samples:
        raise ContractError("cannot collate an empty sample list")
    has_latent = ["latent" in sample and sample["latent"] is not None for sample in samples]
    if any(has_latent) and not all(has_latent):
        raise ContractError("either every sample must contain latent or none may contain latent")
    return DistillBatch(
        lq_rgb=torch.stack([cast(Tensor, sample["lq_rgb"]) for sample in samples]),
        gt_rgb=torch.stack([cast(Tensor, sample["gt_rgb"]) for sample in samples]),
        relative_path=tuple(str(sample["relative_path"]) for sample in samples),
        latent=(
            torch.stack([cast(Tensor, sample["latent"]) for sample in samples])
            if all(has_latent)
            else None
        ),
    )


@dataclass(frozen=True)
class MockDatasetPaths:
    lq_root: Path
    gt_root: Path


def _tensor_to_image(tensor: Tensor, path: Path) -> None:
    array = tensor.clamp(0.0, 1.0).mul(255).round().byte().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def create_mock_dataset(
    output_root: str | Path,
    *,
    count: int = 8,
    size: tuple[int, int] = (64, 64),
    seed: int = 0,
) -> MockDatasetPaths:
    if count <= 0:
        raise ContractError("mock dataset count must be positive")
    height, width = size
    if height <= 0 or width <= 0:
        raise ContractError("mock dataset size must be positive")
    output_root = Path(output_root)
    lq_root = output_root / "lq"
    gt_root = output_root / "gt"
    generator = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0.0, 1.0, height).view(1, height, 1).expand(1, height, width)
    xx = torch.linspace(0.0, 1.0, width).view(1, 1, width).expand(1, height, width)
    for index in range(count):
        noise = torch.rand((1, height, width), generator=generator) * 0.08
        gt = torch.cat(
            (
                (xx + index / max(count, 1) * 0.2).remainder(1.0),
                (yy + noise).clamp(0.0, 1.0),
                ((xx + yy) * 0.5 + noise).clamp(0.0, 1.0),
            ),
            dim=0,
        )
        down_size = (max(1, height // 4), max(1, width // 4))
        lq = F.interpolate(gt.unsqueeze(0), size=down_size, mode="area")
        lq = F.interpolate(lq, size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        relative = Path(f"scene_{index // 2:02d}") / f"frame_{index:04d}.png"
        _tensor_to_image(gt, gt_root / relative)
        _tensor_to_image(lq, lq_root / relative)
    return MockDatasetPaths(lq_root=lq_root, gt_root=gt_root)
