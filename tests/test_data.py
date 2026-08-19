from pathlib import Path

import pytest
import torch
from PIL import Image

from distill_codec.contracts import ContractError, DistillBatch
from distill_codec.data import PairedImageDataset, collate_distill_batch, create_mock_dataset


def _write_rgb(path: Path, size=(8, 8), value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


def test_dataset_matches_nested_relative_paths_and_collates(tmp_path):
    lq_root = tmp_path / "lq"
    gt_root = tmp_path / "gt"
    for relative in (Path("a.png"), Path("scene") / "b.jpg"):
        _write_rgb(lq_root / relative, size=(8, 8), value=64)
        _write_rgb(gt_root / relative, size=(16, 16), value=192)

    dataset = PairedImageDataset(lq_root, gt_root, lq_size=(8, 8), gt_size=(16, 16))
    batch = collate_distill_batch([dataset[0], dataset[1]])

    assert dataset.relative_paths == ("a.png", "scene/b.jpg")
    assert isinstance(batch, DistillBatch)
    assert batch.lq_rgb.shape == (2, 3, 8, 8)
    assert batch.gt_rgb.shape == (2, 3, 16, 16)
    assert 0.0 <= batch.lq_rgb.min() <= batch.lq_rgb.max() <= 1.0


def test_dataset_exposes_preflight_report(tmp_path):
    lq_root = tmp_path / "lq"
    gt_root = tmp_path / "gt"
    _write_rgb(lq_root / "a.png", size=(8, 6), value=64)
    _write_rgb(gt_root / "a.png", size=(16, 12), value=192)
    _write_rgb(lq_root / "nested" / "b.png", size=(8, 6), value=64)
    _write_rgb(gt_root / "nested" / "b.png", size=(16, 12), value=192)

    dataset = PairedImageDataset(lq_root, gt_root)

    assert dataset.preflight_report.pair_count == 2
    assert dataset.preflight_report.relative_paths == ("a.png", "nested/b.png")
    assert dataset.preflight_report.lq_sizes == ((6, 8),)
    assert dataset.preflight_report.gt_sizes == ((12, 16),)


def test_dataset_reports_missing_counterparts(tmp_path):
    _write_rgb(tmp_path / "lq" / "only_lq.png")
    _write_rgb(tmp_path / "gt" / "only_gt.png")

    with pytest.raises(ContractError, match="only in LQ.*only_lq.png.*only in GT.*only_gt.png"):
        PairedImageDataset(tmp_path / "lq", tmp_path / "gt")


def test_dataset_rejects_configured_size_mismatch(tmp_path):
    _write_rgb(tmp_path / "lq" / "a.png", size=(7, 8))
    _write_rgb(tmp_path / "gt" / "a.png", size=(16, 16))
    with pytest.raises(ContractError, match="a.png.*expected LQ size"):
        PairedImageDataset(tmp_path / "lq", tmp_path / "gt", lq_size=(8, 8))


def test_dataset_preflight_rejects_corrupt_image(tmp_path):
    lq = tmp_path / "lq" / "a.png"
    lq.parent.mkdir(parents=True)
    lq.write_bytes(b"not an image")
    _write_rgb(tmp_path / "gt" / "a.png")

    with pytest.raises(ContractError, match="cannot decode image.*a.png"):
        PairedImageDataset(tmp_path / "lq", tmp_path / "gt")


def test_mock_dataset_is_deterministic_and_strictly_paired(tmp_path):
    first = create_mock_dataset(tmp_path / "one", count=3, size=(16, 16), seed=11)
    second = create_mock_dataset(tmp_path / "two", count=3, size=(16, 16), seed=11)

    first_dataset = PairedImageDataset(first.lq_root, first.gt_root, lq_size=(16, 16), gt_size=(16, 16))
    second_dataset = PairedImageDataset(second.lq_root, second.gt_root)

    assert first_dataset.relative_paths == ("scene_00/frame_0000.png", "scene_00/frame_0001.png", "scene_01/frame_0002.png")
    assert torch.equal(first_dataset[0]["lq_rgb"], second_dataset[0]["lq_rgb"])
    assert torch.equal(first_dataset[0]["gt_rgb"], second_dataset[0]["gt_rgb"])
    assert not torch.equal(first_dataset[0]["lq_rgb"], first_dataset[0]["gt_rgb"])
