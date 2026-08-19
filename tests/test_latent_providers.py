from pathlib import Path

import pytest
import torch

from distill_codec.adapters import EncoderAdapter, freeze_module
from distill_codec.contracts import ContractError, DistillBatch, LatentSpec
from distill_codec.latents import CachedLatentProvider, DatasetLatentProvider, TeacherEncoderLatentProvider
from distill_codec.models.mock import MockWanEncoder


SPEC = LatentSpec("mock_wan", 16, "BCHW", 8, 1, "mock_wan")


def _batch(latent=None):
    return DistillBatch(
        lq_rgb=torch.zeros(2, 3, 32, 32),
        gt_rgb=torch.ones(2, 3, 32, 32),
        relative_path=("scene/a.png", "b.png"),
        latent=latent,
    )


def test_teacher_encoder_provider_selects_configured_source():
    encoder = EncoderAdapter(
        freeze_module(MockWanEncoder()), latent_spec=SPEC, input_mode="rgb_video", temporal_frames=1
    )

    latent = TeacherEncoderLatentProvider(encoder, source="gt", latent_spec=SPEC)(_batch())

    assert latent.shape == (2, 16, 4, 4)


def test_teacher_encoder_lq_provider_validates_against_lq_size():
    encoder = EncoderAdapter(
        freeze_module(MockWanEncoder()), latent_spec=SPEC, input_mode="rgb_video", temporal_frames=1
    )
    batch = DistillBatch(
        lq_rgb=torch.zeros(2, 3, 32, 32),
        gt_rgb=torch.ones(2, 3, 64, 64),
        relative_path=("scene/a.png", "b.png"),
    )

    latent = TeacherEncoderLatentProvider(encoder, source="lq", latent_spec=SPEC)(batch)

    assert latent.shape == (2, 16, 4, 4)


def test_cached_provider_reads_relative_pt_files_and_validates_manifest(tmp_path):
    root = tmp_path / "latents"
    (root / "scene").mkdir(parents=True)
    torch.save(torch.zeros(16, 4, 4), root / "scene" / "a.pt")
    torch.save(torch.ones(16, 4, 4), root / "b.pt")
    torch.save({"latent_spec": SPEC.to_dict()}, root / "manifest.pt")

    latent = CachedLatentProvider(root, latent_spec=SPEC)(_batch())

    assert latent.shape == (2, 16, 4, 4)
    assert latent[0].sum() == 0
    assert latent[1].sum() > 0


def test_cached_provider_rejects_manifest_contract_mismatch(tmp_path):
    root = tmp_path / "latents"
    root.mkdir()
    wrong = LatentSpec("other", 16, "BCHW", 8, 1, "other")
    torch.save({"latent_spec": wrong.to_dict()}, root / "manifest.pt")

    with pytest.raises(ContractError, match="incompatible latent contract"):
        CachedLatentProvider(root, latent_spec=SPEC)


def test_cached_provider_preflight_rejects_missing_and_extra_sample_files(tmp_path):
    root = tmp_path / "latents"
    root.mkdir()
    torch.save({"latent_spec": SPEC.to_dict()}, root / "manifest.pt")
    torch.save(torch.zeros(16, 4, 4), root / "extra.pt")
    provider = CachedLatentProvider(root, latent_spec=SPEC)

    with pytest.raises(ContractError, match="missing=.*scene/a.pt.*extra=.*extra.pt"):
        provider.preflight(
            {
                "scene/a.png": (32, 32),
                "b.png": (32, 32),
            }
        )


def test_cached_provider_preflight_validates_every_tensor_contract(tmp_path):
    root = tmp_path / "latents"
    (root / "scene").mkdir(parents=True)
    torch.save({"latent_spec": SPEC.to_dict()}, root / "manifest.pt")
    torch.save(torch.zeros(16, 4, 4), root / "scene" / "a.pt")
    torch.save(torch.zeros(8, 4, 4), root / "b.pt")
    provider = CachedLatentProvider(root, latent_spec=SPEC)

    with pytest.raises(ContractError, match="b.png.*expected channels=16"):
        provider.preflight(
            {
                "scene/a.png": (32, 32),
                "b.png": (32, 32),
            }
        )


def test_cached_provider_preflight_exposes_report(tmp_path):
    root = tmp_path / "latents"
    (root / "scene").mkdir(parents=True)
    torch.save({"latent_spec": SPEC.to_dict()}, root / "manifest.pt")
    torch.save(torch.zeros(16, 4, 4), root / "scene" / "a.pt")
    torch.save(torch.ones(16, 4, 4), root / "b.pt")
    provider = CachedLatentProvider(root, latent_spec=SPEC)

    report = provider.preflight(
        {
            "scene/a.png": (32, 32),
            "b.png": (32, 32),
        }
    )

    assert report.sample_count == 2
    assert report.relative_paths == ("b.png", "scene/a.png")
    assert provider.preflight_report == report


def test_dataset_provider_requires_batch_latent():
    provider = DatasetLatentProvider(latent_spec=SPEC)

    with pytest.raises(ContractError, match="does not contain latent"):
        provider(_batch())

    latent = provider(_batch(torch.zeros(2, 16, 4, 4)))
    assert latent.shape == (2, 16, 4, 4)


def test_cached_provider_uses_safe_tensor_only_load(tmp_path, monkeypatch):
    root = tmp_path / "latents"
    root.mkdir()
    (root / "manifest.pt").write_bytes(b"placeholder")
    (root / "scene").mkdir()
    (root / "scene" / "a.pt").write_bytes(b"placeholder")
    (root / "b.pt").write_bytes(b"placeholder")
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if Path(path).name == "manifest.pt":
            return {"latent_spec": SPEC.to_dict()}
        return torch.zeros(16, 4, 4)

    monkeypatch.setattr(torch, "load", fake_load)

    CachedLatentProvider(root, latent_spec=SPEC)(_batch())

    assert calls
    assert all(call["weights_only"] is True for call in calls)
