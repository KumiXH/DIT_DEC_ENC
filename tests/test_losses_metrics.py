import builtins

import pytest
import torch

from distill_codec.adapters import freeze_module
from distill_codec.contracts import ContractError
from distill_codec.losses import LPIPSLoss, channel_stat_loss, cosine_loss, edge_loss, latent_smooth_l1
from distill_codec.metrics import psnr, save_validation_grid, ssim
from distill_codec.models.mock import MockWanDecoder


def test_latent_losses_are_finite_and_zero_for_equal_tensors():
    latent = torch.rand(2, 16, 8, 8)

    losses = (
        latent_smooth_l1(latent, latent),
        cosine_loss(latent, latent),
        channel_stat_loss(latent, latent),
    )

    assert all(torch.isfinite(loss) for loss in losses)
    assert all(loss.abs() < 1e-6 for loss in losses)


def test_edge_loss_detects_a_structural_difference():
    flat = torch.zeros(1, 3, 8, 8)
    edge = flat.clone()
    edge[:, :, :, 4:] = 1.0

    assert edge_loss(flat, flat) == 0
    assert edge_loss(flat, edge) > 0


def test_frozen_decoder_loss_backpropagates_to_latent():
    decoder = freeze_module(MockWanDecoder())
    latent = torch.rand(1, 16, 8, 8, requires_grad=True)
    target = torch.zeros(1, 3, 64, 64)

    loss = (decoder(latent) - target).abs().mean()
    loss.backward()

    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_psnr_and_ssim_known_bounds():
    black = torch.zeros(1, 3, 8, 8)
    white = torch.ones(1, 3, 8, 8)

    assert torch.isinf(psnr(black, black))
    assert torch.allclose(psnr(black, white), torch.tensor(0.0))
    assert 0.999 <= ssim(black, black) <= 1.0
    assert -1.0 <= ssim(black, white) <= 1.0


def test_validation_grid_writes_five_equal_panels(tmp_path):
    images = {
        "lq": torch.zeros(3, 8, 8),
        "gt": torch.ones(3, 8, 8),
        "teacher": torch.full((3, 8, 8), 0.75),
        "student": torch.full((3, 8, 8), 0.25),
    }
    output = tmp_path / "grid.png"

    save_validation_grid(output, images)

    from PIL import Image

    with Image.open(output) as grid:
        assert grid.size == (8 * 5, 8)


def test_lpips_missing_dependency_has_actionable_install_hint(monkeypatch):
    real_import = builtins.__import__

    def import_without_lpips(name, *args, **kwargs):
        if name == "lpips":
            raise ImportError("lpips unavailable for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_lpips)

    with pytest.raises(ContractError, match=r"install distill-codec\[perceptual\]"):
        LPIPSLoss()
