from pathlib import Path

import pytest
import torch

from distill_codec.integrations.snapshots import (
    FlashVSRConditionInputWrapper,
    FlashVSRTCDecoderWrapper,
    WanDecoderWrapper,
    WanEncoderWrapper,
    create_lq_proj_in,
    load_snapshot_module,
)
from distill_codec.contracts import ContractError


class _FakeWanVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen_video = None
        self.seen_latent = None
        self.seen_encode_device = None
        self.seen_decode_device = None

    def single_encode(self, video, device):
        self.seen_video = video
        self.seen_encode_device = device
        return torch.zeros(video.shape[0], 16, 1, video.shape[-2] // 8, video.shape[-1] // 8)

    def single_decode(self, latent, device):
        self.seen_latent = latent
        self.seen_decode_device = device
        return torch.zeros(latent.shape[0], 3, 1, latent.shape[-2] * 8, latent.shape[-1] * 8)


class _FakeTCDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.latent = None
        self.condition = None

    def decode_video(self, latent, parallel=False, cond=None):
        self.latent = latent
        self.condition = cond
        batch = latent.shape[0]
        return torch.full((batch, 1, 3, cond.shape[-2], cond.shape[-1]), 0.25)

    def clean_mem(self):
        pass


class _FakeConditionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, video):
        self.seen = video
        return [video.mean(dim=(-2, -1)).transpose(1, 2)]


def test_wan_wrappers_convert_ranges_and_time_layout():
    vae = _FakeWanVAE()
    encoder = WanEncoderWrapper(vae)
    decoder = WanDecoderWrapper(vae)

    latent = encoder(torch.ones(2, 3, 1, 32, 32))
    rgb = decoder(torch.zeros(2, 16, 1, 4, 4))

    assert latent.shape == (2, 16, 1, 4, 4)
    assert torch.equal(vae.seen_video, torch.ones_like(vae.seen_video))
    assert rgb.shape == (2, 3, 1, 32, 32)
    assert torch.allclose(rgb, torch.full_like(rgb, 0.5))


def test_wan_wrappers_follow_current_vae_device_after_module_move():
    vae = _FakeWanVAE()
    encoder = WanEncoderWrapper(vae, device="cuda:7").to("cpu")
    decoder = WanDecoderWrapper(vae, device="cuda:7").to("cpu")

    encoder(torch.ones(1, 3, 1, 32, 32))
    decoder(torch.zeros(1, 16, 1, 4, 4))

    assert vae.seen_encode_device == "cpu"
    assert vae.seen_decode_device == "cpu"


def test_tcdecoder_wrapper_repeats_single_lq_frame_for_official_condition():
    decoder = _FakeTCDecoder()
    wrapper = FlashVSRTCDecoderWrapper(decoder, condition_frames=4)

    rgb = wrapper(torch.zeros(2, 16, 4, 4), torch.ones(2, 3, 32, 32))

    assert decoder.latent.shape == (2, 1, 16, 4, 4)
    assert decoder.condition.shape == (2, 3, 4, 32, 32)
    assert torch.equal(decoder.condition, torch.ones_like(decoder.condition))
    assert decoder.latent.dtype == torch.bfloat16
    assert decoder.condition.dtype == torch.bfloat16
    assert rgb.shape == (2, 3, 32, 32)


def test_tcdecoder_wrapper_rejects_condition_spatial_mismatch():
    import pytest

    wrapper = FlashVSRTCDecoderWrapper(_FakeTCDecoder(), condition_frames=4)

    with pytest.raises(Exception, match="8x latent spatial size"):
        wrapper(torch.zeros(1, 16, 4, 4), torch.zeros(1, 3, 40, 32))


def test_flashvsr_condition_wrapper_converts_zero_one_rgb_to_negative_one_one():
    module = _FakeConditionEncoder()
    wrapper = FlashVSRConditionInputWrapper(module)

    wrapper(torch.zeros(1, 3, 5, 8, 8))

    assert torch.equal(module.seen, torch.full_like(module.seen, -1.0))


def test_snapshot_modules_and_lq_proj_factory_import_official_classes():
    wan = load_snapshot_module(Path("third_party/wan/wan_video_vae.py"), "test_wan_snapshot")
    flash = load_snapshot_module(Path("third_party/flashvsr/TCDecoder.py"), "test_flash_snapshot")
    lq_proj = create_lq_proj_in(
        source_file="third_party/flashvsr/utils.py",
        class_name="Causal_LQ4x_Proj",
        in_dim=3,
        out_dim=32,
        layer_num=1,
    )

    assert hasattr(wan, "WanVideoVAE")
    assert hasattr(flash, "build_tcdecoder")
    assert lq_proj.module.__class__.__name__ == "Causal_LQ4x_Proj"


def test_wan_encoder_and_decoder_can_share_one_vae_instance(monkeypatch):
    from distill_codec.integrations import snapshots

    builds = []

    def fake_build(**kwargs):
        builds.append(kwargs)
        return _FakeWanVAE()

    monkeypatch.setattr(snapshots, "_build_wan", fake_build)
    snapshots.clear_wan_cache()

    encoder = snapshots.create_wan_encoder(shared_key="test-shared")
    decoder = snapshots.create_wan_decoder(shared_key="test-shared")

    assert encoder.vae is decoder.vae
    assert len(builds) == 1


def test_wan_shared_key_rejects_different_construction_parameters(monkeypatch):
    from distill_codec.integrations import snapshots

    monkeypatch.setattr(snapshots, "_build_wan", lambda **kwargs: _FakeWanVAE())
    snapshots.clear_wan_cache()
    snapshots.create_wan_encoder(shared_key="teacher", checkpoint="first.pt")

    with pytest.raises(ContractError, match="shared_key.*different construction parameters"):
        snapshots.create_wan_decoder(shared_key="teacher", checkpoint="second.pt")


def test_partial_checkpoint_below_coverage_threshold_is_rejected(tmp_path):
    from distill_codec.integrations import snapshots

    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    checkpoint = tmp_path / "partial.pt"
    torch.save({"0.bias": module[0].bias.detach().clone()}, checkpoint)

    with pytest.raises(ContractError, match="coverage.*below required"):
        snapshots._load_weights(module, checkpoint, strict=False, minimum_coverage=0.5)


def test_snapshot_weight_loader_uses_safe_tensor_only_mode(tmp_path, monkeypatch):
    from distill_codec.integrations import snapshots

    module = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"placeholder")
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return module.state_dict()

    monkeypatch.setattr(torch, "load", fake_load)

    snapshots._load_weights(module, checkpoint, strict=True)

    assert calls == [{"map_location": "cpu", "weights_only": True}]
