from pathlib import Path

import torch

from distill_codec.integrations.snapshots import (
    FlashVSRTCDecoderWrapper,
    WanDecoderWrapper,
    WanEncoderWrapper,
    create_lq_proj_in,
    load_snapshot_module,
)


class _FakeWanVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_video = None
        self.seen_latent = None

    def single_encode(self, video, device):
        self.seen_video = video
        return torch.zeros(video.shape[0], 16, 1, video.shape[-2] // 8, video.shape[-1] // 8)

    def single_decode(self, latent, device):
        self.seen_latent = latent
        return torch.zeros(latent.shape[0], 3, 1, latent.shape[-2] * 8, latent.shape[-1] * 8)


class _FakeTCDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.latent = None
        self.condition = None

    def decode_video(self, latent, parallel=False, cond=None):
        self.latent = latent
        self.condition = cond
        batch = latent.shape[0]
        return torch.full((batch, 1, 3, cond.shape[-2], cond.shape[-1]), 0.25)

    def clean_mem(self):
        pass


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


def test_tcdecoder_wrapper_repeats_single_lq_frame_for_official_condition():
    decoder = _FakeTCDecoder()
    wrapper = FlashVSRTCDecoderWrapper(decoder, condition_frames=4)

    rgb = wrapper(torch.zeros(2, 16, 4, 4), torch.ones(2, 3, 32, 32))

    assert decoder.latent.shape == (2, 1, 16, 4, 4)
    assert decoder.condition.shape == (2, 4, 3, 32, 32)
    assert rgb.shape == (2, 3, 32, 32)


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
    assert lq_proj.__class__.__name__ == "Causal_LQ4x_Proj"

