import pytest
import torch

from distill_codec.adapters import (
    ConditionEncoderAdapter,
    DecoderAdapter,
    EncoderAdapter,
    freeze_module,
    repeat_video_frames,
)
from distill_codec.contracts import ColorSpec, ConditionSpec, ContractError, LatentSpec
from distill_codec.factories import build_from_factory
from distill_codec.models.mock import (
    MockConditionalStudentDecoder,
    MockLQProjIn,
    MockStudentDecoder,
    MockStudentEncoder,
    MockTCDecoder,
    MockWanDecoder,
    MockWanEncoder,
)


LATENT_SPEC = LatentSpec("mock_wan", 16, "BCHW", 8, 1, "mock_wan")


def test_import_string_factory_and_checkpoint_round_trip(tmp_path):
    original = build_from_factory("tests.support_factories:create_encoder", {"channels": 16})
    with torch.no_grad():
        next(original.parameters()).fill_(0.25)
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"state_dict": original.state_dict()}, checkpoint)

    restored = build_from_factory(
        "tests.support_factories:create_encoder",
        {"channels": 16},
        checkpoint=checkpoint,
    )

    assert torch.equal(next(original.parameters()), next(restored.parameters()))


def test_external_factory_checkpoint_uses_safe_tensor_only_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"placeholder")
    source = build_from_factory("tests.support_factories:create_encoder", {"channels": 16})
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return source.state_dict()

    monkeypatch.setattr(torch, "load", fake_load)

    build_from_factory(
        "tests.support_factories:create_encoder",
        {"channels": 16},
        checkpoint=checkpoint,
    )

    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_student_encoder_adapter_uses_packed_input_and_keeps_gradients():
    module = MockStudentEncoder(latent_channels=16)
    adapter = EncoderAdapter(
        module,
        latent_spec=LATENT_SPEC,
        input_mode="packed_6ch",
        color_spec=ColorSpec(),
    )
    rgb = torch.rand(2, 3, 64, 64)

    latent = adapter(rgb)
    latent.mean().backward()

    assert latent.shape == (2, 16, 8, 8)
    assert sum(parameter.grad.abs().sum() for parameter in module.parameters() if parameter.grad is not None) > 0


def test_video_teacher_encoder_normalizes_single_frame_output():
    teacher = MockWanEncoder(latent_channels=16)
    adapter = EncoderAdapter(
        teacher,
        latent_spec=LATENT_SPEC,
        input_mode="rgb_video",
        temporal_frames=3,
    )

    latent = adapter(torch.rand(1, 3, 64, 64))

    assert latent.shape == (1, 16, 8, 8)
    assert repeat_video_frames(torch.zeros(1, 3, 4, 4), 3).shape == (1, 3, 3, 4, 4)


@pytest.mark.parametrize(
    ("frame_selection", "expected"),
    (("first", 0.0), ("center", 2.0), ("last", 3.0)),
)
def test_encoder_adapter_selects_configured_frame_from_5d_output(frame_selection, expected):
    class IndexedVideoEncoder(torch.nn.Module):
        def forward(self, video):
            frames = torch.arange(4, dtype=video.dtype, device=video.device)
            return frames.view(1, 1, 4, 1, 1).expand(video.shape[0], 16, 4, 8, 8)

    adapter = EncoderAdapter(
        IndexedVideoEncoder(),
        latent_spec=LATENT_SPEC,
        input_mode="rgb_video",
        temporal_frames=4,
        frame_selection=frame_selection,
    )

    latent = adapter(torch.rand(1, 3, 64, 64))

    assert torch.all(latent == expected)


def test_encoder_adapter_rejects_invalid_frame_selection():
    with pytest.raises(ContractError, match="frame_selection.*first.*center.*last"):
        EncoderAdapter(
            MockWanEncoder(),
            latent_spec=LATENT_SPEC,
            input_mode="rgb_video",
            frame_selection="middle",
        )


def test_video_teacher_encoder_preserves_bcthw_latent_layout():
    class VideoEncoder(torch.nn.Module):
        def forward(self, video):
            return torch.zeros(video.shape[0], 16, 2, 8, 8)

    adapter = EncoderAdapter(
        VideoEncoder(),
        latent_spec=LatentSpec("video", 16, "BCTHW", 8, 4, "video"),
        input_mode="rgb_video",
        temporal_frames=5,
        frame_selection="last",
    )

    latent = adapter(torch.rand(1, 3, 64, 64))

    assert latent.shape == (1, 16, 2, 8, 8)


def test_video_teacher_encoder_rejects_wrong_bcthw_temporal_size():
    class VideoEncoder(torch.nn.Module):
        def forward(self, video):
            return torch.zeros(video.shape[0], 16, 1, 8, 8)

    adapter = EncoderAdapter(
        VideoEncoder(),
        latent_spec=LatentSpec("video", 16, "BCTHW", 8, 4, "video"),
        input_mode="rgb_video",
        temporal_frames=5,
    )

    with pytest.raises(ContractError, match="expected temporal size=2"):
        adapter(torch.rand(1, 3, 64, 64))


def test_decoder_adapters_support_sparse_unconditional_and_conditional_models():
    latent = torch.rand(1, 16, 8, 8)
    lq = torch.rand(1, 3, 64, 64)
    plain = DecoderAdapter(MockStudentDecoder(), output_mode="sparse_yuv", color_spec=ColorSpec())
    conditional = DecoderAdapter(
        MockConditionalStudentDecoder(),
        output_mode="sparse_yuv",
        color_spec=ColorSpec(),
        accepts_condition=True,
    )
    tc_teacher = DecoderAdapter(MockTCDecoder(), output_mode="rgb", accepts_condition=True)
    wan_teacher = DecoderAdapter(MockWanDecoder(), output_mode="rgb")

    assert plain(latent).shape == (1, 3, 64, 64)
    assert conditional(latent, lq).shape == (1, 3, 64, 64)
    assert tc_teacher(latent, lq).shape == (1, 3, 64, 64)
    assert wan_teacher(latent).shape == (1, 3, 64, 64)


@pytest.mark.parametrize(
    ("frame_selection", "expected"),
    (("first", 0.0), ("center", 2.0), ("last", 3.0)),
)
def test_decoder_adapter_selects_configured_frame_from_5d_output(frame_selection, expected):
    class IndexedVideoDecoder(torch.nn.Module):
        def forward(self, latent):
            frames = torch.arange(4, dtype=latent.dtype, device=latent.device)
            return frames.view(1, 1, 4, 1, 1).expand(latent.shape[0], 3, 4, 64, 64)

    adapter = DecoderAdapter(
        IndexedVideoDecoder(),
        output_mode="rgb",
        frame_selection=frame_selection,
    )

    image = adapter(torch.rand(1, 16, 8, 8))

    assert torch.all(image == expected)


def test_decoder_adapter_rejects_invalid_frame_selection():
    with pytest.raises(ContractError, match="frame_selection.*first.*center.*last"):
        DecoderAdapter(MockWanDecoder(), output_mode="rgb", frame_selection="middle")


def test_condition_adapter_normalizes_tensor_to_named_dictionary():
    adapter = ConditionEncoderAdapter(MockLQProjIn(feature_dim=32), temporal_frames=3)

    condition = adapter(torch.rand(2, 3, 64, 64))

    assert set(condition) == {"features"}
    assert condition["features"].shape[0] == 2
    assert condition["features"].shape[-1] == 32


def test_condition_adapter_normalizes_single_layer_list_to_features_key():
    class SingleLayer(torch.nn.Module):
        def forward(self, video):
            return [torch.zeros(video.shape[0], 8, 32)]

    adapter = ConditionEncoderAdapter(SingleLayer(), temporal_frames=5)

    condition = adapter(torch.rand(2, 3, 64, 64))

    assert set(condition) == {"features"}


def test_condition_adapter_validates_bnc_layout_sampling_and_feature_dim():
    spec = ConditionSpec("mock_lq_proj", "BNC", 32, "lq", "dit", 8, 5)
    adapter = ConditionEncoderAdapter(
        MockLQProjIn(feature_dim=32),
        temporal_frames=5,
        condition_spec=spec,
    )

    condition = adapter(torch.rand(2, 3, 64, 64))

    assert condition["features"].shape == (2, 64, 32)


def test_condition_adapter_rejects_wrong_bnc_layout():
    class ChannelFirstCondition(torch.nn.Module):
        def forward(self, video):
            return torch.zeros(video.shape[0], 32, 64)

    adapter = ConditionEncoderAdapter(
        ChannelFirstCondition(),
        temporal_frames=5,
        condition_spec=ConditionSpec("condition", "BNC", 32, "lq", "dit", 8, 5),
    )

    with pytest.raises(ContractError, match="expected feature_dim=32"):
        adapter(torch.rand(2, 3, 64, 64))


def test_freeze_module_disables_parameter_gradients_but_not_input_gradient():
    decoder = freeze_module(MockWanDecoder())
    latent = torch.rand(1, 16, 8, 8, requires_grad=True)

    decoder(latent).mean().backward()

    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert not decoder.training
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
