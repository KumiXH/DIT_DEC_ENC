import torch

from distill_codec.adapters import (
    ConditionEncoderAdapter,
    DecoderAdapter,
    EncoderAdapter,
    freeze_module,
    repeat_video_frames,
)
from distill_codec.contracts import ColorSpec, LatentSpec
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


def test_condition_adapter_normalizes_tensor_to_named_dictionary():
    adapter = ConditionEncoderAdapter(MockLQProjIn(feature_dim=32), temporal_frames=3)

    condition = adapter(torch.rand(2, 3, 64, 64))

    assert set(condition) == {"features"}
    assert condition["features"].shape[0] == 2
    assert condition["features"].shape[-1] == 32


def test_freeze_module_disables_parameter_gradients_but_not_input_gradient():
    decoder = freeze_module(MockWanDecoder())
    latent = torch.rand(1, 16, 8, 8, requires_grad=True)

    decoder(latent).mean().backward()

    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert not decoder.training
    assert all(not parameter.requires_grad for parameter in decoder.parameters())

