import pytest
import torch

from tests.support_factories import clear_private_bridge_calls, private_bridge_calls


def setup_function():
    clear_private_bridge_calls()


def test_encoder_bridge_forwards_rgb_reference_and_kwargs():
    from private_codec.bridge import PrivateEncoderBridge

    teacher_reference = {"role": "encoder", "inputs": {"rgb": {"shape": [None, 3, 4, 4]}}}
    bridge = PrivateEncoderBridge(
        builder="tests.support_factories:build_private_bridge_network",
        runner="tests.support_factories:run_private_encoder",
        builder_kwargs={"offset": 2.0},
        runner_kwargs={"gain": 3.0},
        teacher_reference=teacher_reference,
    )
    teacher_reference["role"] = "changed_after_construction"
    rgb = torch.ones(1, 3, 4, 4)

    result = bridge(rgb)

    assert torch.equal(result, torch.full_like(rgb, 9.0))
    assert private_bridge_calls[-1]["rgb"] is rgb
    assert private_bridge_calls[-1]["teacher_reference"]["role"] == "encoder"
    assert bridge.teacher_reference["role"] == "encoder"
    assert "runner_mutated" not in bridge.teacher_reference


def test_encoder_bridge_passes_a_fresh_teacher_reference_to_each_call():
    from private_codec.bridge import PrivateEncoderBridge

    bridge = PrivateEncoderBridge(
        builder="tests.support_factories:build_private_bridge_network",
        runner="tests.support_factories:run_private_encoder",
        teacher_reference={"role": "encoder"},
    )
    rgb = torch.ones(1, 3, 4, 4)

    bridge(rgb)
    bridge(rgb)

    assert private_bridge_calls == [
        {
            "network": bridge.network,
            "teacher_reference": {"role": "encoder"},
            "rgb": rgb,
        },
        {
            "network": bridge.network,
            "teacher_reference": {"role": "encoder"},
            "rgb": rgb,
        },
    ]


def test_decoder_bridge_reorders_project_arguments_into_private_keywords():
    from private_codec.bridge import PrivateConditionalDecoderBridge

    bridge = PrivateConditionalDecoderBridge(
        builder="tests.support_factories:build_private_bridge_network",
        runner="tests.support_factories:run_private_decoder",
        builder_kwargs={"offset": 1.0},
        runner_kwargs={"gain": 2.0},
        teacher_reference={"role": "conditional_decoder"},
    )
    dit_latent = torch.randn(1, 16, 2, 2)
    lq_rgb = torch.ones(1, 3, 8, 8)

    result = bridge(dit_latent, lq_rgb)

    assert torch.equal(result, torch.full_like(lq_rgb, 4.0))
    assert private_bridge_calls[-1]["dit_latent"] is dit_latent
    assert private_bridge_calls[-1]["lq_rgb"] is lq_rgb
    assert private_bridge_calls[-1]["teacher_reference"] == {
        "role": "conditional_decoder"
    }


def test_bridge_rejects_builder_that_does_not_return_module():
    from private_codec.bridge import PrivateEncoderBridge

    with pytest.raises(TypeError, match="builder.*object.*torch.nn.Module"):
        PrivateEncoderBridge(
            builder="tests.support_factories:build_private_bridge_non_module",
            runner="tests.support_factories:run_private_encoder",
            teacher_reference={"role": "encoder"},
        )


def test_bridge_rejects_runner_that_does_not_return_tensor():
    from private_codec.bridge import PrivateEncoderBridge

    bridge = PrivateEncoderBridge(
        builder="tests.support_factories:build_private_bridge_network",
        runner="tests.support_factories:run_private_non_tensor",
        teacher_reference={"role": "encoder"},
    )

    with pytest.raises(TypeError, match="runner.*dict.*torch.Tensor"):
        bridge(torch.ones(1, 3, 4, 4))


def test_private_codec_factories_build_stable_bridges():
    from private_codec.bridge import PrivateConditionalDecoderBridge, PrivateEncoderBridge
    from private_codec.factories import create_conditional_decoder, create_encoder

    common = {
        "builder": "tests.support_factories:build_private_bridge_network",
        "builder_kwargs": {"offset": 2.0},
        "teacher_reference": {"role": "test"},
    }
    encoder = create_encoder(
        **common,
        runner="tests.support_factories:run_private_encoder",
    )
    decoder = create_conditional_decoder(
        **common,
        runner="tests.support_factories:run_private_decoder",
    )

    assert isinstance(encoder, PrivateEncoderBridge)
    assert isinstance(decoder, PrivateConditionalDecoderBridge)


@pytest.mark.parametrize(
    ("field", "import_path"),
    (
        ("builder", ":build_private_bridge_network"),
        ("builder", "tests.support_factories:"),
        ("runner", ":run_private_encoder"),
        ("runner", "tests.support_factories:"),
    ),
)
def test_bridge_rejects_malformed_import_paths_with_role(field, import_path):
    from private_codec.bridge import PrivateEncoderBridge

    paths = {
        "builder": "tests.support_factories:build_private_bridge_network",
        "runner": "tests.support_factories:run_private_encoder",
    }
    paths[field] = import_path

    with pytest.raises(ValueError, match=rf"private codec {field}.*module:symbol"):
        PrivateEncoderBridge(
            **paths,
            teacher_reference={"role": "encoder"},
        )


def test_bridge_validates_runner_before_calling_builder():
    from private_codec.bridge import PrivateEncoderBridge

    with pytest.raises(ValueError, match="private codec runner.*module:symbol"):
        PrivateEncoderBridge(
            builder="tests.support_factories:build_private_bridge_should_not_run",
            runner=":run_private_encoder",
            teacher_reference={"role": "encoder"},
        )
