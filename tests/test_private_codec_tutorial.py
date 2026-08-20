from pathlib import Path

import pytest
import torch
import yaml

from distill_codec.config import load_config, preflight_config


TUTORIAL = Path("PRIVATE_CODEC_INTEGRATION_TUTORIAL.md")
PRIVATE_CODEC = Path("src/private_codec")


def test_main_readme_links_private_codec_tutorial():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "[黑盒编解码器接入教程](PRIVATE_CODEC_INTEGRATION_TUTORIAL.md)" in readme


def test_private_codec_v0_is_the_only_copyable_version_template():
    assert (PRIVATE_CODEC / "__init__.py").is_file()
    v0 = PRIVATE_CODEC / "versions" / "v0"
    for path in (
        PRIVATE_CODEC / "versions" / "__init__.py",
        v0 / "__init__.py",
        v0 / "base_network.py",
        v0 / "wrapped_network.py",
        v0 / "entrypoints.py",
    ):
        assert path.is_file()

    for name in ("base_network.py", "wrapped_network.py", "entrypoints.py"):
        source = (v0 / name).read_text(encoding="utf-8")
        assert source.strip()
        assert "COPY" in source
        assert not (PRIVATE_CODEC / name).exists()


def test_private_codec_v0_encoder_runs_through_bridge_with_gradients():
    from private_codec.factories import create_encoder

    encoder = create_encoder(
        builder="private_codec.versions.v0.entrypoints:build_encoder",
        runner="private_codec.versions.v0.entrypoints:run_encoder",
        teacher_reference={"role": "encoder"},
    )
    rgb = torch.randn(2, 3, 32, 40, requires_grad=True)

    latent = encoder(rgb)
    latent.square().mean().backward()

    assert latent.shape == (2, 16, 4, 5)
    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()
    parameter_gradients = [
        parameter.grad for parameter in encoder.parameters() if parameter.requires_grad
    ]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_private_codec_v0_conditional_decoder_runs_through_bridge_with_gradients():
    from private_codec.factories import create_conditional_decoder

    decoder = create_conditional_decoder(
        builder="private_codec.versions.v0.entrypoints:build_decoder",
        runner="private_codec.versions.v0.entrypoints:run_decoder",
        teacher_reference={"role": "conditional_decoder"},
    )
    dit_latent = torch.randn(2, 16, 4, 5, requires_grad=True)
    lq_rgb = torch.randn(2, 3, 32, 40, requires_grad=True)

    output_rgb = decoder(dit_latent, lq_rgb)
    output_rgb.mean().backward()

    assert output_rgb.shape == (2, 3, 32, 40)
    for gradient in (dit_latent.grad, lq_rgb.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    parameter_gradients = [
        parameter.grad for parameter in decoder.parameters() if parameter.requires_grad
    ]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_private_codec_v0_legacy_decoder_is_a_runnable_rgb_example():
    from private_codec.factories import create_decoder

    decoder = create_decoder(
        class_name="V0UnconditionalDecoder",
        init_kwargs={},
    )
    dit_latent = torch.randn(2, 16, 4, 5, requires_grad=True)

    output_rgb = decoder(dit_latent)
    output_rgb.mean().backward()

    assert output_rgb.shape == (2, 3, 32, 40)
    assert dit_latent.grad is not None
    assert torch.isfinite(dit_latent.grad).all()


@pytest.mark.parametrize(
    ("component", "arguments", "message"),
    (
        ("encoder", (torch.randn(1, 1, 32, 32),), "encoder RGB input"),
        ("encoder", (torch.randn(1, 3, 30, 32),), "divisible by 8"),
        (
            "decoder",
            (torch.randn(2, 16, 4, 4), torch.randn(1, 3, 32, 32)),
            "batch size",
        ),
        (
            "decoder",
            (torch.randn(1, 8, 4, 4), torch.randn(1, 3, 32, 32)),
            "latent channels=16",
        ),
    ),
)
def test_private_codec_v0_reports_invalid_project_inputs(component, arguments, message):
    from private_codec.factories import create_conditional_decoder, create_encoder

    if component == "encoder":
        model = create_encoder(
            builder="private_codec.versions.v0.entrypoints:build_encoder",
            runner="private_codec.versions.v0.entrypoints:run_encoder",
            teacher_reference={"role": "encoder"},
        )
    else:
        model = create_conditional_decoder(
            builder="private_codec.versions.v0.entrypoints:build_decoder",
            runner="private_codec.versions.v0.entrypoints:run_decoder",
            teacher_reference={"role": "conditional_decoder"},
        )

    with pytest.raises(ValueError, match=message):
        model(*arguments)


def test_private_codec_factories_construct_the_same_class_with_different_kwargs():
    from private_codec.factories import create_decoder, create_encoder

    encoder = create_encoder(
        module_path="distill_codec.models.mock",
        class_name="MockStudentEncoder",
        init_kwargs={"latent_channels": 7},
    )
    decoder = create_decoder(
        module_path="distill_codec.models.mock",
        class_name="MockStudentDecoder",
        init_kwargs={"latent_channels": 9},
    )

    assert isinstance(encoder, torch.nn.Module)
    assert isinstance(decoder, torch.nn.Module)
    assert encoder.net[-1].out_channels == 7
    assert decoder.net[0].in_channels == 9


def test_private_codec_factory_reports_a_missing_wrapper_class():
    from private_codec.factories import create_encoder

    with pytest.raises(ValueError, match="cannot find class"):
        create_encoder(
            module_path="distill_codec.models.mock",
            class_name="MissingCodecClass",
        )


def test_private_codec_config_templates_preflight():
    for path in (
        "configs/local/private_codec_encoder.yaml",
        "configs/local/private_codec_decoder.yaml",
        "configs/local/private_codec_autoencoder.yaml",
        "configs/local/private_codec_conditional_decoder.yaml",
    ):
        config = load_config(path)
        preflight_config(config)


def test_private_codec_student_config_exposes_rgb_bridge_contracts():
    config = yaml.safe_load(
        Path("configs/students/private_codec.yaml").read_text(encoding="utf-8")
    )

    encoder = config["components"]["student_encoder"]
    assert encoder["factory"] == "private_codec.factories:create_encoder"
    assert encoder["teacher_reference"] == "auto"
    assert encoder["kwargs"] == {
        "builder": "private_codec.versions.v0.entrypoints:build_encoder",
        "runner": "private_codec.versions.v0.entrypoints:run_encoder",
        "builder_kwargs": {},
        "runner_kwargs": {},
    }
    assert encoder["adapter"] == {
        "kind": "encoder",
        "input_mode": "rgb",
        "latent_temporal_frames": "teacher",
    }

    legacy_decoder = config["components"]["student_decoder"]
    assert legacy_decoder["kwargs"] == {
        "module_path": "private_codec.versions.v0.wrapped_network",
        "class_name": "V0UnconditionalDecoder",
        "init_kwargs": {},
    }
    assert legacy_decoder["adapter"] == {
        "kind": "decoder",
        "output_mode": "rgb",
    }

    decoder = config["components"]["conditional_student_decoder"]
    assert decoder["factory"] == "private_codec.factories:create_conditional_decoder"
    assert decoder["teacher_reference"] == "auto"
    assert decoder["kwargs"] == {
        "builder": "private_codec.versions.v0.entrypoints:build_decoder",
        "runner": "private_codec.versions.v0.entrypoints:run_decoder",
        "builder_kwargs": {},
        "runner_kwargs": {},
    }
    assert decoder["adapter"] == {
        "kind": "decoder",
        "output_mode": "rgb",
        "accepts_condition": True,
    }


def test_private_conditional_decoder_config_targets_flashvsr_teacher():
    config = load_config("configs/local/private_codec_conditional_decoder.yaml")

    assert config["recipe"]["name"] == "flashvsr_decoder_conditional_student"
    assert config["latent_provider"] == {"type": "teacher_encoder", "source": "gt"}
    assert set(config["components"]) >= {
        "teacher_encoder",
        "tc_decoder",
        "conditional_student_decoder",
    }


def test_private_codec_tutorial_names_every_fill_in_file_and_probe_command():
    tutorial = TUTORIAL.read_text(encoding="utf-8")

    for marker in (
        "src/private_codec/versions/v0/base_network.py",
        "src/private_codec/versions/v0/wrapped_network.py",
        "src/private_codec/versions/v0/entrypoints.py",
        "configs/students/private_codec.yaml",
        "configs/local/private_codec_encoder.yaml",
        "configs/local/private_codec_conditional_decoder.yaml",
        "private_codec.factories:create_encoder",
        "private_codec.factories:create_conditional_decoder",
        "def build_encoder(",
        "def run_encoder(",
        "def build_decoder(",
        "def run_decoder(",
        "conditional_student_decoder(dit_latent, lq_rgb)",
        "run_decoder(",
        "lq_rgb=lq_rgb",
        "dit_latent=dit_latent",
        "teacher_reference",
        "Copy-Item -Recurse src/private_codec/versions/v0 src/private_codec/versions/v1",
        "private_codec.versions.v0.entrypoints",
        "private_codec.versions.v1.entrypoints",
        "module_path: private_codec.versions.v1.wrapped_network",
        "class_name: V1UnconditionalDecoder",
        "python -m distill_codec.cli probe",
        "output_mode: rgb",
        "框架已经验证",
        "v0 示例网络已经验证",
        "真实私有网络仍需验证",
        "C:\\Users\\xh932\\anaconda3\\Scripts\\conda.exe",
        "$env:PYTHONPATH = (Resolve-Path 'src').Path",
        "python -m py_compile",
        "builder_kwargs",
        "runner_kwargs",
        "BCTHW",
        "torch.cuda.is_available()",
        "test_multi_component_resume_is_stable_across_python_hash_seeds",
    ):
        assert marker in tutorial

    assert "不要修改 `src/private_codec/bridge.py`" in tutorial
    assert "不要修改 `src/private_codec/factories.py`" in tutorial
