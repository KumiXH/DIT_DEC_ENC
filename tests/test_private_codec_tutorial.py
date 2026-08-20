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


def test_private_codec_source_placeholders_are_ready_for_paste():
    assert (PRIVATE_CODEC / "__init__.py").is_file()
    assert (PRIVATE_CODEC / "base_network.py").read_text(encoding="utf-8") == ""
    assert (PRIVATE_CODEC / "wrapped_network.py").read_text(encoding="utf-8") == ""
    assert (PRIVATE_CODEC / "entrypoints.py").read_text(encoding="utf-8") == ""


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
        "builder": "private_codec.entrypoints:build_encoder",
        "runner": "private_codec.entrypoints:run_encoder",
        "builder_kwargs": {},
        "runner_kwargs": {},
    }
    assert encoder["adapter"] == {
        "kind": "encoder",
        "input_mode": "rgb",
        "latent_temporal_frames": "teacher",
    }

    decoder = config["components"]["conditional_student_decoder"]
    assert decoder["factory"] == "private_codec.factories:create_conditional_decoder"
    assert decoder["teacher_reference"] == "auto"
    assert decoder["kwargs"] == {
        "builder": "private_codec.entrypoints:build_decoder",
        "runner": "private_codec.entrypoints:run_decoder",
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
        "src/private_codec/base_network.py",
        "src/private_codec/wrapped_network.py",
        "src/private_codec/entrypoints.py",
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
        "private_codec.versions.v2.entrypoints",
        "python -m distill_codec.cli probe",
        "output_mode: rgb",
        "框架已经验证",
        "真实私有网络尚未验证",
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
