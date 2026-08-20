from pathlib import Path

import pytest
import torch

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
    ):
        config = load_config(path)
        preflight_config(config)


def test_private_codec_tutorial_names_every_fill_in_file_and_probe_command():
    tutorial = TUTORIAL.read_text(encoding="utf-8")

    for marker in (
        "src/private_codec/base_network.py",
        "src/private_codec/wrapped_network.py",
        "configs/students/private_codec.yaml",
        "configs/local/private_codec_encoder.yaml",
        "configs/local/private_codec_decoder.yaml",
        "configs/local/private_codec_autoencoder.yaml",
        "private_codec.factories:create_encoder",
        "private_codec.factories:create_decoder",
        "python -m distill_codec.cli probe",
        "output_mode: rgb",
        "output_mode: sparse_yuv",
    ):
        assert marker in tutorial
