from pathlib import Path

from distill_codec.config import load_config, preflight_config


def test_private_offline_conditional_decoder_config_preflights():
    config = load_config("configs/local/private_codec_conditional_decoder_offline.yaml")

    preflight_config(config)
    assert config["recipe"] == {
        "name": "flashvsr_decoder_conditional_student",
        "weights": {"teacher": 1.0, "gt": 0.0, "edge": 0.1},
    }
    assert config["latent_provider"]["type"] == "cached"
    assert config["teacher_target_provider"] == {"type": "dataset_gt"}
    assert config["data"]["augmentation"] == {"enabled": False}
    assert "teacher_encoder" not in config["components"]
    assert "tc_decoder" not in config["components"]


def test_private_codec_tutorial_documents_offline_teacher_targets():
    tutorial = Path("PRIVATE_CODEC_INTEGRATION_TUTORIAL.md").read_text(encoding="utf-8")

    for marker in (
        "configs/local/private_codec_conditional_decoder_offline.yaml",
        "teacher_target_provider:",
        "type: dataset_gt",
        "~/dit_codec/DIT_LATENT",
        "~/dit_codec/TCDECODER_RGB",
    ):
        assert marker in tutorial
