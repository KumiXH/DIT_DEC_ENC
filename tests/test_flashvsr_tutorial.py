from pathlib import Path
import shutil

import yaml

from distill_codec.config import load_config, preflight_config


TUTORIAL = Path("FLASHVSR_DISTILL_TUTORIAL.md")


def _yaml_after_heading(document: str, heading: str) -> dict:
    heading_offset = document.index(heading)
    block_start = document.index("```yaml\n", heading_offset) + len("```yaml\n")
    block_end = document.index("\n```", block_start)
    values = yaml.safe_load(document[block_start:block_end])
    assert isinstance(values, dict)
    return values


def test_main_readme_links_flashvsr_tutorial():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "[FlashVSR 蒸馏教程](FLASHVSR_DISTILL_TUTORIAL.md)" in readme


def test_tutorial_documents_stepwise_training_and_artifacts():
    tutorial = TUTORIAL.read_text(encoding="utf-8")

    for marker in (
        "configs/local/flashvsr_lq_proj.yaml",
        "configs/local/flashvsr_tcdecoder.yaml",
        "python -m distill_codec.cli probe",
        "python -m distill_codec.cli train",
        "metrics.jsonl",
        "checkpoints/step_XXXXXXXX.pt",
        "validation/step_XXXXXXXX.png",
        "tensorboard/",
        "tail -f",
        "--resume",
        "student_state",
        "预期输出",
        "如何判断成功",
        "shared_across_batch",
        "validation 和 probe",
        "center crop",
        "cached",
    ):
        assert marker in tutorial
    assert "PowerShell" not in tutorial
    assert "C:\\" not in tutorial


def test_tutorial_real_yaml_examples_enable_paired_augmentation():
    tutorial = TUTORIAL.read_text(encoding="utf-8")

    for heading in (
        "### `configs/local/flashvsr_lq_proj.yaml`",
        "### `configs/local/flashvsr_tcdecoder.yaml`",
    ):
        config = _yaml_after_heading(tutorial, heading)
        augmentation = config["data"]["augmentation"]

        assert augmentation == {
            "enabled": True,
            "shared_across_batch": True,
            "crop": {"enabled": True, "mode": "random"},
            "rotation": {
                "enabled": True,
                "mode": "continuous",
                "probability": 0.3,
                "degrees": [-5.0, 5.0],
                "interpolation": "bilinear",
                "padding_mode": "reflection",
            },
            "translation": {
                "enabled": True,
                "probability": 0.3,
                "max_fraction": [0.05, 0.05],
                "padding_mode": "reflection",
            },
        }


def test_tutorial_real_yaml_examples_preflight(tmp_path):
    tutorial = TUTORIAL.read_text(encoding="utf-8")
    lq_config = _yaml_after_heading(
        tutorial, "### `configs/local/flashvsr_lq_proj.yaml`"
    )
    decoder_config = _yaml_after_heading(
        tutorial, "### `configs/local/flashvsr_tcdecoder.yaml`"
    )

    assert lq_config["includes"] == ["../teachers/flashvsr_snapshot.yaml"]
    assert lq_config["recipe"]["name"] == "flashvsr_lq_proj_distill"
    assert "student_condition_encoder" in lq_config["components"]

    assert decoder_config["includes"] == [
        "../teachers/wan_snapshot.yaml",
        "../teachers/flashvsr_snapshot.yaml",
        "../students/private_blackbox.yaml",
    ]
    assert decoder_config["recipe"]["name"] == "flashvsr_decoder_conditional_student"
    assert decoder_config["latent_provider"] == {
        "type": "teacher_encoder",
        "source": "gt",
    }

    config_root = tmp_path / "configs"
    for directory in ("local", "teachers", "students"):
        (config_root / directory).mkdir(parents=True, exist_ok=True)
    for source in (
        Path("configs/teachers/wan_snapshot.yaml"),
        Path("configs/teachers/flashvsr_snapshot.yaml"),
        Path("configs/students/private_blackbox.yaml"),
    ):
        destination = config_root / source.relative_to("configs")
        shutil.copyfile(source, destination)

    for filename, values in (
        ("flashvsr_lq_proj.yaml", lq_config),
        ("flashvsr_tcdecoder.yaml", decoder_config),
    ):
        config_path = config_root / "local" / filename
        config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        preflight_config(load_config(config_path))


def test_tutorial_uses_snapshot_files_that_exist():
    for path in (
        "configs/teachers/wan_snapshot.yaml",
        "configs/teachers/flashvsr_snapshot.yaml",
        "configs/students/private_blackbox.yaml",
        "third_party/wan/wan_video_vae.py",
        "third_party/flashvsr/utils.py",
        "third_party/flashvsr/TCDecoder.py",
    ):
        assert Path(path).is_file(), path
