from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from distill_codec.config import load_config, preflight_config


SMOKE_CONFIGS = (
    "wan_encoder.yaml",
    "wan_decoder.yaml",
    "wan_autoencoder.yaml",
    "flashvsr_vae_encoder.yaml",
    "flashvsr_lq_proj.yaml",
    "flashvsr_decoder_unconditional.yaml",
    "flashvsr_decoder_conditional.yaml",
)

REAL_CONFIG_TEMPLATES = (
    "configs/teachers/wan_snapshot.yaml",
    "configs/teachers/wan_external.yaml",
    "configs/teachers/flashvsr_snapshot.yaml",
    "configs/teachers/flashvsr_external.yaml",
    "configs/students/private_blackbox.yaml",
)


def test_bash_smoke_runner_parses():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")

    result = subprocess.run(
        [bash, "-n", "scripts/run_smoke.sh"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_bash_smoke_runner_contract():
    script = Path("scripts/run_smoke.sh").read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\n")
    for marker in (
        "set -euo pipefail",
        "${PYTHON_BIN:-python3}",
        "make-mock-data",
        " probe ",
        " train ",
        "--resume",
        "trainer.tensorboard=true",
    ):
        assert marker in script
    for config_name in SMOKE_CONFIGS:
        assert config_name in script
    assert "powershell" not in script.lower()
    assert "PYTHONPATH" not in script


def test_readme_documents_complete_ubuntu_workflow():
    readme = Path("README.md").read_text(encoding="utf-8")

    for marker in (
        "source .venv/bin/activate",
        "./scripts/run_smoke.sh",
        "$HOME/dit_codec/LQ",
        "$HOME/dit_codec/GT",
        "$HOME/dit_codec/weights",
        '--resume "$HOME/dit_codec/runs/',
        "curl --fail --location --retry 5",
        "sha256sum -c SHA256SUMS",
        "shared_across_batch: true",
        "validation 和 `probe`",
    ):
        assert marker in readme


def test_readme_names_the_drop_in_flashvsr_replacement_path():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "LQ_proj_in -> student_condition_encoder" in readme
    assert "TCDecoder  -> conditional_student_decoder" in readme
    assert "不是可直接替换 TCDecoder 的等价接口" in readme


def _readme_yaml_block(readme: str, heading: str) -> dict:
    heading_offset = readme.index(heading)
    block_start = readme.index("```yaml\n", heading_offset) + len("```yaml\n")
    block_end = readme.index("\n```", block_start)
    values = yaml.safe_load(readme[block_start:block_end])
    assert isinstance(values, dict)
    return values


def test_readme_only_uses_existing_or_created_config_paths():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "../data/paired_256.yaml" not in readme
    assert "--config config.yaml" not in readme
    assert readme.index("mkdir -p configs/local") < readme.index(
        "--config configs/local/wan_encoder.yaml"
    )


def test_readme_explains_real_probe_component_loading():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Mock 配置的 `probe` 不需要真实教师权重" in readme
    assert "真实配置的 `probe` 会先构造教师和学生组件" in readme


def test_readme_real_flashvsr_configs_match_component_contracts(tmp_path):
    readme = Path("README.md").read_text(encoding="utf-8")
    lq_config = _readme_yaml_block(readme, "#### `configs/local/flashvsr_lq_proj.yaml`")
    decoder_config = _readme_yaml_block(readme, "#### `configs/local/flashvsr_tcdecoder.yaml`")

    assert lq_config["includes"] == ["../teachers/flashvsr_snapshot.yaml"]
    assert lq_config["recipe"]["name"] == "flashvsr_lq_proj_distill"
    assert lq_config["components"]["student_condition_encoder"]["adapter"]["kind"] == (
        "condition_encoder"
    )

    assert decoder_config["includes"] == [
        "../teachers/wan_snapshot.yaml",
        "../teachers/flashvsr_snapshot.yaml",
        "../students/private_blackbox.yaml",
    ]
    assert decoder_config["recipe"]["name"] == "flashvsr_decoder_conditional_student"
    assert decoder_config["latent_provider"] == {"type": "teacher_encoder", "source": "gt"}

    config_root = tmp_path / "configs"
    for directory in ("local", "teachers", "students"):
        (config_root / directory).mkdir(parents=True, exist_ok=True)
    for source in (
        Path("configs/teachers/wan_snapshot.yaml"),
        Path("configs/teachers/flashvsr_snapshot.yaml"),
        Path("configs/students/private_blackbox.yaml"),
    ):
        destination = config_root / source.relative_to("configs")
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for filename, values in (
        ("flashvsr_lq_proj.yaml", lq_config),
        ("flashvsr_tcdecoder.yaml", decoder_config),
    ):
        config_path = config_root / "local" / filename
        config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        preflight_config(load_config(config_path))


def test_real_config_templates_use_posix_example_paths():
    for config_path in REAL_CONFIG_TEMPLATES:
        content = Path(config_path).read_text(encoding="utf-8")
        assert "D:/" not in content, config_path
        assert "~/dit_codec/" in content, config_path


def test_shell_scripts_are_checked_out_with_lf_endings():
    attributes = Path(".gitattributes").read_text(encoding="utf-8")

    assert "*.sh text eol=lf" in attributes
