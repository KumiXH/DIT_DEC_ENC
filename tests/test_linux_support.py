from pathlib import Path
import shutil
import subprocess

import pytest


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
        "/data/dit_codec/LQ",
        "/data/dit_codec/GT",
        "/data/dit_codec/weights",
        "--resume /data/dit_codec/runs/",
        "curl --fail --location --retry 5",
        "sha256sum -c SHA256SUMS",
    ):
        assert marker in readme


def test_readme_names_the_drop_in_flashvsr_replacement_path():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "LQ_proj_in -> student_condition_encoder" in readme
    assert "TCDecoder  -> conditional_student_decoder" in readme
    assert "不是可直接替换 TCDecoder 的等价接口" in readme


def test_real_config_templates_use_posix_example_paths():
    for config_path in REAL_CONFIG_TEMPLATES:
        content = Path(config_path).read_text(encoding="utf-8")
        assert "D:/" not in content, config_path
        assert "/data/dit_codec/" in content, config_path


def test_shell_scripts_are_checked_out_with_lf_endings():
    attributes = Path(".gitattributes").read_text(encoding="utf-8")

    assert "*.sh text eol=lf" in attributes
