import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from distill_codec.contracts import ContractError
from distill_codec.config import build_components, load_config
from distill_codec.data import create_mock_dataset
from distill_codec.recipes import build_recipe
from distill_codec.trainer import Trainer


def _training_config(tmp_path: Path, max_steps: int):
    paths = create_mock_dataset(tmp_path / "data", count=4, size=(32, 32), seed=5)
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["data"].update(
        {
            "lq_root": str(paths.lq_root),
            "gt_root": str(paths.gt_root),
            "lq_size": [32, 32],
            "gt_size": [32, 32],
        }
    )
    config["run"]["output_dir"] = str(tmp_path / "run")
    config["trainer"].update(
        {
            "max_steps": max_steps,
            "batch_size": 2,
            "validate_every": 1,
            "checkpoint_every": 1,
            "tensorboard": False,
        }
    )
    return config


def _make_trainer(config):
    components = build_components(config)
    recipe = build_recipe(
        config["recipe"]["name"],
        components,
        config["recipe"].get("weights"),
        source=config["recipe"].get("source", "gt"),
        compatibility_every=config["recipe"].get("compatibility_every", 1),
    )
    return Trainer(config, recipe)


def test_trainer_updates_student_saves_artifacts_and_resumes(tmp_path):
    config = _training_config(tmp_path, max_steps=2)
    trainer = _make_trainer(config)
    first_parameter = next(trainer.recipe.trainable_parameters()).detach().clone()

    result = trainer.fit()

    changed_parameter = next(trainer.recipe.trainable_parameters()).detach()
    assert result.global_step == 2
    assert not torch.equal(first_parameter, changed_parameter)
    assert result.latest_checkpoint.is_file()
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert sorted((tmp_path / "run" / "validation").glob("step_*.png"))

    events = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert {event["phase"] for event in events} >= {"train", "validation"}
    assert all("global_step" in event for event in events)

    payload = torch.load(result.latest_checkpoint, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 2
    assert set(payload["student_state"]) == {"student_encoder"}
    assert "teacher_encoder" not in payload["student_state"]
    assert payload["contracts"]["latent_spec"]["channels"] == 16

    resumed_config = deepcopy(config)
    resumed_config["trainer"]["max_steps"] = 3
    resumed_trainer = _make_trainer(resumed_config)
    resumed = resumed_trainer.fit(resume=result.latest_checkpoint)

    assert resumed.global_step == 3
    assert resumed.start_step == 2


def test_multi_component_resume_is_stable_across_python_hash_seeds(tmp_path):
    paths = create_mock_dataset(tmp_path / "data", count=4, size=(32, 32), seed=5)
    config_path = Path("configs/smoke/wan_autoencoder.yaml").resolve()
    output_dir = tmp_path / "run"
    pythonpath = os.pathsep.join(
        filter(None, (str(Path("src").resolve()), os.environ.get("PYTHONPATH")))
    )

    def run_training(hash_seed: int, max_steps: int, resume: Path | None = None):
        command = [
            sys.executable,
            "-m",
            "distill_codec.cli",
            "train",
            "--config",
            str(config_path),
            "--set",
            f"data.lq_root={paths.lq_root}",
            "--set",
            f"data.gt_root={paths.gt_root}",
            "--set",
            "data.lq_size=[32,32]",
            "--set",
            "data.gt_size=[32,32]",
            "--set",
            f"run.output_dir={output_dir}",
            "--set",
            f"trainer.max_steps={max_steps}",
            "--set",
            "trainer.tensorboard=false",
        ]
        if resume is not None:
            command.extend(("--resume", str(resume)))
        env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "PYTHONPATH": pythonpath}
        return subprocess.run(command, capture_output=True, text=True, env=env, check=False)

    first = run_training(hash_seed=1, max_steps=1)
    assert first.returncode == 0, first.stdout + first.stderr
    checkpoint = output_dir / "checkpoints" / "step_00000001.pt"

    resumed = run_training(hash_seed=2, max_steps=2, resume=checkpoint)

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (output_dir / "checkpoints" / "step_00000002.pt").is_file()


def test_trainer_builds_configured_adam_optimizer(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    config["trainer"]["optimizer"] = "adam"

    trainer = _make_trainer(config)

    assert type(trainer.optimizer) is torch.optim.Adam


def test_trainer_rejects_unknown_optimizer(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    config["trainer"]["optimizer"] = "sgd"

    with pytest.raises(ContractError, match="unsupported optimizer 'sgd'"):
        _make_trainer(config)


def test_trainer_preflights_all_cached_latents_before_training(tmp_path):
    paths = create_mock_dataset(tmp_path / "data", count=2, size=(32, 32), seed=5)
    latent_root = tmp_path / "latents"
    latent_root.mkdir()
    config = load_config("configs/smoke/wan_decoder.yaml")
    config["data"].update(
        {
            "lq_root": str(paths.lq_root),
            "gt_root": str(paths.gt_root),
            "lq_size": [32, 32],
            "gt_size": [32, 32],
        }
    )
    config["run"]["output_dir"] = str(tmp_path / "run")
    config["latent_provider"] = {"type": "cached", "root": str(latent_root)}
    torch.save({"latent_spec": config["latent_spec"]}, latent_root / "manifest.pt")

    with pytest.raises(ContractError, match="cached latent sample set mismatch.*missing"):
        _make_trainer(config)


def test_trainer_keeps_only_latest_configured_checkpoints(tmp_path):
    config = _training_config(tmp_path, max_steps=3)
    config["trainer"]["keep_last_checkpoints"] = 2

    _make_trainer(config).fit()

    checkpoints = sorted((tmp_path / "run" / "checkpoints").glob("step_*.pt"))
    assert [path.name for path in checkpoints] == ["step_00000002.pt", "step_00000003.pt"]


def test_trainer_rejects_nonpositive_checkpoint_retention(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    config["trainer"]["keep_last_checkpoints"] = 0

    with pytest.raises(ContractError, match="keep_last_checkpoints must be positive"):
        _make_trainer(config)


def test_resume_rejects_changed_optimizer(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    checkpoint = _make_trainer(config).fit().latest_checkpoint
    incompatible = deepcopy(config)
    incompatible["trainer"]["max_steps"] = 2
    incompatible["trainer"]["optimizer"] = "adam"

    with pytest.raises(ValueError, match="incompatible training contract.*optimizer"):
        _make_trainer(incompatible).fit(resume=checkpoint)


@pytest.mark.parametrize(
    ("path", "value", "field"),
    (
        (("trainer", "batch_size"), 1, "batch_size"),
        (("trainer", "gradient_accumulation"), 2, "gradient_accumulation"),
        (("run", "seed"), 99, "seed"),
        (("data", "lq_root"), "__alternate_existing_lq__", "lq_root"),
        (("recipe", "weights"), {"latent": 2.0}, "weights"),
    ),
)
def test_resume_rejects_changed_reproducibility_contract(tmp_path, path, value, field):
    config = _training_config(tmp_path, max_steps=1)
    checkpoint = _make_trainer(config).fit().latest_checkpoint
    incompatible = deepcopy(config)
    incompatible["trainer"]["max_steps"] = 2
    if value == "__alternate_existing_lq__":
        alternate = create_mock_dataset(tmp_path / "alternate", count=4, size=(32, 32), seed=6)
        value = str(alternate.lq_root)
    incompatible[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=rf"incompatible training contract.*{field}"):
        _make_trainer(incompatible).fit(resume=checkpoint)


def test_resume_allows_explicit_semantic_contract_override_but_never_shape_override(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    checkpoint = _make_trainer(config).fit().latest_checkpoint
    semantic_override = deepcopy(config)
    semantic_override["trainer"]["max_steps"] = 2
    semantic_override["trainer"]["allow_contract_override"] = True
    semantic_override["latent_spec"]["family"] = "intentionally_changed_family"
    semantic_override["color"]["matrix"] = "bt601"

    resumed = _make_trainer(semantic_override).fit(resume=checkpoint)

    assert resumed.start_step == 1
    shape_override = deepcopy(semantic_override)
    shape_override["latent_spec"]["channels"] = 8
    with pytest.raises(ValueError, match="latent shape contract"):
        _make_trainer(shape_override).fit(resume=checkpoint)


def test_resume_rejects_changed_optimizer_parameter_order(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    checkpoint = _make_trainer(config).fit().latest_checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["optimizer_parameter_names"] = ["unexpected.parameter"]
    incompatible_checkpoint = checkpoint.with_name("incompatible_parameter_order.pt")
    torch.save(payload, incompatible_checkpoint)
    resumed_config = deepcopy(config)
    resumed_config["trainer"]["max_steps"] = 2

    with pytest.raises(ValueError, match="optimizer parameter order"):
        _make_trainer(resumed_config).fit(resume=incompatible_checkpoint)


def test_cosine_resume_matches_uninterrupted_with_fixed_scheduler_horizon(tmp_path):
    uninterrupted_config = _training_config(tmp_path / "uninterrupted_cosine", max_steps=4)
    uninterrupted_config["trainer"].update(
        {"scheduler": "cosine", "scheduler_max_steps": 4, "validate_every": 4}
    )
    uninterrupted = _make_trainer(uninterrupted_config).fit()
    uninterrupted_payload = torch.load(uninterrupted.latest_checkpoint, map_location="cpu", weights_only=False)

    staged_config = _training_config(tmp_path / "staged_cosine", max_steps=2)
    staged_config["trainer"].update(
        {"scheduler": "cosine", "scheduler_max_steps": 4, "validate_every": 2}
    )
    first_stage = _make_trainer(staged_config).fit()
    resumed_config = deepcopy(staged_config)
    resumed_config["trainer"].update({"max_steps": 4, "validate_every": 4})
    resumed = _make_trainer(resumed_config).fit(resume=first_stage.latest_checkpoint)
    resumed_payload = torch.load(resumed.latest_checkpoint, map_location="cpu", weights_only=False)

    assert resumed_payload["scheduler"]["T_max"] == 4
    for component, state in uninterrupted_payload["student_state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed_payload["student_state"][component][name])


def test_resume_rejects_changed_cosine_scheduler_horizon(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    config["trainer"].update({"scheduler": "cosine", "scheduler_max_steps": 4})
    checkpoint = _make_trainer(config).fit().latest_checkpoint
    incompatible = deepcopy(config)
    incompatible["trainer"].update({"max_steps": 2, "scheduler_max_steps": 8})

    with pytest.raises(ValueError, match="incompatible training contract.*scheduler_max_steps"):
        _make_trainer(incompatible).fit(resume=checkpoint)


def test_resume_discards_checkpoints_from_abandoned_future_trajectory(tmp_path):
    config = _training_config(tmp_path, max_steps=3)
    completed = _make_trainer(config).fit()
    checkpoint_one = completed.output_dir / "checkpoints" / "step_00000001.pt"
    resumed_config = deepcopy(config)
    resumed_config["trainer"].update({"max_steps": 2, "keep_last_checkpoints": 2})

    _make_trainer(resumed_config).fit(resume=checkpoint_one)

    checkpoints = sorted((completed.output_dir / "checkpoints").glob("step_*.pt"))
    assert [path.name for path in checkpoints] == ["step_00000001.pt", "step_00000002.pt"]


def test_noop_resume_applies_checkpoint_retention(tmp_path):
    config = _training_config(tmp_path, max_steps=3)
    completed = _make_trainer(config).fit()
    resumed_config = deepcopy(config)
    resumed_config["trainer"]["keep_last_checkpoints"] = 2

    resumed = _make_trainer(resumed_config).fit(resume=completed.latest_checkpoint)

    assert resumed.start_step == resumed.global_step == 3
    checkpoints = sorted((completed.output_dir / "checkpoints").glob("step_*.pt"))
    assert [path.name for path in checkpoints] == ["step_00000002.pt", "step_00000003.pt"]


def test_resume_rejects_changed_latent_contract(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    checkpoint = _make_trainer(config).fit().latest_checkpoint
    incompatible = deepcopy(config)
    incompatible["latent_spec"]["family"] = "different_family"

    trainer = _make_trainer(incompatible)

    with pytest.raises(ValueError, match="incompatible latent contract"):
        trainer.fit(resume=checkpoint)


def test_resume_matches_uninterrupted_training_data_order_and_parameters(tmp_path):
    uninterrupted_config = _training_config(tmp_path / "uninterrupted", max_steps=4)
    uninterrupted_config["trainer"].update({"gradient_accumulation": 2, "validate_every": 4})
    uninterrupted = _make_trainer(uninterrupted_config).fit()
    uninterrupted_payload = torch.load(uninterrupted.latest_checkpoint, map_location="cpu", weights_only=False)

    staged_config = _training_config(tmp_path / "staged", max_steps=2)
    staged_config["trainer"].update({"gradient_accumulation": 2, "validate_every": 2})
    first_stage = _make_trainer(staged_config).fit()
    resumed_config = deepcopy(staged_config)
    resumed_config["trainer"].update({"max_steps": 4, "validate_every": 4})
    resumed = _make_trainer(resumed_config).fit(resume=first_stage.latest_checkpoint)
    resumed_payload = torch.load(resumed.latest_checkpoint, map_location="cpu", weights_only=False)

    assert resumed_payload["data_batches_consumed"] == 8
    for component, state in uninterrupted_payload["student_state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed_payload["student_state"][component][name])
