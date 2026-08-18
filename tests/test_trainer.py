import json
from copy import deepcopy
from pathlib import Path

import torch

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


def test_resume_rejects_changed_latent_contract(tmp_path):
    config = _training_config(tmp_path, max_steps=1)
    checkpoint = _make_trainer(config).fit().latest_checkpoint
    incompatible = deepcopy(config)
    incompatible["latent_spec"]["family"] = "different_family"

    trainer = _make_trainer(incompatible)

    import pytest

    with pytest.raises(ValueError, match="incompatible latent contract"):
        trainer.fit(resume=checkpoint)

