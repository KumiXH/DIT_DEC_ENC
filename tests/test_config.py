from pathlib import Path

import pytest
import yaml

from distill_codec.config import apply_overrides, build_components, load_config
from distill_codec.contracts import ContractError
from distill_codec.recipes import build_recipe


SMOKE_CONFIGS = (
    "wan_encoder.yaml",
    "wan_decoder.yaml",
    "flashvsr_lq_proj.yaml",
    "flashvsr_decoder_unconditional.yaml",
    "flashvsr_decoder_conditional.yaml",
)


@pytest.mark.parametrize("filename", SMOKE_CONFIGS)
def test_smoke_configs_construct_recipe_components(filename):
    config = load_config(Path("configs/smoke") / filename)

    components = build_components(config)
    recipe = build_recipe(
        config["recipe"]["name"],
        components,
        config["recipe"].get("weights"),
        source=config["recipe"].get("source", "gt"),
    )

    assert recipe.name == config["recipe"]["name"]
    assert list(recipe.trainable_parameters())


def test_load_config_resolves_declared_paths_relative_to_yaml(tmp_path):
    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"lq_root": "../data/lq", "gt_root": "../data/gt"},
                "run": {"output_dir": "runs/example"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["data"]["lq_root"] == str((config_path.parent / "../data/lq").resolve())
    assert config["run"]["output_dir"] == str((config_path.parent / "runs/example").resolve())


def test_dotted_overrides_parse_yaml_scalars_and_lists():
    config = {"trainer": {"max_steps": 3}, "data": {"gt_size": [64, 64]}}

    updated = apply_overrides(config, ["trainer.max_steps=7", "data.gt_size=[256, 256]"])

    assert updated["trainer"]["max_steps"] == 7
    assert updated["data"]["gt_size"] == [256, 256]


def test_unknown_backend_has_actionable_error():
    with pytest.raises(ContractError, match="unknown backend.*mock.*external.*snapshot"):
        build_components(
            {
                "latent_spec": {
                    "family": "mock",
                    "channels": 16,
                    "layout": "BCHW",
                    "spatial_downsample": 8,
                    "temporal_downsample": 1,
                    "normalization": "mock",
                },
                "color": {},
                "components": {
                    "student_encoder": {
                        "backend": "mystery",
                        "adapter": {"kind": "encoder", "input_mode": "packed_6ch"},
                    }
                },
            }
        )

