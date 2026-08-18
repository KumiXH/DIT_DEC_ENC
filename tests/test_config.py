from pathlib import Path

import pytest
import yaml

from distill_codec.config import apply_overrides, build_components, load_config
from distill_codec.contracts import ContractError
from distill_codec.recipes import build_recipe


SMOKE_CONFIGS = (
    "wan_encoder.yaml",
    "wan_decoder.yaml",
    "wan_autoencoder.yaml",
    "flashvsr_vae_encoder.yaml",
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


def test_flashvsr_lq_proj_repeats_five_frames_for_causal_warmup():
    config = load_config("configs/smoke/flashvsr_lq_proj.yaml")
    components = build_components(config)

    assert components["teacher_condition_encoder"].temporal_frames == 5


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


def test_config_includes_merge_layers_and_main_file_wins(tmp_path):
    teacher = tmp_path / "teacher.yaml"
    student = tmp_path / "student.yaml"
    main = tmp_path / "main.yaml"
    teacher.write_text(
        yaml.safe_dump({"components": {"teacher": {"backend": "mock"}}, "trainer": {"max_steps": 10}}),
        encoding="utf-8",
    )
    student.write_text(
        yaml.safe_dump({"components": {"student": {"backend": "mock"}}, "trainer": {"batch_size": 2}}),
        encoding="utf-8",
    )
    main.write_text(
        yaml.safe_dump(
            {
                "includes": ["teacher.yaml", "student.yaml"],
                "trainer": {"max_steps": 3},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(main)

    assert set(config["components"]) == {"teacher", "student"}
    assert config["trainer"] == {"max_steps": 3, "batch_size": 2}


def test_config_include_cycle_is_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("includes: [second.yaml]\n", encoding="utf-8")
    second.write_text("includes: [first.yaml]\n", encoding="utf-8")

    with pytest.raises(ContractError, match="include cycle"):
        load_config(first)


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


def test_cached_latent_provider_constructs_without_teacher_encoder(tmp_path):
    latent_root = tmp_path / "latents"
    latent_root.mkdir()
    latent_spec = {
        "family": "mock",
        "channels": 16,
        "layout": "BCHW",
        "spatial_downsample": 8,
        "temporal_downsample": 1,
        "normalization": "mock",
    }
    import torch

    torch.save({"latent_spec": latent_spec}, latent_root / "manifest.pt")
    config = {
        "latent_spec": latent_spec,
        "color": {},
        "latent_provider": {"type": "cached", "root": str(latent_root)},
        "components": {
            "teacher_decoder": {
                "backend": "mock",
                "model": "wan_decoder",
                "freeze": True,
                "adapter": {"kind": "decoder", "output_mode": "rgb"},
            },
            "student_decoder": {
                "backend": "mock",
                "model": "student_decoder",
                "adapter": {"kind": "decoder", "output_mode": "sparse_yuv"},
            },
        },
    }

    components = build_components(config)
    recipe = build_recipe("wan_decoder_distill", components)

    assert "latent_provider" in recipe.components


def test_standard_trainer_rejects_dataset_latent_provider(tmp_path):
    from distill_codec.data import create_mock_dataset
    from distill_codec.trainer import Trainer

    paths = create_mock_dataset(tmp_path / "data", count=2, size=(32, 32))
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
    config["latent_provider"] = {"type": "dataset"}
    components = build_components(config)
    recipe = build_recipe("wan_decoder_distill", components)

    with pytest.raises(ContractError, match="standard paired-image Trainer.*dataset latent provider"):
        Trainer(config, recipe)


def test_component_builder_skips_unused_broken_factories_for_selected_recipe():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["components"]["unused_decoder"] = {
        "backend": "external",
        "factory": "package_that_does_not_exist:create_decoder",
        "adapter": {"kind": "decoder", "output_mode": "rgb"},
    }

    components = build_components(config)

    assert "unused_decoder" not in components


def test_component_builder_rejects_checkpoint_sha256_mismatch(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["components"]["student_encoder"]["checkpoint"] = str(checkpoint)
    config["components"]["student_encoder"]["sha256"] = "0" * 64

    with pytest.raises(ContractError, match="SHA256 mismatch.*student_encoder"):
        build_components(config)
