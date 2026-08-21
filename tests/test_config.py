from pathlib import Path

import pytest
import torch
import yaml

from distill_codec.config import apply_overrides, build_components, load_config, preflight_config
from distill_codec.contracts import ContractError
from distill_codec.recipes import build_recipe
from private_codec.bridge import PrivateConditionalDecoderBridge, PrivateEncoderBridge
from tests.support_factories import private_bridge_calls


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
        compatibility_every=config["recipe"].get("compatibility_every", 1),
    )

    assert recipe.name == config["recipe"]["name"]
    assert list(recipe.trainable_parameters())


def test_flashvsr_lq_proj_repeats_five_frames_for_causal_warmup():
    config = load_config("configs/smoke/flashvsr_lq_proj.yaml")
    components = build_components(config)

    assert components["teacher_condition_encoder"].temporal_frames == 5


def test_flashvsr_replacement_recipes_target_lq_proj_and_tc_decoder():
    lq_config = load_config("configs/smoke/flashvsr_lq_proj.yaml")
    decoder_config = load_config("configs/smoke/flashvsr_decoder_conditional.yaml")

    assert lq_config["recipe"]["name"] == "flashvsr_lq_proj_distill"
    assert set(lq_config["components"]) >= {
        "teacher_condition_encoder",
        "student_condition_encoder",
    }
    assert decoder_config["recipe"]["name"] == "flashvsr_decoder_conditional_student"
    assert set(decoder_config["components"]) >= {
        "tc_decoder",
        "conditional_student_decoder",
    }


def test_recipe_compatibility_interval_is_loaded_from_config():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["recipe"]["compatibility_every"] = 4

    components = build_components(config)
    recipe = build_recipe(
        config["recipe"]["name"],
        components,
        config["recipe"].get("weights"),
        source=config["recipe"].get("source", "gt"),
        compatibility_every=config["recipe"].get("compatibility_every", 1),
    )

    assert recipe.compatibility_every == 4


def test_component_builder_requires_explicit_color_contract():
    config = load_config("configs/smoke/wan_encoder.yaml")
    del config["color"]

    with pytest.raises(ContractError, match="config requires color"):
        build_components(config)


def _private_encoder_component():
    return {
        "backend": "external",
        "factory": "private_codec.factories:create_encoder",
        "teacher_reference": "auto",
        "kwargs": {
            "builder": "tests.support_factories:build_private_bridge_network",
            "runner": "tests.support_factories:run_private_encoder",
        },
        "adapter": {
            "kind": "encoder",
            "input_mode": "rgb",
            "latent_temporal_frames": "teacher",
        },
    }


def _private_conditional_decoder_component():
    return {
        "backend": "external",
        "factory": "private_codec.factories:create_conditional_decoder",
        "teacher_reference": "auto",
        "kwargs": {
            "builder": "tests.support_factories:build_private_bridge_network",
            "runner": "tests.support_factories:run_private_decoder",
        },
        "adapter": {
            "kind": "decoder",
            "output_mode": "rgb",
            "accepts_condition": True,
        },
    }


def test_component_builder_derives_encoder_teacher_reference_without_mutating_config():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["data"].update({"lq_size": [128, 192], "gt_size": [256, 320]})
    config["components"]["student_encoder"] = _private_encoder_component()

    components = build_components(config)

    bridge = components["student_encoder"].module
    assert isinstance(bridge, PrivateEncoderBridge)
    assert bridge.teacher_reference == {
        "role": "encoder",
        "inputs": {
            "rgb": {
                "layout": "BCHW",
                "shape": [None, 3, 256, 320],
                "source": "gt",
            }
        },
        "outputs": {
            "latent": {
                "layout": "BCHW",
                "shape": [None, 16, 32, 40],
            }
        },
    }
    assert "teacher_reference" not in config["components"]["student_encoder"]["kwargs"]


def test_component_builder_derives_conditional_decoder_teacher_reference():
    config = load_config("configs/smoke/flashvsr_decoder_conditional.yaml")
    config["data"].update({"lq_size": [128, 192], "gt_size": [256, 320]})
    config["components"]["conditional_student_decoder"] = (
        _private_conditional_decoder_component()
    )

    components = build_components(config)

    bridge = components["conditional_student_decoder"].module
    assert isinstance(bridge, PrivateConditionalDecoderBridge)
    assert bridge.teacher_reference == {
        "role": "conditional_decoder",
        "inputs": {
            "lq_rgb": {
                "layout": "BCHW",
                "shape": [None, 3, 256, 320],
                "source": "teacher_condition",
            },
            "dit_latent": {
                "layout": "BCHW",
                "shape": [None, 16, 32, 40],
            },
        },
        "outputs": {
            "rgb": {
                "layout": "BCHW",
                "shape": [None, 3, 256, 320],
            }
        },
    }


def test_component_builder_derives_bcthw_teacher_latent_reference():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["data"]["gt_size"] = [256, 320]
    config["latent_spec"].update({"layout": "BCTHW", "temporal_downsample": 2})
    config["components"]["student_encoder"] = _private_encoder_component()

    components = build_components(config)

    bridge = components["student_encoder"].module
    assert bridge.teacher_reference["outputs"]["latent"] == {
        "layout": "BCTHW",
        "shape": [None, 16, 2, 32, 40],
    }


def test_private_rgb_encoder_accepts_teacher_compatible_bcthw_output():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["data"]["gt_size"] = [256, 320]
    config["latent_spec"].update({"layout": "BCTHW", "temporal_downsample": 2})
    component = _private_encoder_component()
    component["kwargs"]["runner"] = "tests.support_factories:run_private_video_encoder"
    config["components"]["student_encoder"] = component

    components = build_components(config)
    rgb = torch.zeros(1, 3, 256, 320)
    latent = components["student_encoder"](rgb)

    assert latent.shape == (1, 16, 2, 32, 40)
    assert private_bridge_calls[-1]["rgb"] is rgb


def test_private_bcthw_validation_is_independent_of_teacher_reference_mode():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["latent_spec"].update({"layout": "BCTHW", "temporal_downsample": 2})
    auto_component = _private_encoder_component()
    auto_component["kwargs"]["runner"] = (
        "tests.support_factories:run_private_two_frame_encoder"
    )
    config["components"]["student_encoder"] = auto_component

    auto_encoder = build_components(config)["student_encoder"]
    auto_latent = auto_encoder(torch.zeros(1, 3, 64, 64))

    manual_component = _private_encoder_component()
    manual_component.pop("teacher_reference")
    manual_component["kwargs"].update(
        runner="tests.support_factories:run_private_two_frame_encoder",
        teacher_reference={"role": "manual_debug_only"},
    )
    config["components"]["student_encoder"] = manual_component

    manual_encoder = build_components(config)["student_encoder"]
    manual_latent = manual_encoder(torch.zeros(1, 3, 64, 64))

    assert auto_latent.shape == manual_latent.shape == (1, 16, 2, 8, 8)
    assert auto_encoder.latent_temporal_frames == manual_encoder.latent_temporal_frames == 3


@pytest.mark.parametrize("value", (0, -1, True, False, "two", 1.5))
def test_component_builder_rejects_invalid_latent_temporal_frames(value):
    config = load_config("configs/smoke/wan_encoder.yaml")
    component = _private_encoder_component()
    component["adapter"]["latent_temporal_frames"] = value
    config["components"]["student_encoder"] = component

    with pytest.raises(
        ContractError,
        match="student_encoder.*latent_temporal_frames.*teacher.*positive integer",
    ):
        build_components(config)


def test_component_builder_does_not_inject_teacher_reference_without_opt_in():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["components"]["student_encoder"] = {
        "backend": "external",
        "factory": "tests.support_factories:create_encoder",
        "kwargs": {"channels": 16},
        "adapter": {"kind": "encoder", "input_mode": "packed_6ch"},
    }

    components = build_components(config)

    assert components["student_encoder"].module.net[-1].out_channels == 16


def test_component_builder_rejects_automatic_and_explicit_teacher_reference():
    config = load_config("configs/smoke/wan_encoder.yaml")
    component = _private_encoder_component()
    component["kwargs"]["teacher_reference"] = {"role": "explicit"}
    config["components"]["student_encoder"] = component

    with pytest.raises(ContractError, match="student_encoder.*teacher_reference.*both"):
        build_components(config)


@pytest.mark.parametrize("size", (None, [256], [256, 0], [256, "320"]))
def test_component_builder_rejects_invalid_teacher_reference_image_size(size):
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["components"]["student_encoder"] = _private_encoder_component()
    if size is None:
        config["data"].pop("gt_size")
    else:
        config["data"]["gt_size"] = size

    with pytest.raises(ContractError, match=r"data\.gt_size.*height.*width"):
        build_components(config)


def test_teacher_components_are_frozen_even_if_config_omits_freeze():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["components"]["teacher_encoder"].pop("freeze")
    config["components"]["teacher_decoder"].pop("freeze")

    components = build_components(config)
    recipe = build_recipe("wan_encoder_distill", components)
    recipe.train()

    for name in ("teacher_encoder", "teacher_decoder"):
        component = recipe.components[name]
        assert not component.training
        assert all(not parameter.requires_grad for parameter in component.parameters())


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


@pytest.mark.parametrize(
    ("mutation", "field"),
    (
        (lambda config: config.pop("recipe"), "recipe.name"),
        (lambda config: config["data"].pop("gt_root"), "data.gt_root"),
        (
            lambda config: config["components"].pop("student_encoder"),
            "components.student_encoder",
        ),
    ),
)
def test_config_preflight_reports_missing_field_with_config_path(mutation, field):
    config = load_config("configs/smoke/wan_encoder.yaml")
    mutation(config)

    with pytest.raises(ContractError, match=rf"{field}.*wan_encoder.yaml"):
        preflight_config(config)


def test_config_preflight_rejects_unknown_recipe_before_component_construction():
    config = load_config("configs/smoke/wan_encoder.yaml")
    config["recipe"]["name"] = "unknown_recipe"

    with pytest.raises(ContractError, match="recipe.name.*unknown_recipe.*available"):
        preflight_config(config)


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


def test_offline_flashvsr_teacher_targets_construct_without_teacher_modules(tmp_path):
    latent_root = tmp_path / "latents"
    latent_root.mkdir()
    config = load_config("configs/smoke/flashvsr_decoder_conditional.yaml")
    torch.save({"latent_spec": config["latent_spec"]}, latent_root / "manifest.pt")
    config["latent_provider"] = {"type": "cached", "root": str(latent_root)}
    config["teacher_target_provider"] = {"type": "dataset_gt"}
    config["components"].pop("teacher_encoder")
    config["components"].pop("tc_decoder")

    preflight_config(config)
    components = build_components(config)
    recipe = build_recipe("flashvsr_decoder_conditional_student", components)

    assert set(recipe.components) == {
        "conditional_student_decoder",
        "latent_provider",
        "teacher_target_provider",
    }


def test_dataset_gt_teacher_targets_require_cached_latents():
    config = load_config("configs/smoke/flashvsr_decoder_conditional.yaml")
    config["teacher_target_provider"] = {"type": "dataset_gt"}

    with pytest.raises(
        ContractError,
        match="teacher_target_provider.*dataset_gt.*latent_provider.*cached",
    ):
        preflight_config(config)

    with pytest.raises(
        ContractError,
        match="teacher_target_provider.*dataset_gt.*latent_provider.*cached",
    ):
        build_components(config)


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
