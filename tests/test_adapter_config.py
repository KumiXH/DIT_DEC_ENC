from distill_codec.config import build_components, load_config


def test_flashvsr_condition_sampling_contract_is_loaded_from_config():
    config = load_config("configs/smoke/flashvsr_lq_proj.yaml")

    components = build_components(config)

    condition_spec = components["teacher_condition_encoder"].condition_spec
    assert condition_spec.spatial_downsample == 8
    assert condition_spec.temporal_downsample == 5


def test_component_builder_passes_frame_selection_to_encoder_and_decoder():
    config = load_config("configs/smoke/wan_autoencoder.yaml")
    config["components"]["teacher_encoder"]["adapter"]["frame_selection"] = "first"
    config["components"]["teacher_decoder"]["adapter"]["frame_selection"] = "last"

    components = build_components(config)

    assert components["teacher_encoder"].frame_selection == "first"
    assert components["teacher_decoder"].frame_selection == "last"
