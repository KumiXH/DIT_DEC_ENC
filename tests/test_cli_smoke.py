import json
from pathlib import Path

from distill_codec.cli import main


def test_cli_make_data_probe_train_and_resume(tmp_path, capsys):
    data_root = tmp_path / "data"
    assert main(["make-mock-data", "--output", str(data_root), "--count", "4", "--size", "32"]) == 0
    assert (data_root / "lq" / "scene_00" / "frame_0000.png").is_file()

    run_root = tmp_path / "run"
    common = [
        "--config",
        "configs/smoke/flashvsr_decoder_unconditional.yaml",
        "--set",
        f"data.lq_root={data_root / 'lq'}",
        "--set",
        f"data.gt_root={data_root / 'gt'}",
        "--set",
        "data.lq_size=[32,32]",
        "--set",
        "data.gt_size=[32,32]",
        "--set",
        f"run.output_dir={run_root}",
    ]

    assert main(["probe", *common]) == 0
    probe = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert probe["recipe"] == "flashvsr_decoder_unconditional_student"
    assert probe["trainable_parameters"] > 0

    assert main(["train", *common, "--set", "trainer.max_steps=1"]) == 0
    checkpoint = run_root / "checkpoints" / "step_00000001.pt"
    assert checkpoint.is_file()

    assert main(
        [
            "train",
            *common,
            "--set",
            "trainer.max_steps=2",
            "--resume",
            str(checkpoint),
        ]
    ) == 0
    assert (run_root / "checkpoints" / "step_00000002.pt").is_file()


def test_cli_probe_conditional_and_lq_proj_recipes(tmp_path):
    data_root = tmp_path / "data"
    main(["make-mock-data", "--output", str(data_root), "--count", "2", "--size", "32"])
    for config in (
        "configs/smoke/flashvsr_decoder_conditional.yaml",
        "configs/smoke/flashvsr_lq_proj.yaml",
    ):
        code = main(
            [
                "probe",
                "--config",
                config,
                "--set",
                f"data.lq_root={data_root / 'lq'}",
                "--set",
                f"data.gt_root={data_root / 'gt'}",
                "--set",
                "data.lq_size=[32,32]",
                "--set",
                "data.gt_size=[32,32]",
            ]
        )
        assert code == 0

