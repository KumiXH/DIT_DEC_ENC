# FlashVSR Golden YAML Design

## Goal

Add two self-contained, runnable Ubuntu/Linux YAML examples for distilling FlashVSR `LQ_proj_in` and `TCDecoder`, plus one fully annotated reference YAML that documents all relevant configuration fields and choices.

## Deliverables

### Runnable golden examples

- `configs/examples/flashvsr_lq_proj_in_golden.yaml`
- `configs/examples/flashvsr_tcdecoder_golden.yaml`

Each golden example will be self-contained. It will not use `includes`, so the teacher source path, teacher checkpoint, student factory, optional student checkpoint, image paths, augmentation, trainer, and output directory are visible in one file.

Both examples will use Ubuntu paths under `~/dit_codec/` and will pass `load_config` plus `preflight_config` without requiring the referenced weight files to exist. Component construction and `probe` will still require the real source packages, factories, and checkpoints.

### Annotated reference

- `configs/reference/flashvsr_distillation_all_options.yaml`

This file is a documentation dictionary, not a training entrypoint. It will be valid YAML and will explicitly say not to pass it to `distill_codec.cli train`. It will separate:

- shared tensor, color, data, augmentation, trainer, and run settings;
- the complete `LQ_proj_in` teacher and student component options;
- the complete `TCDecoder` teacher, Wan latent provider, and student component options;
- supported alternatives and valid values.

Every meaningful configuration line will have an adjacent Chinese comment explaining its purpose. Where a field has a finite option set, the comment will list the valid choices. Numeric starting values and allowed ranges will be stated where the framework enforces them.

## LQ_proj_in Golden Contract

The LQ example will use:

- recipe `flashvsr_lq_proj_distill`;
- frozen snapshot teacher `teacher_condition_encoder` created by `distill_codec.integrations.snapshots:create_lq_proj_in`;
- teacher source `third_party/flashvsr/utils.py`;
- teacher checkpoint `~/dit_codec/weights/LQ_proj_in.ckpt`;
- external student factory `my_encrypted_package.models:create_condition_encoder`;
- student checkpoint `null` by default, with an inline example for `~/dit_codec/weights/student_lq_proj_in.pth`;
- a five-frame adapter formed by repeating the single LQ image;
- `BNC`, feature dimension `1536`, spatial downsample `16`, and temporal downsample `5` condition semantics;
- paired LQ/GT paths at `~/dit_codec/LQ` and `~/dit_codec/GT`;
- `256x256` patches with paired crop, continuous rotation, and translation;
- CUDA training output at `~/dit_codec/runs/flashvsr_lq_proj_in_golden`.

The golden example will document the launch command:

```bash
python -m distill_codec.cli probe --config configs/examples/flashvsr_lq_proj_in_golden.yaml
python -m distill_codec.cli train --config configs/examples/flashvsr_lq_proj_in_golden.yaml
```

## TCDecoder Golden Contract

The decoder example will use:

- recipe `flashvsr_decoder_conditional_student`;
- frozen Wan teacher encoder as the online latent provider;
- Wan source `third_party/wan/wan_video_vae.py`;
- Wan checkpoint `~/dit_codec/weights/Wan2.1_VAE.pth`;
- frozen snapshot `tc_decoder` created by `distill_codec.integrations.snapshots:create_tc_decoder`;
- TCDecoder source `third_party/flashvsr/TCDecoder.py`;
- TCDecoder checkpoint `~/dit_codec/weights/TCDecoder.ckpt`;
- external conditional student factory `my_encrypted_package.models:create_conditional_decoder`;
- student checkpoint `null` by default, with an inline example for `~/dit_codec/weights/student_tcdecoder.pth`;
- teacher-encoder latent provider sourced from GT;
- sparse-YUV student output converted to RGB by the decoder adapter;
- the same paired `256x256` augmentation contract as the LQ example;
- CUDA training output at `~/dit_codec/runs/flashvsr_tcdecoder_golden`.

The golden example will document the launch command:

```bash
python -m distill_codec.cli probe --config configs/examples/flashvsr_tcdecoder_golden.yaml
python -m distill_codec.cli train --config configs/examples/flashvsr_tcdecoder_golden.yaml
```

## Student Weight Behavior

Both examples will use `checkpoint: null` for the trainable student. This means the external factory determines initialization. An inline comment will show the optional pretrained student checkpoint path. When a non-null student checkpoint is supplied, the framework loads it using the configured `strict` behavior before training.

Teacher checkpoints remain non-null because they define the distillation targets.

## Data And Augmentation Behavior

The two golden examples will use identical data defaults:

```yaml
data:
  lq_root: ~/dit_codec/LQ
  gt_root: ~/dit_codec/GT
  lq_size: [256, 256]
  gt_size: [256, 256]
```

Paired augmentation will be enabled with batch-shared parameters:

- random crop to `256x256` during training;
- continuous rotation with probability `0.3` and range `[-5.0, 5.0]` degrees;
- translation with probability `0.3` and maximum vertical/horizontal fractions `[0.05, 0.05]`;
- bilinear interpolation and reflection padding;
- deterministic center crop with no rotation or translation during validation and `probe`.

LQ and GT source images must have matching dimensions. Images smaller than the target patch fail preflight; larger images are cropped.

## Documentation Integration

`README.md` and `FLASHVSR_DISTILL_TUTORIAL.md` will link the three files. Documentation will state that:

- the two golden examples are direct training entrypoints;
- users must replace the external student factory with their encrypted package factory;
- `checkpoint: null` starts from factory initialization;
- the all-options YAML is reference-only;
- `probe` should be run before training.

## Validation

Tests will verify:

- both golden example files exist;
- both files parse through `load_config` and pass `preflight_config`;
- the LQ example exposes the expected teacher and student paths and recipe;
- the decoder example exposes the expected Wan, TCDecoder, student, and latent-provider paths;
- both student checkpoints default to `null`;
- both examples contain the agreed paired augmentation values;
- the reference file is valid YAML, marked as non-runnable, and contains comments for every meaningful configuration line;
- README and the FlashVSR tutorial link all three files.

Full verification will run the focused tests, complete Pytest suite, Ruff, mypy, Bash syntax check, and `git diff --check`.

## Out Of Scope

- Downloading or redistributing FlashVSR, Wan, or private student weights.
- Guessing the encrypted student package's actual Python module name or constructor arguments.
- Combining both recipes into one runnable training process.
- Changing model, adapter, recipe, or checkpoint loading behavior.
