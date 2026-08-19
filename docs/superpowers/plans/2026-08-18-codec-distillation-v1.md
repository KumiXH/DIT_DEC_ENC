# Codec Distillation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable PyTorch codec-distillation framework with mock students/teachers, paired LQ/GT data, Wan and FlashVSR recipes, external factories, color adapters, checkpoints, validation, and documented third-party snapshot provenance.

**Architecture:** Components communicate through explicit batch, latent, condition, and color contracts. Model-specific adapters normalize Wan, FlashVSR, mock, and encrypted external modules behind recipe interfaces, while one trainer executes encoder, decoder, and condition distillation without knowing model internals.

**Tech Stack:** Python 3.10+, PyTorch 2.x, PyYAML, Pillow, pytest, TensorBoard (optional runtime dependency with an explicit error when requested but unavailable).

---

## File Map

- `pyproject.toml`: packaging, dependencies, CLI entry points, pytest settings.
- `src/distill_codec/contracts.py`: dataclass contracts and compatibility validation.
- `src/distill_codec/color.py`: differentiable RGB/YUV conversion and the user's packed/sparse layouts.
- `src/distill_codec/data.py`: strict paired-image dataset and mock-data generator.
- `src/distill_codec/factories.py`: import-string factories and checkpoint loading.
- `src/distill_codec/models/mock.py`: trainable mock students and deterministic frozen teachers.
- `src/distill_codec/adapters.py`: student/teacher wrappers and temporal/output normalization.
- `src/distill_codec/losses.py`: latent, image, edge, and statistics losses.
- `src/distill_codec/metrics.py`: PSNR, SSIM, and validation image grids.
- `src/distill_codec/recipes.py`: Wan encoder/decoder, FlashVSR condition, and two TCDecoder recipes.
- `src/distill_codec/checkpoint.py`: complete training-state round trip with contract checks.
- `src/distill_codec/config.py`: YAML loading, overrides, and component construction.
- `src/distill_codec/trainer.py`: single-device training, AMP, accumulation, validation, logging, and resume.
- `src/distill_codec/cli.py`: `train`, `probe`, and `make-mock-data` commands.
- `configs/`: runnable smoke configs plus external Wan/FlashVSR/private templates.
- `third_party/`: pinned-source manifests, licenses, and vendored required files.
- `tests/`: unit and end-to-end smoke tests.

### Task 1: Package Skeleton and Contract Types

**Files:**
- Create: `pyproject.toml`
- Create: `src/distill_codec/__init__.py`
- Create: `src/distill_codec/contracts.py`
- Test: `tests/test_contracts.py`

- [ ] Write tests constructing compatible and incompatible `LatentSpec`, `ConditionSpec`, `ColorSpec`, and `DistillBatch` values.
- [ ] Run `pytest tests/test_contracts.py -q` and confirm failure because the package does not exist.
- [ ] Implement frozen dataclasses, `to_dict/from_dict`, tensor-shape validation, and detailed `ContractError` messages.
- [ ] Run the contract tests and confirm all pass.
- [ ] Commit with `feat: add distillation contracts`.

### Task 2: Differentiable Color Boundaries

**Files:**
- Create: `src/distill_codec/color.py`
- Test: `tests/test_color.py`

- [ ] Write tests for `[B,3,H,W] -> [B,6,H/2,W/2]`, exact `Y00/Y01/Y10/Y11/U/V` ordering, sparse top-left U/V extraction, RGB round-trip tolerance, BT.601/709 range selection, and autograd.
- [ ] Run `pytest tests/test_color.py -q` and confirm import/function failures.
- [ ] Implement matrix-based full-range RGB/YUV conversion, limited-range scaling, 2x2 chroma averaging for packing, and configurable nearest/bilinear chroma reconstruction.
- [ ] Run color tests and confirm all pass.
- [ ] Commit with `feat: add packed yuv color adapters`.

### Task 3: Strict Paired Dataset and Mock Images

**Files:**
- Create: `src/distill_codec/data.py`
- Test: `tests/test_data.py`

- [ ] Write tests for nested relative-path matching, missing counterparts, RGB loading, incompatible image sizes, and deterministic mock LQ/GT generation.
- [ ] Run `pytest tests/test_data.py -q` and confirm failure.
- [ ] Implement `PairedImageDataset`, an explicit preflight report, image-extension filtering, and `create_mock_dataset` using Pillow.
- [ ] Run data tests and confirm all pass.
- [ ] Commit with `feat: add paired image dataset`.

### Task 4: Factories, Mock Models, and Adapters

**Files:**
- Create: `src/distill_codec/factories.py`
- Create: `src/distill_codec/models/__init__.py`
- Create: `src/distill_codec/models/mock.py`
- Create: `src/distill_codec/adapters.py`
- Test: `tests/test_models_and_adapters.py`

- [ ] Write tests for import-string factory loading, optional state-dict checkpoints, encoder output `[B,16,H/8,W/8]`, sparse decoder output, condition encoder dictionaries, conditional/unconditional TCDecoder calls, single-frame temporal repetition, teacher freezing, and nonzero student gradients.
- [ ] Run the targeted tests and confirm failure.
- [ ] Implement compact convolutional mock modules and adapters with deterministic seeds, forward signature normalization, contract probes, and clear factory errors.
- [ ] Run targeted tests and confirm all pass.
- [ ] Commit with `feat: add model factories and mock adapters`.

### Task 5: Losses, Metrics, and Validation Images

**Files:**
- Create: `src/distill_codec/losses.py`
- Create: `src/distill_codec/metrics.py`
- Test: `tests/test_losses_metrics.py`

- [ ] Write tests for finite SmoothL1/cosine/channel-stat losses, edge loss, differentiability through a frozen decoder, known PSNR values, bounded SSIM, and a five-panel validation grid.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement the loss functions, lightweight global SSIM, PSNR, absolute-error heatmap, and Pillow grid saving.
- [ ] Run targeted tests and confirm all pass.
- [ ] Commit with `feat: add distillation losses and metrics`.

### Task 6: Recipe Implementations

**Files:**
- Create: `src/distill_codec/recipes.py`
- Test: `tests/test_recipes.py`

- [ ] Write one-step tests for `wan_encoder_distill`, `wan_decoder_distill`, `wan_autoencoder_distill`, `flashvsr_vae_encoder_distill`, `flashvsr_lq_proj_distill`, `flashvsr_decoder_unconditional_student`, and `flashvsr_decoder_conditional_student`.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement a recipe registry and structured `RecipeOutput(total_loss, losses, images, metrics)`; ensure compatibility loss freezes teacher weights without detaching student latent.
- [ ] Run recipe tests and confirm all pass.
- [ ] Commit with `feat: add wan and flashvsr recipes`.

### Task 7: Configuration and Component Construction

**Files:**
- Create: `src/distill_codec/config.py`
- Create: `configs/smoke/wan_encoder.yaml`
- Create: `configs/smoke/wan_decoder.yaml`
- Create: `configs/smoke/flashvsr_decoder_unconditional.yaml`
- Create: `configs/smoke/flashvsr_decoder_conditional.yaml`
- Create: `configs/smoke/flashvsr_lq_proj.yaml`
- Create: `configs/students/private_blackbox.yaml`
- Create: `configs/teachers/wan_external.yaml`
- Create: `configs/teachers/flashvsr_external.yaml`
- Test: `tests/test_config.py`

- [ ] Write tests loading each smoke config, resolving relative paths from the config file, applying dotted CLI overrides, rejecting unknown recipes/backends, and constructing all mock components.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement YAML loading, recursive merge/overrides, typed accessors, backend dispatch (`mock`, `external`, `snapshot`), and configuration preflight.
- [ ] Run config tests and confirm all pass.
- [ ] Commit with `feat: add configurable component construction`.

### Task 8: Trainer, Checkpoints, Logging, and Resume

**Files:**
- Create: `src/distill_codec/checkpoint.py`
- Create: `src/distill_codec/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] Write an end-to-end CPU test that trains, validates, updates parameters, saves a checkpoint, restores optimizer/model/RNG/contracts, resumes one step, emits scalar logs, and writes validation grids.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement single-device AMP-aware training, accumulation, clipping, finite-loss checks with sample paths, JSONL logging, optional TensorBoard, checkpoint retention, and strict resume validation.
- [ ] Run trainer tests and confirm all pass.
- [ ] Commit with `feat: add training engine and checkpoints`.

### Task 9: CLI and Full Mock Smoke

**Files:**
- Create: `src/distill_codec/cli.py`
- Create: `scripts/run_smoke.ps1`
- Test: `tests/test_cli_smoke.py`

- [ ] Write CLI tests for `make-mock-data`, `probe`, `train`, and `--resume`, including encoder and both FlashVSR decoder modes.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement argparse subcommands and the PowerShell smoke runner without assuming `python` is on PATH.
- [ ] Run CLI tests, then run `scripts/run_smoke.ps1 -PythonExe <python>` and confirm artifacts and resumed steps exist.
- [ ] Commit with `feat: add runnable codec distillation cli`.

### Task 10: Wan and FlashVSR Source Snapshot and External Templates

**Files:**
- Create: `third_party/README.md`
- Create: `third_party/wan/SOURCE.md`
- Create: `third_party/flashvsr/SOURCE.md`
- Create: `third_party/patches/README.md`
- Create or copy: only the Wan VAE and FlashVSR `LQ_proj_in`/`TCDecoder` source files required by the chosen pinned upstream revisions
- Preserve: upstream license files covering all copied source
- Test: `tests/test_third_party.py`

- [ ] Write tests that require each snapshot manifest to contain repository URL, immutable revision, retrieval date, copied-file list, license path, and known adapter limitations.
- [ ] Run targeted tests and confirm failure.
- [ ] Fetch pinned upstream source, copy only required implementation/dependency files, retain copyright/license notices, and record SHA256 values; do not fetch model weights.
- [ ] Add external/snapshot factory examples and verify import/probe errors name the missing dependency or weight clearly.
- [ ] Run third-party tests and confirm all pass.
- [ ] Commit with `chore: vendor pinned wan and flashvsr source snapshots`.

### Task 11: Documentation and Release Verification

**Files:**
- Create: `README.md`
- Create: `docs/ADDING_A_BACKEND.md`
- Modify: `.gitignore`

- [ ] Document installation, tensor contracts, paired directory layout, all recipes, mock smoke commands, private factory replacement points, real-weight setup, and limitations.
- [ ] Run `pytest -q` from a clean process and confirm zero failures.
- [ ] Run all smoke recipes on CPU and inspect generated checkpoints, JSONL/TensorBoard logs, and validation grids.
- [ ] Run `python -m build` and install the wheel into a temporary virtual environment; verify `distill-codec --help`.
- [ ] Run `git diff --check`, inspect `git status`, and confirm no weights or run artifacts are tracked.
- [ ] Commit with `docs: add codec distillation usage guide`.

