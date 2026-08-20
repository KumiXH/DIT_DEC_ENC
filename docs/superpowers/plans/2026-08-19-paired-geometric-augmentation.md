# Paired Geometric Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add YAML-configurable, batch-shared paired crop, rotation, and translation for LQ/GT training while keeping probe and validation deterministic.

**Architecture:** A new augmentation module parses and validates the YAML contract, converts unstacked paired samples into the existing `DistillBatch`, and applies one shared affine transform. Dataset preflight validates raw images; Trainer and CLI use a raw collator so oversized or differently sized source images can be cropped before stacking. Recipes and model adapters remain unchanged.

**Tech Stack:** PyTorch `affine_grid`/`grid_sample`, PIL-backed dataset, PyYAML, pytest

---

### Task 1: Configuration contract

**Files:**
- Create: `src/distill_codec/augmentation.py`
- Modify: `src/distill_codec/config.py`
- Modify: `src/distill_codec/checkpoint.py`
- Test: `tests/test_augmentation.py`
- Test: `tests/test_config.py`

- [ ] Write failing tests for valid continuous augmentation, invalid probabilities/ranges, unequal LQ/GT target sizes, unsupported per-sample mode, right-angle rectangular targets, and cached/dataset latent rejection.
- [ ] Run focused tests and confirm failures are caused by missing augmentation parsing.
- [ ] Implement immutable augmentation specification classes and `paired_augmentation_from_config(config)`.
- [ ] Call augmentation validation from `preflight_config`.
- [ ] Add the normalized augmentation mapping to `training_contract` so resume rejects configuration drift.
- [ ] Run focused tests until green.

### Task 2: Raw batch and paired crop

**Files:**
- Modify: `src/distill_codec/data.py`
- Test: `tests/test_data.py`
- Test: `tests/test_augmentation.py`

- [ ] Write failing tests proving small images fail, large same-size pairs pass preflight, legacy exact-size checks remain when augmentation is disabled, and raw batches can contain different source dimensions.
- [ ] Add `RawPairedBatch` and `collate_raw_paired_batch` without changing existing `collate_distill_batch`.
- [ ] Let `PairedImageDataset` validate minimum dimensions and equal LQ/GT pair sizes when crop augmentation is enabled.
- [ ] Implement batch-shared normalized random crop and deterministic center crop, producing a fixed-size `DistillBatch`.
- [ ] Verify exact-size and larger-image cases.

### Task 3: Rotation and translation

**Files:**
- Modify: `src/distill_codec/augmentation.py`
- Test: `tests/test_augmentation.py`

- [ ] Write failing tests for shared continuous angle/translation, identical LQ/GT geometry, final shape, fixed-seed reproducibility, different-step variation, and deterministic validation.
- [ ] Implement continuous affine sampling with `affine_grid` and `grid_sample`.
- [ ] Implement square-only right-angle rotation with `torch.rot90`, followed by optional translation.
- [ ] Use one local CPU generator seeded from run seed, phase, global step, and micro-step.
- [ ] Run augmentation tests until green.

### Task 4: Trainer and probe integration

**Files:**
- Modify: `src/distill_codec/trainer.py`
- Modify: `src/distill_codec/cli.py`
- Test: `tests/test_cli_smoke.py`
- Test: `tests/test_trainer.py`

- [ ] Write failing integration tests showing probe center-crops oversized images and Trainer random-crops before recipes.
- [ ] Switch paired-image DataLoaders to `collate_raw_paired_batch`.
- [ ] Prepare train batches with random augmentation immediately before device recipe execution.
- [ ] Prepare validation and probe batches with deterministic center crop and affine disabled.
- [ ] Preserve sampler replay and derive augmentation from step/micro-step so resumed training matches uninterrupted training.
- [ ] Run CLI and Trainer integration tests until green.

### Task 5: Documentation and examples

**Files:**
- Modify: `README.md`
- Modify: `FLASHVSR_DISTILL_TUTORIAL.md`
- Modify: `tests/test_flashvsr_tutorial.py`
- Modify: `tests/test_linux_support.py`

- [ ] Add failing documentation tests requiring the augmentation YAML and semantics.
- [ ] Update both FlashVSR YAML examples with crop, continuous `[-5,5]` rotation, and 5% translation.
- [ ] Explain small-image failure, large-image crop, batch sharing, deterministic validation/probe, resume restriction, and cached-latent limitation.
- [ ] Run documentation tests until green.

### Task 6: Full verification

**Files:**
- All modified files

- [ ] Run `pytest -q` and require zero failures.
- [ ] Run `ruff check .` and `python -m mypy src tests`.
- [ ] Run `bash -n scripts/run_smoke.sh` and `git diff --check`.
- [ ] Review the final diff for unrelated changes, private paths, weights, or training artifacts.
