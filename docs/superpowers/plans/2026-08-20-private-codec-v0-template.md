# Private Codec V0 Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable, heavily commented `private_codec.versions.v0` convolutional encoder/decoder template that can be copied for future private network versions.

**Architecture:** Shared bridge and factory modules remain version-independent. The v0 package owns base convolution modules, inheritance-based wrappers, and four stable builder/runner entrypoints; YAML selects v0 by import path. The old root-level empty placeholders are removed so there is only one obvious template layout.

**Tech Stack:** Python 3.13, PyTorch, PyYAML, pytest, Ruff, mypy

---

## File Structure

- Create `src/private_codec/versions/__init__.py`: marks the version namespace.
- Create `src/private_codec/versions/v0/__init__.py`: documents v0 as a copyable example.
- Create `src/private_codec/versions/v0/base_network.py`: reusable convolutional encoder and decoder bases with input validation.
- Create `src/private_codec/versions/v0/wrapped_network.py`: inheritance wrappers showing version-specific defaults and forward methods.
- Create `src/private_codec/versions/v0/entrypoints.py`: builder/runner boundary where project tensors are mapped to private calls.
- Delete `src/private_codec/base_network.py`: obsolete root placeholder.
- Delete `src/private_codec/wrapped_network.py`: obsolete root placeholder.
- Delete `src/private_codec/entrypoints.py`: obsolete root placeholder.
- Modify `configs/students/private_codec.yaml`: select v0 entrypoints and v0 legacy decoder class.
- Modify `tests/test_private_codec_tutorial.py`: assert v0 structure, runtime behavior, configuration, and documentation.
- Modify `PRIVATE_CODEC_INTEGRATION_TUTORIAL.md`: make v0 the executable copy-and-edit tutorial.

### Task 1: Define the v0 artifact and runtime contract with failing tests

**Files:**
- Modify: `tests/test_private_codec_tutorial.py`

- [ ] **Step 1: Replace root-placeholder assertions with version-layout assertions**

Require all five v0 package files to exist, require the three old root placeholders to be absent, and require non-empty commented Python sources for the three implementation files.

- [ ] **Step 2: Add encoder behavior and gradient test**

Construct the real shared encoder bridge with v0 entrypoints, pass a `[2,3,32,40]` tensor with gradients, assert output shape `[2,16,4,5]`, backpropagate `latent.square().mean()`, and assert both input and trainable network gradients exist and are finite.

- [ ] **Step 3: Add conditional decoder behavior and gradient test**

Construct the real conditional bridge with v0 entrypoints, pass latent `[2,16,4,5]` and LQ RGB `[2,3,32,40]`, assert RGB output shape `[2,3,32,40]`, backpropagate the mean, and assert gradients reach both project inputs and network parameters.

- [ ] **Step 4: Add malformed-input tests**

Assert useful `ValueError` messages for non-RGB encoder input, image dimensions not divisible by 8, decoder batch mismatch, and decoder latent-channel mismatch.

- [ ] **Step 5: Update YAML and tutorial marker assertions**

Require all default builder/runner paths to use `private_codec.versions.v0.entrypoints`, require the legacy decoder class path to use `private_codec.versions.v0.wrapped_network`, and require tutorial text to describe copying `versions/v0` to `versions/v1`.

- [ ] **Step 6: Run focused tests and verify RED**

Run:

```powershell
& 'C:\Users\xh932\anaconda3\Scripts\conda.exe' run --no-capture-output -n base python -m pytest tests/test_private_codec_tutorial.py -q
```

Expected: failures because `private_codec.versions.v0` and the updated paths do not exist yet.

### Task 2: Implement the minimal runnable v0 package and configuration

**Files:**
- Create: `src/private_codec/versions/__init__.py`
- Create: `src/private_codec/versions/v0/__init__.py`
- Create: `src/private_codec/versions/v0/base_network.py`
- Create: `src/private_codec/versions/v0/wrapped_network.py`
- Create: `src/private_codec/versions/v0/entrypoints.py`
- Delete: `src/private_codec/base_network.py`
- Delete: `src/private_codec/wrapped_network.py`
- Delete: `src/private_codec/entrypoints.py`
- Modify: `configs/students/private_codec.yaml`

- [ ] **Step 1: Implement validated base convolution modules**

Implement `ConvEncoderBase` with three `Conv2d(kernel_size=3, stride=2, padding=1)` stages and `ConvConditionalDecoderBase` with latent/LQ projections, bilinear latent interpolation to the LQ size, feature fusion, and a three-channel sigmoid output. Add clear shape checks at the private boundary.

- [ ] **Step 2: Implement inheritance-based wrappers**

Implement `V0Encoder`, `V0ConditionalDecoder`, and `V0UnconditionalDecoder`. Keep version defaults in constructors and private forward signatures in this file so a future copied version has obvious edit points.

- [ ] **Step 3: Implement all four entrypoints with copy instructions**

Implement `build_encoder`, `run_encoder`, `build_decoder`, and `run_decoder`. Comments must identify where to import real classes, set initialization arguments, transform project tensors, call nested private networks, and inspect but not obey `teacher_reference`.

- [ ] **Step 4: Switch configuration imports to v0**

Point encoder and conditional decoder builder/runner paths to the v0 entrypoint module. Point the legacy unconditional decoder at `V0UnconditionalDecoder` in the v0 wrapper module.

- [ ] **Step 5: Remove the obsolete root placeholders**

Delete the three zero-byte files so future users cannot accidentally fill a layout no longer referenced by YAML.

- [ ] **Step 6: Run focused tests and verify remaining failures are documentation-only**

Run the Task 1 command. Expected: network, gradient, input-error, and configuration tests pass; tutorial marker assertions may still fail until Task 3.

### Task 3: Rewrite the second tutorial around the runnable v0 template

**Files:**
- Modify: `PRIVATE_CODEC_INTEGRATION_TUTORIAL.md`

- [ ] **Step 1: Replace empty-file instructions with v0 walkthrough**

Document the actual `versions/v0` tree, what each implementation file owns, and how the supplied convolution example maps RGB to latent and latent plus LQ RGB back to RGB.

- [ ] **Step 2: Add exact v0 probe commands**

Use the v0 builder/runner import paths and random tensors compatible with the default 16-channel, 1/8-spatial latent contract.

- [ ] **Step 3: Add copy-to-v1 instructions**

Provide PowerShell commands to copy the complete v0 directory to v1, then state the three files to edit and the four YAML import paths to change. Explicitly state that shared `bridge.py` and `factories.py` are not copied or edited.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: all tests in `tests/test_private_codec_tutorial.py` pass.

- [ ] **Step 5: Commit implementation batch**

```powershell
git add -- src/private_codec/versions configs/students/private_codec.yaml tests/test_private_codec_tutorial.py PRIVATE_CODEC_INTEGRATION_TUTORIAL.md
git add -u -- src/private_codec/base_network.py src/private_codec/wrapped_network.py src/private_codec/entrypoints.py
git commit -m "feat: add runnable private codec v0 template"
```

### Task 4: Verify, review, and integrate

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run focused private codec tests**

```powershell
& 'C:\Users\xh932\anaconda3\Scripts\conda.exe' run --no-capture-output -n base python -m pytest tests/test_private_codec_bridge.py tests/test_private_codec_tutorial.py tests/test_config.py tests/test_recipes.py -q
```

- [ ] **Step 2: Run full repository tests with the recorded environment exclusion**

```powershell
& 'C:\Users\xh932\anaconda3\Scripts\conda.exe' run --no-capture-output -n base python -c "import numpy, torch, pytest, sys; sys.exit(pytest.main(['-q','-k','not test_multi_component_resume_is_stable_across_python_hash_seeds']))"
```

- [ ] **Step 3: Run static and repository checks**

```powershell
& 'C:\Users\xh932\anaconda3\Scripts\conda.exe' run --no-capture-output -n base python -m ruff check .
& 'C:\Users\xh932\anaconda3\Scripts\conda.exe' run --no-capture-output -n base python -m mypy src tests
git diff --check main...HEAD
```

Also run `bash -n scripts/run_smoke.sh` through the installed Git Bash if present.

- [ ] **Step 4: Request code review and address all important findings**

Review the complete diff against `docs/superpowers/specs/2026-08-20-private-codec-v0-template-design.md`, with emphasis on copyability, tensor contracts, gradients, import paths, and tutorial accuracy.

- [ ] **Step 5: Merge locally and re-run focused verification**

Merge `codex/private-codec-v0` into `main`, remove the worktree and feature branch, then re-run focused private codec tests on the merged result before reporting completion.
