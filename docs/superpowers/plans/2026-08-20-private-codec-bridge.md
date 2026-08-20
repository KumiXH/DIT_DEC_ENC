# Private Codec Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable RGB/latent bridge for versioned private encoders and conditional decoders, with automatically derived teacher shape references that never alter real tensors.

**Architecture:** Project adapters keep the public calls `encoder(rgb)` and `decoder(dit_latent, lq_rgb)`. `private_codec.bridge` imports version-specific builder and runner functions, owns the private `nn.Module`, and forwards real tensors by keyword. `distill_codec.config` injects a read-only teacher reference only for components that explicitly declare `teacher_reference: auto`.

**Tech Stack:** Python 3.13, PyTorch, PyYAML, pytest, Ruff, mypy

---

## File Map

- Create `src/private_codec/bridge.py`: stable import, construction, and forward boundary.
- Create `src/private_codec/entrypoints.py`: empty user fill-in file for the first private version.
- Modify `src/private_codec/factories.py`: construct bridges while retaining the legacy unconditional decoder factory.
- Modify `src/distill_codec/config.py`: derive and opt-in inject teacher references.
- Modify `src/distill_codec/recipes.py`: send aligned LQ only to the FlashVSR teacher and raw LQ to the private student.
- Modify `configs/students/private_codec.yaml`: select builders/runners and RGB adapters.
- Create `configs/local/private_codec_conditional_decoder.yaml`: runnable FlashVSR conditional-decoder template.
- Modify `PRIVATE_CODEC_INTEGRATION_TUTORIAL.md`: document first integration and version expansion.
- Modify `tests/support_factories.py`: importable real builder/runner fixtures.
- Create `tests/test_private_codec_bridge.py`: bridge contract coverage.
- Modify `tests/test_config.py`: teacher-reference derivation and opt-in coverage.
- Modify `tests/test_recipes.py`: raw-versus-aligned LQ behavior.
- Modify `tests/test_private_codec_tutorial.py`: fill-in files, YAML, and tutorial coverage.

## Test Command Convention

The existing Conda base environment is used through `conda run`. On this Windows host, NumPy and PyTorch must be imported before `pytest.main()` to avoid a NumPy BLAS FPE abort during pytest collection:

```powershell
& 'C:\Users\xh932\anaconda3\Scripts\conda.exe' run -n base python -c "import numpy, torch, pytest, sys; sys.exit(pytest.main(['-q', '<test>']))"
```

The baseline has one environment-specific failure already reproduced on `main`: `tests/test_trainer.py::test_multi_component_resume_is_stable_across_python_hash_seeds` aborts its child process because two OpenMP runtimes are loaded. All other baseline tests pass.

### Task 1: Stable private bridge

**Files:**
- Create: `tests/test_private_codec_bridge.py`
- Modify: `tests/support_factories.py`
- Create: `src/private_codec/bridge.py`
- Modify: `src/private_codec/factories.py`

- [ ] **Step 1: Write failing bridge tests**

Add importable test builders/runners and tests equivalent to:

```python
def test_encoder_bridge_forwards_rgb_reference_and_kwargs():
    bridge = PrivateEncoderBridge(
        builder="tests.support_factories:build_private_bridge_network",
        runner="tests.support_factories:run_private_encoder",
        builder_kwargs={"offset": 2.0},
        runner_kwargs={"gain": 3.0},
        teacher_reference={"role": "encoder"},
    )
    rgb = torch.ones(1, 3, 4, 4)
    result = bridge(rgb)
    assert torch.equal(result, torch.full_like(rgb, 9.0))
```

```python
def test_decoder_bridge_reorders_project_arguments_into_private_keywords():
    result = bridge(dit_latent, lq_rgb)
    call = private_bridge_calls[-1]
    assert call["dit_latent"] is dit_latent
    assert call["lq_rgb"] is lq_rgb
```

Also cover defensive reference copies, a non-`nn.Module` builder, and a non-Tensor runner result.

- [ ] **Step 2: Run the focused test and verify RED**

Run the command convention with `tests/test_private_codec_bridge.py`.

Expected: collection fails because `private_codec.bridge` does not exist.

- [ ] **Step 3: Implement the minimal bridge**

Implement:

```python
class PrivateEncoderBridge(_PrivateBridge):
    def forward(self, rgb: Tensor) -> Tensor:
        return self._run(rgb=rgb)


class PrivateConditionalDecoderBridge(_PrivateBridge):
    def forward(self, dit_latent: Tensor, lq_rgb: Tensor) -> Tensor:
        return self._run(lq_rgb=lq_rgb, dit_latent=dit_latent)
```

The shared base imports `module:symbol` callables, constructs the network once, deep-copies the reference at storage and call time, merges runner kwargs, and validates builder/runner return types.

Update factories to expose `create_encoder()` and `create_conditional_decoder()` for bridges. Retain the existing class-import implementation as `create_decoder()` for the legacy unconditional Wan templates.

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: all tests in `tests/test_private_codec_bridge.py` pass.

- [ ] **Step 5: Commit**

```powershell
git add src/private_codec/bridge.py src/private_codec/factories.py tests/support_factories.py tests/test_private_codec_bridge.py
git commit -m "feat: add private codec bridges"
```

### Task 2: Automatic teacher shape references

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/distill_codec/config.py`

- [ ] **Step 1: Write failing config tests**

Add tests that construct bridge-backed components and assert exact dictionaries:

```python
assert bridge.teacher_reference == {
    "role": "encoder",
    "inputs": {
        "rgb": {"layout": "BCHW", "shape": [None, 3, 256, 256], "source": "gt"}
    },
    "outputs": {
        "latent": {"layout": "BCHW", "shape": [None, 16, 32, 32]}
    },
}
```

```python
assert bridge.teacher_reference["inputs"]["lq_rgb"]["shape"] == [None, 3, 256, 256]
assert bridge.teacher_reference["inputs"]["dit_latent"]["shape"] == [None, 16, 32, 32]
```

Also prove that a component without `teacher_reference: auto` receives no injected keyword, that an explicit `kwargs.teacher_reference` conflicts with auto mode, and that missing/malformed image sizes raise `ContractError`.

- [ ] **Step 2: Run the new config tests and verify RED**

Expected: bridge factory construction fails because no reference is injected.

- [ ] **Step 3: Implement derivation helpers and opt-in injection**

Add focused helpers to:

- validate `[height, width]` sizes;
- derive latent BCHW/BCTHW shapes using `LatentSpec`;
- select encoder source from `recipe.source`, defaulting to `gt`;
- select decoder latent source from `latent_provider.source`, defaulting to `gt`;
- describe FlashVSR teacher condition/output using `data.gt_size`;
- inject only when the component field is exactly `teacher_reference: auto`.

Pass injected kwargs to `_build_module()` without modifying the original config mapping. Reject ambiguous explicit and automatic references.

- [ ] **Step 4: Run config and bridge tests and verify GREEN**

Expected: the new tests pass and existing `tests/test_config.py` remains green.

- [ ] **Step 5: Commit**

```powershell
git add src/distill_codec/config.py tests/test_config.py
git commit -m "feat: inject teacher shape references"
```

### Task 3: Preserve raw LQ for the private decoder

**Files:**
- Modify: `tests/test_recipes.py`
- Modify: `src/distill_codec/recipes.py`

- [ ] **Step 1: Write the failing recipe test**

Create recording teacher and student conditional decoders. Run a batch with LQ `32x32` and GT `64x64`, then assert:

```python
assert teacher_decoder.received_condition.shape[-2:] == (64, 64)
assert student_decoder.received_condition is batch.lq_rgb
assert student_decoder.received_condition.shape[-2:] == (32, 32)
```

- [ ] **Step 2: Run the focused test and verify RED**

Expected: the student currently receives the teacher-aligned `64x64` condition.

- [ ] **Step 3: Split teacher and student condition variables**

Implement:

```python
student_condition_rgb = batch.lq_rgb
teacher_condition_rgb = student_condition_rgb
if flashvsr and teacher_condition_rgb.shape[-2:] != batch.gt_rgb.shape[-2:]:
    teacher_condition_rgb = F.interpolate(...)
```

Use `teacher_condition_rgb` only for `tc_decoder` and `student_condition_rgb` only for `conditional_student_decoder`. Preserve existing metadata key semantics for the teacher condition shape.

- [ ] **Step 4: Run recipe tests and verify GREEN**

Expected: all tests in `tests/test_recipes.py` pass.

- [ ] **Step 5: Commit**

```powershell
git add src/distill_codec/recipes.py tests/test_recipes.py
git commit -m "fix: preserve raw LQ for private decoder"
```

### Task 4: Fill-in templates, extensible YAML, and tutorial

**Files:**
- Create: `src/private_codec/entrypoints.py`
- Modify: `configs/students/private_codec.yaml`
- Create: `configs/local/private_codec_conditional_decoder.yaml`
- Modify: `PRIVATE_CODEC_INTEGRATION_TUTORIAL.md`
- Modify: `tests/test_private_codec_tutorial.py`

- [ ] **Step 1: Write failing artifact tests**

Require:

- `base_network.py`, `wrapped_network.py`, and `entrypoints.py` are present as empty fill-in files;
- student encoder uses `input_mode: rgb` and `teacher_reference: auto`;
- conditional decoder uses `accepts_condition: true`, `output_mode: rgb`, and `teacher_reference: auto`;
- the new local conditional decoder config preflights;
- the tutorial contains all four entry function signatures, the exact public/private argument order, automatic teacher reference semantics, and a V2 folder/YAML example.

- [ ] **Step 2: Run tutorial tests and verify RED**

Expected: missing `entrypoints.py`, conditional config, and updated YAML/tutorial markers.

- [ ] **Step 3: Add the empty entrypoint and update templates**

Use bridge paths:

```yaml
factory: private_codec.factories:create_encoder
teacher_reference: auto
kwargs:
  builder: private_codec.entrypoints:build_encoder
  runner: private_codec.entrypoints:run_encoder
  builder_kwargs: {}
  runner_kwargs: {}
```

and the equivalent `create_conditional_decoder` block. Keep the legacy unconditional `student_decoder` block solely for the existing Wan decoder/autoencoder examples.

Create a FlashVSR conditional local config that includes Wan teacher, FlashVSR teacher, and the private student include.

- [ ] **Step 4: Rewrite the tutorial around the bridge boundary**

Document that the project supplies raw RGB/latent tensors; only the user's entrypoint module converts them. Include exact extension steps for `private_codec.versions.v2.entrypoints` without editing the shared factory.

- [ ] **Step 5: Run tutorial/config tests and verify GREEN**

Expected: private tutorial and config tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/private_codec/entrypoints.py configs/students/private_codec.yaml configs/local/private_codec_conditional_decoder.yaml PRIVATE_CODEC_INTEGRATION_TUTORIAL.md tests/test_private_codec_tutorial.py
git commit -m "docs: add extensible private codec template"
```

### Task 5: Full verification and branch completion

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run focused feature tests**

Run bridge, config, recipe, and tutorial tests together. Expected: all pass.

- [ ] **Step 2: Run the complete pytest suite**

Expected: all feature tests pass; compare any failure to the recorded baseline OpenMP failure. Do not claim a clean suite if the existing child-process failure remains.

- [ ] **Step 3: Run static and repository checks**

```powershell
& 'C:\Users\xh932\anaconda3\Scripts\conda.exe' run -n base python -m ruff check .
& 'C:\Users\xh932\anaconda3\Scripts\conda.exe' run -n base python -m mypy src tests
git diff --check ff0afbb...HEAD
```

Use Git Bash for `bash -n scripts/run_smoke.sh` if available.

- [ ] **Step 4: Review requirements against the design**

Confirm public/private signatures, raw LQ behavior, opt-in injection, reference-only semantics, error behavior, version extension, and fill-in artifacts.

- [ ] **Step 5: Use `superpowers:finishing-a-development-branch`**

Present verified integration options. Push only after the user selects or has already explicitly requested publication.
