# FlashVSR Distillation Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Linux-first, command-by-command tutorial for distilling FlashVSR `LQ_proj_in` and conditional `TCDecoder`, with verified YAML examples, expected outputs, artifact locations, recovery commands, and parameter explanations.

**Architecture:** The tutorial is a standalone root-level Markdown document linked from the main README. It uses the repository snapshot teacher configurations and private student factories, while tests extract and preflight the embedded YAML so documentation cannot silently drift from the configuration contract.

**Tech Stack:** Markdown, Mermaid, Bash, YAML, PyYAML, pytest, project `load_config` and `preflight_config`

---

### Task 1: Add failing tutorial contract tests

**Files:**
- Create: `tests/test_flashvsr_tutorial.py`
- Reference: `tests/test_linux_support.py`
- Reference: `src/distill_codec/config.py`

- [ ] **Step 1: Write tests for the tutorial file and README link**

Create tests that require:

```python
def test_main_readme_links_flashvsr_tutorial():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "[FlashVSR 蒸馏教程](FLASHVSR_DISTILL_TUTORIAL.md)" in readme


def test_tutorial_documents_training_artifacts():
    tutorial = Path("FLASHVSR_DISTILL_TUTORIAL.md").read_text(encoding="utf-8")
    for marker in (
        "metrics.jsonl",
        "checkpoints/step_XXXXXXXX.pt",
        "validation/step_XXXXXXXX.png",
        "tensorboard/",
        "tail -f",
        "--resume",
        "student_state",
    ):
        assert marker in tutorial
```

- [ ] **Step 2: Add YAML extraction and preflight tests**

Extract YAML blocks following headings for `configs/local/flashvsr_lq_proj.yaml` and `configs/local/flashvsr_tcdecoder.yaml`. Copy the existing teacher/student include files into a temporary `configs` tree, write the extracted blocks, then call:

```python
preflight_config(load_config(config_path))
```

Assert the LQ config contains `student_condition_encoder`, and the TCDecoder config includes Wan snapshot, FlashVSR snapshot, private student config, and a teacher-encoder latent provider.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_flashvsr_tutorial.py
```

Expected: FAIL because `FLASHVSR_DISTILL_TUTORIAL.md` and the main README link do not exist.

### Task 2: Write the executable FlashVSR tutorial

**Files:**
- Create: `FLASHVSR_DISTILL_TUTORIAL.md`
- Reference: `README.md`
- Reference: `configs/teachers/wan_snapshot.yaml`
- Reference: `configs/teachers/flashvsr_snapshot.yaml`
- Reference: `configs/students/private_blackbox.yaml`
- Reference: `src/distill_codec/cli.py`
- Reference: `src/distill_codec/trainer.py`
- Reference: `src/distill_codec/checkpoint.py`

- [ ] **Step 1: Write the workflow overview and pre-run checklist**

Start after environment installation. Include a Mermaid flow showing mock verification, PNG data, weight checks, private student factories, separate LQ/TCDecoder training, and result inspection.

- [ ] **Step 2: Write command-by-command mock and input checks**

For each Bash command, document expected stable output shape, generated files, success criteria, and likely failure causes. Use structure examples rather than fixed random losses.

- [ ] **Step 3: Write the private student factory contract**

Show callable examples for:

```python
create_condition_encoder(**kwargs) -> torch.nn.Module
create_conditional_decoder(**kwargs) -> torch.nn.Module
```

Explain that factories must return trainable modules and how component-level checkpoint loading works.

- [ ] **Step 4: Write both complete local YAML configurations**

Add complete, parseable sections headed:

```markdown
### `configs/local/flashvsr_lq_proj.yaml`
### `configs/local/flashvsr_tcdecoder.yaml`
```

Use snapshot includes and output directories:

```text
~/dit_codec/runs/flashvsr_lq_proj
~/dit_codec/runs/flashvsr_tcdecoder
```

- [ ] **Step 5: Write probe, train, monitoring, resume, and extraction commands**

Document:

```bash
python -m distill_codec.cli probe --config ...
python -m distill_codec.cli train --config ...
tail -f "$HOME/dit_codec/runs/.../metrics.jsonl"
tensorboard --logdir ...
python -m distill_codec.cli train --config ... --resume ...
```

Show the final CLI JSON and exact artifact tree. Explain that `.pt` is a full training checkpoint and provide a Python extraction command for the relevant entry in `student_state`.

- [ ] **Step 6: Write the YAML parameter dictionary**

Explain every field used by the two tutorial YAMLs, including defaults and resume-sensitive values. Distinguish tensor contracts from tunable optimization parameters.

- [ ] **Step 7: Add troubleshooting and quick reference**

Cover missing checkpoint/source file, factory import failure, CUDA unavailable, empty or mismatched PNG pairs, shape/condition contract errors, OOM, non-finite loss, and incompatible resume contract.

### Task 3: Link and verify the tutorial

**Files:**
- Modify: `README.md`
- Test: `tests/test_flashvsr_tutorial.py`

- [ ] **Step 1: Add the main README link**

Add this near the first run/training documentation entry:

```markdown
详细的 Linux 分步操作见 [FlashVSR 蒸馏教程](FLASHVSR_DISTILL_TUTORIAL.md)。
```

- [ ] **Step 2: Run focused tests and verify GREEN**

Run:

```bash
pytest -q tests/test_flashvsr_tutorial.py tests/test_linux_support.py
```

Expected: all tests pass.

- [ ] **Step 3: Run full verification**

Run:

```bash
pytest -q
ruff check .
python -m mypy src tests
bash -n scripts/run_smoke.sh
git diff --check
```

Expected: zero failures and zero static-analysis errors.

- [ ] **Step 4: Review final diff**

Confirm the change set contains only the tutorial, README link, documentation tests, and approved specification/plan files. Do not modify training behavior or commit private paths and credentials.
