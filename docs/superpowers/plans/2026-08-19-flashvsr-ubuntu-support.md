# FlashVSR Target and Ubuntu Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `LQ_proj_in` plus conditional `TCDecoder` the documented FlashVSR replacement path and provide complete Ubuntu commands and a Bash smoke runner.

**Architecture:** Keep the existing recipe and adapter boundaries because they already separate DiT condition features from the main latent. Add a POSIX shell entry point equivalent to the PowerShell smoke runner, then make README Linux commands the primary executable examples while retaining a short Windows alternative.

**Tech Stack:** Python 3.10+, PyTorch, pytest, Bash, PowerShell, YAML, Markdown.

---

### Task 1: Lock the Linux and FlashVSR contracts with tests

**Files:**
- Create: `tests/test_linux_support.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing Linux support tests**

Create tests that require `scripts/run_smoke.sh`, validate it with `bash -n`, assert `PYTHON_BIN`, `set -euo pipefail`, `probe`, `train`, `--resume`, and all seven smoke configs, and require README markers for `source .venv/bin/activate`, `./scripts/run_smoke.sh`, `curl --fail --location`, `sha256sum -c`, Linux training paths, and resume.

- [ ] **Step 2: Write the FlashVSR replacement mapping test**

Load `configs/smoke/flashvsr_lq_proj.yaml` and `configs/smoke/flashvsr_decoder_conditional.yaml`, then assert the recipe/component mappings are exactly:

```python
assert lq_config["recipe"]["name"] == "flashvsr_lq_proj_distill"
assert set(lq_config["components"]) >= {"teacher_condition_encoder", "student_condition_encoder"}
assert decoder_config["recipe"]["name"] == "flashvsr_decoder_conditional_student"
assert set(decoder_config["components"]) >= {"tc_decoder", "conditional_student_decoder"}
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_linux_support.py tests/test_config.py
```

Expected: Linux support tests fail because `scripts/run_smoke.sh` and the required README commands do not exist; the existing FlashVSR config assertions pass.

### Task 2: Add the Bash smoke runner

**Files:**
- Create: `scripts/run_smoke.sh`
- Modify: `scripts/run_smoke.ps1`
- Test: `tests/test_linux_support.py`

- [ ] **Step 1: Implement the Bash runner**

Use this execution contract:

```bash
#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
output_root="${1:-work/smoke}"
```

Resolve a relative output root under the repository, `cd` to the repository root, generate mock data, probe each smoke config, train to step 1, and resume to step 2. Quote every path and override argument.

- [ ] **Step 2: Keep the PowerShell runner behavior aligned**

Add one `probe` call for each config before its first training call, using the same data roots and image sizes. Preserve the mandatory `-PythonExe` parameter and existing output layout.

- [ ] **Step 3: Mark the shell script executable**

Run:

```bash
git update-index --chmod=+x scripts/run_smoke.sh
git ls-files --stage scripts/run_smoke.sh
```

Expected mode: `100755`.

- [ ] **Step 4: Run shell and PowerShell parser tests**

Run:

```bash
bash -n scripts/run_smoke.sh
python -m pytest -q tests/test_linux_support.py tests/test_cli_smoke.py
```

Expected: script syntax tests pass; README assertions remain red until Task 3.

### Task 3: Make Ubuntu the primary documented command path

**Files:**
- Modify: `README.md`
- Test: `tests/test_linux_support.py`

- [ ] **Step 1: Clarify the FlashVSR replacement path**

State near the recipe table and FlashVSR teacher section that an interface-compatible replacement trains:

```text
LQ_proj_in -> student_condition_encoder
TCDecoder  -> conditional_student_decoder
```

Keep Wan VAE under the independent DiT main-latent examples. Label the unconditional decoder as output distillation only, not a drop-in TCDecoder replacement.

- [ ] **Step 2: Add Ubuntu environment and smoke commands**

Use Bash examples with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,test]"
PYTHON_BIN="$(command -v python)" ./scripts/run_smoke.sh work/smoke
```

Keep a compact Windows block that points to `scripts/run_smoke.ps1`.

- [ ] **Step 3: Convert data, training, and resume examples to Ubuntu paths**

Use `/data/dit_codec/LQ`, `/data/dit_codec/GT`, `/data/dit_codec/weights`, and `/data/dit_codec/runs` consistently. Use Bash backslash continuation for every multi-line command and replace `New-Item`/`Copy-Item` with `mkdir -p`/`cp` in the primary workflow.

- [ ] **Step 4: Replace weight download and verification with Linux commands**

Use `curl --fail --location --retry 5 --retry-delay 3 --continue-at -` for the three files. Generate a checksum file with the three fixed hashes and verify it from the weight directory using:

```bash
(cd "$WEIGHT_DIR" && sha256sum -c SHA256SUMS)
```

Retain the official Hugging Face URL as a commented alternative to the mirror. Use `du -ch` and `stat -c` for Linux size inspection.

- [ ] **Step 5: Run focused documentation tests**

Run:

```bash
python -m pytest -q tests/test_linux_support.py tests/test_config.py
```

Expected: all focused tests pass.

### Task 4: Audit platform assumptions and verify the final tree

**Files:**
- Inspect: `src/`, `configs/`, `scripts/`, `README.md`

- [ ] **Step 1: Search for Windows-only runtime assumptions**

Run:

```bash
rg -n "powershell|powershell.exe|D:/|\\\\|\.venv\\Scripts" src configs scripts README.md
```

Expected: no Windows-only path or command in Python runtime code; Windows references in README are confined to the explicit Windows section; `run_smoke.ps1` is the retained Windows runner; template paths are documented as user-replaceable.

- [ ] **Step 2: Run the full verification suite**

Run:

```bash
python -m pytest -q
ruff check .
mypy src tests
bash -n scripts/run_smoke.sh
git diff --check
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 3: Verify Git scope and ignored artifacts**

Check that only the design/plan, README, two smoke scripts, and focused tests changed. Confirm no `.pt`, `.pth`, `.ckpt`, `.safetensors`, `runs/`, or `work/` file is tracked.

- [ ] **Step 4: Commit the implementation**

Stage the exact intended paths and commit with:

```bash
git commit -m "feat: add Ubuntu FlashVSR training workflow"
```

