# FlashVSR Target Clarification and Ubuntu Support Design

## Goal

Clarify the FlashVSR distillation target and make the repository's documented and scripted workflows usable on Ubuntu without changing the Windows workflow.

## Scope

FlashVSR's replaceable distillation path is:

```text
LQ RGB/video -> LQ_proj_in -> DiT condition features
latent + condition -> TCDecoder -> RGB/YUV output
```

The project keeps these as two separately trainable recipes:

- `flashvsr_lq_proj_distill` distills `LQ_proj_in` into `student_condition_encoder`.
- `flashvsr_decoder_conditional_student` distills `TCDecoder` into `conditional_student_decoder`.

The unconditional decoder recipe remains available for a black-box student that only accepts the main latent, but its documentation must state that it is not an interface-equivalent TCDecoder replacement. Wan VAE encoder/decoder recipes remain as an independent DiT main-latent case and are not presented as the FlashVSR condition path.

## Ubuntu workflow

Add a POSIX shell smoke runner at `scripts/run_smoke.sh`. It must:

- resolve the repository root from the script location;
- use `${PYTHON_BIN:-python3}`;
- fail on unset variables and command errors with `set -euo pipefail`;
- invoke the same mock-data, probe, smoke training, and resume checks as the PowerShell runner;
- write generated artifacts under an ignored `work/` directory.

README command examples will include Ubuntu instructions for:

- Python virtual environment and editable installation;
- CPU/CUDA PyTorch installation selection;
- mock data generation and probing;
- smoke training and resume;
- real training configuration overrides;
- downloading `Wan2.1_VAE.pth`, `LQ_proj_in.ckpt`, and `TCDecoder.ckpt` with `curl` retry/resume options;
- SHA-256 verification with `sha256sum`.

The current PowerShell commands remain documented as a Windows option. Weight URLs and expected hashes remain unchanged; only the command form changes.

## Compatibility boundary

The audit will inspect Python imports, path handling, config loading, subprocess calls, and shell entry points for Windows-only assumptions. It will not claim a CUDA or Ubuntu runtime test from this Windows environment. The acceptance evidence is:

- shell syntax validation for `scripts/run_smoke.sh` when a Bash checker is available;
- the existing Python test suite on the final tree;
- static checks for README references, script paths, and ignored output locations;
- Ruff and mypy checks.

## Testing

Add focused tests that verify:

1. the Linux smoke script exists, is executable in a Unix checkout, references `python3`/`PYTHON_BIN`, and does not use PowerShell syntax;
2. the README contains the Linux install, smoke, training, resume, weight download, and checksum commands;
3. the FlashVSR recipe/config mapping names `LQ_proj_in` and `TCDecoder` as the replaceable teacher components.

The full suite must continue to pass, and no model weights or run artifacts may become tracked files.

