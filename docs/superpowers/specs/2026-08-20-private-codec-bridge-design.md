# Private Codec Bridge Design

**Date:** 2026-08-20
**Status:** Ready for implementation after user review

## Goal

Provide a stable project-facing bridge for private encoder and conditional decoder networks. The project passes only the tensors it owns; each private network version owns all conversion, preprocessing, nesting, and version-specific call logic behind the bridge.

The public contracts are:

```python
dit_latent = encoder(rgb)
rgb = decoder(dit_latent, lq_rgb)
```

The private entry functions are:

```python
dit_latent = run_encoder(
    network=network,
    rgb=rgb,
    teacher_reference=teacher_reference,
)

rgb = run_decoder(
    network=network,
    lq_rgb=lq_rgb,
    dit_latent=dit_latent,
    teacher_reference=teacher_reference,
)
```

`teacher_reference` contains theoretical teacher input and output shapes for inspection and debugging. It never resizes, pads, crops, validates, or otherwise changes a real tensor.

## Confirmed Ownership Boundary

### Encoder

The project supplies one RGB tensor in `BCHW` layout. It does not convert the RGB tensor to packed YUV, normalize it for the private network, resize it, pad it, or inspect the private network's nested modules.

The private implementation receives `rgb`, performs every required internal transformation, and returns the latent tensor expected by the selected project `latent_spec`.

### Conditional decoder

The project-facing adapter keeps its existing argument order:

```python
conditional_student_decoder(dit_latent, lq_rgb)
```

The bridge deliberately converts that call to keyword arguments for the private runner:

```python
run_decoder(
    network=network,
    lq_rgb=lq_rgb,
    dit_latent=dit_latent,
    teacher_reference=teacher_reference,
)
```

This means the project contract and the private function signature can use different argument order without ambiguity. The private implementation owns all conversion, normalization, resize, padding, fusion, and nested-network invocation, then returns RGB to the project.

## Approaches Considered

### Selected: stable Bridge plus version-specific builder and runner functions

`src/private_codec/factories.py` always constructs one of two stable bridge classes. YAML selects importable builder and runner functions. Adding a network version means adding a module or package with four entry functions and pointing YAML at it; the shared factory is not edited.

Advantages:

- private preprocessing and nested-network details remain private;
- the project sees one stable encoder and decoder interface;
- new versions do not create an expanding list of factory functions;
- builder, runner, and bridge behavior can be tested independently;
- keyword forwarding prevents decoder argument-order mistakes.

### Rejected: register every version in `factories.py`

This would require editing a central file for every V2, V3, or experimental network and would turn the factory into a version registry. It creates unnecessary merge conflicts and couples private network evolution to project infrastructure.

### Rejected: make every private class conform directly to project adapters

This would force project-specific `forward` conventions and preprocessing into every private class. It would also make nested wrappers harder to change and repeat integration logic across versions.

## Components

### `src/private_codec/bridge.py`

This module provides:

```python
class PrivateEncoderBridge(nn.Module): ...
class PrivateConditionalDecoderBridge(nn.Module): ...
```

Both bridges accept these constructor values:

```python
builder: str
runner: str
builder_kwargs: Mapping[str, Any] | None
runner_kwargs: Mapping[str, Any] | None
teacher_reference: Mapping[str, Any]
```

`builder` and `runner` use the existing `"package.module:symbol"` notation. The bridge imports the builder, constructs the private network once, and requires the builder result to be `torch.nn.Module`.

On every forward call, the bridge invokes the runner with keyword arguments. The runner result must be a `torch.Tensor`. The bridge does not inspect, convert, detach, or clamp that tensor.

The bridge stores a defensive copy of `teacher_reference` and passes a fresh copy to each runner call. Private code may annotate or inspect its copy without changing project configuration or later calls.

### `src/private_codec/factories.py`

The factory remains version-independent and exposes:

```python
create_encoder(...)
create_conditional_decoder(...)
```

It constructs the corresponding bridge. It does not import a concrete private network class itself and does not contain V1/V2/V3 branches.

### Private version entry module

The first fill-in template remains under `src/private_codec/`. The user's two network files remain:

```text
src/private_codec/base_network.py
src/private_codec/wrapped_network.py
```

An entry module supplies the integration layer:

```python
def build_encoder(**kwargs) -> nn.Module: ...
def run_encoder(*, network, rgb, teacher_reference, **kwargs) -> Tensor: ...
def build_decoder(**kwargs) -> nn.Module: ...
def run_decoder(
    *, network, lq_rgb, dit_latent, teacher_reference, **kwargs
) -> Tensor: ...
```

This is the only place that needs to know how a specific private version is initialized and called. A later version can live in a new module or package, for example `private_codec.versions.v2.entrypoints`, and YAML can select it without a factory edit.

### `src/distill_codec/config.py`

Teacher references are opt-in at component level:

```yaml
teacher_reference: auto
```

Only a component that declares this field receives a generated `teacher_reference` in its factory keyword arguments. Existing external factories receive no new keyword and remain compatible.

`build_components()` derives the reference from `recipe`, `data`, `latent_provider`, and `latent_spec` after includes and overrides have been applied. Explicit factory `kwargs.teacher_reference` is rejected when `teacher_reference: auto` is also present, because two sources would be ambiguous.

### `src/distill_codec/recipes.py`

For a FlashVSR conditional decoder, the teacher and student inputs are separated:

```text
raw_lq_rgb = batch.lq_rgb
teacher_lq_rgb = raw_lq_rgb, or bicubic-aligned to GT for the teacher

tc_decoder(dit_latent, teacher_lq_rgb)
conditional_student_decoder(dit_latent, raw_lq_rgb)
```

The teacher keeps the current alignment behavior. The private student always receives the original LQ RGB tensor and decides for itself whether and how to resize it.

## Teacher Reference Schema

The reference contains Python lists so it is easy to print, log, compare, or serialize. The unknown batch dimension is `None`.

Encoder example:

```python
{
    "role": "encoder",
    "inputs": {
        "rgb": {
            "layout": "BCHW",
            "shape": [None, 3, 256, 256],
            "source": "gt",
        },
    },
    "outputs": {
        "latent": {
            "layout": "BCHW",
            "shape": [None, 16, 32, 32],
        },
    },
}
```

Conditional decoder example:

```python
{
    "role": "conditional_decoder",
    "inputs": {
        "lq_rgb": {
            "layout": "BCHW",
            "shape": [None, 3, 256, 256],
            "source": "teacher_condition",
        },
        "dit_latent": {
            "layout": "BCHW",
            "shape": [None, 16, 32, 32],
        },
    },
    "outputs": {
        "rgb": {
            "layout": "BCHW",
            "shape": [None, 3, 256, 256],
        },
    },
}
```

### Derivation rules

- `data.lq_size` and `data.gt_size` are interpreted as `[height, width]`.
- Encoder RGB size comes from `recipe.source`; the existing default is `gt`.
- Decoder output RGB size comes from `data.gt_size`.
- FlashVSR teacher condition RGB size is `data.gt_size`, matching the existing teacher-side alignment behavior.
- Decoder latent source comes from `latent_provider.source`; the existing default is `gt`.
- Spatial latent size is `[height // spatial_downsample, width // spatial_downsample]`, matching `LatentSpec.validate_tensor()`.
- Latent channel count and layout come from `latent_spec`.
- For `BCTHW`, the reference includes the theoretical temporal dimension derived from the relevant configured temporal frame count and `temporal_downsample`.
- Missing or malformed sizes produce a `ContractError` during component construction rather than silently generating an incomplete reference.

These values describe the teacher path only. In particular, the real `lq_rgb` passed to the private decoder may have a different spatial size from `teacher_reference["inputs"]["lq_rgb"]["shape"]`.

## Configuration Shape

The student include will use RGB adapters and the stable bridge factories:

```yaml
components:
  student_encoder:
    backend: external
    factory: private_codec.factories:create_encoder
    checkpoint: null
    teacher_reference: auto
    kwargs:
      builder: private_codec.entrypoints:build_encoder
      runner: private_codec.entrypoints:run_encoder
      builder_kwargs: {}
      runner_kwargs: {}
    adapter:
      kind: encoder
      input_mode: rgb

  conditional_student_decoder:
    backend: external
    factory: private_codec.factories:create_conditional_decoder
    checkpoint: null
    teacher_reference: auto
    kwargs:
      builder: private_codec.entrypoints:build_decoder
      runner: private_codec.entrypoints:run_decoder
      builder_kwargs: {}
      runner_kwargs: {}
    adapter:
      kind: decoder
      accepts_condition: true
      output_mode: rgb
```

The private network is responsible for loading any special internal checkpoint format in its builder. The project's standard `checkpoint` field remains available for a conventional `state_dict` loaded onto the completed bridge.

## Adding Another Private Version

Adding a version requires no change to `src/private_codec/factories.py`, `src/distill_codec/adapters.py`, or `src/distill_codec/recipes.py`.

1. Add the version's network files and entry module, for example:

   ```text
   src/private_codec/versions/v2/
   ├── __init__.py
   ├── base_network.py
   ├── wrapped_network.py
   └── entrypoints.py
   ```

2. Implement the same four builder and runner functions in that version's `entrypoints.py`.
3. Copy the student YAML and change only the `builder`, `runner`, and their kwargs.

The runner is the intended middle zone. It may translate project RGB to any private representation, choose among nested networks, alter call order, and translate private output back to the required project tensor.

## Error Handling

- An invalid `module:symbol` path reports which builder or runner could not be imported.
- A builder that does not return `nn.Module` raises `TypeError` during component construction.
- A runner that does not return `Tensor` raises `TypeError` on forward.
- The decoder bridge requires both `dit_latent` and `lq_rgb` through the existing conditional adapter contract.
- Bridge errors do not fall back to implicit packing, resizing, output unwrapping, or color conversion.
- Existing `EncoderAdapter` latent validation and `DecoderAdapter` RGB shape validation remain the final project-side contract checks.

## Testing

Implementation follows red-green-refactor and adds focused coverage for:

- encoder builder and keyword runner forwarding;
- conditional decoder conversion from project argument order to private keyword arguments;
- `builder_kwargs` and `runner_kwargs` forwarding;
- rejection of non-`nn.Module` builders;
- rejection of non-Tensor runner outputs;
- automatic encoder teacher-reference derivation;
- automatic conditional-decoder teacher-reference derivation;
- injection only when `teacher_reference: auto` is declared;
- preservation of real tensor identity and shape through the reference-only bridge;
- teacher use of aligned LQ while the private student receives original LQ;
- validity of the fill-in YAML and tutorial examples.

After focused tests pass, the complete repository verification is:

```text
pytest -q
ruff check .
python -m mypy src tests
bash -n scripts/run_smoke.sh
git diff --check
```

## Out of Scope

- The project will not implement private normalization, color conversion, resize, padding, cropping, or nested-network selection.
- The project will not enforce that real tensors match `teacher_reference`.
- The project will not infer private constructor arguments.
- The project will not add a central registry of private network versions.
- The project will not change the teacher model's existing preprocessing semantics.
- This change does not add or benchmark fast attention; it is independent of the private codec interface.
