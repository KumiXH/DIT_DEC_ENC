# Private Codec V0 Template Design

**Date:** 2026-08-20
**Status:** Ready for implementation after user review

## Goal

Replace the ambiguous root-level empty private-codec placeholders with one runnable,
well-commented `v0` example that the user can copy to `v1`, `v2`, and later versions.
The example must exercise the existing public encoder and conditional-decoder bridge
contracts without adding version-specific branches to the shared factory.

## Directory Layout

The shared infrastructure remains version-independent:

```text
src/private_codec/
|-- __init__.py
|-- bridge.py
|-- factories.py
`-- versions/
    |-- __init__.py
    `-- v0/
        |-- __init__.py
        |-- base_network.py
        |-- wrapped_network.py
        `-- entrypoints.py
```

The old root-level `base_network.py`, `wrapped_network.py`, and `entrypoints.py`
placeholders are removed. Keeping both layouts would create two apparent default
entrypoints and make future copies ambiguous.

## V0 Example Network

`base_network.py` contains small reusable convolution blocks rather than any project
integration logic. The encoder uses three stride-2 convolutions and returns BCHW
latent with configurable channel count, defaulting to 16. Thus an RGB tensor of
shape `[B, 3, H, W]` produces `[B, 16, H/8, W/8]` for image sizes divisible by 8.

The decoder derives an example target size from the latent spatial dimensions and
the configured 8x scale, projects both latent and LQ RGB to features, interpolates
both streams to that target, and returns three-channel RGB. This supports unequal
LQ/GT super-resolution examples while documenting that the private implementation,
not the shared bridge or `teacher_reference`, owns size handling.

`wrapped_network.py` demonstrates the user's real pattern: wrapper classes inherit
the base networks, select different initialization defaults, and expose private
forward signatures. This file contains no project factory or adapter logic.

## Entrypoints

`entrypoints.py` implements the four stable integration functions:

```python
build_encoder(**kwargs) -> nn.Module
run_encoder(*, network, rgb, teacher_reference, **kwargs) -> Tensor
build_decoder(**kwargs) -> nn.Module
run_decoder(
    *, network, lq_rgb, dit_latent, teacher_reference, **kwargs
) -> Tensor
```

Comments identify the exact copy-and-edit locations:

- imports and concrete wrapper class names;
- network construction and initialization arguments in `build_*`;
- project-to-private input conversion and private forward calls in `run_*`;
- optional inspection of `teacher_reference`, without using it to transform tensors.

The v0 functions keep all operations differentiable and do not detach outputs or
wrap training forward passes in `torch.no_grad()`.

## Configuration

`configs/students/private_codec.yaml` points the encoder and conditional decoder to:

```yaml
private_codec.versions.v0.entrypoints:build_encoder
private_codec.versions.v0.entrypoints:run_encoder
private_codec.versions.v0.entrypoints:build_decoder
private_codec.versions.v0.entrypoints:run_decoder
```

The legacy unconditional decoder also points at a usable v0 decoder class so the
shared student include has no dead import path. Version-specific hyperparameters
remain in `builder_kwargs` and `runner_kwargs`; `factories.py` remains unchanged.

## Tutorial

`PRIVATE_CODEC_INTEGRATION_TUTORIAL.md` is updated so v0 is the executable reference,
not an empty fill-in target. It explains how to:

1. run the supplied v0 bridge probes;
2. copy the complete `versions/v0` directory to `versions/v1`;
3. replace or edit the three version-owned modules;
4. change only the YAML import paths and version parameters;
5. leave `bridge.py` and `factories.py` unchanged.

The tutorial retains the public/private tensor contracts and the reference-only
meaning of `teacher_reference`.

## Error Handling

The example networks reject malformed ranks, non-RGB inputs, mismatched batch sizes,
unexpected latent channel counts, and image sizes that cannot produce the declared
1/8 encoder latent. Errors include actual shapes so copied versions are easy to
debug. The shared bridge continues to validate builder and runner return types.

## Testing

Tests first establish the desired artifact and behavior contract:

- v0 modules exist and root-level placeholders do not;
- default YAML imports v0 entrypoints;
- v0 encoder returns `[B, 16, H/8, W/8]` and supports gradients;
- v0 conditional decoder returns RGB at the latent-declared 8x target size and
  supports gradients through both latent and LQ inputs;
- v0 entrypoints work through the real shared bridges;
- invalid shapes fail with useful messages;
- tutorial paths and copy instructions match the actual tree.

Focused tests are followed by the repository test suite and static checks already
used by this project. The known OpenMP child-process baseline failure remains a
separate environment limitation and is not treated as a v0 failure.
