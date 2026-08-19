# Third-party source snapshots

This directory contains unmodified, minimal source snapshots for reproducibility and offline inspection. Model weights are intentionally excluded.

- `wan/`: Wan VAE implementation snapshot from DiffSynth-Studio.
- `flashvsr/`: FlashVSR `TCDecoder` and LQ projection implementations.
- `patches/`: optional documented patches; empty in v1.

Each snapshot has a machine-readable `manifest.yaml`. Compatibility code belongs in `src/distill_codec/integrations`, not in the vendored files.

