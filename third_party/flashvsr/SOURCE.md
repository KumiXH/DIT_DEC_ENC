# FlashVSR snapshot

Source: https://github.com/OpenImagingLab/FlashVSR/tree/6dd38e57203af4efca97df82c659f5d5a2dcf51a

Copied files are byte-for-byte versions of:

- `examples/WanVSR/utils/TCDecoder.py`
- `examples/WanVSR/utils/utils.py`

The v1.1 official inference script creates `Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)` and builds TCDecoder with `new_latent_channels=16+768`. The snapshot factory preserves that explicit interface; it does not synthesize the extra 768 channels.

