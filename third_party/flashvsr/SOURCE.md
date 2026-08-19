# FlashVSR snapshot

Source: https://github.com/OpenImagingLab/FlashVSR/tree/6dd38e57203af4efca97df82c659f5d5a2dcf51a

Copied files are byte-for-byte versions of:

- `examples/WanVSR/utils/TCDecoder.py`
- `examples/WanVSR/utils/utils.py`

The v1.1 official inference script creates `Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)` and builds TCDecoder with `new_latent_channels=16+768`. In `TCDecoder.py`, the 768 channels are produced from the LQ condition by `4x8x8` space-time-to-channel rearrangement and concatenated with the 16-channel latent. The wrapper accepts a 16-channel latent plus RGB LQ condition and performs this official concatenation inside TCDecoder.
