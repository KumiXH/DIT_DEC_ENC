import pytest
import torch

from distill_codec.color import (
    packed_6ch_to_rgb,
    rgb_to_packed_6ch,
    rgb_to_yuv,
    sparse_yuv420_to_rgb,
    yuv_to_rgb,
)
from distill_codec.contracts import ColorSpec, ContractError


def test_packed_layout_is_y00_y01_y10_y11_u_v():
    rgb = torch.tensor(
        [[[[0.0, 0.2], [0.4, 0.6]], [[0.1, 0.3], [0.5, 0.7]], [[0.2, 0.4], [0.6, 0.8]]]]
    )
    spec = ColorSpec(matrix="bt709", range="full")
    yuv = rgb_to_yuv(rgb, spec)

    packed = rgb_to_packed_6ch(rgb, spec)

    assert packed.shape == (1, 6, 1, 1)
    assert torch.allclose(packed[0, :4, 0, 0], yuv[0, 0].reshape(-1))
    assert torch.allclose(packed[0, 4:, 0, 0], yuv[0, 1:].mean(dim=(-2, -1)))


@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("value_range", ["full", "limited"])
def test_rgb_yuv_round_trip(matrix, value_range):
    torch.manual_seed(3)
    rgb = torch.rand(2, 3, 8, 8)
    spec = ColorSpec(matrix=matrix, range=value_range)

    restored = yuv_to_rgb(rgb_to_yuv(rgb, spec), spec)

    assert torch.allclose(restored, rgb, atol=2e-5, rtol=1e-5)


def test_sparse_decoder_reads_only_top_left_chroma_samples():
    spec = ColorSpec(matrix="bt709", range="full", chroma_upsample="nearest")
    sparse = torch.zeros(1, 3, 4, 4)
    sparse[:, 0] = 0.5
    sparse[:, 1, 0::2, 0::2] = 0.25
    sparse[:, 2, 0::2, 0::2] = 0.75
    sparse[:, 1, 1::2, 1::2] = 1.0
    sparse[:, 2, 1::2, 1::2] = 0.0

    rgb = sparse_yuv420_to_rgb(sparse, spec)
    expected_yuv = torch.cat(
        [
            torch.full((1, 1, 4, 4), 0.5),
            torch.full((1, 1, 4, 4), 0.25),
            torch.full((1, 1, 4, 4), 0.75),
        ],
        dim=1,
    )

    assert torch.allclose(rgb, yuv_to_rgb(expected_yuv, spec))


def test_packed_conversion_is_differentiable():
    rgb = torch.rand(1, 3, 8, 8, requires_grad=True)

    restored = packed_6ch_to_rgb(rgb_to_packed_6ch(rgb), output_size=(8, 8))
    restored.square().mean().backward()

    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()
    assert rgb.grad.abs().sum() > 0


def test_packing_rejects_odd_spatial_size():
    with pytest.raises(ContractError, match="even"):
        rgb_to_packed_6ch(torch.zeros(1, 3, 7, 8))

