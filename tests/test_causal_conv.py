"""Tests for causal convolution layers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from charavoice_engine.models.causal_conv import CausalConv1d, CausalResBlock1, CausalTransposedConv1d

# ---------------------------------------------------------------------------
# CausalConv1d
# ---------------------------------------------------------------------------

CONV_PARAMS = [
    pytest.param(3, 1, id="k3d1"),
    pytest.param(3, 3, id="k3d3"),
    pytest.param(3, 5, id="k3d5"),
    pytest.param(7, 1, id="k7d1"),
    pytest.param(7, 3, id="k7d3"),
    pytest.param(7, 5, id="k7d5"),
    pytest.param(11, 1, id="k11d1"),
    pytest.param(11, 3, id="k11d3"),
    pytest.param(11, 5, id="k11d5"),
]


@pytest.mark.parametrize("kernel_size,dilation", CONV_PARAMS)
def test_causal_conv1d_output_shape(kernel_size: int, dilation: int) -> None:
    """Output length must equal input length (stride=1)."""
    c = CausalConv1d(8, 16, kernel_size, dilation=dilation)
    x = torch.randn(2, 8, 100)
    y = c(x)
    assert y.shape == (2, 16, 100)


@pytest.mark.parametrize("kernel_size,dilation", CONV_PARAMS)
def test_causal_conv1d_causality(kernel_size: int, dilation: int) -> None:
    """Changing the last k input frames must not affect earlier outputs."""
    c = CausalConv1d(8, 16, kernel_size, dilation=dilation)
    c.eval()

    T = 100
    k = 5  # perturb last k frames
    x = torch.randn(1, 8, T)
    x_pert = x.clone()
    x_pert[:, :, -k:] = torch.randn(1, 8, k)

    with torch.no_grad():
        y = c(x)
        y_pert = c(x_pert)

    # Earlier outputs must be identical
    torch.testing.assert_close(y[:, :, :-k], y_pert[:, :, :-k])
    # Later outputs must differ (with overwhelming probability)
    assert not torch.allclose(y[:, :, -k:], y_pert[:, :, -k:])


@pytest.mark.parametrize("kernel_size,dilation", CONV_PARAMS)
def test_causal_conv1d_weight_compat(kernel_size: int, dilation: int) -> None:
    """Weight shape must match standard Conv1d."""
    causal = CausalConv1d(8, 16, kernel_size, dilation=dilation)
    standard = nn.Conv1d(8, 16, kernel_size, dilation=dilation)
    assert causal.conv.weight.shape == standard.weight.shape
    assert causal.conv.bias is not None and standard.bias is not None
    assert causal.conv.bias.shape == standard.bias.shape


# ---------------------------------------------------------------------------
# CausalTransposedConv1d
# ---------------------------------------------------------------------------

TCONV_PARAMS = [
    pytest.param(16, 8, id="k16s8"),
    pytest.param(4, 2, id="k4s2"),
]


@pytest.mark.parametrize("kernel_size,stride", TCONV_PARAMS)
def test_causal_transposed_conv1d_output_shape(kernel_size: int, stride: int) -> None:
    """Output length must equal input length * stride."""
    c = CausalTransposedConv1d(16, 8, kernel_size, stride)
    T = 50
    x = torch.randn(2, 16, T)
    y = c(x)
    assert y.shape == (2, 8, T * stride)


@pytest.mark.parametrize("kernel_size,stride", TCONV_PARAMS)
def test_causal_transposed_conv1d_causality(kernel_size: int, stride: int) -> None:
    """Changing the last input frame must not affect earlier outputs."""
    c = CausalTransposedConv1d(16, 8, kernel_size, stride)
    c.eval()

    T = 50
    k = 1  # perturb last k input frames
    x = torch.randn(1, 16, T)
    x_pert = x.clone()
    x_pert[:, :, -k:] = torch.randn(1, 16, k)

    with torch.no_grad():
        y = c(x)
        y_pert = c(x_pert)

    # Output samples before the perturbed region must be identical.
    # Last k input frames affect last k*stride output samples.
    safe = T * stride - k * stride
    torch.testing.assert_close(y[:, :, :safe], y_pert[:, :, :safe])
    # Perturbed region must differ
    assert not torch.allclose(y[:, :, safe:], y_pert[:, :, safe:])


@pytest.mark.parametrize("kernel_size,stride", TCONV_PARAMS)
def test_causal_transposed_conv1d_weight_compat(kernel_size: int, stride: int) -> None:
    """Weight shape must match standard ConvTranspose1d."""
    causal = CausalTransposedConv1d(16, 8, kernel_size, stride)
    standard = nn.ConvTranspose1d(16, 8, kernel_size, stride)
    assert causal.conv_t.weight.shape == standard.weight.shape
    assert causal.conv_t.bias is not None and standard.bias is not None
    assert causal.conv_t.bias.shape == standard.bias.shape


# ---------------------------------------------------------------------------
# CausalResBlock1
# ---------------------------------------------------------------------------

RESBLOCK_PARAMS = [
    pytest.param(3, (1, 3, 5), id="k3"),
    pytest.param(7, (1, 3, 5), id="k7"),
    pytest.param(11, (1, 3, 5), id="k11"),
]


@pytest.mark.parametrize("kernel_size,dilation", RESBLOCK_PARAMS)
def test_causal_resblock1_output_shape(kernel_size: int, dilation: tuple[int, int, int]) -> None:
    """Output shape must equal input shape (residual block preserves dimensions)."""
    block = CausalResBlock1(32, kernel_size, dilation)
    x = torch.randn(2, 32, 100)
    y = block(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("kernel_size,dilation", RESBLOCK_PARAMS)
def test_causal_resblock1_causality(kernel_size: int, dilation: tuple[int, int, int]) -> None:
    """Causality must hold through the entire ResBlock."""
    block = CausalResBlock1(32, kernel_size, dilation)
    block.eval()

    T = 100
    k = 10  # perturb last k frames (large enough to cover max receptive field)
    x = torch.randn(1, 32, T)
    x_pert = x.clone()
    x_pert[:, :, -k:] = torch.randn(1, 32, k)

    with torch.no_grad():
        y = block(x)
        y_pert = block(x_pert)

    # Earlier outputs must be identical
    torch.testing.assert_close(y[:, :, :-k], y_pert[:, :, :-k])
    # Later outputs must differ
    assert not torch.allclose(y[:, :, -k:], y_pert[:, :, -k:])


def test_causal_resblock1_weight_compat() -> None:
    """State dict key structure must match original ResBlock1."""
    from charavoice_engine.models.modules import ResBlock1

    causal = CausalResBlock1(32, 3, (1, 3, 5))
    original = ResBlock1(32, 3, (1, 3, 5))

    causal_keys = {k.rsplit(".", 1)[-1] for k in causal.state_dict()}
    original_keys = {k.rsplit(".", 1)[-1] for k in original.state_dict()}
    assert causal_keys == original_keys

    # Verify convs1/convs2 structure: same number of sublayers
    assert len(causal.convs1) == len(original.convs1)
    assert len(causal.convs2) == len(original.convs2)

    # Verify individual weight shapes match
    for c_mod, o_mod in zip(causal.convs1, original.convs1, strict=True):
        assert c_mod.conv.weight.shape == o_mod.weight.shape  # type: ignore[union-attr]
    for c_mod, o_mod in zip(causal.convs2, original.convs2, strict=True):
        assert c_mod.conv.weight.shape == o_mod.weight.shape  # type: ignore[union-attr]
