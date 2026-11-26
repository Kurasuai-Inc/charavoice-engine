"""Causal convolution layers for streaming HiFi-GAN decoder.

Left-padded Conv1d / right-trimmed ConvTranspose1d ensure outputs depend
only on past inputs, enabling overlap-save streaming inference.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class CausalConv1d(nn.Module):
    """Conv1d with causal (left-only) padding.

    Output length equals input length for stride=1.
    Weight shape is identical to standard Conv1d (allows direct weight transfer).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.pad_size = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            **kwargs,  # type: ignore[arg-type]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_size, 0))
        return self.conv(x)

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.conv)


class CausalTransposedConv1d(nn.Module):
    """ConvTranspose1d with causal right-trimming.

    Uses padding=0 to get full output, then trims the right side so that
    output length = input length * stride.
    Weight shape is identical to standard ConvTranspose1d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.conv_t = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            **kwargs,  # type: ignore[arg-type]
        )
        self.trim_right = kernel_size - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_t(x)
        if self.trim_right > 0:
            out = out[:, :, : -self.trim_right]
        return out

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.conv_t)


def _init_conv_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Init weights for nn.Conv1d / nn.ConvTranspose1d (skips wrapper classes)."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)


class CausalResBlock1(nn.Module):
    """Causal version of ResBlock1 from modules.py.

    Replaces symmetric-padded Conv1d with CausalConv1d.
    Structure: 3 pairs of (convs1, convs2) with residual connections.
    State dict keys match the original ResBlock1 (convs1.{0,1,2}, convs2.{0,1,2}).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        super().__init__()

        def _make_causal(d: int) -> CausalConv1d:
            c = CausalConv1d(channels, channels, kernel_size, dilation=d)
            weight_norm(c.conv)
            return c

        self.convs1 = nn.ModuleList([_make_causal(dilation[0]), _make_causal(dilation[1]), _make_causal(dilation[2])])
        self.convs1.apply(_init_conv_weights)

        self.convs2 = nn.ModuleList([_make_causal(1), _make_causal(1), _make_causal(1)])
        self.convs2.apply(_init_conv_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        for l in self.convs1:
            l.remove_weight_norm()  # type: ignore[union-attr]
        for l in self.convs2:
            l.remove_weight_norm()  # type: ignore[union-attr]
