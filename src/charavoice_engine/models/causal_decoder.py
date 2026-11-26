"""Causal HiFi-GAN Decoder for streaming inference.

CausalGenerator mirrors the original Generator (models_bilingual_jp_extra.py) but uses
CausalConv1d / CausalTransposedConv1d / CausalResBlock1 for overlap-save
streaming. Weight transfer from the original Generator is supported via
remap_generator_state_dict().
"""

from __future__ import annotations

import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from charavoice_engine.models.causal_conv import (
    LRELU_SLOPE,
    CausalConv1d,
    CausalResBlock1,
    CausalTransposedConv1d,
    _init_conv_weights,
)

# Regex patterns for remapping Generator state_dict keys to CausalGenerator.
# More specific patterns first; each key is matched against patterns in order.
_REMAP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # resblocks.{i}.convs{1|2}.{j}.{param} → resblocks.{i}.convs{1|2}.{j}.conv.{param}
    (re.compile(r"^(resblocks\.\d+\.convs[12]\.\d+)\.(.+)$"), r"\1.conv.\2"),
    # ups.{i}.{param} → ups.{i}.conv_t.{param}
    (re.compile(r"^(ups\.\d+)\.(.+)$"), r"\1.conv_t.\2"),
    # conv_pre.{param} → conv_pre.conv.{param}
    (re.compile(r"^(conv_pre)\.(.+)$"), r"\1.conv.\2"),
    # conv_post.{param} → conv_post.conv.{param}
    (re.compile(r"^(conv_post)\.(.+)$"), r"\1.conv.\2"),
]


def remap_generator_state_dict(
    src_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap Generator state_dict keys for CausalGenerator.

    The original Generator uses bare nn.Conv1d / nn.ConvTranspose1d,
    while CausalGenerator wraps them in CausalConv1d (.conv) and
    CausalTransposedConv1d (.conv_t). This function inserts the wrapper
    attribute name into each key.

    Keys starting with ``cond.`` are left unchanged (1x1 conv, no wrapper).

    Args:
        src_state_dict: State dict from original Generator.

    Returns:
        New state dict with remapped keys, suitable for
        ``CausalGenerator.load_state_dict()``.
    """
    dst: dict[str, torch.Tensor] = {}
    for key, value in src_state_dict.items():
        new_key = key
        for pattern, replacement in _REMAP_PATTERNS:
            result, n = pattern.subn(replacement, key)
            if n > 0:
                new_key = result
                break
        dst[new_key] = value
    return dst


class CausalGenerator(nn.Module):
    """Causal HiFi-GAN Generator for overlap-save streaming inference.

    Architecture mirrors Generator from models_bilingual_jp_extra.py:

    - conv_pre (CausalConv1d, k=7)
    - N upsample stages: CausalTransposedConv1d + 3x CausalResBlock1 (MRF)
    - conv_post (CausalConv1d, k=7, no bias) + tanh
    - Optional speaker conditioning via 1x1 Conv1d
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Pre-conv: no weight_norm (matches original)
        self.conv_pre = CausalConv1d(initial_channel, upsample_initial_channel, 7)

        # Upsample layers: weight_norm on inner conv_t
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            ct = CausalTransposedConv1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                k,
                u,
            )
            weight_norm(ct.conv_t)
            self.ups.append(ct)

        # ResBlocks (MRF): num_upsamples * num_kernels blocks
        self.resblocks = nn.ModuleList()
        ch = None
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(CausalResBlock1(ch, k, tuple(d)))  # type: ignore[arg-type]

        assert ch is not None

        # Post-conv: no weight_norm, no bias (matches original)
        self.conv_post = CausalConv1d(ch, 1, 7, bias=False)

        # Init upsample weights
        self.ups.apply(_init_conv_weights)

        # Speaker conditioning: plain 1x1 Conv1d (no causal wrapper needed)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            assert xs is not None
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            layer.remove_weight_norm()  # type: ignore[union-attr]
        for layer in self.resblocks:
            layer.remove_weight_norm()  # type: ignore[union-attr]
