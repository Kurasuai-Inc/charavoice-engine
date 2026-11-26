"""Tests for CausalGenerator (Causal HiFi-GAN Decoder)."""

from __future__ import annotations

import torch

from charavoice_engine.models.causal_decoder import CausalGenerator, remap_generator_state_dict

# JP-Extra default config (from config.py HyperParametersModel defaults)
JP_EXTRA_CONFIG: dict[str, object] = {
    "initial_channel": 192,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 8, 2, 2],
    "gin_channels": 512,
}
HOP_SIZE = 512  # product of upsample_rates: 8*8*2*2*2


def _make_generator(**overrides: object) -> CausalGenerator:
    config = {**JP_EXTRA_CONFIG, **overrides}
    return CausalGenerator(**config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_output_shape_no_g() -> None:
    """Output must be [b, 1, T_in * hop_size] without speaker conditioning."""
    gen = _make_generator()
    gen.eval()
    T = 20
    x = torch.randn(1, 192, T)
    with torch.no_grad():
        y = gen(x)
    assert y.shape == (1, 1, T * HOP_SIZE)


def test_output_shape_with_g() -> None:
    """Speaker conditioning must not change output shape."""
    gen = _make_generator()
    gen.eval()
    T = 20
    x = torch.randn(1, 192, T)
    g = torch.randn(1, 512, 1)
    with torch.no_grad():
        y = gen(x, g=g)
    assert y.shape == (1, 1, T * HOP_SIZE)


def test_output_range() -> None:
    """Output values must be in [-1, 1] due to tanh."""
    gen = _make_generator()
    gen.eval()
    x = torch.randn(1, 192, 20)
    with torch.no_grad():
        y = gen(x)
    assert y.min() >= -1.0
    assert y.max() <= 1.0


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_causality() -> None:
    """Changing the last k input frames must not affect earlier outputs."""
    gen = _make_generator()
    gen.eval()

    T = 30
    k = 3  # perturb last k input frames
    x = torch.randn(1, 192, T)
    x_pert = x.clone()
    x_pert[:, :, -k:] = torch.randn(1, 192, k)

    with torch.no_grad():
        y = gen(x)
        y_pert = gen(x_pert)

    # Earlier outputs must be identical
    safe_samples = (T - k) * HOP_SIZE
    torch.testing.assert_close(y[:, :, :safe_samples], y_pert[:, :, :safe_samples])
    # Later outputs must differ
    assert not torch.allclose(y[:, :, safe_samples:], y_pert[:, :, safe_samples:])


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------


def _make_original_generator() -> torch.nn.Module:
    from charavoice_engine.models.models_bilingual_jp_extra import Generator

    return Generator(
        initial_channel=192,
        resblock_str="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        gin_channels=512,
    )


def test_remap_key_count() -> None:
    """Remap must produce keys matching CausalGenerator's state_dict exactly."""
    orig = _make_original_generator()
    causal = _make_generator()

    remapped_keys = set(remap_generator_state_dict(orig.state_dict()).keys())
    causal_keys = set(causal.state_dict().keys())

    assert remapped_keys == causal_keys


def test_weight_transfer() -> None:
    """Remapped Generator state dict must load into CausalGenerator without errors."""
    orig = _make_original_generator()
    causal = _make_generator()

    remapped = remap_generator_state_dict(orig.state_dict())
    result = causal.load_state_dict(remapped, strict=True)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0


def test_weight_transfer_valid_output() -> None:
    """CausalGenerator with transferred weights must produce valid output."""
    orig = _make_original_generator()
    causal = _make_generator()

    remapped = remap_generator_state_dict(orig.state_dict())
    causal.load_state_dict(remapped, strict=True)
    causal.eval()

    x = torch.randn(1, 192, 20)
    g = torch.randn(1, 512, 1)
    with torch.no_grad():
        y = causal(x, g=g)

    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert y.abs().max() <= 1.0
    assert y.abs().max() > 0.0  # not all zeros


# ---------------------------------------------------------------------------
# Speaker conditioning
# ---------------------------------------------------------------------------


def test_speaker_conditioning_changes_output() -> None:
    """Speaker conditioning must change the output."""
    gen = _make_generator()
    gen.eval()

    x = torch.randn(1, 192, 20)
    g = torch.randn(1, 512, 1)

    with torch.no_grad():
        y_no_g = gen(x)
        y_with_g = gen(x, g=g)

    assert not torch.allclose(y_no_g, y_with_g)


def test_no_gin_channels() -> None:
    """CausalGenerator must work without speaker conditioning (gin_channels=0)."""
    gen = _make_generator(gin_channels=0)
    gen.eval()

    x = torch.randn(1, 192, 20)
    with torch.no_grad():
        y = gen(x)
    assert y.shape == (1, 1, 20 * HOP_SIZE)


# ---------------------------------------------------------------------------
# Weight norm management
# ---------------------------------------------------------------------------


def test_remove_weight_norm() -> None:
    """remove_weight_norm must complete without errors and remove parametrizations."""
    gen = _make_generator()
    gen.remove_weight_norm()

    # After removing, ups should have plain .weight (not weight_g/weight_v)
    for up_layer in gen.ups:
        assert hasattr(up_layer.conv_t, "weight")  # type: ignore[union-attr]
        assert not hasattr(up_layer.conv_t, "weight_g")  # type: ignore[union-attr]
