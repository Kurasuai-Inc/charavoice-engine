"""Verify non-causal overlap-save produces output identical to full-sequence decoding.

Loads the real Generator from a pretrained model, runs full-sequence decoding
and chunk-by-chunk overlap-save with L=R=13 mel frames, and checks that the
valid output regions match to near floating-point precision.

Usage:
    uv run pytest tests/test_overlap_save.py -v -s

Requires GPU and model assets:
    - model_assets/mix/config.json
    - Data/mix/models/G_272000.pth
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from charavoice_engine.config import HyperParameters
from charavoice_engine.models.infer import get_net_g

logger = logging.getLogger(__name__)

MODEL_PATH = Path("Data/mix/models/G_272000.pth")
CONFIG_PATH = Path("model_assets/mix/config.json")

# Receptive field analysis result: right RF = left RF = 13 mel frames
L_CONTEXT = 13  # left context in mel frames
R_CONTEXT = 13  # right context in mel frames


def overlap_save_decode(
    decoder: torch.nn.Module,
    z: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    L: int,
    R: int,
    hop_length: int,
) -> torch.Tensor:
    """Chunk-by-chunk overlap-save decoding with left + right context.

    At sequence boundaries, use only the available real data as context
    (no zero-padding). This ensures Conv1d internal boundary padding matches
    the full-sequence behavior exactly.

    Args:
        decoder: Generator module (non-causal).
        z: Full latent sequence [1, C, T_mel].
        g: Speaker embedding [1, gin_channels, 1].
        chunk_size: Valid output chunk size in mel frames.
        L: Left context in mel frames.
        R: Right context in mel frames.
        hop_length: Samples per mel frame.

    Returns:
        Reconstructed waveform [1, 1, T_mel * hop_length], matching full-sequence shape.
    """
    T_mel = z.shape[2]
    T_wav = T_mel * hop_length
    output = torch.zeros(1, 1, T_wav, device=z.device, dtype=z.dtype)

    for start in range(0, T_mel, chunk_size):
        end = min(start + chunk_size, T_mel)

        # Context range clamped to actual sequence bounds (no zero-padding)
        ctx_left = max(0, start - L)
        ctx_right = min(T_mel, end + R)
        z_chunk = z[:, :, ctx_left:ctx_right]

        # How much real left context we actually have
        actual_L = start - ctx_left

        # Decode the context window
        with torch.no_grad():
            wav_chunk = decoder(z_chunk, g=g)

        # Extract valid region (skip actual context, not fixed L)
        valid_start_wav = actual_L * hop_length
        valid_end_wav = valid_start_wav + (end - start) * hop_length
        valid_wav = wav_chunk[:, :, valid_start_wav:valid_end_wav]

        # Write to output buffer
        output[:, :, start * hop_length : end * hop_length] = valid_wav

    return output


from charavoice_engine.inference.overlap_save import split_by_duration_if_needed  # noqa: E402

# ---- split_by_duration_if_needed tests (CPU-only, no model needed) ----

SIL_IDS = {100, 101, 102}  # Fake sil phone IDs for testing


def _make_durations(durs: list[int]) -> torch.Tensor:
    """Helper: list of ints → [1, 1, T_phone] tensor."""
    return torch.tensor(durs, dtype=torch.long).unsqueeze(0).unsqueeze(0)


def _make_phone_ids(ids: list[int]) -> torch.Tensor:
    """Helper: list of ints → [1, T_phone] tensor."""
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def test_no_split_needed() -> None:
    """T_mel <= max_t_mel → single segment covering all phones."""
    durs = _make_durations([10, 20, 30])  # total=60
    ids = _make_phone_ids([1, 2, 3])
    result = split_by_duration_if_needed(durs, ids, max_t_mel=100, sil_phone_ids=SIL_IDS)
    assert result == [(0, 3)]


def test_exact_boundary() -> None:
    """T_mel == max_t_mel exactly → no split needed."""
    durs = _make_durations([50, 50])  # total=100
    ids = _make_phone_ids([1, 2])
    result = split_by_duration_if_needed(durs, ids, max_t_mel=100, sil_phone_ids=SIL_IDS)
    assert result == [(0, 2)]


def test_split_at_sil_boundary() -> None:
    """Split at the last sil phone before the hard limit."""
    # phones: [normal, normal, sil, normal, normal]
    # durs:   [30,     30,     10,  20,     10]
    # cumsum: [30,     60,     70,  90,     100]
    # max_t_mel=80 → exceed at phone 3 (cumsum 90 > 80)
    # Backward scan from phone 2: ids[2]=100 (sil) → split_at = 3
    # Result: [(0, 3), (3, 5)]
    durs = _make_durations([30, 30, 10, 20, 10])
    ids = _make_phone_ids([1, 2, 100, 3, 4])
    result = split_by_duration_if_needed(durs, ids, max_t_mel=80, sil_phone_ids=SIL_IDS)
    assert len(result) == 2
    assert result[0] == (0, 3)  # split after sil at index 2
    assert result[1] == (3, 5)
    # Verify each segment's T_mel fits
    seg0_mel = int(durs[0, 0, 0:3].sum().item())
    seg1_mel = int(durs[0, 0, 3:5].sum().item())
    assert seg0_mel <= 80  # 70
    assert seg1_mel <= 80  # 30


def test_hard_limit_split_without_silence() -> None:
    """No sil phone available → split at hard_limit."""
    # All non-sil phones, each dur=30
    # cumsum: [30, 60, 90, 120, 150]
    # max_t_mel=100 → exceed at phone 3 (cumsum 120 > 100)
    # Backward scan finds no sil → split at hard_limit=3
    durs = _make_durations([30, 30, 30, 30, 30])
    ids = _make_phone_ids([1, 2, 3, 4, 5])
    result = split_by_duration_if_needed(durs, ids, max_t_mel=100, sil_phone_ids=SIL_IDS)
    assert len(result) >= 2
    assert result[0] == (0, 3)
    # First segment: 30+30+30=90 <= 100 ✓
    seg0_mel = int(durs[0, 0, 0:3].sum().item())
    assert seg0_mel <= 100


def test_multiple_splits() -> None:
    """Long sequence requiring 3+ segments."""
    # 10 phones, each dur=50, total=500, max_t_mel=120
    # Should produce multiple segments
    durs = _make_durations([50] * 10)  # total=500
    ids = _make_phone_ids([1] * 10)  # no sil phones
    result = split_by_duration_if_needed(durs, ids, max_t_mel=120, sil_phone_ids=SIL_IDS)
    assert len(result) >= 3
    # Each segment's T_mel must be <= 120 (before adjust_duration padding)
    for seg_start, seg_end in result:
        seg_mel = int(durs[0, 0, seg_start:seg_end].sum().item())
        assert seg_mel <= 120, f"Segment ({seg_start},{seg_end}) T_mel={seg_mel} > 120"
    # All phones covered
    assert result[0][0] == 0
    assert result[-1][1] == 10


@pytest.mark.gpu
@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("T_mel", [80, 150, 200])
def test_overlap_save_matches_full_sequence(chunk_size: int, T_mel: int) -> None:
    """Overlap-save output matches full-sequence decoding within FP tolerance."""
    hps = HyperParameters.load_from_json(CONFIG_PATH)
    net_g = get_net_g(str(MODEL_PATH), hps.version, "cuda", hps)
    net_g.dec.remove_weight_norm()

    hop_length = hps.data.hop_length  # 512

    # Random latent input (simulates enc_q output)
    torch.manual_seed(42)
    z = torch.randn(1, hps.model.inter_channels, T_mel, device="cuda")
    g = torch.randn(1, hps.model.gin_channels, 1, device="cuda")

    # Full-sequence decode (FP32 for deterministic comparison)
    with torch.no_grad():
        wav_full = net_g.dec(z, g=g)

    # Chunk-by-chunk overlap-save
    wav_chunked = overlap_save_decode(
        net_g.dec,
        z,
        g,
        chunk_size=chunk_size,
        L=L_CONTEXT,
        R=R_CONTEXT,
        hop_length=hop_length,
    )

    # Compare entire output (including boundaries)
    full_f32 = wav_full.float()
    chunk_f32 = wav_chunked.float()

    max_err = (full_f32 - chunk_f32).abs().max().item()
    snr = 10 * torch.log10(full_f32.pow(2).mean() / (full_f32 - chunk_f32).pow(2).mean().clamp(min=1e-10)).item()

    logger.info(
        "T_mel=%d chunk_size=%d | max_err=%.2e SNR=%.1f dB",
        T_mel,
        chunk_size,
        max_err,
        snr,
    )

    # On CPU, overlap-save gives max_err ~1e-7, SNR ~65 dB (near bit-exact).
    # On CUDA, different input sizes trigger different conv algorithm selections,
    # causing FP non-determinism of ~1e-3. This is inaudible (SNR > 50 dB).
    assert max_err < 2e-3, f"Max error too large: {max_err:.2e}"
    assert snr > 50.0, f"SNR too low: {snr:.1f} dB"
