"""Overlap-save chunking utilities for non-causal HiFi-GAN decoder.

Pure functions — no ONNX/TRT dependency. All chunk manipulation is done
with torch tensors so the streaming engine can call these directly.

Parameters (proven in test_overlap_save.py):
    L = R = 13 mel frames (receptive field of Generator)
    cs = 32 mel frames (valid output per chunk)
    hop_length = 512 (mel frame → sample conversion)
    TRT fixed input: [1, 192, 58] = [B, C, L+cs+R]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True)
class OverlapSaveConfig:
    """Immutable config for overlap-save chunking."""

    chunk_size: int = 32
    left_context: int = 13
    right_context: int = 13
    hop_length: int = 512

    @property
    def input_frames(self) -> int:
        """Total mel frames per decoder input (L + cs + R)."""
        return self.left_context + self.chunk_size + self.right_context


def adjust_duration_for_chunk_boundary(durations: Tensor, chunk_size: int) -> Tensor:
    """Pad the last silence duration so total mel frames is a multiple of chunk_size.

    The last token in Style-Bert-VITS2 is always a silence (sil). We extend its duration
    rather than zero-padding, which avoids Conv1d bias artifacts.

    Args:
        durations: [1, 1, T_phone] integer durations (w_ceil from DDP).
        chunk_size: Valid output chunk size in mel frames.

    Returns:
        Modified durations tensor (same shape, last element may be increased).
    """
    durations = durations.clone()
    total = durations.sum().item()
    remainder = int(total) % chunk_size
    if remainder != 0:
        pad = chunk_size - remainder
        # Extend last sil token
        durations[0, 0, -1] += pad
    return durations


def split_by_duration_if_needed(
    durations: Tensor,
    phone_ids: Tensor,
    max_t_mel: int,
    sil_phone_ids: set[int] | None = None,
) -> list[tuple[int, int]]:
    """Split phone sequence into segments so each segment's T_mel <= max_t_mel.

    Used when long text produces T_mel exceeding the Flow TRT profile max.
    Prefers splitting at silence/punctuation phone boundaries.

    Args:
        durations: [1, 1, T_phone] integer durations (w_ceil from DDP).
        phone_ids: [1, T_phone] phone ID tensor.
        max_t_mel: Maximum mel frames per segment (e.g. 2048).
        sil_phone_ids: Set of phone IDs that are silence/punctuation.
            If None, imports SIL_PHONEMES_IDS from charavoice_engine.nlp.symbols.

    Returns:
        List of (start, end) tuples — phone index ranges [start, end) for each segment.
    """
    dur = durations[0, 0]  # [T_phone]
    T_phone = dur.shape[0]
    total_mel = int(dur.sum().item())

    # Fast path: no split needed
    if total_mel <= max_t_mel:
        return [(0, T_phone)]

    if sil_phone_ids is None:
        from charavoice_engine.nlp.symbols import SIL_PHONEMES_IDS

        sil_phone_ids = set(SIL_PHONEMES_IDS)

    ids = phone_ids[0]  # [T_phone]
    cumsum = dur.cumsum(0)  # [T_phone]

    segments: list[tuple[int, int]] = []
    seg_start = 0

    while seg_start < T_phone:
        # Remaining duration from seg_start
        offset = int(cumsum[seg_start - 1].item()) if seg_start > 0 else 0
        remaining = total_mel - offset

        if remaining <= max_t_mel:
            # Last segment fits entirely
            segments.append((seg_start, T_phone))
            break

        # Find hard_limit: first phone where cumsum - offset > max_t_mel
        seg_cumsum = cumsum[seg_start:] - offset  # cumsum relative to segment start
        exceed_mask = seg_cumsum > max_t_mel
        if not exceed_mask.any():
            # Defensive split guard for inconsistent duration inputs.
            segments.append((seg_start, T_phone))
            break

        hard_limit = int(exceed_mask.int().argmax().item()) + seg_start  # first True index (absolute)

        # Scan backward from hard_limit to find a sil/pau phone for clean split
        split_at = hard_limit
        for i in range(hard_limit - 1, seg_start, -1):
            if int(ids[i].item()) in sil_phone_ids:
                split_at = i + 1  # split after the sil phone
                break

        segments.append((seg_start, split_at))
        seg_start = split_at

    return segments


def build_decoder_chunks(z_all: Tensor, config: OverlapSaveConfig) -> list[Tensor]:
    """Split z_all into fixed-size [1, C, input_frames] chunks with overlap.

    Boundary handling:
        - First chunk: left side zero-padded by L frames.
        - Last chunk: right side zero-padded by R frames.
        - Interior chunks: real context on both sides.

    Args:
        z_all: [1, C, T_mel] full latent sequence (T_mel must be multiple of chunk_size).
        config: Overlap-save parameters.

    Returns:
        List of [1, C, input_frames] tensors ready for decoder.
    """
    T_mel = z_all.shape[2]
    cs = config.chunk_size
    L = config.left_context
    R = config.right_context
    C = z_all.shape[1]
    input_frames = config.input_frames

    assert T_mel % cs == 0, f"T_mel={T_mel} must be multiple of chunk_size={cs}"
    n_chunks = T_mel // cs

    chunks: list[Tensor] = []
    for i in range(n_chunks):
        start = i * cs  # start of valid region in z_all
        # Context range in z_all (may go out of bounds)
        ctx_start = start - L
        ctx_end = start + cs + R

        chunk = torch.zeros(1, C, input_frames, device=z_all.device, dtype=z_all.dtype)

        # Source slice (clamped to z_all bounds)
        src_start = max(0, ctx_start)
        src_end = min(T_mel, ctx_end)

        # Destination offset in chunk
        dst_start = src_start - ctx_start
        dst_end = dst_start + (src_end - src_start)

        chunk[:, :, dst_start:dst_end] = z_all[:, :, src_start:src_end]
        chunks.append(chunk)

    return chunks


def build_decoder_chunks_numpy(z_np: NDArray, config: OverlapSaveConfig) -> list[NDArray]:
    """Split z_np into fixed-size [C, input_frames] chunks with overlap (numpy, CPU).

    Same logic as build_decoder_chunks but operates on numpy arrays.
    No torch dependency — suitable for use after GPU→CPU transfer.

    Args:
        z_np: [C, T_mel] latent sequence (T_mel must be multiple of chunk_size).
        config: Overlap-save parameters.

    Returns:
        List of [C, input_frames] numpy arrays ready for decoder.
    """
    T_mel = z_np.shape[1]
    cs = config.chunk_size
    L = config.left_context
    R = config.right_context
    C = z_np.shape[0]
    input_frames = config.input_frames

    assert T_mel % cs == 0, f"T_mel={T_mel} must be multiple of chunk_size={cs}"
    n_chunks = T_mel // cs

    chunks: list[NDArray] = []
    for i in range(n_chunks):
        start = i * cs
        ctx_start = start - L
        ctx_end = start + cs + R

        chunk = np.zeros((C, input_frames), dtype=z_np.dtype)

        src_start = max(0, ctx_start)
        src_end = min(T_mel, ctx_end)

        dst_start = src_start - ctx_start
        dst_end = dst_start + (src_end - src_start)

        chunk[:, dst_start:dst_end] = z_np[:, src_start:src_end]
        chunks.append(chunk)

    return chunks


def build_decoder_chunks_with_lengths(
    z_np: NDArray, config: OverlapSaveConfig, output_block_frames: int = 32
) -> tuple[list[NDArray], list[int]]:
    """Pack larger decode windows while retaining the original output block grid.

    The last 128-frame decode can contain only 1-3 valid 32-frame blocks. Its
    padding supplies convolution context; it must never lengthen the response or
    its watermark plan.
    """
    block_frames = min(output_block_frames, config.chunk_size)
    frames = z_np.shape[-1]
    if config.chunk_size % block_frames or frames % block_frames:
        raise ValueError("Decoder and latent lengths must align to the output block grid")
    if config.chunk_size == block_frames:
        chunks = build_decoder_chunks_numpy(z_np, config)
        return chunks, [1] * len(chunks)
    if frames == 0:
        return [], []
    count = (frames + config.chunk_size - 1) // config.chunk_size
    padded = np.pad(z_np, ((0, 0), (config.left_context, count * config.chunk_size - frames + config.right_context)))
    views = np.lib.stride_tricks.sliding_window_view(padded, config.input_frames, axis=1)
    packed = np.ascontiguousarray(views[:, :: config.chunk_size, :].transpose(1, 0, 2))
    counts = [min(config.chunk_size, frames - i * config.chunk_size) // block_frames for i in range(count)]
    return list(packed), counts


def extract_valid_audio(wav: Tensor, config: OverlapSaveConfig) -> Tensor:
    """Extract the valid cs*hop_length audio samples from a decoded chunk.

    The decoder output contains L+cs+R mel frames worth of audio.
    We extract only the middle cs frames (the valid region).

    For boundary chunks, the context region may have been zero-padded,
    but we still extract from the same position — the small boundary error
    is inaudible (verified in test_overlap_save.py).

    Args:
        wav: [1, 1, input_frames * hop_length] decoder output.
        config: Overlap-save parameters.

    Returns:
        [valid_samples] 1D tensor of audio samples.
    """
    L = config.left_context
    hop = config.hop_length
    cs = config.chunk_size

    valid_start = L * hop
    valid_end = valid_start + cs * hop
    return wav[0, 0, valid_start:valid_end]


def extract_valid_audio_batch_numpy(wav_batch: NDArray, config: OverlapSaveConfig) -> NDArray:
    """Extract the valid audio region from a batched decoder output.

    Args:
        wav_batch: [B, 1, input_frames * hop_length] decoder outputs.
        config: Overlap-save parameters.

    Returns:
        [B, valid_samples] numpy view over the valid region.
    """
    L = config.left_context
    hop = config.hop_length
    cs = config.chunk_size

    valid_start = L * hop
    valid_end = valid_start + cs * hop
    return wav_batch[:, 0, valid_start:valid_end]
