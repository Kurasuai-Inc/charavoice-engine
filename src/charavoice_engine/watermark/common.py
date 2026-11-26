from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

MESSAGE_BITS = 16
MESSAGE_MAX = (1 << MESSAGE_BITS) - 1


@runtime_checkable
class BatchedWatermarkBackend(Protocol):
    """Runtime interface for chunk-local batched watermark backends."""

    backend_name: str
    chunk_samples: int
    max_batch_size: int | None

    def apply_batch(
        self,
        audio_batch: NDArray[np.float32],
        message_bits: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """Apply watermarking to a batch of mono chunks shaped [B, T]."""
        ...


def validate_watermark_word(word: int, field_name: str = "watermark_core16") -> int:
    """Validate a 16-bit AudioSeal word and return it unchanged."""
    if not 0 <= word <= MESSAGE_MAX:
        raise ValueError(f"{field_name} must be between 0 and {MESSAGE_MAX}")
    return word


def pack_watermark_words(words: Sequence[int]) -> NDArray[np.int64]:
    """Pack 16-bit words into AudioSeal's [B, 16] binary tensor format."""
    if not words:
        return np.empty((0, MESSAGE_BITS), dtype=np.int64)

    values = np.asarray([validate_watermark_word(int(w)) for w in words], dtype=np.uint16)[:, None]
    shifts = np.arange(MESSAGE_BITS - 1, -1, -1, dtype=np.uint16)
    return ((values >> shifts) & 1).astype(np.int64, copy=False)


def unpack_watermark_bits(bits: Sequence[int] | NDArray[np.int64]) -> int:
    """Convert a single AudioSeal 16-bit row back into an integer word."""
    bit_array = np.asarray(bits, dtype=np.int64)
    if bit_array.shape != (MESSAGE_BITS,):
        raise ValueError(f"bits must have shape ({MESSAGE_BITS},), got {bit_array.shape}")
    weights = (1 << np.arange(MESSAGE_BITS - 1, -1, -1, dtype=np.int64)).astype(np.int64)
    value = int((bit_array * weights).sum())
    return validate_watermark_word(value)
