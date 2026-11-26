from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from charavoice_engine.watermark.common import MESSAGE_BITS, BatchedWatermarkBackend

# AudioSeal models (audioseal_wm_16bits) are trained at 16 kHz.  The TTS
# decoder outputs 44.1 kHz audio, but we intentionally pass sample_rate=_AUDIOSEAL_SR
# so that AudioSeal skips its internal resampler and applies the watermark
# directly on native-rate chunks.  Embedding and detection are symmetric —
# both use the same declared rate — so the watermark round-trips correctly.
_AUDIOSEAL_SR = 16000


@dataclass
class WatermarkConfig:
    """Configuration for eager AudioSeal watermarking."""

    alpha: float = 1.0
    device: str = "cuda"
    max_concurrency: int = 4


class AudioSealWatermarker(BatchedWatermarkBackend):
    """Eager PyTorch AudioSeal backend supporting batched fixed chunks."""

    backend_name = "audioseal"
    chunk_samples = 16384
    max_batch_size: int | None = None

    def __init__(self, config: WatermarkConfig) -> None:
        self.config = config
        self._generator = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._semaphore = (
            threading.Semaphore(config.max_concurrency)
            if config.max_concurrency and config.max_concurrency > 0
            else None
        )

    def _load(self) -> None:
        if self._generator is not None:
            return
        with self._load_lock:
            if self._generator is not None:
                return
            os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
            try:
                from audioseal import AudioSeal  # pyright: ignore[reportMissingImports, reportPrivateImportUsage]
            except ImportError as e:  # pragma: no cover - optional dependency
                raise RuntimeError("audioseal is not installed. Please `pip install audioseal`.") from e

            generator = AudioSeal.load_generator("audioseal_wm_16bits")
            generator.eval()
            generator.to(self.config.device)
            self._generator = generator

    def warmup(self) -> None:
        """Pre-load the AudioSeal generator model."""
        self._load()

    def apply_batch(
        self,
        audio_batch: NDArray[np.float32],
        message_bits: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """Apply watermark to a chunk batch shaped [B, T]."""
        self._load()
        assert self._generator is not None

        if audio_batch.ndim != 2:
            raise ValueError(f"audio_batch must have shape [B, T], got {audio_batch.shape}")
        if message_bits.shape != (audio_batch.shape[0], MESSAGE_BITS):
            raise ValueError(
                f"message_bits must have shape [{audio_batch.shape[0]}, {MESSAGE_BITS}], got {message_bits.shape}"
            )

        with torch.inference_mode():
            wav = torch.from_numpy(audio_batch[:, None, :].astype(np.float32, copy=False)).to(self.config.device)
            msg = torch.from_numpy(message_bits.astype(np.int64, copy=False)).to(self.config.device)

            # The batched pipeline operates on native fixed decoder chunks.
            # v1 keeps them at their original length and skips any extra resampling.
            wav_in = wav

            if self._semaphore:
                self._semaphore.acquire()
            try:
                with self._run_lock:
                    watermark = self._generator.get_watermark(wav_in, sample_rate=_AUDIOSEAL_SR, message=msg)
            finally:
                if self._semaphore:
                    self._semaphore.release()

            watermarked = wav_in + self.config.alpha * watermark
            return torch.clamp(watermarked, -1.0, 1.0).squeeze(1).detach().cpu().numpy()
