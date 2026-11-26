"""Audio encoding utilities for TTS streaming.

Provides WAV/PCM/OGG-Opus conversion helpers and buffering for streaming audio responses.
"""

from __future__ import annotations

import io
import struct
from collections.abc import AsyncIterator
from contextlib import suppress

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 44100
OPUS_BITRATE = 48_000


def _make_wav_header(n_samples: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Create a WAV file header for 16-bit mono PCM."""
    data_size = n_samples * 2  # 16-bit = 2 bytes per sample
    file_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM format
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header


def audio_to_wav_bytes(audio: NDArray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert float32 audio to 16-bit WAV bytes."""
    # Normalize and convert to int16
    if audio.dtype in (np.float32, np.float64):
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        audio = (audio * 32767).astype(np.int16)

    header = _make_wav_header(len(audio), sample_rate)
    return header + audio.tobytes()


def audio_chunk_to_pcm_bytes(audio: NDArray) -> bytes:
    """Convert float32 audio chunk to raw 16-bit PCM bytes (no header)."""
    if audio.dtype in (np.float32, np.float64):
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        audio = (audio * 32767).astype(np.int16)
    return audio.tobytes()


def audio_chunk_to_pcm_int16_clamped(audio: NDArray) -> bytes:
    """Fixed-gain conversion for streaming (no per-chunk normalization).

    Uses hard clipping at [-1, 1] instead of per-chunk normalization
    to avoid volume jumps between chunks in streaming mode.
    """
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def audio_to_ogg_opus_bytes(
    audio: NDArray,
    sample_rate: int = SAMPLE_RATE,
    bitrate: int = OPUS_BITRATE,
) -> bytes:
    """Convert float32 audio to OGG/Opus bytes.

    Uses PyAV (FFmpeg) for encoding. Handles resampling to 48kHz internally
    since Opus requires 48000 Hz input.

    Note: PyAV's encode loop acquires/releases the GIL per Opus frame (~500
    times for 10s audio). Under concurrent inference this causes severe GIL
    contention. Callers should run this in a ProcessPoolExecutor to isolate
    from the inference GIL.
    """
    import av

    # Normalize to int16 (same as WAV path)
    if audio.dtype in (np.float32, np.float64):
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
    audio_s16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=48000)
        stream.bit_rate = bitrate
        stream.layout = "mono"
        stream.codec_context.options["compression_level"] = "0"

        frame = av.AudioFrame.from_ndarray(audio_s16.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = 0

        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    return buf.getvalue()


async def buffer_streaming_chunks(
    source: AsyncIterator[bytes],
    min_chunks: int = 10,
) -> AsyncIterator[bytes]:
    """Buffer min_chunks PCM chunks before yielding. Flush remainder on end."""
    if min_chunks < 1:
        raise ValueError("min_chunks must be >= 1")
    buf: list[bytes] = []
    try:
        async for chunk in source:
            buf.append(chunk)
            if len(buf) >= min_chunks:
                yield b"".join(buf)
                buf.clear()
        if buf:
            yield b"".join(buf)
    finally:
        await _close_source(source)


async def _close_source(source: object) -> None:
    import anyio

    close = getattr(source, "aclose", None)
    if close is not None:
        with anyio.CancelScope(shield=True):
            await close()


class OggOpusStreamEncoder:
    """Incremental OGG/Opus encoder that converts float32 PCM chunks to OGG pages."""

    def __init__(
        self, sample_rate: int = SAMPLE_RATE, bitrate: int = OPUS_BITRATE, page_duration_ms: float | None = None
    ) -> None:
        import av

        self._buf = io.BytesIO()
        if page_duration_ms is not None and page_duration_ms <= 0:
            raise ValueError("page_duration_ms must be positive")
        options = {"page_duration": str(round(page_duration_ms * 1000))} if page_duration_ms is not None else {}
        self._container = av.open(self._buf, mode="w", format="ogg", options=options)
        self._closed = False
        self._stream = self._container.add_stream("libopus", rate=48000)
        self._stream.bit_rate = bitrate
        self._stream.layout = "mono"
        self._stream.codec_context.options["compression_level"] = "0"
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        self._input_rate = sample_rate
        self._pos = 0
        self._pts = 0

    def encode_chunk(self, audio: NDArray[np.floating]) -> bytes:
        import av

        audio_s16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(audio_s16.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self._input_rate
        frame.pts = self._pts
        self._pts += len(audio_s16)

        for resampled in self._resampler.resample(frame):
            for packet in self._stream.encode(resampled):
                self._container.mux(packet)

        data = self._buf.getvalue()[self._pos :]
        self._pos += len(data)
        return data

    def finalize(self) -> bytes:
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self.close()
        data = self._buf.getvalue()[self._pos :]
        return data

    def close(self) -> None:
        if not self._closed:
            try:
                self._container.close()
            finally:
                self._closed = True


async def ogg_streaming_chunks(
    source: AsyncIterator[NDArray],
    sample_rate: int,
    page_duration_ms: float | None = None,
) -> AsyncIterator[bytes]:
    """Wrap an async iterator of float32 NDArrays into OGG/Opus byte chunks."""
    encoder = OggOpusStreamEncoder(sample_rate=sample_rate, page_duration_ms=page_duration_ms)
    try:
        async for audio in source:
            data = encoder.encode_chunk(audio)
            if data:
                yield data
        tail = encoder.finalize()
        if tail:
            yield tail
    finally:
        await _close_source(source)
        with suppress(Exception):
            encoder.close()
