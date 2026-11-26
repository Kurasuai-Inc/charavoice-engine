"""Tests for charavoice_engine.server.streaming WAV encoding functions."""

from __future__ import annotations

import struct

import numpy as np

from charavoice_engine.server.streaming import (
    SAMPLE_RATE,
    _make_wav_header,
    audio_chunk_to_pcm_bytes,
    audio_to_wav_bytes,
    buffer_streaming_chunks,
)


class TestMakeWavHeader:
    def test_header_size(self):
        header = _make_wav_header(44100)
        assert len(header) == 44  # standard WAV header

    def test_riff_marker(self):
        header = _make_wav_header(100)
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"

    def test_fmt_chunk(self):
        header = _make_wav_header(100)
        assert header[12:16] == b"fmt "
        # PCM format = 1
        fmt_tag = struct.unpack_from("<H", header, 20)[0]
        assert fmt_tag == 1
        # Mono
        channels = struct.unpack_from("<H", header, 22)[0]
        assert channels == 1
        # Sample rate
        sr = struct.unpack_from("<I", header, 24)[0]
        assert sr == SAMPLE_RATE

    def test_data_size(self):
        n_samples = 44100
        header = _make_wav_header(n_samples)
        data_size = struct.unpack_from("<I", header, 40)[0]
        assert data_size == n_samples * 2  # 16-bit

    def test_file_size(self):
        n_samples = 1000
        header = _make_wav_header(n_samples)
        file_size = struct.unpack_from("<I", header, 4)[0]
        assert file_size == 36 + n_samples * 2

    def test_custom_sample_rate(self):
        header = _make_wav_header(100, sample_rate=22050)
        sr = struct.unpack_from("<I", header, 24)[0]
        assert sr == 22050


class TestAudioToWavBytes:
    def test_output_starts_with_riff(self, sample_audio_float32):
        wav = audio_to_wav_bytes(sample_audio_float32)
        assert wav[:4] == b"RIFF"

    def test_output_length(self, sample_audio_float32):
        wav = audio_to_wav_bytes(sample_audio_float32)
        expected = 44 + len(sample_audio_float32) * 2  # header + 16-bit samples
        assert len(wav) == expected

    def test_silent_audio(self, sample_audio_silent):
        wav = audio_to_wav_bytes(sample_audio_silent)
        # Header + zero samples
        pcm_data = wav[44:]
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        assert np.all(samples == 0)

    def test_normalization(self):
        # Audio with max value 0.5 should be normalized to full range
        audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        wav = audio_to_wav_bytes(audio)
        pcm_data = wav[44:]
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        assert samples[0] == 32767
        assert samples[1] == -32767

    def test_int16_passthrough(self):
        # Already int16 audio should pass through without normalization
        audio = np.array([100, -100, 0], dtype=np.int16)
        wav = audio_to_wav_bytes(audio)
        pcm_data = wav[44:]
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        np.testing.assert_array_equal(samples, audio)


class TestAudioChunkToPcmBytes:
    def test_no_header(self, sample_audio_float32):
        pcm = audio_chunk_to_pcm_bytes(sample_audio_float32)
        # No WAV header, just raw PCM
        assert pcm[:4] != b"RIFF"
        assert len(pcm) == len(sample_audio_float32) * 2

    def test_silent_audio(self, sample_audio_silent):
        pcm = audio_chunk_to_pcm_bytes(sample_audio_silent)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(samples == 0)

    def test_normalization(self):
        audio = np.array([1.0, -1.0], dtype=np.float32)
        pcm = audio_chunk_to_pcm_bytes(audio)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == 32767
        assert samples[1] == -32767


class TestBufferStreamingChunks:
    @staticmethod
    async def _async_iter(items: list[bytes]):
        for item in items:
            yield item

    async def test_basic(self):
        chunks = [b"x" * 100] * 25
        result = [b async for b in buffer_streaming_chunks(self._async_iter(chunks), min_chunks=10)]
        assert len(result) == 3  # 10 + 10 + 5
        assert result[0] == b"x" * 1000
        assert result[1] == b"x" * 1000
        assert result[2] == b"x" * 500

    async def test_min_chunks_one_yields_each_chunk(self):
        chunks = [b"a", b"b", b"c"]
        result = [b async for b in buffer_streaming_chunks(self._async_iter(chunks), min_chunks=1)]
        assert result == chunks

    async def test_flush(self):
        chunks = [b"a", b"b", b"c"]
        result = [b async for b in buffer_streaming_chunks(self._async_iter(chunks), min_chunks=10)]
        assert len(result) == 1
        assert result[0] == b"abc"

    async def test_rejects_invalid_min_chunks(self):
        chunks = [b"a"]
        try:
            _ = [b async for b in buffer_streaming_chunks(self._async_iter(chunks), min_chunks=0)]
        except ValueError as error:
            assert "min_chunks" in str(error)
        else:
            raise AssertionError("buffer_streaming_chunks accepted min_chunks=0")

    async def test_empty(self):
        result = [b async for b in buffer_streaming_chunks(self._async_iter([]), min_chunks=10)]
        assert result == []


class TestSampleRate:
    def test_value(self):
        assert SAMPLE_RATE == 44100
