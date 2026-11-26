from __future__ import annotations

import io

import av
import numpy as np
import pytest

from charavoice_engine.server.responses import ClosingStreamingResponse
from charavoice_engine.server.streaming import OggOpusStreamEncoder, buffer_streaming_chunks, ogg_streaming_chunks


async def test_http_disconnect_closes_iterator():
    closed = []

    async def body():
        try:
            yield b"audio"
            yield b"more audio"
        finally:
            closed.append(True)

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    response = ClosingStreamingResponse(body())
    await response({"type": "http", "asgi": {"spec_version": "2.4"}}, None, send)
    assert closed == [True]


@pytest.mark.parametrize("encoding", ["pcm", "ogg"])
async def test_closing_wrapper_closes_audio_source(encoding):
    closed = []

    async def source():
        try:
            while True:
                yield b"pcm" if encoding == "pcm" else np.zeros(16384, dtype=np.float32)
        finally:
            closed.append(True)

    wrapped = buffer_streaming_chunks(source(), 1) if encoding == "pcm" else ogg_streaming_chunks(source(), 44100, 200)
    await anext(wrapped)
    await wrapped.aclose()
    assert closed == [True]


def test_200ms_pages_preserve_opus_packets_and_emit_early():
    audio = (0.2 * np.sin(np.arange(44100 * 2) * 0.05)).astype(np.float32)

    def encode(duration):
        encoder = OggOpusStreamEncoder(page_duration_ms=duration)
        chunks = [encoder.encode_chunk(part) for part in np.array_split(audio, 20)]
        chunks.append(encoder.finalize())
        encoder.close()  # close is idempotent after successful finalization
        data = b"".join(chunks)
        with av.open(io.BytesIO(data)) as container:
            packets = [bytes(packet) for packet in container.demux(audio=0) if packet.size]
        return chunks, packets

    chunks, packets = encode(200)
    _, original_packets = encode(None)
    assert packets == original_packets
    # After the initial header, an audio page is available before one second.
    assert any(chunks[1:5])
