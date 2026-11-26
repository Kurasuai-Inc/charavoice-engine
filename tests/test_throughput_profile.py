"""Regression coverage for the accepted profile's duration and watermark contract."""

from __future__ import annotations

import asyncio
from unittest.mock import PropertyMock

import numpy as np
import pytest
import torch

from charavoice_engine.config import FastInferenceConfig, InferenceProfile
from charavoice_engine.inference.overlap_save import OverlapSaveConfig, build_decoder_chunks_with_lengths
from charavoice_engine.server.batcher import DynamicBatcher
from charavoice_engine.watermark.protocol import build_watermark_plan
from tests.test_batcher import _FakeWatermarker, _make_engine, _make_request


def test_profile_keeps_default_compatible_and_fixes_adopted_policy():
    default = FastInferenceConfig(batch_timeout_ms=42)
    assert default.decoder_chunk_frames == 32
    assert default.batch_timeout_ms == 42
    assert default.adaptive_batch_timeout
    adopted = FastInferenceConfig(profile=InferenceProfile.THROUGHPUT128, batch_timeout_ms=42)
    assert adopted.decoder_chunk_frames == 128
    assert adopted.batch_timeout_ms == 0
    assert not adopted.adaptive_batch_timeout
    with pytest.raises(ValueError):
        FastInferenceConfig(profile="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("frames", [32, 64, 96, 128, 160, 256, 288])
def test_large_decoder_context_and_partial_tail(frames):
    latent = np.arange(2 * frames, dtype=np.float32).reshape(2, frames)
    oc = OverlapSaveConfig(chunk_size=128)
    chunks, counts = build_decoder_chunks_with_lengths(latent, oc)
    assert sum(counts) == frames // 32
    reconstructed = np.concatenate(
        [chunk[:, 13 : 13 + count * 32] for chunk, count in zip(chunks, counts, strict=True)], axis=1
    )
    np.testing.assert_array_equal(reconstructed, latent)
    for index, chunk in enumerate(chunks):
        start = index * 128 - 13
        expected = np.zeros((2, 154), dtype=np.float32)
        for offset in range(154):
            if 0 <= start + offset < frames:
                expected[:, offset] = latent[:, start + offset]
        np.testing.assert_array_equal(chunk, expected)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("max_watermark_batch", [4, 8, 16])
async def test_128_frame_tail_preserves_watermark_words_and_length(streaming, max_watermark_batch):
    frames = 288  # two complete decoder windows plus one valid output block
    oc = OverlapSaveConfig(chunk_size=128)
    engine = _make_engine()
    type(engine).overlap_config = PropertyMock(return_value=oc)
    engine.run_stages_0_4.side_effect = lambda requests: (
        [torch.zeros(1, 192, frames) for _ in requests],
        np.zeros((len(requests), 512, 1), dtype=np.float32),
        (0.0,) * 5,
    )
    engine.decode_chunk_batch.side_effect = lambda z, g: np.zeros((len(z), 1, 154 * 512), dtype=np.float32)
    watermarker = _FakeWatermarker()
    watermarker.max_batch_size = max_watermark_batch
    config = FastInferenceConfig(profile=InferenceProfile.THROUGHPUT128)
    batcher = DynamicBatcher(engine, config, watermark_backend=watermarker, watermark_batch_max=max_watermark_batch)
    request = _make_request()
    request.watermark_core16 = 0x1234
    request.watermark_extension = "beef"
    plan = build_watermark_plan(core16=0x1234, extension_bytes=bytes.fromhex("beef"), chunk_count=9)
    await batcher.start()
    try:
        if streaming:

            async def collect():
                return [chunk async for chunk in batcher.submit_streaming(request)]

            chunks = await asyncio.wait_for(collect(), 5)
            assert len(chunks) == 9
            audio = np.concatenate(chunks)
        else:
            audio = (await asyncio.wait_for(batcher.submit(request), 5)).audio
        assert len(audio) == frames * 512
        assert [int(audio[index * 16384]) for index in range(9)] == [
            plan.frame_words[index % len(plan.frame_words)] for index in range(9)
        ]
        assert all(len(call) <= max_watermark_batch for call in watermarker.calls)
    finally:
        await batcher.stop()


async def test_close_releases_a_full_queue_after_request_future_completed():
    batcher = DynamicBatcher(_make_engine(), FastInferenceConfig())
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    pending = batcher._build_pending(
        request=_make_request(), future=future, is_streaming=True, chunk_queue=asyncio.Queue(maxsize=1)
    )
    pending.chunk_queue.put_nowait(np.zeros(1))
    future.set_result(None)
    delivery = asyncio.create_task(batcher._put_stream_item(pending, None))
    await asyncio.sleep(0)
    assert not delivery.done()
    pending.stream_closed.set()
    assert await asyncio.wait_for(delivery, 1) is False


async def test_closed_slow_consumer_does_not_block_next_request():
    engine = _make_engine()
    engine.run_stages_0_4.side_effect = lambda requests: (
        [torch.zeros(1, 192, 512) for _ in requests],
        np.zeros((len(requests), 512, 1), dtype=np.float32),
        (0.0,) * 5,
    )
    batcher = DynamicBatcher(engine, FastInferenceConfig(adaptive_batch_timeout=False), batch_timeout_ms=0)
    await batcher.start()
    try:
        source = batcher.submit_streaming(_make_request())
        await asyncio.wait_for(anext(source), 2)
        await asyncio.sleep(0.03)  # allow the producer to fill its bounded queue
        await source.aclose()
        result = await asyncio.wait_for(batcher.submit(_make_request()), 2)
        assert len(result.audio) == 512 * 512
    finally:
        await batcher.stop()
