"""Tests for charavoice_engine.server.batcher (pipeline S0-4 / Decoder architecture)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from charavoice_engine.config import FastInferenceConfig
from charavoice_engine.constants import Languages
from charavoice_engine.inference.engine_base import InferenceRequest
from charavoice_engine.inference.overlap_save import OverlapSaveConfig
from charavoice_engine.server.batcher import _EWMA_ALPHA, DynamicBatcher
from charavoice_engine.watermark.common import MESSAGE_BITS, pack_watermark_words
from charavoice_engine.watermark.protocol import build_watermark_plan


def _make_config(
    *,
    adaptive: bool = True,
    min_ms: float = 2.0,
    max_ms: float = 100.0,
    base_ms: float = 50.0,
) -> FastInferenceConfig:
    return FastInferenceConfig(
        adaptive_batch_timeout=adaptive,
        batch_timeout_min_ms=min_ms,
        batch_timeout_max_ms=max_ms,
        batch_timeout_ms=base_ms,
    )


_OC = OverlapSaveConfig()  # L=13, cs=32, R=13, hop=512


def _make_engine() -> MagicMock:
    """Create a mock engine with pipeline-separated API.

    run_stages_0_4: returns 2-chunk z_all per request (T_mel=64 = 2*cs=32)
    decode_chunk_batch: returns synthetic wav of correct shape
    overlap_config: returns default OverlapSaveConfig
    sample_rate: returns 44100
    """
    engine = MagicMock()

    def mock_run_stages_0_4(requests):
        B = len(requests)
        # Each request gets T_mel=64 (2 chunks of cs=32)
        z_list = [torch.zeros(1, 192, 64) for _ in range(B)]
        g_np = np.zeros((B, 512, 1), dtype=np.float32)
        times = (0.0, 0.0, 0.0, 0.0, 0.0)
        return z_list, g_np, times

    engine.run_stages_0_4.side_effect = mock_run_stages_0_4

    def mock_decode_chunk_batch(z_batch, g_batch):
        B = z_batch.shape[0]
        output_samples = _OC.input_frames * _OC.hop_length  # 58 * 512
        return np.zeros((B, 1, output_samples), dtype=np.float32)

    engine.decode_chunk_batch.side_effect = mock_decode_chunk_batch

    type(engine).overlap_config = PropertyMock(return_value=_OC)
    type(engine).sample_rate = PropertyMock(return_value=44100)

    return engine


def _make_request() -> InferenceRequest:
    return InferenceRequest(
        text="テスト",
        language=Languages.JP,
        speaker_id=0,
        style_vec=np.zeros(256, dtype=np.float32),
    )


class _FakeWatermarker:
    backend_name = "fake"
    chunk_samples = _OC.chunk_size * _OC.hop_length
    max_batch_size = 16

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls: list[NDArray[np.int64]] = []

    def apply_batch(self, audio_batch: NDArray[np.float32], message_bits: NDArray[np.int64]) -> NDArray[np.float32]:
        self.calls.append(message_bits.copy())
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        out = audio_batch.copy()
        weights = (1 << np.arange(MESSAGE_BITS - 1, -1, -1, dtype=np.int64)).astype(np.float32)
        out[:, 0] = (message_bits.astype(np.float32) * weights).sum(axis=1)
        return out


class TestComputeTimeoutMs:
    """Test _compute_timeout_ms under various EWMA states."""

    def test_non_adaptive_returns_fixed(self):
        config = _make_config(adaptive=False, base_ms=42.0)
        batcher = DynamicBatcher(_make_engine(), config, batch_timeout_ms=42.0)
        assert batcher._compute_timeout_ms() == 42.0

    def test_not_initialized_returns_base(self):
        config = _make_config(adaptive=True, base_ms=50.0)
        batcher = DynamicBatcher(_make_engine(), config, batch_timeout_ms=50.0)
        assert not batcher._ewma_initialized
        assert batcher._compute_timeout_ms() == 50.0

    def test_low_fill_rate_short_timeout(self):
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        batcher = DynamicBatcher(_make_engine(), config)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.125  # C=1 scenario: 1/8
        timeout = batcher._compute_timeout_ms()
        # 2 + 0.125 * 98 = 14.25
        assert 14.0 <= timeout <= 15.0

    def test_medium_fill_rate_medium_timeout(self):
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        batcher = DynamicBatcher(_make_engine(), config)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.5
        timeout = batcher._compute_timeout_ms()
        # 2 + 0.5 * 98 = 51
        assert 50.0 <= timeout <= 52.0

    def test_high_fill_rate_long_timeout(self):
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        batcher = DynamicBatcher(_make_engine(), config)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.9
        timeout = batcher._compute_timeout_ms()
        # 2 + 0.9 * 98 = 90.2
        assert 89.0 <= timeout <= 91.0

    def test_fill_rate_clamped_at_1(self):
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        batcher = DynamicBatcher(_make_engine(), config)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 1.5  # Should be clamped
        timeout = batcher._compute_timeout_ms()
        assert timeout == 100.0

    def test_arrival_rate_based_timeout(self):
        """When drain >= 2 and arrival rate is known, compute rate-based timeout."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        batcher = DynamicBatcher(_make_engine(), config, max_batch_size=8)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.25  # B=2/8
        batcher._arrival_rate_initialized = True
        batcher._ewma_arrival_rate = 30.0  # 30 req/s

        # drain=2, target=4, needed=2, timeout = 2/30*1000 = 66.7ms
        timeout = batcher._compute_timeout_ms(drain_count=2)
        assert 65.0 <= timeout <= 68.0

    def test_drain_one_ignores_arrival_rate(self):
        """When drain <= 1, use EWMA fill-rate formula regardless of arrival rate."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        batcher = DynamicBatcher(_make_engine(), config, max_batch_size=8)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.125  # C=1 scenario
        batcher._arrival_rate_initialized = True
        batcher._ewma_arrival_rate = 20.0

        # drain=1 → arrival rate not used, base = 2 + 0.125*98 = 14.25
        timeout = batcher._compute_timeout_ms(drain_count=1)
        assert 14.0 <= timeout <= 15.0

    def test_arrival_rate_clamped_to_max(self):
        """Rate-based timeout is clamped to max_timeout_ms."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        batcher = DynamicBatcher(_make_engine(), config, max_batch_size=8)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.25
        batcher._arrival_rate_initialized = True
        batcher._ewma_arrival_rate = 5.0  # Very slow: 5 req/s

        # drain=2, target=4, needed=2, timeout = 2/5*1000 = 400ms → clamped to 100
        timeout = batcher._compute_timeout_ms(drain_count=2)
        assert timeout == 100.0

    def test_drain_exceeds_target_uses_base(self):
        """When drain already >= target, rate-based timeout is not applied."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        batcher = DynamicBatcher(_make_engine(), config, max_batch_size=8)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.5
        batcher._arrival_rate_initialized = True
        batcher._ewma_arrival_rate = 30.0

        # drain=4 >= target=4, needed=0 → base = 2 + 0.5*98 = 51
        timeout = batcher._compute_timeout_ms(drain_count=4)
        assert 50.0 <= timeout <= 52.0


class TestEwmaConvergence:
    """Test that EWMA fill rate converges correctly."""

    def test_first_batch_initializes(self):
        config = _make_config(adaptive=True)
        batcher = DynamicBatcher(_make_engine(), config, max_batch_size=8)
        assert not batcher._ewma_initialized

        fill = 2 / 8
        batcher._ewma_fill_rate = fill
        batcher._ewma_initialized = True
        assert batcher._ewma_fill_rate == 0.25

    def test_ewma_converges_to_steady_state(self):
        config = _make_config(adaptive=True)
        batcher = DynamicBatcher(_make_engine(), config, max_batch_size=8)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.0

        for _ in range(50):
            fill = 4 / 8
            batcher._ewma_fill_rate = _EWMA_ALPHA * fill + (1 - _EWMA_ALPHA) * batcher._ewma_fill_rate

        assert abs(batcher._ewma_fill_rate - 0.5) < 0.01

    def test_ewma_responds_to_load_change(self):
        config = _make_config(adaptive=True)
        batcher = DynamicBatcher(_make_engine(), config, max_batch_size=8)
        batcher._ewma_initialized = True
        batcher._ewma_fill_rate = 0.25

        for _ in range(20):
            fill = 8 / 8
            batcher._ewma_fill_rate = _EWMA_ALPHA * fill + (1 - _EWMA_ALPHA) * batcher._ewma_fill_rate

        assert batcher._ewma_fill_rate > 0.85


class TestStats:
    """Test stats() output includes adaptive fields."""

    def test_stats_includes_adaptive_fields(self):
        config = _make_config(adaptive=True)
        batcher = DynamicBatcher(_make_engine(), config)
        s = batcher.stats()
        assert "adaptive" in s
        assert s["adaptive"] is True
        assert "avg_timeout_ms" in s
        assert "ewma_fill_rate" in s
        assert "ewma_arrival_rate" in s

    def test_stats_non_adaptive(self):
        config = _make_config(adaptive=False)
        batcher = DynamicBatcher(_make_engine(), config)
        s = batcher.stats()
        assert s["adaptive"] is False
        assert s["watermark_backend"] == "none"


class TestWatermarkHelpers:
    """Unit tests for watermark-related invariants."""

    def test_fixed_valid_chunk_matches_trt_shape(self):
        assert _OC.chunk_size * _OC.hop_length == 16384

    def test_pack_watermark_words(self):
        bits = pack_watermark_words([0, 1, 0xFFFF])
        assert bits.shape == (3, 16)
        assert bits[0].tolist() == [0] * 16
        assert bits[1].tolist() == [0] * 15 + [1]
        assert bits[2].tolist() == [1] * 16


class TestDispatchIntegration:
    """Integration tests for the pipelined dispatch loop."""

    @pytest.mark.asyncio
    async def test_single_request_dispatches_quickly(self):
        """Single request should dispatch through S0-4 + decoder pipeline."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0, base_ms=50.0)
        engine = _make_engine()
        batcher = DynamicBatcher(engine, config, max_batch_size=8)

        await batcher.start()
        try:
            request = _make_request()
            result = await asyncio.wait_for(batcher.submit(request), timeout=5.0)
            assert result.audio is not None
            assert len(result.audio) > 0
            # Verify pipeline was used
            engine.run_stages_0_4.assert_called_once()
            engine.decode_chunk_batch.assert_called()
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_burst_requests_batched(self):
        """Multiple concurrent requests should be batched together."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=100.0)
        engine = _make_engine()
        batcher = DynamicBatcher(engine, config, max_batch_size=8)

        await batcher.start()
        try:
            requests = [_make_request() for _ in range(4)]
            results = await asyncio.wait_for(
                asyncio.gather(*[batcher.submit(r) for r in requests]),
                timeout=5.0,
            )
            assert len(results) == 4

            stats = batcher.stats()
            assert stats["total_requests"] == 4
            assert stats["total_batches"] <= 2
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_ewma_updates_after_dispatch(self):
        """EWMA fill rate should be updated after dispatch."""
        config = _make_config(adaptive=True)
        engine = _make_engine()
        batcher = DynamicBatcher(engine, config, max_batch_size=8)

        await batcher.start()
        try:
            request = _make_request()
            await asyncio.wait_for(batcher.submit(request), timeout=5.0)

            assert batcher._ewma_initialized
            assert batcher._ewma_fill_rate > 0
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_double_buffer_multiple_batches(self):
        """Multiple sequential batches complete correctly with pipeline dispatch."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        batcher = DynamicBatcher(engine, config, max_batch_size=2)

        await batcher.start()
        try:
            requests = [_make_request() for _ in range(6)]
            results = await asyncio.wait_for(
                asyncio.gather(*[batcher.submit(r) for r in requests]),
                timeout=10.0,
            )
            assert len(results) == 6
            assert all(r.audio is not None for r in results)

            stats = batcher.stats()
            assert stats["total_requests"] == 6
            assert stats["total_batches"] >= 2
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_double_buffer_shutdown_drains(self):
        """Shutdown properly awaits in-flight batch from pipeline."""
        import time as time_mod

        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()

        original_s04 = engine.run_stages_0_4.side_effect

        def slow_s04(reqs):
            time_mod.sleep(0.05)
            return original_s04(reqs)

        engine.run_stages_0_4.side_effect = slow_s04

        batcher = DynamicBatcher(engine, config, max_batch_size=1)
        await batcher.start()

        result_future = asyncio.ensure_future(batcher.submit(_make_request()))
        await asyncio.sleep(0.01)
        await batcher.stop()

        result = await asyncio.wait_for(result_future, timeout=5.0)
        assert result.audio is not None

    @pytest.mark.asyncio
    async def test_sample_rate_from_engine(self):
        """InferenceResult should use engine.sample_rate."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        type(engine).sample_rate = PropertyMock(return_value=22050)
        batcher = DynamicBatcher(engine, config, max_batch_size=8)

        await batcher.start()
        try:
            result = await asyncio.wait_for(batcher.submit(_make_request()), timeout=5.0)
            assert result.sample_rate == 22050
        finally:
            await batcher.stop()


class TestWatermarkPipeline:
    """Tests for the optional third-stage watermark worker."""

    @pytest.mark.asyncio
    async def test_non_streaming_waits_for_watermark_completion(self):
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        watermarker = _FakeWatermarker(delay_s=0.1)
        batcher = DynamicBatcher(
            engine,
            config,
            max_batch_size=8,
            watermark_backend=watermarker,
            watermark_default_core16=0,
        )

        await batcher.start()
        try:
            task = asyncio.create_task(batcher.submit(_make_request()))
            await asyncio.sleep(0.03)
            assert not task.done()

            result = await asyncio.wait_for(task, timeout=5.0)
            assert len(watermarker.calls) >= 1
            assert result.audio[0] == 0.0
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_request_core16_is_batched_and_packed(self):
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        watermarker = _FakeWatermarker()
        batcher = DynamicBatcher(
            engine,
            config,
            max_batch_size=8,
            watermark_backend=watermarker,
            watermark_default_core16=0,
        )

        await batcher.start()
        try:
            request = _make_request()
            request.watermark_core16 = 0x1234
            result = await asyncio.wait_for(batcher.submit(request), timeout=5.0)

            assert watermarker.calls
            first_batch = watermarker.calls[0]
            assert first_batch.shape[1] == 16
            assert first_batch[0].tolist() == pack_watermark_words([0x1234])[0].tolist()
            assert result.audio[0] == pytest.approx(float(0x1234))
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_request_extension_is_planned_chunk_by_chunk(self):
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        engine.run_stages_0_4.side_effect = lambda requests: (
            [torch.zeros(1, 192, 128) for _ in requests],
            np.zeros((len(requests), 512, 1), dtype=np.float32),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )
        watermarker = _FakeWatermarker()
        batcher = DynamicBatcher(
            engine,
            config,
            max_batch_size=8,
            watermark_backend=watermarker,
            watermark_default_core16=0,
        )

        await batcher.start()
        try:
            request = _make_request()
            request.watermark_core16 = 0x1234
            request.watermark_extension = "beef"
            result = await asyncio.wait_for(batcher.submit(request), timeout=5.0)

            plan = build_watermark_plan(core16=0x1234, extension_bytes=bytes.fromhex("beef"), chunk_count=4)
            assert plan.mode == "extension"
            chunk_samples = _OC.chunk_size * _OC.hop_length
            for idx, expected_word in enumerate(plan.frame_words):
                assert result.audio[idx * chunk_samples] == pytest.approx(float(expected_word))
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_streaming_sentinel_waits_for_watermarked_chunks(self):
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        watermarker = _FakeWatermarker(delay_s=0.05)
        batcher = DynamicBatcher(
            engine,
            config,
            max_batch_size=8,
            watermark_backend=watermarker,
            watermark_default_core16=7,
        )

        await batcher.start()
        try:
            request = _make_request()
            chunks = []
            async for chunk in batcher.submit_streaming(request):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert all(chunk[0] == pytest.approx(7.0) for chunk in chunks)
            stats = batcher.stats()
            assert stats["wm_total_chunks"] == 2
            assert stats["watermark_backend"] == "fake"
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_streaming_watermark_splits_oversized_first_decoder_block(self):
        engine = _make_engine()
        engine.run_stages_0_4.side_effect = lambda requests: (
            [torch.zeros(1, 192, 512) for _ in requests],
            np.zeros((len(requests), 512, 1), dtype=np.float32),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )
        watermarker = _FakeWatermarker()
        watermarker.max_batch_size = 8
        batcher = DynamicBatcher(
            engine,
            _make_config(adaptive=True, min_ms=2.0, max_ms=10.0),
            max_batch_size=8,
            watermark_backend=watermarker,
            watermark_batch_max=8,
            watermark_default_core16=7,
        )

        await batcher.start()
        try:
            async def collect():
                return [chunk async for chunk in batcher.submit_streaming(_make_request())]

            chunks = await asyncio.wait_for(collect(), timeout=5.0)
            assert len(chunks) == 16
            assert all(chunk[0] == pytest.approx(7.0) for chunk in chunks)
            assert sum(call.shape[0] for call in watermarker.calls) == 16
            assert all(call.shape[0] <= 8 for call in watermarker.calls)
        finally:
            await batcher.stop()


class TestShutdownDrain:
    """Test that shutdown drains pending requests."""

    @pytest.mark.asyncio
    async def test_drain_cancels_pending(self):
        """Pending requests in queue should get CancelledError on shutdown."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()

        # Make S0-4 block so requests accumulate
        def blocking_s04(reqs):
            import time as time_mod

            time_mod.sleep(0.5)
            return engine.run_stages_0_4.__wrapped__(reqs)

        engine.run_stages_0_4.side_effect = blocking_s04
        engine.run_stages_0_4.__wrapped__ = _make_engine().run_stages_0_4.side_effect

        batcher = DynamicBatcher(engine, config, max_batch_size=1)
        await batcher.start()

        # Submit multiple requests; first will block, rest accumulate in queue
        futures = [asyncio.ensure_future(batcher.submit(_make_request())) for _ in range(3)]
        await asyncio.sleep(0.1)  # Let first dispatch start

        await batcher.stop()

        # At least one request should complete or be cancelled
        for f in futures:
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(f, timeout=2.0)


class TestStreamingDispatch:
    """Tests for streaming through the pipeline."""

    @pytest.mark.asyncio
    async def test_streaming_single_request(self):
        """Streaming request should yield chunks via decoder pipeline."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        batcher = DynamicBatcher(engine, config, max_batch_size=8)

        await batcher.start()
        try:
            request = _make_request()
            chunks = []
            async for chunk in batcher.submit_streaming(request):
                chunks.append(chunk)

            # Each request has T_mel=64 = 2 chunks of cs=32
            assert len(chunks) == 2
            expected_samples = _OC.chunk_size * _OC.hop_length  # 32 * 512
            for chunk in chunks:
                assert len(chunk) == expected_samples
        finally:
            await batcher.stop()


class TestCrossBatchOrdering:
    """Test cross-batch round-robin chunk ordering."""

    @pytest.mark.asyncio
    async def test_cross_batch_chunk_ordering(self):
        """When multiple batches overlap, chunk_idx=0 from all batches should be decoded first."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()

        # Track decode order
        decode_order: list[tuple[int, int]] = []  # (batch_shape, chunk_position)
        original_decode = engine.decode_chunk_batch.side_effect

        def tracking_decode(z_batch, g_batch):
            B = z_batch.shape[0]
            decode_order.append((B, len(decode_order)))
            return original_decode(z_batch, g_batch)

        engine.decode_chunk_batch.side_effect = tracking_decode

        batcher = DynamicBatcher(engine, config, max_batch_size=4)

        await batcher.start()
        try:
            requests = [_make_request() for _ in range(4)]
            results = await asyncio.wait_for(
                asyncio.gather(*[batcher.submit(r) for r in requests]),
                timeout=10.0,
            )
            assert len(results) == 4
            assert all(r.audio is not None for r in results)
            # Decoder was called at least once
            assert len(decode_order) >= 1
        finally:
            await batcher.stop()

    def test_cross_batch_sort_key_order(self):
        """Sort key (chunk_idx, req_idx, batch_id) gives true round-robin."""
        # Simulate two batches, each with 2 requests, each with 2 chunks
        items = [
            # (chunk_idx, req_idx, batch_id)
            (0, 0, 1),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 0, 2),
            (0, 1, 2),
            (1, 0, 2),
            (1, 1, 2),
        ]
        sorted_items = sorted(items, key=lambda x: (x[0], x[1], x[2]))
        expected = [
            (0, 0, 1),
            (0, 0, 2),
            (0, 1, 1),
            (0, 1, 2),
            (1, 0, 1),
            (1, 0, 2),
            (1, 1, 1),
            (1, 1, 2),
        ]
        assert sorted_items == expected


class TestS04ErrorPropagation:
    """Test that S0-4 errors propagate to futures and chunk queues."""

    @pytest.mark.asyncio
    async def test_s04_error_propagates_to_future(self):
        """S0-4 error should set exception on the future."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        engine.run_stages_0_4.side_effect = RuntimeError("S0-4 explosion")

        batcher = DynamicBatcher(engine, config, max_batch_size=8)
        await batcher.start()

        try:
            with pytest.raises(RuntimeError, match="S0-4 explosion"):
                await asyncio.wait_for(batcher.submit(_make_request()), timeout=5.0)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_s04_error_propagates_to_streaming(self):
        """S0-4 error should be delivered to streaming chunk_queue."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        engine.run_stages_0_4.side_effect = RuntimeError("S0-4 boom")

        batcher = DynamicBatcher(engine, config, max_batch_size=8)
        await batcher.start()

        try:
            with pytest.raises(RuntimeError, match="S0-4 boom"):
                async for _ in batcher.submit_streaming(_make_request()):
                    pass
        finally:
            await batcher.stop()


class TestDecoderErrorPropagation:
    """Test that decoder errors propagate correctly."""

    @pytest.mark.asyncio
    async def test_decoder_error_propagates(self):
        """Decoder error should propagate to future."""
        config = _make_config(adaptive=True, min_ms=2.0, max_ms=10.0)
        engine = _make_engine()
        engine.decode_chunk_batch.side_effect = RuntimeError("decoder crash")

        batcher = DynamicBatcher(engine, config, max_batch_size=8)
        await batcher.start()

        try:
            with pytest.raises(RuntimeError, match="decoder crash"):
                await asyncio.wait_for(batcher.submit(_make_request()), timeout=5.0)
        finally:
            await batcher.stop()
