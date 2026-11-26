"""Dynamic batching engine with pipelined S0-4 / Decoder / Watermark threads.

Accumulates incoming requests and dispatches them as batches.
Stage 0-4 (text -> latent), Stage 5 (decoder), and optional fixed-chunk
watermarking run on separate threads, enabling cross-batch round-robin
decode and ordered chunk-local watermark microbatching.

Supports adaptive batch timeout: short timeout at low load for minimal
latency, longer timeout at high load for larger batches and better GPU
efficiency. Controlled by EWMA fill rate (batch_size / max_batch_size).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import random
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from charavoice_engine.config import FastInferenceConfig
from charavoice_engine.inference.engine_base import InferenceEngine, InferenceRequest, InferenceResult
from charavoice_engine.inference.overlap_save import (
    OverlapSaveConfig,
    build_decoder_chunks_with_lengths,
    extract_valid_audio_batch_numpy,
)
from charavoice_engine.watermark.common import BatchedWatermarkBackend, validate_watermark_word
from charavoice_engine.watermark.protocol import WatermarkPlan, build_watermark_plan, extension_hex_to_bytes

logger = logging.getLogger(__name__)

_EWMA_ALPHA = 0.15  # EWMA smoothing factor (responds in ~7 batches)


@dataclass
class _PendingRequest:
    """Internal wrapper for a request waiting in the batch queue."""

    request: InferenceRequest
    future: asyncio.Future[InferenceResult]
    submit_time: float = field(default_factory=time.monotonic)
    is_streaming: bool = False
    chunk_queue: asyncio.Queue[NDArray[Any] | BaseException | None] | None = None
    resolved_watermark_core16: int | None = None
    resolved_watermark_extension: bytes = b""
    resolved_watermark_plan: WatermarkPlan | None = None
    stream_closed: asyncio.Event = field(default_factory=asyncio.Event)
    delivery_tasks: set[asyncio.Task[Any]] = field(default_factory=set)


@dataclass
class _DecoderJob:
    """Work item passed from S0-4 thread to decoder thread."""

    batch_id: int
    # (req_idx_in_batch, chunk_idx, chunk_numpy [C, F])
    chunks: list[tuple[int, int, NDArray[Any]]]
    g_np: NDArray[Any]  # [B, 512, 1]
    pending_requests: list[_PendingRequest]
    chunks_per_request: list[int]
    audio_parts: dict[int, list[NDArray[Any]]]
    delivered_counts: dict[int, int]
    s04_times: tuple[float, float, float, float, float]
    chunk_block_counts: dict[tuple[int, int], int] = field(default_factory=dict)


@dataclass
class _WatermarkBlock:
    """Contiguous decoder batch awaiting watermark processing."""

    items: list[tuple[_DecoderJob, int, int]]
    audio_batch: NDArray[np.float32]
    message_batch: NDArray[np.int64]

    @property
    def batch_size(self) -> int:
        return int(self.audio_batch.shape[0])

    @property
    def is_streaming(self) -> bool:
        return any(job.pending_requests[req_idx].is_streaming for job, req_idx, _chunk_idx in self.items)

    def split(self, take: int) -> tuple[_WatermarkBlock, _WatermarkBlock]:
        """Split the block into a prefix and suffix while preserving order."""
        if take <= 0 or take >= self.batch_size:
            raise ValueError(f"split size must be in (0, {self.batch_size}), got {take}")

        return (
            _WatermarkBlock(
                items=self.items[:take],
                audio_batch=self.audio_batch[:take],
                message_batch=self.message_batch[:take],
            ),
            _WatermarkBlock(
                items=self.items[take:],
                audio_batch=self.audio_batch[take:],
                message_batch=self.message_batch[take:],
            ),
        )


class DynamicBatcher:
    """Accumulates requests and dispatches batched inference.

    Architecture: dedicated threads for GPU work:
    - s04_executor: runs Stages 0-4 (enc_p, dp, flow) -- one batch at a time
    - decoder_thread: runs Stage 5 (decoder) -- drains job queue, cross-batch round-robin
    - watermark_thread: optional fixed-chunk batched watermark stage

    This pipeline lets S0-4 for batch N+1 overlap with Decoder for batch N,
    and keeps chunk watermarking out of the FastAPI request path so the
    decoder's existing fixed chunk seam remains the scheduling source.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        config: FastInferenceConfig,
        max_batch_size: int = 8,
        batch_timeout_ms: float = 50.0,
        max_queue_size: int = 32,
        queue_deadline: float = 5.0,
        watermark_backend: BatchedWatermarkBackend | None = None,
        watermark_batch_max: int = 16,
        watermark_timeout_ms: float = 0.5,
        watermark_default_core16: int | None = None,
        watermark_default_extension: str | None = None,
    ):
        self.engine = engine
        self.config = config
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = config.batch_timeout_ms if config.throughput_optimized else batch_timeout_ms
        self.max_queue_size = max_queue_size
        self.queue_deadline = queue_deadline

        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue(maxsize=max_queue_size)
        self._s04_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s04-infer")
        self._decoder_job_queue: queue.Queue[_DecoderJob | None] = queue.Queue(maxsize=2)
        self._decoder_thread: threading.Thread | None = None
        self._dispatch_task: asyncio.Task[None] | None = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._batch_counter = 0

        self._watermark_backend = watermark_backend
        self._watermark_thread: threading.Thread | None = None
        self._watermark_queue: queue.Queue[_WatermarkBlock | None] | None = None
        self._watermark_timeout_ms = max(watermark_timeout_ms, 0.0)
        self._watermark_default_core16 = (
            validate_watermark_word(watermark_default_core16, field_name="watermark_default_core16")
            if watermark_default_core16 is not None
            else None
        )
        self._watermark_default_extension = extension_hex_to_bytes(watermark_default_extension)
        self._message_rng = random.Random()

        if watermark_backend is not None:
            backend_batch_max = watermark_backend.max_batch_size
            if backend_batch_max is not None and watermark_batch_max > backend_batch_max:
                logger.warning(
                    "watermark_batch_max=%d exceeds backend max %d; clipping",
                    watermark_batch_max,
                    backend_batch_max,
                )
                watermark_batch_max = backend_batch_max

            oc = self._get_overlap_config()
            expected_chunk_samples = min(32, oc.chunk_size) * oc.hop_length
            if watermark_backend.chunk_samples != expected_chunk_samples:
                raise ValueError(
                    f"watermark backend chunk size must be {expected_chunk_samples}, "
                    f"got {watermark_backend.chunk_samples}"
                )
            self._watermark_queue = queue.Queue()

        self._watermark_batch_max = max(1, watermark_batch_max)
        self._watermark_streaming_batch_max = min(32, self._watermark_batch_max)

        # Adaptive timeout state
        self._adaptive = config.adaptive_batch_timeout
        self._min_timeout_ms = config.batch_timeout_min_ms
        self._max_timeout_ms = config.batch_timeout_max_ms
        self._ewma_fill_rate: float = 0.0
        self._ewma_initialized: bool = False

        # Arrival rate estimation for rate-based timeout
        self._ewma_arrival_rate: float = 0.0  # requests/sec per batcher
        self._arrival_rate_initialized: bool = False
        self._last_collect_time: float = 0.0

        # Stats
        self._total_requests = 0
        self._total_batches = 0
        self._total_batch_items = 0
        self._total_timeout_ms: float = 0.0
        self._total_text_length: int = 0
        self._total_dropped_stale: int = 0

        self._wm_total_batches = 0
        self._wm_total_batch_items = 0
        self._wm_total_chunks = 0
        self._wm_total_latency_ms = 0.0

    async def start(self) -> None:
        """Start the dispatch loop and worker threads."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()

        if self._watermark_backend is not None:
            self._watermark_thread = threading.Thread(
                target=self._run_watermark_worker,
                name="wm-infer",
                daemon=True,
            )
            self._watermark_thread.start()

        self._decoder_thread = threading.Thread(
            target=self._run_decoder_worker,
            name="dec-infer",
            daemon=True,
        )
        self._decoder_thread.start()

        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        mode = "adaptive" if self._adaptive else "fixed"
        if self._adaptive:
            logger.info(
                "DynamicBatcher started (pipeline, %s, max_batch=%d, timeout=%.1f-%.1fms, "
                "queue_deadline=%.1fs, queue_size=%d, watermark=%s)",
                mode,
                self.max_batch_size,
                self._min_timeout_ms,
                self._max_timeout_ms,
                self.queue_deadline,
                self.max_queue_size,
                self._watermark_backend.backend_name if self._watermark_backend is not None else "none",
            )
        else:
            logger.info(
                "DynamicBatcher started (pipeline, %s, max_batch=%d, timeout=%.1fms, "
                "queue_deadline=%.1fs, queue_size=%d, watermark=%s)",
                mode,
                self.max_batch_size,
                self.batch_timeout_ms,
                self.queue_deadline,
                self.max_queue_size,
                self._watermark_backend.backend_name if self._watermark_backend is not None else "none",
            )

    async def stop(self) -> None:
        """Stop the dispatch loop and wait for pending requests."""
        self._running = False
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatch_task

        # Shut down S0-4 executor first (no more jobs will be produced)
        self._s04_executor.shutdown(wait=True)

        # Drain decoder work, then watermark work
        self._decoder_job_queue.put(None)
        if self._decoder_thread is not None:
            self._decoder_thread.join(timeout=30)

        if self._watermark_queue is not None:
            self._watermark_queue.put(None)
        if self._watermark_thread is not None:
            self._watermark_thread.join(timeout=30)

        self._drain_queue_on_shutdown()
        logger.info("DynamicBatcher stopped")

    def _drain_queue_on_shutdown(self) -> None:
        """Cancel all pending requests still in the queue on shutdown."""
        cancelled = 0
        while True:
            try:
                pending = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(asyncio.CancelledError())
            if pending.is_streaming and pending.chunk_queue is not None:
                with contextlib.suppress(asyncio.QueueFull):
                    pending.chunk_queue.put_nowait(asyncio.CancelledError())
            cancelled += 1
        if cancelled:
            logger.info("Shutdown: cancelled %d pending requests", cancelled)

    async def submit(self, request: InferenceRequest) -> InferenceResult:
        """Submit a request and wait for the result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResult] = loop.create_future()
        pending = self._build_pending(request=request, future=future)

        await self._queue.put(pending)
        self._total_requests += 1
        self._total_text_length += len(request.text)
        return await future

    async def submit_streaming(self, request: InferenceRequest) -> AsyncGenerator[NDArray[Any]]:
        """Submit a streaming request. Yields audio chunks as they're decoded."""
        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue[NDArray[Any] | BaseException | None] = asyncio.Queue(maxsize=4)
        future: asyncio.Future[InferenceResult] = loop.create_future()
        future.add_done_callback(self._consume_future_outcome)
        pending = self._build_pending(request=request, future=future, is_streaming=True, chunk_queue=chunk_queue)

        try:
            await self._queue.put(pending)
            self._total_requests += 1
            self._total_text_length += len(request.text)
            while True:
                item = await chunk_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            pending.stream_closed.set()
            if not future.done():
                future.cancel()

    def _build_pending(
        self,
        request: InferenceRequest,
        future: asyncio.Future[InferenceResult],
        is_streaming: bool = False,
        chunk_queue: asyncio.Queue[NDArray[Any] | BaseException | None] | None = None,
    ) -> _PendingRequest:
        resolved_core16 = None
        resolved_extension = b""
        if self._watermark_backend is not None:
            resolved = self._resolve_watermark_payload(
                core16=request.watermark_core16,
                extension=request.watermark_extension,
            )
            if resolved is not None:
                resolved_core16, resolved_extension = resolved
        return _PendingRequest(
            request=request,
            future=future,
            is_streaming=is_streaming,
            chunk_queue=chunk_queue,
            resolved_watermark_core16=resolved_core16,
            resolved_watermark_extension=resolved_extension,
        )

    def _resolve_watermark_payload(
        self,
        *,
        core16: int | None,
        extension: str | None,
    ) -> tuple[int, bytes] | None:
        if core16 is not None:
            resolved_core16 = validate_watermark_word(core16)
        elif self._watermark_default_core16 is not None:
            resolved_core16 = self._watermark_default_core16
        else:
            return None

        if extension is not None:
            resolved_extension = extension_hex_to_bytes(extension)
        else:
            resolved_extension = self._watermark_default_extension
        return resolved_core16, resolved_extension

    def _compute_timeout_ms(self, drain_count: int = 0) -> float:
        """Compute batch accumulation timeout based on current load."""
        if not self._adaptive:
            return self.batch_timeout_ms

        if not self._ewma_initialized:
            return self.batch_timeout_ms

        t = min(self._ewma_fill_rate, 1.0)
        base = self._min_timeout_ms + t * (self._max_timeout_ms - self._min_timeout_ms)

        if drain_count >= 2 and self._arrival_rate_initialized and self._ewma_arrival_rate > 0:
            target = self.max_batch_size // 2
            needed = target - drain_count
            if needed > 0:
                rate_based_ms = (needed / self._ewma_arrival_rate) * 1000
                rate_based_ms = max(self._min_timeout_ms, min(rate_based_ms, self._max_timeout_ms))
                return max(base, rate_based_ms)

        return base

    async def _dispatch_loop(self) -> None:
        """Main loop: collect requests into batches and dispatch S0-4."""
        prev_task: asyncio.Task[None] | None = None
        batch: list[_PendingRequest] = []
        try:
            while self._running:
                batch = []

                try:
                    first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    batch.append(first)
                except TimeoutError:
                    continue

                while len(batch) < self.max_batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                now = time.monotonic()
                drain_count = len(batch)
                if self._last_collect_time > 0 and drain_count > 0:
                    interval = now - self._last_collect_time
                    if interval > 0.001:
                        rate = drain_count / interval
                        if self._arrival_rate_initialized:
                            self._ewma_arrival_rate = _EWMA_ALPHA * rate + (1 - _EWMA_ALPHA) * self._ewma_arrival_rate
                        else:
                            self._ewma_arrival_rate = rate
                            self._arrival_rate_initialized = True
                self._last_collect_time = now

                if len(batch) < self.max_batch_size:
                    timeout_ms = self._compute_timeout_ms(drain_count=drain_count)
                    self._total_timeout_ms += timeout_ms
                    deadline = time.monotonic() + (timeout_ms / 1000.0)

                    while len(batch) < self.max_batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                            batch.append(item)
                        except TimeoutError:
                            break

                if batch:
                    if prev_task is not None:
                        await prev_task

                    if self.config.throughput_optimized:
                        while len(batch) < self.max_batch_size:
                            try:
                                batch.append(self._queue.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                    fill = len(batch) / self.max_batch_size
                    if self._ewma_initialized:
                        self._ewma_fill_rate = _EWMA_ALPHA * fill + (1 - _EWMA_ALPHA) * self._ewma_fill_rate
                    else:
                        self._ewma_fill_rate = fill
                        self._ewma_initialized = True

                    if self.queue_deadline > 0:
                        now = time.monotonic()
                        stale = [p for p in batch if now - p.submit_time > self.queue_deadline]
                        if stale:
                            self._total_dropped_stale += len(stale)
                            logger.warning(
                                "Dropping %d/%d stale requests (deadline=%.1fs)",
                                len(stale),
                                len(batch),
                                self.queue_deadline,
                            )
                            self._fail_batch_from_loop(
                                stale,
                                TimeoutError(f"Request exceeded queue deadline ({self.queue_deadline}s)"),
                            )
                            batch = [p for p in batch if now - p.submit_time <= self.queue_deadline]

                    if not batch:
                        prev_task = None
                        continue

                    prev_task = asyncio.create_task(self._dispatch_batch(batch))
        except asyncio.CancelledError:
            if batch:
                self._fail_batch_from_loop(batch, asyncio.CancelledError())
            raise
        finally:
            if prev_task is not None and not prev_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await prev_task

    def _run_s04_and_build_job(
        self,
        batch: list[_PendingRequest],
        batch_id: int,
    ) -> _DecoderJob:
        """Run S0-4 and build a decoder job on the S0-4 worker thread."""
        requests = [p.request for p in batch]
        z_list, g_np, s04_times = self.engine.run_stages_0_4(requests)

        oc = self._get_overlap_config()
        chunk_pool: list[tuple[int, int, NDArray[Any]]] = []
        chunks_per_request: list[int] = []
        chunk_block_counts: dict[tuple[int, int], int] = {}

        for req_idx, z_all in enumerate(z_list):
            z_np = z_all[0].cpu().numpy().astype(np.float32)
            chunks, valid_blocks = build_decoder_chunks_with_lengths(z_np, oc)
            chunks_per_request.append(sum(valid_blocks))
            if self._watermark_backend is not None:
                pending = batch[req_idx]
                if pending.resolved_watermark_core16 is not None:
                    pending.resolved_watermark_plan = build_watermark_plan(
                        core16=pending.resolved_watermark_core16,
                        extension_bytes=pending.resolved_watermark_extension,
                        chunk_count=sum(valid_blocks),
                    )
            for chunk_idx, chunk_np in enumerate(chunks):
                chunk_pool.append((req_idx, chunk_idx, chunk_np))
                chunk_block_counts[req_idx, chunk_idx] = valid_blocks[chunk_idx]

        chunk_pool.sort(key=lambda x: (x[1], x[0]))

        audio_parts: dict[int, list[NDArray[Any]]] = {i: [] for i in range(len(batch)) if not batch[i].is_streaming}
        delivered_counts = {i: 0 for i in range(len(batch))}

        return _DecoderJob(
            batch_id=batch_id,
            chunks=chunk_pool,
            g_np=g_np,
            pending_requests=batch,
            chunks_per_request=chunks_per_request,
            audio_parts=audio_parts,
            delivered_counts=delivered_counts,
            s04_times=s04_times,
            chunk_block_counts=chunk_block_counts,
        )

    async def _dispatch_batch(self, batch: list[_PendingRequest]) -> None:
        """Run S0-4 on the worker, then enqueue a decoder job."""
        self._total_batches += 1
        self._total_batch_items += len(batch)
        batch_size = len(batch)

        batch.sort(key=lambda p: p.request.seq_len_hint)

        self._batch_counter += 1
        batch_id = self._batch_counter

        loop = asyncio.get_running_loop()
        timeout = self.config.inference_timeout or None

        try:
            coro = loop.run_in_executor(self._s04_executor, self._run_s04_and_build_job, batch, batch_id)
            job = await (asyncio.wait_for(coro, timeout=timeout) if timeout else coro)
        except asyncio.CancelledError:
            self._fail_batch_from_loop(batch, asyncio.CancelledError())
            raise
        except TimeoutError:
            logger.error("S0-4 timed out after %ss (batch_size=%d)", timeout, batch_size)
            exc = TimeoutError(f"S0-4 timed out after {timeout}s")
            self._fail_batch_from_loop(batch, exc)
            return
        except Exception as e:
            self._log_batch_error(e, batch_size)
            self._fail_batch_from_loop(batch, e)
            return

        logger.debug(
            "[s04] B=%d batch_id=%d chunks=%d g2p=%.1fms bert=%.1fms enc+dp=%.1fms lenreg=%.1fms flow=%.1fms",
            batch_size,
            batch_id,
            len(job.chunks),
            job.s04_times[0],
            job.s04_times[1],
            job.s04_times[2],
            job.s04_times[3],
            job.s04_times[4],
        )

        await asyncio.to_thread(self._decoder_job_queue.put, job)

    def _get_overlap_config(self) -> OverlapSaveConfig:
        oc = getattr(self.engine, "overlap_config", None)
        if oc is not None:
            return oc
        return OverlapSaveConfig()

    def _run_decoder_worker(self) -> None:
        """Decoder thread main loop: drain jobs, cross-batch round-robin decode."""
        loop = self._loop
        assert loop is not None

        oc = self._get_overlap_config()
        batch_max = self.config.trt.decoder_batch_max

        while True:
            first_job = self._decoder_job_queue.get()
            if first_job is None:
                return

            jobs: list[_DecoderJob] = [first_job]
            while True:
                try:
                    job = self._decoder_job_queue.get_nowait()
                    if job is None:
                        self._decode_jobs(jobs, oc, batch_max, loop)
                        return
                    jobs.append(job)
                except queue.Empty:
                    break

            self._decode_jobs(jobs, oc, batch_max, loop)

    def _decode_jobs(
        self,
        jobs: list[_DecoderJob],
        oc: OverlapSaveConfig,
        batch_max: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Decode one or more jobs with cross-batch round-robin ordering."""
        t_dec_start = time.perf_counter()
        do_profile = logger.isEnabledFor(logging.DEBUG)

        if len(jobs) == 1:
            self._decode_single_job(jobs[0], oc, batch_max, loop)
        else:
            self._decode_cross_batch(jobs, oc, batch_max, loop)

        if do_profile:
            dec_ms = (time.perf_counter() - t_dec_start) * 1000
            total_chunks = sum(len(j.chunks) for j in jobs)
            logger.debug("[dec] jobs=%d chunks=%d dec=%.1fms", len(jobs), total_chunks, dec_ms)

    def _decode_single_job(
        self,
        job: _DecoderJob,
        oc: OverlapSaveConfig,
        batch_max: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Decode all chunks from a single job."""
        try:
            total_chunks = len(job.chunks)

            for batch_start in range(0, total_chunks, batch_max):
                batch_end = min(batch_start + batch_max, total_chunks)
                batch_items = job.chunks[batch_start:batch_end]

                z_batch = np.stack([chunk_np for _req_idx, _chunk_idx, chunk_np in batch_items])
                g_batch = np.stack([job.g_np[req_idx] for req_idx, _chunk_idx, _chunk_np in batch_items])
                wav_batch = self.engine.decode_chunk_batch(z_batch, g_batch)
                audio_batch = extract_valid_audio_batch_numpy(wav_batch, oc)
                items = [(job, req_idx, chunk_idx) for req_idx, chunk_idx, _chunk_np in batch_items]
                self._dispatch_decoded_batch(items, audio_batch, loop)

        except Exception as e:
            self._log_batch_error(e, len(job.pending_requests))
            self._fail_batch_from_thread(job.pending_requests, e, loop)

    def _decode_cross_batch(
        self,
        jobs: list[_DecoderJob],
        oc: OverlapSaveConfig,
        batch_max: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Decode chunks from multiple jobs with cross-batch round-robin."""
        try:
            unified: list[tuple[int, int, int, _DecoderJob, NDArray[Any]]] = []
            for job in jobs:
                for req_idx, chunk_idx, chunk_np in job.chunks:
                    unified.append((chunk_idx, req_idx, job.batch_id, job, chunk_np))

            unified.sort(key=lambda x: (x[0], x[1], x[2]))

            total = len(unified)
            for batch_start in range(0, total, batch_max):
                batch_end = min(batch_start + batch_max, total)
                batch_items = unified[batch_start:batch_end]

                z_batch = np.stack([chunk_np for _ci, _ri, _bi, _job, chunk_np in batch_items])
                g_batch = np.stack([job.g_np[req_idx] for _ci, req_idx, _bi, job, _chunk_np in batch_items])
                wav_batch = self.engine.decode_chunk_batch(z_batch, g_batch)
                audio_batch = extract_valid_audio_batch_numpy(wav_batch, oc)
                items = [(job, req_idx, chunk_idx) for chunk_idx, req_idx, _batch_id, job, _chunk_np in batch_items]
                self._dispatch_decoded_batch(items, audio_batch, loop)

        except Exception as e:
            for job in jobs:
                self._log_batch_error(e, len(job.pending_requests))
                self._fail_batch_from_thread(job.pending_requests, e, loop)

    def _dispatch_decoded_batch(
        self,
        items: list[tuple[_DecoderJob, int, int]],
        audio_batch: NDArray[Any],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        oc = self._get_overlap_config()
        factor = oc.chunk_size // min(32, oc.chunk_size)
        if factor > 1:
            block_samples = min(32, oc.chunk_size) * oc.hop_length
            expanded: list[tuple[_DecoderJob, int, int]] = []
            audio_parts = []
            for index, (job, req_idx, chunk_idx) in enumerate(items):
                for block in range(job.chunk_block_counts[req_idx, chunk_idx]):
                    expanded.append((job, req_idx, chunk_idx * factor + block))
                    audio_parts.append(audio_batch[index, block * block_samples : (block + 1) * block_samples])
            for start in range(0, len(expanded), 16):
                self._dispatch_output_blocks(
                    expanded[start : start + 16], np.stack(audio_parts[start : start + 16]), loop
                )
            return
        self._dispatch_output_blocks(items, audio_batch, loop)

    def _dispatch_output_blocks(
        self, items: list[tuple[_DecoderJob, int, int]], audio_batch: NDArray[Any], loop: asyncio.AbstractEventLoop
    ) -> None:
        audio_f32 = np.asarray(audio_batch, dtype=np.float32)

        if self._watermark_backend is None:
            for idx, (job, req_idx, chunk_idx) in enumerate(items):
                p = job.pending_requests[req_idx]
                if p.future.done():
                    continue
                self._deliver_chunk(job, req_idx, chunk_idx, audio_f32[idx], loop)
            return

        # Split items into watermark vs bypass based on whether a plan exists
        wm_items: list[tuple[_DecoderJob, int, int]] = []
        wm_indices: list[int] = []
        for idx, (job, req_idx, chunk_idx) in enumerate(items):
            p = job.pending_requests[req_idx]
            if p.future.done():
                continue
            if p.resolved_watermark_plan is not None:
                wm_items.append((job, req_idx, chunk_idx))
                wm_indices.append(idx)
            else:
                self._deliver_chunk(job, req_idx, chunk_idx, audio_f32[idx], loop)

        if not wm_items:
            return

        block = self._build_watermark_block(wm_items, audio_f32[wm_indices])
        if block is None:
            return

        assert self._watermark_queue is not None
        self._watermark_queue.put(block)

    def _build_watermark_block(
        self,
        items: list[tuple[_DecoderJob, int, int]],
        audio_batch: NDArray[Any],
    ) -> _WatermarkBlock | None:
        live_items: list[tuple[_DecoderJob, int, int]] = []
        live_indices: list[int] = []
        message_rows: list[NDArray[np.int64]] = []

        for idx, (job, req_idx, chunk_idx) in enumerate(items):
            pending = job.pending_requests[req_idx]
            if pending.future.done():
                continue
            if pending.resolved_watermark_plan is None:
                raise RuntimeError("watermark plan must be resolved before entering the watermark stage")
            live_items.append((job, req_idx, chunk_idx))
            live_indices.append(idx)
            message_rows.append(pending.resolved_watermark_plan.bits_for_chunk(chunk_idx))

        if not live_items:
            return None

        return _WatermarkBlock(
            items=live_items,
            audio_batch=np.asarray(audio_batch[live_indices], dtype=np.float32),
            message_batch=np.stack(message_rows).astype(np.int64, copy=False),
        )

    def _run_watermark_worker(self) -> None:
        """Optional watermark thread: preserve decoder order while microbatching chunks."""
        if self._watermark_backend is None or self._watermark_queue is None:
            return

        loop = self._loop
        assert loop is not None
        carry_block: _WatermarkBlock | None = None

        try:
            while True:
                if carry_block is not None:
                    first_block = carry_block
                    carry_block = None
                else:
                    first_block = self._watermark_queue.get()

                if first_block is None:
                    return

                saw_shutdown = False
                deadline = time.monotonic() + (self._watermark_timeout_ms / 1000.0)
                batch_cap = (
                    self._watermark_streaming_batch_max if first_block.is_streaming else self._watermark_batch_max
                )
                # A single decoder dispatch may contain more chunks than the
                # watermark TensorRT profile accepts. Split the first block as
                # well as later blocks, preserving its tail for the next pass.
                if first_block.batch_size > batch_cap:
                    first_block, carry_block = first_block.split(batch_cap)
                blocks = [first_block]
                total_items = first_block.batch_size

                while total_items < batch_cap:
                    if carry_block is not None:
                        block = carry_block
                        carry_block = None
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            block = self._watermark_queue.get(timeout=remaining)
                        except queue.Empty:
                            break

                    if block is None:
                        saw_shutdown = True
                        break

                    if total_items + block.batch_size <= batch_cap:
                        blocks.append(block)
                        total_items += block.batch_size
                        continue

                    take = batch_cap - total_items
                    if take <= 0:
                        carry_block = block
                        break

                    head, tail = block.split(take)
                    blocks.append(head)
                    total_items += head.batch_size
                    carry_block = tail
                    break

                self._apply_watermark_batch(blocks, loop)
                if saw_shutdown:
                    return
        except Exception:
            logger.critical("watermark worker thread crashed", exc_info=True)
            self._drain_watermark_queue_on_crash(loop)

    def _apply_watermark_batch(
        self,
        blocks: list[_WatermarkBlock],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        assert self._watermark_backend is not None

        try:
            t0 = time.perf_counter()
            if len(blocks) == 1:
                meta = blocks[0].items
                audio_batch = np.ascontiguousarray(blocks[0].audio_batch, dtype=np.float32)
                messages = blocks[0].message_batch
            else:
                meta = [item for block in blocks for item in block.items]
                audio_batch = np.concatenate([block.audio_batch for block in blocks], axis=0).astype(
                    np.float32, copy=False
                )
                messages = np.concatenate([block.message_batch for block in blocks], axis=0)
            out_batch = self._watermark_backend.apply_batch(audio_batch, messages)
            latency_ms = (time.perf_counter() - t0) * 1000

            if out_batch.shape != audio_batch.shape:
                raise RuntimeError(f"watermark backend returned shape {out_batch.shape}, expected {audio_batch.shape}")

            self._wm_total_batches += 1
            self._wm_total_batch_items += len(meta)
            self._wm_total_chunks += len(meta)
            self._wm_total_latency_ms += latency_ms

            if self.config.throughput_optimized:
                delivery = asyncio.run_coroutine_threadsafe(self._deliver_batch(meta, out_batch, loop), loop)
                try:
                    delivery.result(timeout=10)
                except BaseException:
                    delivery.cancel()
                    raise
            else:
                for (job, req_idx, chunk_idx), out_audio in zip(meta, out_batch, strict=True):
                    self._deliver_chunk(job, req_idx, chunk_idx, out_audio, loop)

        except Exception as e:
            pending_requests: list[_PendingRequest] = []
            seen: set[int] = set()
            for block in blocks:
                for job, req_idx, _chunk_idx in block.items:
                    pending = job.pending_requests[req_idx]
                    pid = id(pending)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    pending_requests.append(pending)
            self._log_batch_error(e, len(pending_requests))
            self._fail_batch_from_thread(pending_requests, e, loop)

    def _drain_watermark_queue_on_crash(self, loop: asyncio.AbstractEventLoop) -> None:
        """Fail all remaining watermark blocks after the worker thread crashes."""
        assert self._watermark_queue is not None
        exc = RuntimeError("watermark worker thread crashed")
        while True:
            try:
                block = self._watermark_queue.get_nowait()
            except queue.Empty:
                break
            if block is None:
                break
            pending: list[_PendingRequest] = []
            seen: set[int] = set()
            for job, req_idx, _chunk_idx in block.items:
                p = job.pending_requests[req_idx]
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    pending.append(p)
            self._fail_batch_from_thread(pending, exc, loop)

    @staticmethod
    async def _put_stream_item(pending: _PendingRequest, item: NDArray[Any] | BaseException | None) -> bool:
        """Release backpressure as soon as the consumer closes, including after completion."""
        assert pending.chunk_queue is not None
        if pending.stream_closed.is_set() or pending.future.cancelled():
            return False
        if not pending.chunk_queue.full():
            pending.chunk_queue.put_nowait(item)
            return True
        put = asyncio.create_task(pending.chunk_queue.put(item))
        closed = asyncio.create_task(pending.stream_closed.wait())
        try:
            done, _ = await asyncio.wait((put, closed), return_when=asyncio.FIRST_COMPLETED)
            if closed in done:
                return False
            await put
            return not pending.stream_closed.is_set()
        finally:
            for task in (put, closed):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put, closed, return_exceptions=True)

    async def _deliver_batch(
        self, items: list[tuple[_DecoderJob, int, int]], audio: NDArray[Any], loop: asyncio.AbstractEventLoop
    ) -> None:
        for (job, req_idx, _chunk_idx), value in zip(items, audio, strict=True):
            pending = job.pending_requests[req_idx]
            if pending.future.done():
                continue
            if pending.is_streaming:
                if not await self._put_stream_item(pending, value):
                    continue
            else:
                job.audio_parts[req_idx].append(value)
            job.delivered_counts[req_idx] += 1
            if job.delivered_counts[req_idx] == job.chunks_per_request[req_idx]:
                self._finalize_request(job, req_idx, loop)

    def _deliver_chunk(
        self,
        job: _DecoderJob,
        req_idx: int,
        chunk_idx: int,
        audio_np: NDArray[np.float32],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        p = job.pending_requests[req_idx]
        if p.future.done():
            return

        if p.is_streaming:
            assert p.chunk_queue is not None
            delivery = asyncio.run_coroutine_threadsafe(self._put_stream_item(p, audio_np), loop)
            try:
                if not delivery.result(timeout=10):
                    return
            except Exception as e:
                delivery.cancel()
                logger.warning("chunk_queue.put failed (req=%d, chunk=%d): %s", req_idx, chunk_idx, e)
                job.audio_parts.pop(req_idx, None)
                self._fail_batch_from_thread([p], RuntimeError(f"Failed to deliver streaming chunk: {e}"), loop)
                return
        else:
            job.audio_parts[req_idx].append(audio_np)

        job.delivered_counts[req_idx] += 1
        if job.delivered_counts[req_idx] == job.chunks_per_request[req_idx]:
            self._finalize_request(job, req_idx, loop)

    def _finalize_request(self, job: _DecoderJob, req_idx: int, loop: asyncio.AbstractEventLoop) -> None:
        """Finalize a fully delivered request."""
        p = job.pending_requests[req_idx]
        if p.future.done():
            return

        sr = self.engine.sample_rate
        if p.is_streaming:
            assert p.chunk_queue is not None

            def finish_stream() -> None:
                if not p.future.done():
                    p.future.set_result(InferenceResult(audio=np.array([], dtype=np.float32), sample_rate=sr))
                    self._schedule_stream_item(p, None)

            loop.call_soon_threadsafe(finish_stream)
            return

        parts = job.audio_parts.get(req_idx, [])
        audio = np.concatenate(parts) if parts else np.array([], dtype=np.float32)

        def finish() -> None:
            if not p.future.done():
                p.future.set_result(InferenceResult(audio=audio, sample_rate=sr))

        loop.call_soon_threadsafe(finish)

    @staticmethod
    def _schedule_stream_item(pending: _PendingRequest, item: NDArray[Any] | None | BaseException) -> None:
        task = asyncio.create_task(DynamicBatcher._put_stream_item(pending, item))
        pending.delivery_tasks.add(task)
        task.add_done_callback(pending.delivery_tasks.discard)

    @staticmethod
    def _consume_future_outcome(future: asyncio.Future[Any]) -> None:
        """Suppress un-awaited streaming future warnings."""
        if future.cancelled():
            return
        with contextlib.suppress(Exception):
            future.exception()

    @staticmethod
    def _fail_batch_from_loop(
        batch: list[_PendingRequest],
        exc: BaseException,
    ) -> None:
        """Deliver error to all pending requests (called from event loop thread)."""
        for p in batch:
            if p.future.done():
                continue
            if p.is_streaming and p.chunk_queue is not None:
                if not p.chunk_queue.full() and not p.stream_closed.is_set():
                    p.chunk_queue.put_nowait(exc)
                else:
                    DynamicBatcher._schedule_stream_item(p, exc)
            p.future.set_exception(exc)

    @staticmethod
    def _fail_batch_from_thread(
        batch: list[_PendingRequest],
        exc: BaseException,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Deliver error to all pending requests (called from worker thread)."""
        loop.call_soon_threadsafe(DynamicBatcher._fail_batch_from_loop, batch, exc)

    @staticmethod
    def _log_batch_error(e: Exception, batch_size: int) -> None:
        err_msg = str(e).lower()
        if any(kw in err_msg for kw in ("out of memory", "oom", "cuda error")):
            logger.warning("GPU memory error during batch inference (batch_size=%d): %s", batch_size, e)
        else:
            logger.error("Batch inference failed: %s", e)

    def stats(self) -> dict[str, Any]:
        """Return batcher statistics."""
        avg_batch = self._total_batch_items / self._total_batches if self._total_batches > 0 else 0.0
        avg_timeout = self._total_timeout_ms / self._total_batches if self._total_batches > 0 else 0.0
        wm_avg_batch = self._wm_total_batch_items / self._wm_total_batches if self._wm_total_batches > 0 else 0.0
        wm_avg_latency = self._wm_total_latency_ms / self._wm_total_batches if self._wm_total_batches > 0 else 0.0
        return {
            "inference_profile": self.config.profile.value,
            "decoder_chunk_frames": self._get_overlap_config().chunk_size,
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": round(avg_batch, 2),
            "queue_size": self._queue.qsize(),
            "max_batch_size": self.max_batch_size,
            "adaptive": self._adaptive,
            "avg_timeout_ms": round(avg_timeout, 2),
            "ewma_fill_rate": round(self._ewma_fill_rate, 3),
            "ewma_arrival_rate": round(self._ewma_arrival_rate, 1),
            "total_text_length": self._total_text_length,
            "total_dropped_stale": self._total_dropped_stale,
            "watermark_backend": self._watermark_backend.backend_name
            if self._watermark_backend is not None
            else "none",
            "wm_total_batches": self._wm_total_batches,
            "wm_avg_batch_size": round(wm_avg_batch, 2),
            "wm_queue_size": self._watermark_queue.qsize() if self._watermark_queue is not None else 0,
            "wm_total_chunks": self._wm_total_chunks,
            "wm_avg_latency_ms": round(wm_avg_latency, 2),
            "wm_batch_max": self._watermark_batch_max if self._watermark_backend is not None else 0,
            "wm_timeout_ms": round(self._watermark_timeout_ms, 3) if self._watermark_backend is not None else 0.0,
        }
