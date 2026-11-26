"""Prometheus metrics for CHARAVOICE Engine."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

if TYPE_CHECKING:
    from charavoice_engine.server.app import LoadedModel

logger = logging.getLogger(__name__)

# --- Request metrics (updated by middleware in app.py) ---

REQUEST_COUNT = Counter(
    "charavoice_request_total",
    "Total /voice requests",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "charavoice_request_duration_seconds",
    "/voice request latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

IN_FLIGHT = Gauge(
    "charavoice_in_flight_requests",
    "Number of in-flight /voice requests",
)

# --- GPU metrics (updated by background task) ---

GPU_UTIL = Gauge(
    "charavoice_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
)

GPU_MEM_USED = Gauge(
    "charavoice_gpu_memory_used_mb",
    "GPU VRAM used in MB",
    ["gpu_id"],
)

# --- Batcher metrics (updated by background task) ---

QUEUE_DEPTH = Gauge(
    "charavoice_batcher_queue_depth",
    "Number of requests waiting in batcher queue",
    ["model"],
)

BATCH_TOTAL = Counter(
    "charavoice_batcher_batch_total",
    "Total number of batches processed",
    ["model"],
)

DROPPED_STALE = Counter(
    "charavoice_batcher_dropped_stale_total",
    "Total requests dropped due to queue dwell timeout",
    ["model"],
)


async def update_gauges_loop(models: dict[str, LoadedModel], interval: float = 15.0) -> None:
    """Periodically update GPU and batcher gauge metrics."""
    # Track previous batch totals to compute deltas for the Counter
    prev_batch_totals: dict[str, int] = {}
    prev_stale_totals: dict[str, int] = {}

    try:
        while True:
            _update_gpu_gauges()
            _update_batcher_gauges(models, prev_batch_totals, prev_stale_totals)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


def _update_gpu_gauges() -> None:
    """Read GPU utilization and memory from torch/GPUtil."""
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            # Memory
            allocated = torch.cuda.memory_allocated(i)
            GPU_MEM_USED.labels(gpu_id=str(i)).set(allocated / (1024 * 1024))

            # Utilization via GPUtil if available
            try:
                import GPUtil  # type: ignore[reportMissingImports]

                gpus = GPUtil.getGPUs()
                if i < len(gpus):
                    GPU_UTIL.labels(gpu_id=str(i)).set(gpus[i].load * 100)
            except ImportError:
                pass
    except Exception as e:
        logger.debug("Failed to update GPU gauges: %s", e)


def _update_batcher_gauges(
    models: dict[str, LoadedModel],
    prev_totals: dict[str, int],
    prev_stale_totals: dict[str, int],
) -> None:
    """Read batcher queue depth and batch counts."""
    for name, model in models.items():
        stats = model.batcher.stats()
        QUEUE_DEPTH.labels(model=name).set(stats.get("queue_size", 0))

        current_total = stats.get("total_batches", 0)
        prev = prev_totals.get(name, 0)
        if current_total > prev:
            BATCH_TOTAL.labels(model=name).inc(current_total - prev)
        prev_totals[name] = current_total

        current_stale = stats.get("total_dropped_stale", 0)
        prev_stale = prev_stale_totals.get(name, 0)
        if current_stale > prev_stale:
            DROPPED_STALE.labels(model=name).inc(current_stale - prev_stale)
        prev_stale_totals[name] = current_stale
