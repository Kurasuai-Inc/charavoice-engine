"""G2P process pool for parallel pyopenjtalk execution.

Replaces the serial TCP worker + ThreadPoolExecutor pattern with
ProcessPoolExecutor(spawn) for true multicore parallelism.

Each spawned worker imports pyopenjtalk directly because process-pool workers
do not use the TCP worker client.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

_pool: ProcessPoolExecutor | None = None
_n_workers: int = 0


def _noop() -> None:
    """Pickle-safe startup probe for forcing worker initialization."""


def _worker_init() -> None:
    """Per-worker initializer: import pyopenjtalk and warm up OpenJTalk."""
    import pyopenjtalk

    # Warmup: trigger lazy init (mecab model load etc.)
    pyopenjtalk.run_frontend("テスト", use_vanilla=True)


def init_pool(n_workers: int | None = None) -> ProcessPoolExecutor:
    """Initialize the G2P process pool.

    Args:
        n_workers: Number of worker processes. Default: min(cpu_count, 4).

    Returns:
        The initialized ProcessPoolExecutor.
    """
    global _pool, _n_workers

    if _pool is not None:
        logger.warning("G2P pool already initialized, returning existing pool")
        return _pool

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 4)

    ctx = multiprocessing.get_context("spawn")
    _pool = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_worker_init,
    )
    _n_workers = n_workers

    # Force all workers to start now (avoid cold-start on first request)
    futures = [_pool.submit(_noop) for _ in range(n_workers)]
    for f in futures:
        f.result(timeout=120)

    logger.info("G2P process pool initialized with %d workers", n_workers)
    return _pool


def get_pool() -> ProcessPoolExecutor | None:
    """Return the G2P process pool, or None if not initialized."""
    return _pool


def shutdown_pool() -> None:
    """Shutdown the G2P process pool."""
    global _pool, _n_workers
    if _pool is not None:
        _pool.shutdown(wait=True, cancel_futures=False)
        logger.info("G2P process pool shut down")
        _pool = None
        _n_workers = 0
