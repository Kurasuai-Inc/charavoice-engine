"""Tests for charavoice_engine.nlp.japanese.g2p_pool."""

from __future__ import annotations

import pytest

from charavoice_engine.nlp.japanese.g2p_pool import get_pool, init_pool, shutdown_pool


@pytest.fixture(autouse=True)
def _cleanup_pool():
    """Ensure pool is shut down after each test."""
    yield
    shutdown_pool()


class TestG2pPoolLifecycle:
    def test_not_initialized_returns_none(self):
        assert get_pool() is None

    @pytest.mark.slow
    def test_init_and_shutdown(self):
        pool = init_pool(n_workers=1)
        assert pool is not None
        assert get_pool() is pool
        shutdown_pool()
        assert get_pool() is None

    @pytest.mark.slow
    def test_double_init_returns_same_pool(self):
        pool1 = init_pool(n_workers=1)
        pool2 = init_pool(n_workers=1)
        assert pool1 is pool2


class TestG2pPoolExecution:
    @pytest.mark.slow
    def test_g2p_in_spawned_worker(self):
        """check_split_and_run_g2p runs correctly in a spawned process."""
        from charavoice_engine.config import BertBackend
        from charavoice_engine.constants import Languages
        from charavoice_engine.inference.text_processing import TextProcessingConfig, check_split_and_run_g2p

        pool = init_pool(n_workers=1)

        config = TextProcessingConfig(
            add_blank=True,
            max_seq_len=768,
            device="cpu",
            bert_backend=BertBackend.ONNX_CUDA,
            onnx_providers=("CPUExecutionProvider",),
        )
        future = pool.submit(check_split_and_run_g2p, "こんにちは", Languages.JP, config)
        result = future.result(timeout=30)

        assert len(result) >= 1
        text, seq_len, g2p = result[0]
        assert text == "こんにちは"
        assert seq_len > 0
        assert g2p is not None
        assert len(g2p.phone) > 0

    @pytest.mark.slow
    def test_parallel_g2p(self):
        """Multiple G2P tasks run in parallel across workers."""
        from charavoice_engine.config import BertBackend
        from charavoice_engine.constants import Languages
        from charavoice_engine.inference.text_processing import TextProcessingConfig, check_split_and_run_g2p

        pool = init_pool(n_workers=2)

        config = TextProcessingConfig(
            add_blank=True,
            max_seq_len=768,
            device="cpu",
            bert_backend=BertBackend.ONNX_CUDA,
            onnx_providers=("CPUExecutionProvider",),
        )
        texts = ["こんにちは", "世界", "テスト", "音声合成"]
        futures = [pool.submit(check_split_and_run_g2p, t, Languages.JP, config) for t in texts]
        results = [f.result(timeout=30) for f in futures]

        assert len(results) == 4
        for r in results:
            assert len(r) >= 1
            assert r[0][2] is not None  # G2pResult exists
