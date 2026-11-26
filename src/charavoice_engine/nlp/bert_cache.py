"""Thread-safe LRU cache for BERT feature tensors.

Since BERT feature extraction is the #2 bottleneck (~22ms, 17% of total),
caching eliminates this cost for repeated texts.

Cache keys: SHA256(text + language + assist_text + word2ph)
Storage: CPU tensors (to save GPU memory), transferred to GPU on retrieval.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import torch


class BertFeatureCache:
    """LRU cache for BERT features. Thread-safe for server use."""

    def __init__(self, max_size: int = 10000, device: str = "cuda"):
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._max_size = max_size
        self._device = device
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(
        text: str,
        language: str,
        assist_text: str | None = None,
        word2ph: list[int] | None = None,
    ) -> str:
        word2ph_key = ",".join(str(value) for value in word2ph) if word2ph is not None else ""
        raw = f"{text}|{language}|{assist_text or ''}|{word2ph_key}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        text: str,
        language: str,
        assist_text: str | None = None,
        word2ph: list[int] | None = None,
    ) -> torch.Tensor | None:
        """Retrieve cached BERT features, or None on cache miss."""
        key = self._make_key(text, language, assist_text, word2ph)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                # Transfer from CPU storage to target device
                return self._cache[key].to(self._device)
            self._misses += 1
            return None

    def put(
        self,
        text: str,
        language: str,
        feature: torch.Tensor,
        assist_text: str | None = None,
        word2ph: list[int] | None = None,
    ) -> None:
        """Store BERT features in cache (stored on CPU to save GPU memory)."""
        key = self._make_key(text, language, assist_text, word2ph)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Evict oldest
            self._cache[key] = feature.detach().cpu()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "enabled": True,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1%}",
            }
