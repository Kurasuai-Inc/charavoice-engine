"""Shared test fixtures for charavoice_engine tests.

All fixtures here are CPU-only and do not require GPU or model files.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_audio_float32() -> np.ndarray:
    """Generate a 1-second sine wave at 440Hz, float32, 44100 Hz."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sample_audio_silent() -> np.ndarray:
    """Generate 0.5 seconds of silence, float32."""
    return np.zeros(22050, dtype=np.float32)
