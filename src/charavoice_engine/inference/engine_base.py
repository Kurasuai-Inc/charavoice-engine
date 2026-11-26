"""Abstract base class for inference engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from charavoice_engine.constants import Languages

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from charavoice_engine.inference.text_processing import G2pResult, InputChunk


@dataclass
class InferenceRequest:
    """A single TTS inference request with all required parameters."""

    text: str
    language: Languages
    speaker_id: int
    style_vec: NDArray[Any]
    noise_scale: float = 0.6
    length_scale: float = 1.0
    assist_text: str | None = None
    assist_text_weight: float = 0.7
    skip_start: bool = False
    skip_end: bool = False
    watermark_core16: int | None = None
    watermark_extension: str | None = None
    seq_len_hint: int = 0  # Estimated phone seq_len (post-intersperse). 0 = unknown.
    g2p_result: G2pResult | None = None  # Pre-computed G2P (skip engine G2P if set)
    chunks: Sequence[InputChunk] | None = None  # Mixed-language chunk input; bypasses request-level language.


@dataclass
class InferenceResult:
    """Result of a TTS inference."""

    audio: NDArray[Any]  # float32 waveform
    sample_rate: int = 44100


class InferenceEngine(ABC):
    """Abstract inference engine interface."""

    @abstractmethod
    def load(
        self,
        model_path: str,
        config_path: str,
        style_vec_path: str,
        device: str = "cuda",
    ) -> None:
        """Load model, config, and style vectors."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        ...

    def check_and_split_text(
        self,
        text: str,
        language: Languages,
    ) -> list[str]:
        """Check if text exceeds max_seq_len and split if needed.

        Default implementation returns [text] (no splitting).
        """
        return [text]

    def check_and_split_text_with_len(
        self,
        text: str,
        language: Languages,
    ) -> list[tuple[str, int]]:
        """Check if text exceeds max_seq_len and split if needed.

        Returns:
            List of (chunk_text, seq_len_estimate) tuples.
            Default implementation returns [(text, 0)] (no splitting, unknown length).
        """
        return [(text, 0)]

    @property
    def text_config(self) -> Any:
        """Return text processing config if available, None otherwise."""
        return None

    @property
    def vram_breakdown(self) -> Any:
        """Return VRAM breakdown if available, None otherwise."""
        return None

    @property
    def sample_rate(self) -> int:
        """Audio sample rate. Default 44100."""
        return 44100

    @abstractmethod
    def run_stages_0_4(
        self,
        requests: list[InferenceRequest],
    ) -> tuple[list[torch.Tensor], NDArray[Any], tuple[float, float, float, float, float]]:
        """Run Stages 0-4 (text processing → flow).

        Returns:
            (z_list, g_np, (g2p_ms, bert_ms, enc_dp_ms, lenreg_ms, flow_ms))
        """
        ...

    @abstractmethod
    def decode_chunk_batch(
        self,
        z_batch: NDArray[Any],
        g_batch: NDArray[Any],
    ) -> NDArray[Any]:
        """Decode a batch of chunks.

        Args:
            z_batch: [B, 192, input_frames] latent chunks.
            g_batch: [B, 512, 1] speaker embeddings.

        Returns:
            wav_batch: [B, 1, output_samples] raw decoder output.
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        ...
