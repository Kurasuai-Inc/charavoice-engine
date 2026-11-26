from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from charavoice_engine.config import TensorRtConfig
from charavoice_engine.inference.onnx_session import preload_tensorrt_libs
from charavoice_engine.watermark.common import MESSAGE_BITS, BatchedWatermarkBackend

logger = logging.getLogger(__name__)

# AudioSeal models (audioseal_wm_16bits) are trained at 16 kHz.  The TTS
# decoder outputs 44.1 kHz audio, but we intentionally pass sample_rate=_AUDIOSEAL_SR
# so that AudioSeal skips its internal resampler and applies the watermark
# directly on native-rate chunks.  Embedding and detection are symmetric —
# both use the same declared rate — so the watermark round-trips correctly.
_AUDIOSEAL_SR = 16000


@dataclass(frozen=True)
class AudioSealTRTConfig:
    """TensorRT backend configuration for fixed AudioSeal chunk inference."""

    alpha: float = 1.0
    chunk_samples: int = 16384
    batch_min: int = 1
    batch_opt: int = 8
    batch_max: int = 16
    cuda_graph_enable: bool = False
    onnx_filename: str = "audioseal_chunk_batched.onnx"


class AudioSealTRTWatermarker(BatchedWatermarkBackend):
    """Fixed-chunk AudioSeal backend backed by ORT TensorRT EP."""

    backend_name = "audioseal_trt"

    def __init__(
        self,
        trt_config: TensorRtConfig,
        config: AudioSealTRTConfig | None = None,
    ) -> None:
        self._trt_config = trt_config
        self._config = config or AudioSealTRTConfig()
        self.chunk_samples = self._config.chunk_samples
        self.max_batch_size = self._config.batch_max
        self.alpha = self._config.alpha
        self._session = None
        self._run_lock = threading.Lock()

        if not preload_tensorrt_libs():
            raise RuntimeError("TensorRT libraries could not be preloaded for audioseal_trt")

        cache_suffix = f"audioseal_chunk_b{self._config.batch_max}_opt{self._config.batch_opt}"
        if self._config.cuda_graph_enable:
            cache_suffix += "_graph"
        self._cache_dir = Path(self._trt_config.engine_cache_path) / cache_suffix
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._onnx_path = self._cache_dir / self._config.onnx_filename

        self._ensure_onnx_export()
        self._session = self._create_session()
        self._warmup()

    def apply_batch(
        self,
        audio_batch: NDArray[np.float32],
        message_bits: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """Apply watermark to a fixed-length chunk batch shaped [B, T]."""
        if audio_batch.ndim != 2:
            raise ValueError(f"audio_batch must have shape [B, T], got {audio_batch.shape}")
        if audio_batch.shape[1] != self.chunk_samples:
            raise ValueError(f"audio_batch chunk length must be {self.chunk_samples}, got {audio_batch.shape[1]}")
        if audio_batch.shape[0] > self.max_batch_size:
            raise ValueError(
                f"audio_batch batch size {audio_batch.shape[0]} exceeds TRT profile max {self.max_batch_size}"
            )
        if message_bits.shape != (audio_batch.shape[0], MESSAGE_BITS):
            raise ValueError(
                f"message_bits must have shape [{audio_batch.shape[0]}, {MESSAGE_BITS}], got {message_bits.shape}"
            )

        assert self._session is not None
        ort_inputs = {
            "x": audio_batch[:, None, :].astype(np.float32, copy=False),
            "message": message_bits.astype(np.int64, copy=False),
        }
        with self._run_lock:
            outputs = self._session.run(None, ort_inputs)
        if not outputs:
            raise RuntimeError("AudioSeal TRT session returned no outputs")
        watermark = outputs[0]
        watermarked = audio_batch + self.alpha * watermark[:, 0, :]
        return np.clip(watermarked, -1.0, 1.0).astype(np.float32, copy=False)

    def _ensure_onnx_export(self) -> None:
        if self._onnx_path.exists():
            logger.info("Using cached AudioSeal ONNX at %s", self._onnx_path)
            return

        try:
            os.environ.setdefault("NO_TORCH_COMPILE", "1")
            import torch
            from audioseal import AudioSeal  # pyright: ignore[reportMissingImports, reportPrivateImportUsage]
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("audioseal_trt requires torch and audioseal to export the ONNX graph") from e

        class _Wrapper(torch.nn.Module):
            def __init__(self, generator):
                super().__init__()
                self.generator = generator

            def forward(self, x, message):
                return self.generator.get_watermark(x, sample_rate=_AUDIOSEAL_SR, message=message)

        logger.info("Exporting fixed-chunk AudioSeal ONNX to %s", self._onnx_path)
        generator = AudioSeal.load_generator("audioseal_wm_16bits")
        generator.eval().cuda()
        model = _Wrapper(generator).eval().cuda()
        example_x = torch.randn(1, 1, self.chunk_samples, device="cuda")
        example_message = torch.zeros(1, MESSAGE_BITS, dtype=torch.int64, device="cuda")

        torch.onnx.export(
            model,
            (example_x, example_message),
            self._onnx_path,
            input_names=["x", "message"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "batch"},
                "message": {0: "batch"},
                "y": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    def _create_session(self) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("audioseal_trt requires onnxruntime-gpu") from e

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cache_path = str(self._cache_dir)
        trt_opts: dict[str, Any] = {
            "trt_fp16_enable": self._trt_config.fp16,
            "trt_max_workspace_size": self._trt_config.max_workspace_size,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_path,
            "trt_builder_optimization_level": 3,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache_path,
            "trt_profile_min_shapes": f"x:{self._config.batch_min}x1x{self.chunk_samples},message:{self._config.batch_min}x{MESSAGE_BITS}",
            "trt_profile_opt_shapes": f"x:{self._config.batch_opt}x1x{self.chunk_samples},message:{self._config.batch_opt}x{MESSAGE_BITS}",
            "trt_profile_max_shapes": f"x:{self._config.batch_max}x1x{self.chunk_samples},message:{self._config.batch_max}x{MESSAGE_BITS}",
        }
        providers: list[str | tuple[str, dict[str, Any]]] = [
            ("TensorrtExecutionProvider", trt_opts),
            "CUDAExecutionProvider",
        ]

        try:
            session = ort.InferenceSession(str(self._onnx_path), so, providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to create AudioSeal TRT session: {e}") from e

        session_providers = session.get_providers()
        if not session_providers or session_providers[0] != "TensorrtExecutionProvider":
            raise RuntimeError(f"AudioSeal TRT session did not start with TensorRT EP: {session_providers}")
        return session

    def _warmup(self) -> None:
        assert self._session is not None
        try:
            import torch
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("audioseal_trt warmup requires torch") from e

        for batch_size in (self._config.batch_min, self._config.batch_opt):
            warmup_x = np.zeros((batch_size, self.chunk_samples), dtype=np.float32)
            warmup_message = np.zeros((batch_size, MESSAGE_BITS), dtype=np.int64)
            self.apply_batch(warmup_x, warmup_message)
            torch.cuda.synchronize()

        logger.info(
            "AudioSeal TRT backend ready (chunk=%d, batch=%d..%d, cuda_graph=%s, cache=%s)",
            self.chunk_samples,
            self._config.batch_min,
            self._config.batch_max,
            self._config.cuda_graph_enable,
            self._cache_dir,
        )
