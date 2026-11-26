"""ONNX Runtime session configuration utilities.

TensorRT library preloading for the streaming inference engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_trt_libs_preloaded = False


def run_cuda_bound(
    session: Any, inputs: dict[str, Any], output_shapes: list[tuple[int, ...]], device: str
) -> list[Any]:
    """Keep tensors alive through explicit PyTorch/TensorRT synchronization."""
    import numpy as np
    import torch

    held = {
        name: (value if isinstance(value, torch.Tensor) else torch.from_numpy(value)).to(device).contiguous()
        for name, value in inputs.items()
    }
    cuda_device = torch.device(device)
    device_id = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
    binding = session.io_binding()
    for name, value in held.items():
        if value.dtype != torch.float32:
            raise ValueError(f"Flow I/O binding requires float32, got {value.dtype} for {name}")
        binding.bind_input(name, "cuda", device_id, np.float32, tuple(value.shape), value.data_ptr())
    outputs = []
    for info, shape in zip(session.get_outputs(), output_shapes, strict=True):
        value = torch.empty(shape, device=device, dtype=torch.float32)
        binding.bind_output(info.name, "cuda", device_id, np.float32, shape, value.data_ptr())
        outputs.append(value)
    torch.cuda.synchronize(cuda_device)
    session.run_with_iobinding(binding)
    torch.cuda.synchronize(cuda_device)
    return outputs


def preload_tensorrt_libs() -> bool:
    """TRT EP 用に libnvinfer.so.10 等をプリロード。

    pip の tensorrt-cu12 は libnvinfer を site-packages/tensorrt_libs/ に配置するが、
    LD_LIBRARY_PATH に含まれない。tensorrt_libs パッケージの import で ctypes.CDLL が
    実行され、ORT の dlopen が成功するようになる。
    """
    global _trt_libs_preloaded
    if _trt_libs_preloaded:
        return True
    try:
        import tensorrt_libs  # pyright: ignore[reportMissingImports]

        _trt_libs_preloaded = True
        logger.info("TensorRT libs preloaded from %s", Path(tensorrt_libs.__path__[0]))
        return True
    except ImportError:
        logger.warning(
            "tensorrt_libs not found. Install: uv add tensorrt-cu12, or set LD_LIBRARY_PATH to tensorrt_libs directory."
        )
        return False
