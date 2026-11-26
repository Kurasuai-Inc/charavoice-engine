from __future__ import annotations

import time

import numpy as np
import torch

from charavoice_engine.config import TensorRtConfig
from charavoice_engine.watermark.audioseal_trt import AudioSealTRTConfig, AudioSealTRTWatermarker
from charavoice_engine.watermark.common import pack_watermark_messages

T = 16384
B_LIST = [1, 2, 4, 8, 16, 32]
MESSAGE = 0x1234
SAMPLE_RATE = 44100


def main() -> None:
    trt_config = TensorRtConfig(fp16=True, engine_cache_path="./trt_cache", max_workspace_size=2 << 30)
    backend = AudioSealTRTWatermarker(
        trt_config=trt_config,
        config=AudioSealTRTConfig(alpha=1.0, batch_opt=16, batch_max=32),
    )

    reps = 30
    warm = 5
    np.random.seed(0)

    for batch_size in B_LIST:
        audio = np.random.randn(batch_size, T).astype(np.float32)
        messages = pack_watermark_messages([MESSAGE] * batch_size)

        for _ in range(warm):
            backend.apply_batch(audio, messages)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(reps):
            backend.apply_batch(audio, messages)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / reps * 1000
        audio_sec = (T / SAMPLE_RATE) * batch_size
        throughput = audio_sec / ((t1 - t0) / reps)
        print(f"B={batch_size:>2} avg={avg_ms:6.2f} ms | throughput={throughput:7.1f}x")


if __name__ == "__main__":
    main()
