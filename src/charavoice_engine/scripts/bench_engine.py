"""Standalone engine benchmark: measure S0-4 and decoder at controlled batch sizes.

Loads a single model and directly calls run_stages_0_4() and decode_chunk_batch()
with fixed text inputs and exact batch sizes. No server, no batcher, no HTTP.

Usage:
    uv run python -m charavoice_engine.scripts.bench_engine --dir model_assets
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for _name in ("numba", "matplotlib", "PIL"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Fixed test texts (JP) — short / medium / long
TEXTS = {
    "short": "こんにちは",  # ~10 phones
    "medium": "今日は天気がとても良くて、散歩に出かけたくなりますね。気持ちのいい一日です。",  # ~50 chars
    "long": "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",  # ~100 chars
}

S04_BATCH_SIZES = [1, 2, 4, 8]
DEC_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
S04_WARMUP = 5
S04_ITERS = 15
DEC_WARMUP = 10
DEC_ITERS = 40


def main() -> None:
    parser = argparse.ArgumentParser(description="Engine batch-size benchmark")
    parser.add_argument("--dir", "-d", type=str, default="model_assets", help="Model directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--trt-cache", type=str, default="./trt_cache")
    args = parser.parse_args()

    # --- 1. Initialize ---
    from charavoice_engine.config import BertBackend, FastInferenceConfig, TensorRtConfig
    from charavoice_engine.inference.engine_base import InferenceRequest
    from charavoice_engine.inference.onnx_session import preload_tensorrt_libs
    from charavoice_engine.inference.overlap_save import build_decoder_chunks_numpy
    from charavoice_engine.inference.streaming_engine import StreamingEngine

    preload_tensorrt_libs()

    # BERT preload
    from concurrent.futures import ThreadPoolExecutor

    trt_config = TensorRtConfig(fp16=args.fp16, engine_cache_path=args.trt_cache)

    def _preload_bert() -> None:
        from charavoice_engine.constants import Languages
        from charavoice_engine.nlp.fast_bert import preload_batch_bert_session

        providers: tuple[str | tuple[str, dict], ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
        for lang in Languages:
            preload_batch_bert_session(lang, providers, trt_config)

    bert_exec = ThreadPoolExecutor(1)
    bert_fut = bert_exec.submit(_preload_bert)
    logger.info("BERT preload started")

    # pyopenjtalk
    from charavoice_engine.nlp.japanese import pyopenjtalk_worker as pyopenjtalk

    pyopenjtalk.initialize_worker()

    # G2P pool
    import atexit

    from charavoice_engine.nlp.japanese.g2p_pool import init_pool, shutdown_pool

    init_pool(n_workers=4)
    atexit.register(shutdown_pool)

    # Scan and load first model
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from charavoice_engine.scripts.run_server import scan_models

    model_root = Path(args.dir)
    found = scan_models(model_root)
    if not found:
        logger.error("No models found")
        sys.exit(1)

    model_info = found[0]
    logger.info(f"Using model: {model_info['name']}")

    config = FastInferenceConfig(bert_backend=BertBackend.ONNX_CUDA, device=args.device, trt=trt_config)
    engine = StreamingEngine(config=config)

    bert_fut.result()
    logger.info("BERT preload done")

    engine.load(
        model_path=model_info["model_path"],
        config_path=model_info["config_path"],
        style_vec_path=model_info["style_vec_path"],
        device=args.device,
        preload_bert=False,
    )
    logger.info("Engine loaded")

    style_vec = np.load(model_info["style_vec_path"])[0]  # first style
    from charavoice_engine.constants import Languages

    # --- 2. Build requests ---
    def make_requests(text: str, batch_size: int) -> list[InferenceRequest]:
        return [
            InferenceRequest(text=text, language=Languages.JP, speaker_id=0, style_vec=style_vec)
            for _ in range(batch_size)
        ]

    # Warmup G2P + BERT + TRT
    logger.info("Warming up...")
    for _ in range(3):
        engine.run_stages_0_4(make_requests(TEXTS["medium"], 1))
    torch.cuda.synchronize()
    logger.info("Warmup done")

    # --- 3. S0-4 Benchmark ---
    print("\n" + "=" * 70)
    print(" S0-4 Benchmark (run_stages_0_4)")
    print("=" * 70)

    for text_label, text in TEXTS.items():
        print(f"\n--- Text: {text_label} ({len(text)} chars) ---")
        print(f"  {'B':>3}  {'total':>8}  {'g2p':>7}  {'bert':>7}  {'enc+dp':>7}  {'lenreg':>7}  {'flow':>7}")

        for B in S04_BATCH_SIZES:
            reqs = make_requests(text, B)
            times_all: list[tuple[float, ...]] = []

            for i in range(S04_WARMUP + S04_ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _, _, s04_times = engine.run_stages_0_4(reqs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                total_ms = (t1 - t0) * 1000
                if i >= S04_WARMUP:
                    times_all.append((total_ms, *s04_times))

            arr = np.array(times_all)
            means = arr.mean(axis=0)
            print(
                f"  {B:>3}  {means[0]:>7.1f}  {means[1]:>7.1f}  {means[2]:>7.1f}  "
                f"{means[3]:>7.1f}  {means[4]:>7.1f}  {means[5]:>7.1f}"
            )

    # --- 4. Decoder Benchmark ---
    print("\n" + "=" * 70)
    print(" Decoder Benchmark (decode_chunk_batch)")
    print("=" * 70)

    # Generate real chunks from medium text at B=8
    reqs_for_chunks = make_requests(TEXTS["medium"], 8)
    z_list, g_np, _ = engine.run_stages_0_4(reqs_for_chunks)
    torch.cuda.synchronize()

    oc = engine.overlap_config
    all_chunks: list[np.ndarray] = []
    all_g: list[np.ndarray] = []
    for req_idx, z_all in enumerate(z_list):
        z_np = z_all[0].cpu().numpy().astype(np.float32)
        chunks = build_decoder_chunks_numpy(z_np, oc)
        for chunk_np in chunks:
            all_chunks.append(chunk_np)
            all_g.append(g_np[req_idx])

    logger.info(f"Generated {len(all_chunks)} decoder chunks from 8 requests")

    print(f"\n  {'B':>3}  {'mean':>8}  {'P50':>8}  {'P99':>8}  {'per_chunk':>10}")

    for B in DEC_BATCH_SIZES:
        # Pick first B chunks (cycle if needed)
        z_batch = np.stack([all_chunks[i % len(all_chunks)] for i in range(B)])
        g_batch = np.stack([all_g[i % len(all_g)] for i in range(B)])

        times_dec: list[float] = []
        for i in range(DEC_WARMUP + DEC_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.decode_chunk_batch(z_batch, g_batch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= DEC_WARMUP:
                times_dec.append((t1 - t0) * 1000)

        arr = np.array(times_dec)
        print(
            f"  {B:>3}  {arr.mean():>7.1f}  {np.percentile(arr, 50):>7.1f}  {np.percentile(arr, 99):>7.1f}  {arr.mean() / B:>9.2f}"
        )

    print()


if __name__ == "__main__":
    main()
