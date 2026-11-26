#!/usr/bin/env python3
"""CLI entrypoint for CHARAVOICE Engine.

Usage:
    uv run python -m charavoice_engine.scripts.run_server \
        --dir ../model_assets --port 5000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from charavoice_engine.config import InferenceProfile
from charavoice_engine.constants import Languages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress uvicorn's noisy access/error logs (we handle our own logging)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class ModelScanError(RuntimeError):
    """Raised when model directory scanning cannot satisfy the requested model set."""


def _first_env(*names: str) -> str | None:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value
    return None


def _parse_csv(value: str | None) -> list[str] | None:
    """Parse a comma-separated environment/CLI value."""
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_model_keys(value: str | None) -> list[str] | None:
    """Parse model key filters from CLI/env."""
    keys = _parse_csv(value)
    if keys is None:
        return None
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate model keys are not allowed")
    return keys


def parse_bert_languages(value: str | None) -> list[Languages]:
    """Parse BERT preload languages from CLI/env."""
    names = _parse_csv(value) or ["JP", "EN"]
    result: list[Languages] = []
    seen: set[Languages] = set()
    for raw_name in names:
        name = raw_name.upper()
        try:
            lang = Languages[name]
        except KeyError as e:
            allowed = ", ".join(lang.name for lang in (Languages.JP, Languages.EN))
            raise ValueError(f"Unsupported BERT language {raw_name!r}; allowed: {allowed}") from e
        if lang not in (Languages.JP, Languages.EN):
            raise ValueError(f"Unsupported BERT language {raw_name!r}; allowed: JP, EN")
        if lang in seen:
            raise ValueError(f"Duplicate BERT language is not allowed: {lang.name}")
        seen.add(lang)
        result.append(lang)
    return result


def _find_pytorch_model(model_dir: Path) -> Path | None:
    """Find a PyTorch model in a model directory (.safetensors > .pth)."""
    ignored_prefixes = ("D_", "DUR_", "WD_")
    for ext in (".safetensors", ".pth"):
        candidates = sorted(
            (file for file in model_dir.glob(f"*{ext}") if not file.name.startswith(ignored_prefixes)),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def scan_models(model_root: Path, requested_models: list[str] | None = None) -> list[dict]:
    """Scan model directory for loadable PyTorch models.

    Args:
        model_root: Root directory containing model subdirectories.
        requested_models: Optional model directory names to load. When provided,
            every requested model must exist and be valid. Other directories are ignored.

    Returns:
        List of dicts with: name, config_path, model_path, style_vec_path, hps
    """
    from charavoice_engine.config import HyperParameters

    models = []
    all_model_dirs = sorted([d for d in model_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if requested_models is None:
        model_dirs = all_model_dirs
    else:
        by_name = {d.name: d for d in all_model_dirs}
        missing = [name for name in requested_models if name not in by_name]
        if missing:
            available = ", ".join(sorted(by_name)) or "(none)"
            raise ModelScanError(f"Requested model(s) missing: {', '.join(missing)}; available: {available}")
        model_dirs = [by_name[name] for name in requested_models]

    for model_dir in model_dirs:
        config_path = model_dir / "config.json"
        style_vec_path = model_dir / "style_vectors.npy"

        if not config_path.exists():
            raise ModelScanError(f"No config.json in {model_dir}")
        if not style_vec_path.exists():
            raise ModelScanError(f"No style_vectors.npy in {model_dir}")

        model_path = _find_pytorch_model(model_dir)
        if model_path is None:
            raise ModelScanError(f"No generator PyTorch model (.safetensors/.pth) in {model_dir}")

        # Validate config loads
        try:
            hps = HyperParameters.load_from_json(str(config_path))
        except Exception as e:
            raise ModelScanError(f"Failed to load config {config_path}: {e}") from e

        models.append(
            {
                "name": model_dir.name,
                "config_path": str(config_path),
                "model_path": str(model_path),
                "style_vec_path": str(style_vec_path),
                "hps": hps,
            }
        )
        logger.info(f"Found model: {model_dir.name} -> {model_path.name}")

    if requested_models is not None and len(models) != len(requested_models):
        loaded = {model["name"] for model in models}
        missing = [name for name in requested_models if name not in loaded]
        raise ModelScanError(f"Requested model(s) were not loaded: {', '.join(missing)}")

    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CHARAVOICE Engine TTS server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir", "-d", type=str, default="model_assets", help="Model directory")
    parser.add_argument(
        "--models",
        type=str,
        default=_first_env("CHARAVOICE_MODEL_KEYS", "MODEL_KEYS"),
        help="Comma-separated model directory names to load. Defaults to CHARAVOICE_MODEL_KEYS or MODEL_KEYS.",
    )
    parser.add_argument(
        "--bert-langs",
        type=str,
        default=_first_env("CHARAVOICE_LANGS"),
        help="Comma-separated BERT languages to preload. Defaults to CHARAVOICE_LANGS or JP,EN.",
    )
    parser.add_argument("--worker-id", type=str, default=os.environ.get("WORKER_ID", ""))
    parser.add_argument("--worker-site", type=str, default=os.environ.get("WORKER_SITE", ""))
    parser.add_argument(
        "--worker-capacity",
        type=float,
        default=os.environ.get("WORKER_CAPACITY_AUDIO_SEC_PER_SEC", "0") or "0",
        help="Worker capacity in audio-sec/sec for dispatcher status.",
    )
    parser.add_argument(
        "--bert-backend",
        type=str,
        default="onnx_cuda",
        choices=["onnx_cuda"],
        help="BERT inference backend",
    )
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size")
    parser.add_argument(
        "--inference-profile",
        choices=[p.value for p in InferenceProfile],
        default="default",
        help="throughput128: same-weight 128-frame decoder, fixed-zero batching, GPU Flow I/O and 200ms Ogg pages",
    )
    parser.add_argument("--batch-timeout", type=float, default=50.0, help="Fixed batch timeout ms")
    parser.add_argument("--batch-timeout-min", type=float, default=2.0, help="Adaptive: minimum batch timeout (ms)")
    parser.add_argument("--batch-timeout-max", type=float, default=100.0, help="Adaptive: maximum batch timeout (ms)")
    parser.add_argument(
        "--no-adaptive-timeout", action="store_true", help="Disable adaptive batch timeout (use fixed --batch-timeout)"
    )
    parser.add_argument("--port", type=int, default=5000, help="Listen port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen host")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--trt-cache", type=str, default="./trt_cache", help="TRT engine cache dir")
    parser.add_argument("--trt-workspace", type=int, default=4, help="TRT max workspace size (GB)")
    parser.add_argument("--max-seq-len", type=int, default=768, help="TRT profile max sequence length (~200字)")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TensorRT FP16 inference (default: enabled)",
    )
    parser.add_argument("--queue-deadline", type=float, default=5.0, help="Queue dwell timeout in seconds (0=disabled)")
    parser.add_argument("--max-queue-size", type=int, default=32, help="Maximum pending requests in batcher queue")
    parser.add_argument(
        "--stream-buffer-chunks",
        type=int,
        default=10,
        help="Number of PCM decoder chunks to coalesce before each streaming HTTP write.",
    )
    parser.add_argument("--no-worker", action="store_true", help="Skip pyopenjtalk worker, use direct import")
    parser.add_argument("--g2p-workers", type=int, default=4, help="G2P process pool workers")
    parser.add_argument(
        "--watermark",
        type=str,
        default="audioseal",
        choices=["none", "audioseal", "audioseal_trt"],
        help="Watermark backend (default: audioseal). Use 'none' to disable.",
    )
    parser.add_argument(
        "--watermark-core16",
        type=int,
        default=None,
        help="Optional default 16-bit core watermark word (0-65535).",
    )
    parser.add_argument(
        "--watermark-extension",
        type=str,
        default=None,
        help="Optional default extension payload as hex string. Excess bytes are dropped to fit each utterance.",
    )
    parser.add_argument("--watermark-alpha", type=float, default=1.0, help="Watermark strength (AudioSeal alpha).")
    parser.add_argument(
        "--watermark-batch-max",
        type=int,
        default=16,
        help="Maximum microbatch size for the watermark worker.",
    )
    parser.add_argument(
        "--watermark-timeout-ms",
        type=float,
        default=0.5,
        help="Additional wait time to form a watermark microbatch.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging (includes per-batch profiling)")
    args = parser.parse_args()

    # Fail before loading models (minutes) when the license/source information is missing.
    from charavoice_engine.server import legal

    try:
        legal.validate_startup()
    except legal.LegalConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    inference_profile = InferenceProfile(args.inference_profile)
    if inference_profile == InferenceProfile.THROUGHPUT128:
        args.batch_timeout = 0.0
        args.no_adaptive_timeout = True

    from charavoice_engine.watermark.protocol import normalize_extension_hex

    try:
        requested_models = parse_model_keys(args.models)
        bert_languages = parse_bert_languages(args.bert_langs)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.watermark_core16 is not None and not (0 <= args.watermark_core16 <= 0xFFFF):
        logger.error("--watermark-core16 must be between 0 and 65535")
        sys.exit(1)
    if args.g2p_workers <= 0:
        logger.error("--g2p-workers must be >= 1")
        sys.exit(1)
    if args.stream_buffer_chunks <= 0:
        logger.error("--stream-buffer-chunks must be >= 1")
        sys.exit(1)
    try:
        args.watermark_extension = normalize_extension_hex(args.watermark_extension)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    if args.watermark_batch_max <= 0:
        logger.error("--watermark-batch-max must be >= 1")
        sys.exit(1)
    if args.watermark_timeout_ms < 0:
        logger.error("--watermark-timeout-ms must be >= 0")
        sys.exit(1)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Keep noisy third-party loggers quiet
        for _name in ("numba", "matplotlib", "PIL"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    # 1. Start BERT TRT preload ASAP (longest single operation, ~14s cached / ~80s build)
    import time
    from concurrent.futures import Future, ThreadPoolExecutor

    from charavoice_engine.config import TensorRtConfig
    from charavoice_engine.inference.onnx_session import preload_tensorrt_libs

    trt_config = TensorRtConfig(
        fp16=args.fp16,
        engine_cache_path=args.trt_cache,
        max_workspace_size=args.trt_workspace << 30,
        max_seq_len=args.max_seq_len,
    )
    preload_tensorrt_libs()

    def _preload_bert_background() -> None:
        from charavoice_engine.nlp.fast_bert import preload_batch_bert_session

        bert_providers: tuple[str | tuple[str, dict], ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
        for lang in bert_languages:
            preload_batch_bert_session(lang, bert_providers, trt_config)

    bert_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bert-preload")
    bert_future: Future[None] = bert_executor.submit(_preload_bert_background)
    bert_start = time.monotonic()
    logger.info("BERT TRT preload started in background thread")
    if inference_profile == InferenceProfile.THROUGHPUT128:
        # TensorRT engine construction and G2P worker spawning must not race the
        # BERT constructor. This ordering also matches the measured profile.
        bert_future.result()

    # 2. Initialize pyopenjtalk worker + user dict (runs concurrently with BERT preload)
    if not args.no_worker:
        logger.info("Initializing pyopenjtalk worker...")
        from charavoice_engine.nlp.japanese import pyopenjtalk_worker as pyopenjtalk

        pyopenjtalk.initialize_worker()
        logger.info("pyopenjtalk worker started (TCP socket mode)")
    else:
        logger.info("pyopenjtalk worker disabled, using direct import")

    logger.info("pyopenjtalk ready")

    # 2b. Initialize G2P process pool (parallel pyopenjtalk via spawn)
    import atexit

    from charavoice_engine.nlp.japanese.g2p_pool import init_pool, shutdown_pool

    logger.info(f"Starting G2P process pool ({args.g2p_workers} workers)...")
    init_pool(n_workers=args.g2p_workers)
    atexit.register(shutdown_pool)

    # 3. Scan model directory
    model_root = Path(args.dir)
    if not model_root.exists():
        logger.error(f"Model directory not found: {model_root}")
        sys.exit(1)

    try:
        found_models = scan_models(model_root, requested_models)
    except ModelScanError as e:
        logger.error(str(e))
        sys.exit(1)
    if not found_models:
        logger.error(f"No valid models found in {model_root}")
        sys.exit(1)

    # 4. Build config and create engines
    import numpy as np

    from charavoice_engine.config import (
        BertBackend,
        FastInferenceConfig,
    )
    from charavoice_engine.inference.streaming_engine import StreamingEngine
    from charavoice_engine.server.app import LoadedModel, set_models, set_streaming_config, set_worker_metadata
    from charavoice_engine.server.batcher import DynamicBatcher

    bert_backend = BertBackend.ONNX_CUDA

    models_dict: dict[str, LoadedModel] = {}
    models_list: list[LoadedModel] = []

    watermark_backend = None

    # 4a. Optional watermark backend
    if args.watermark == "audioseal":
        from charavoice_engine.watermark.audioseal import AudioSealWatermarker, WatermarkConfig

        wm_cfg = WatermarkConfig(alpha=args.watermark_alpha, device=args.device)
        try:
            watermark_backend = AudioSealWatermarker(wm_cfg)
            watermark_backend.warmup()
            logger.info(
                "AudioSeal eager watermark backend initialized (alpha=%.2f, default_core16=%s, extension_bytes=%d)",
                args.watermark_alpha,
                "random" if args.watermark_core16 is None else "configured",
                len(bytes.fromhex(args.watermark_extension)) if args.watermark_extension is not None else 0,
            )
        except Exception as e:  # pragma: no cover - optional dependency
            logger.error("Failed to initialize AudioSeal eager backend: %s", e)
            sys.exit(1)
    elif args.watermark == "audioseal_trt":
        if args.device != "cuda":
            logger.error("audioseal_trt requires --device cuda")
            sys.exit(1)
        from charavoice_engine.watermark.audioseal_trt import AudioSealTRTConfig, AudioSealTRTWatermarker

        try:
            wm_batch_max = max(1, args.watermark_batch_max)
            wm_batch_opt = min(32, wm_batch_max)
            watermark_backend = AudioSealTRTWatermarker(
                trt_config=trt_config,
                config=AudioSealTRTConfig(
                    alpha=args.watermark_alpha,
                    batch_opt=wm_batch_opt,
                    batch_max=wm_batch_max,
                ),
            )
            logger.info(
                "AudioSeal TRT watermark backend initialized (alpha=%.2f, batch_max=%d, default_core16=%s, extension_bytes=%d)",
                args.watermark_alpha,
                wm_batch_max,
                "random" if args.watermark_core16 is None else "configured",
                len(bytes.fromhex(args.watermark_extension)) if args.watermark_extension is not None else 0,
            )
        except Exception as e:
            logger.error("Failed to initialize AudioSeal TRT backend: %s", e)
            sys.exit(1)

    for model_info in found_models:
        name = model_info["name"]
        hps = model_info["hps"]

        logger.info(f"Loading model: {name} (streaming)...")

        # Create engine config
        config = FastInferenceConfig(
            profile=inference_profile,
            bert_backend=bert_backend,
            device=args.device,
            max_batch_size=args.max_batch_size,
            batch_timeout_ms=args.batch_timeout,
            adaptive_batch_timeout=not args.no_adaptive_timeout,
            batch_timeout_min_ms=args.batch_timeout_min,
            batch_timeout_max_ms=args.batch_timeout_max,
            trt=trt_config,
        )

        # Create engine
        engine = StreamingEngine(config=config)

        # Load model (skip BERT preload — handled by background thread)
        engine.load(
            model_path=model_info["model_path"],
            config_path=model_info["config_path"],
            style_vec_path=model_info["style_vec_path"],
            device=args.device,
            preload_bert=False,
        )

        # Extract metadata from HyperParameters
        spk2id: dict[str, int] = hps.data.spk2id
        id2spk: dict[int, str] = {v: k for k, v in spk2id.items()}
        style2id: dict[str, int] = (
            hps.data.style2id if hasattr(hps.data, "style2id") else {str(i): i for i in range(hps.data.num_styles)}
        )

        # Create batcher
        batcher = DynamicBatcher(
            engine=engine,
            config=config,
            max_batch_size=args.max_batch_size,
            batch_timeout_ms=args.batch_timeout,
            max_queue_size=args.max_queue_size,
            queue_deadline=args.queue_deadline,
            watermark_backend=watermark_backend,
            watermark_batch_max=args.watermark_batch_max,
            watermark_timeout_ms=args.watermark_timeout_ms,
            watermark_default_core16=args.watermark_core16,
            watermark_default_extension=args.watermark_extension,
        )

        loaded = LoadedModel(
            name=name,
            engine=engine,
            batcher=batcher,
            style_vectors=np.load(model_info["style_vec_path"]),
            spk2id=spk2id,
            id2spk=id2spk,
            style2id=style2id,
            config_path=model_info["config_path"],
            model_path=model_info["model_path"],
            device=args.device,
        )

        models_dict[name] = loaded
        models_list.append(loaded)
        logger.info(f"Model loaded: {name} (speakers={len(spk2id)}, styles={len(style2id)})")

    # 4b. Wait for BERT preload to finish + warmup
    bert_future.result()  # raises if background thread failed
    bert_elapsed = time.monotonic() - bert_start
    logger.info(f"BERT TRT preload completed in {bert_elapsed:.1f}s")
    bert_executor.shutdown(wait=False)

    from charavoice_engine.nlp.fast_bert import warmup_batch_bert_sessions

    warmup_batch_bert_sessions(trt_config)

    # 5. Wire models into FastAPI app
    set_models(models_dict, models_list)
    set_streaming_config(
        stream_buffer_chunks=args.stream_buffer_chunks,
        ogg_page_duration_ms=200.0 if inference_profile == InferenceProfile.THROUGHPUT128 else None,
    )
    set_worker_metadata(
        {
            "worker_id": args.worker_id,
            "worker_site": args.worker_site,
            "capacity_audio_sec_per_sec": args.worker_capacity,
            "requested_model_keys": requested_models or [],
            "bert_languages": [lang.name for lang in bert_languages],
        }
    )

    # 6. Start server
    import uvicorn

    from charavoice_engine.server.app import app

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"API docs: http://127.0.0.1:{args.port}/docs")
    logger.info(f"Loaded {len(models_list)} model(s): {[m.name for m in models_list]}")
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
