#!/usr/bin/env python3
"""HTTP benchmark for CHARAVOICE Engine.

Sends concurrent requests with random speakers, styles, and text lengths.

Two modes:
  - Concurrency mode (default): fire N requests with semaphore-based concurrency
  - Rate mode (--rps): send requests at a fixed rate with drift compensation

Usage:
    # Concurrency mode
    uv run python -m charavoice_engine.scripts.bench_server \
        --url http://localhost:5000 -c 8 -n 200

    # Rate mode (33 req/s = 30ms interval)
    uv run python -m charavoice_engine.scripts.bench_server \
        --url http://localhost:5000 --rps 33 -n 500 --streaming
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import random
import struct
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import aiohttp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Japanese kana for random text generation
KANA = (
    "あいうえおかきくけこさしすせそたちつてとなにぬねの"
    "はひふへほまみむめもやゆよらりるれろわをん"
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
    "ぁぃぅぇぉっゃゅょ"
)

ENGLISH_WORDS = [
    "hello",
    "voice",
    "model",
    "happy",
    "bright",
    "calm",
    "quick",
    "story",
    "morning",
    "evening",
    "little",
    "friend",
    "answer",
    "wonder",
    "gentle",
    "strong",
    "careful",
    "natural",
    "service",
    "message",
    "today",
    "tomorrow",
    "simple",
    "useful",
    "speech",
    "emotion",
    "speaker",
    "quality",
    "listen",
    "example",
    "sentence",
    "sound",
    "clear",
    "warm",
    "surprise",
    "fear",
    "sadness",
    "anger",
    "joy",
]


def random_japanese_text(min_len: int = 3, max_len: int = 100) -> str:
    """Generate random Japanese kana text with completely random length."""
    length = random.randint(min_len, max_len)
    return "".join(random.choices(KANA, k=length))


def random_english_text(min_len: int = 3, max_len: int = 100) -> str:
    """Generate random English text within an approximate character range."""
    target = random.randint(min_len, max_len)
    words: list[str] = []
    while len(" ".join(words)) < target:
        words.append(random.choice(ENGLISH_WORDS))
    text = " ".join(words)
    if len(text) > max_len and len(words) > 1:
        text = " ".join(words[:-1])
    return text[:max_len].rstrip(" ,") + "."


def random_text(min_len: int = 3, max_len: int = 100) -> str:
    """Generate random benchmark text for the selected language."""
    if _language == "EN":
        return random_english_text(min_len, max_len)
    return random_japanese_text(min_len, max_len)


# Global text length range (set from CLI args)
_text_min_len = 3
_text_max_len = 100
_language = "JP"
_input_mode = "text"
_model_id_filter: set[int] | None = None
_audio_format = "wav"


def ogg_opus_duration(data: bytes) -> float:
    """Return decoded Opus duration from Ogg page granules, excluding pre-skip.

    Ogg/Opus granules count decoded samples at 48 kHz. Encoded byte count is
    unrelated to duration, so it must never be interpreted as PCM length.
    """
    pos = 0
    last_granule = None
    pre_skip = None
    while pos + 27 <= len(data):
        if data[pos : pos + 4] != b"OggS":
            raise ValueError(f"Invalid Ogg page at byte {pos}")
        segments = data[pos + 26]
        if pos + 27 + segments > len(data):
            raise ValueError("Truncated Ogg page segments")
        body_start = pos + 27 + segments
        body_end = body_start + sum(data[pos + 27 : body_start])
        if body_end > len(data):
            raise ValueError("Truncated Ogg page body")
        granule = struct.unpack_from("<Q", data, pos + 6)[0]
        if granule != 0xFFFFFFFFFFFFFFFF:
            last_granule = granule
        if pre_skip is None and body_end >= body_start + 12 and data[body_start : body_start + 8] == b"OpusHead":
            pre_skip = struct.unpack_from("<H", data, body_start + 10)[0]
        pos = body_end
    if pos != len(data) or pre_skip is None or last_granule is None:
        raise ValueError("Incomplete Ogg/Opus response")
    return max(0.0, (last_granule - pre_skip) / 48000.0)


@dataclass
class ModelSpec:
    """Per-model speaker/style info for request generation."""

    model_id: int
    speakers: list[int]
    styles: list[str]


@dataclass
class BenchResult:
    """Single request result."""

    latency_ms: float
    status: int
    audio_bytes: int
    text_len: int
    model_id: int
    speaker_id: int
    style: str
    ttfb_ms: float = 0.0  # Time to first byte (streaming only)
    n_chunks: int = 0  # Number of chunks received (streaming only)
    submit_time: float = 0.0  # monotonic time when request was actually sent
    audio_duration: float = 0.0  # Audio duration in seconds (from header or computed)
    error: str = ""  # Short response body/error text for failed requests


@dataclass
class _InflightTracker:
    """Track in-flight request count for rate-limited mode."""

    current: int = 0
    peak: int = 0
    samples: list[tuple[float, int]] = field(default_factory=list)  # (time, count)


def _normalize_base_url(url: str) -> str:
    """Normalize a base URL without changing scheme/host."""
    return url.rstrip("/")


def _parse_header(raw_header: str) -> tuple[str, str]:
    """Parse a CLI header value in `Name: value` form."""
    if ":" not in raw_header:
        raise argparse.ArgumentTypeError("--header must be in 'Name: value' form")
    name, value = raw_header.split(":", 1)
    name = name.strip()
    value = value.strip()
    if not name:
        raise argparse.ArgumentTypeError("--header name must not be empty")
    return name, value


def _build_headers(
    *,
    api_key: str,
    api_key_env: str,
    api_key_header: str,
    extra_headers: list[tuple[str, str]],
) -> dict[str, str]:
    """Build HTTP headers for metadata and benchmark requests."""
    headers = {name: value for name, value in extra_headers}

    resolved_api_key = api_key
    if not resolved_api_key and api_key_env:
        for env_name in [item.strip() for item in api_key_env.split(",") if item.strip()]:
            resolved_api_key = os.environ.get(env_name, "")
            if resolved_api_key:
                break

    if resolved_api_key:
        headers[api_key_header] = resolved_api_key
    return headers


def _make_client_session(
    *,
    headers: dict[str, str],
    timeout_s: float,
    insecure_tls: bool,
) -> aiohttp.ClientSession:
    """Create an aiohttp session with benchmark-wide connection settings."""
    connector = aiohttp.TCPConnector(ssl=False) if insecure_tls else None
    return aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
        connector=connector,
    )


def _response_error_text(body: bytes, limit: int = 240) -> str:
    """Decode a short response body snippet for failed requests."""
    if not body:
        return ""
    text = body.decode("utf-8", errors="replace").replace("\n", " ")
    return text[:limit]


def _audio_duration_from_header(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _build_model_specs(models_info: dict[str, object]) -> list[ModelSpec]:
    """Build benchmark request targets from /models/info payload."""
    model_specs: list[ModelSpec] = []
    for mid_str, raw_minfo in sorted(models_info.items()):
        mid = int(mid_str)
        if _model_id_filter is not None and mid not in _model_id_filter:
            continue
        if not isinstance(raw_minfo, dict):
            logger.warning("Skipping malformed model info for model %s", mid_str)
            continue

        id2spk = raw_minfo.get("id2spk", {})
        style2id = raw_minfo.get("style2id", {})
        if not isinstance(id2spk, dict) or not isinstance(style2id, dict):
            logger.warning("Skipping model %s: missing id2spk/style2id", mid)
            continue

        speakers = sorted(int(speaker_id) for speaker_id in id2spk)
        styles = [str(name) for name, _ in sorted(style2id.items(), key=lambda item: int(item[1]))]
        if not speakers or not styles:
            logger.warning("Skipping model %s: empty speakers/styles", mid)
            continue

        model_specs.append(ModelSpec(model_id=mid, speakers=speakers, styles=styles))
        logger.info("Model %s: %s speakers, %s styles", mid, len(speakers), len(styles))
    return model_specs


async def _fetch_models_info(
    base_url: str,
    *,
    headers: dict[str, str],
    timeout_s: float,
    insecure_tls: bool,
) -> dict[str, object]:
    """Fetch /models/info and raise a useful error on failure."""
    async with (
        _make_client_session(
            headers=headers,
            timeout_s=timeout_s,
            insecure_tls=insecure_tls,
        ) as session,
        session.get(f"{base_url}/models/info") as resp,
    ):
        body = await resp.read()
        if resp.status != 200:
            raise RuntimeError(f"Failed to get /models/info: HTTP {resp.status}: {_response_error_text(body)}")
        data = await resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("/models/info did not return a JSON object")
        return data


def _full_control_chunks(language: str) -> tuple[str, list[dict[str, object]]]:
    if language == "EN":
        return "SQL", [
            {
                "language": "EN",
                "surface_text": "SQL",
                "phones": ["s", "iy", "k", "w", "ah", "l"],
                "tones": [0, 1, 0, 0, 1, 0],
                "word2ph": [2, 2, 2],
            }
        ]
    return "今日", [
        {
            "language": "JP",
            "surface_text": "今日",
            "phones": ["ky", "o:"],
            "tones": [1, 0],
            "word2ph": [1, 1],
        }
    ]


def _make_request_payload(
    model_specs: list[ModelSpec], streaming: bool = False, audio_format: str = "wav"
) -> tuple[str, dict[str, object]]:
    """Generate a JSON request body."""
    text = random_text(_text_min_len, _text_max_len)
    spec = random.choice(model_specs)
    speaker_id = random.choice(spec.speakers)
    style_name = random.choice(spec.styles)
    style_weight = round(random.uniform(0.5, 3.0), 1)

    payload: dict[str, object] = {
        "model_id": spec.model_id,
        "speaker_id": speaker_id,
        "style": style_name,
        "style_weight": style_weight,
        "language": _language,
        "streaming": streaming,
        "format": audio_format,
    }
    if _input_mode == "full-control":
        text, chunks = _full_control_chunks(_language)
        payload["chunks"] = chunks
    else:
        payload["text"] = text
    return text, payload


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model_specs: list[ModelSpec],
    sem: asyncio.Semaphore,
    audio_format: str = "wav",
) -> BenchResult:
    """Send a single TTS request with random model/speaker/style."""
    text, payload = _make_request_payload(model_specs, audio_format=audio_format)
    spec_id = int(payload["model_id"])  # type: ignore[arg-type]

    async with sem:
        t0 = time.monotonic()
        async with session.post(f"{base_url}/voice", json=payload) as resp:
            body = await resp.read()
            t1 = time.monotonic()
            duration_hdr = resp.headers.get("X-Audio-Duration", "")

    audio_dur = _audio_duration_from_header(duration_hdr)
    return BenchResult(
        latency_ms=(t1 - t0) * 1000,
        status=resp.status,
        audio_bytes=len(body),
        text_len=len(text),
        model_id=spec_id,
        speaker_id=int(payload["speaker_id"]),  # type: ignore[arg-type]
        style=str(payload["style"]),
        submit_time=t0,
        audio_duration=audio_dur,
        error=_response_error_text(body) if resp.status != 200 else "",
    )


async def send_streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model_specs: list[ModelSpec],
    sem: asyncio.Semaphore,
    audio_format: str = "pcm",
) -> BenchResult:
    """Send a streaming TTS request and measure TTFB + total latency."""
    text, payload = _make_request_payload(model_specs, streaming=True, audio_format=audio_format)
    spec_id = int(payload["model_id"])  # type: ignore[arg-type]

    async with sem:
        t0 = time.monotonic()
        ttfb = 0.0
        total_bytes = 0
        ogg_bytes = bytearray() if audio_format == "ogg" else None

        async with session.post(f"{base_url}/voice", json=payload) as resp:
            first_chunk = True
            n_chunks = 0
            error_body = bytearray()
            async for chunk in resp.content.iter_any():
                if first_chunk:
                    ttfb = (time.monotonic() - t0) * 1000
                    first_chunk = False
                total_bytes += len(chunk)
                if ogg_bytes is not None:
                    ogg_bytes.extend(chunk)
                n_chunks += 1
                if resp.status != 200 and len(error_body) < 512:
                    error_body.extend(chunk)
            t1 = time.monotonic()

    return BenchResult(
        latency_ms=(t1 - t0) * 1000,
        status=resp.status,
        audio_bytes=total_bytes,
        text_len=len(text),
        model_id=spec_id,
        speaker_id=int(payload["speaker_id"]),  # type: ignore[arg-type]
        style=str(payload["style"]),
        ttfb_ms=ttfb,
        n_chunks=n_chunks,
        submit_time=t0,
        audio_duration=ogg_opus_duration(bytes(ogg_bytes)) if resp.status == 200 and ogg_bytes is not None else 0.0,
        error=_response_error_text(bytes(error_body)) if resp.status != 200 else "",
    )


def wav_bytes_to_audio_seconds(audio_bytes: int, is_streaming: bool = False) -> float:
    """Convert audio data size to audio duration (44100Hz, 16-bit mono)."""
    BYTES_PER_SECOND = 44100 * 2  # 16-bit mono
    if is_streaming:
        # Raw PCM, no header
        return audio_bytes / BYTES_PER_SECOND
    WAV_HEADER = 44
    pcm_bytes = max(0, audio_bytes - WAV_HEADER)
    return pcm_bytes / BYTES_PER_SECOND


def print_stats(
    results: list[BenchResult],
    label: str,
    wall_time_s: float,
    is_streaming: bool = False,
    send_times: list[float] | None = None,
    target_rps: float | None = None,
    inflight_tracker: _InflightTracker | None = None,
) -> None:
    """Print latency statistics."""
    ok = [r for r in results if r.status == 200]
    fail = [r for r in results if r.status != 200]
    if not results:
        print(f"\n{'=' * 60}")
        print(f" {label}")
        print(f"{'=' * 60}")
        print("  Requests:     0 completed")
        print("  No results were collected.")
        print(f"{'=' * 60}\n")
        return
    if not ok:
        print(f"\n{'=' * 60}")
        print(f" {label}")
        print(f"{'=' * 60}")
        print(f"  Requests:     0 OK, {len(fail)} failed")
        status_counts: dict[int, int] = {}
        for r in fail:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        print("  Failed statuses:")
        for status_code, count in sorted(status_counts.items()):
            print(f"    {status_code}: {count}")
        for r in fail[:5]:
            error_suffix = f", error={r.error}" if r.error else ""
            print(f"    sample status={r.status}, text_len={r.text_len}{error_suffix}")
        print(f"{'=' * 60}\n")
        return

    latencies = np.array([r.latency_ms for r in ok])
    text_lens = np.array([r.text_len for r in ok])
    audio_durations = np.array(
        [
            r.audio_duration
            if r.audio_duration > 0
            else (float("nan") if _audio_format == "ogg" else wav_bytes_to_audio_seconds(r.audio_bytes, is_streaming))
            for r in ok
        ]
    )

    n = len(ok)
    rps = n / wall_time_s if wall_time_s > 0 else 0
    total_audio_s = audio_durations.sum()
    audio_per_second = total_audio_s / wall_time_s if wall_time_s > 0 else 0

    print(f"\n{'=' * 60}")
    print(f" {label}")
    print(f"{'=' * 60}")
    # Model distribution
    model_counts: dict[int, int] = {}
    for r in ok:
        model_counts[r.model_id] = model_counts.get(r.model_id, 0) + 1
    model_dist = ", ".join(f"model {mid}={cnt}" for mid, cnt in sorted(model_counts.items()))

    print(f"  Requests:     {n} OK, {len(fail)} failed ({model_dist})")
    print(f"  Wall time:    {wall_time_s:.2f}s")
    print(f"  Throughput:   {rps:.1f} RPS")
    print(f"  Text length:  mean={text_lens.mean():.0f}, min={text_lens.min()}, max={text_lens.max()}")
    print("  Audio duration (per request):")
    print(f"    mean:  {audio_durations.mean():.2f}s")
    print(f"    min:   {audio_durations.min():.2f}s")
    print(f"    max:   {audio_durations.max():.2f}s")
    print("  Audio throughput:")
    print(f"    total:     {total_audio_s:.1f}s of audio in {wall_time_s:.2f}s")
    print(f"    rate:      {audio_per_second:.1f} audio sec / wall sec")
    print(f"    realtime:  {audio_per_second:.0f}x real-time")

    # Rate control stats
    if send_times is not None and target_rps is not None and len(send_times) > 1:
        target_interval_ms = 1000.0 / target_rps
        intervals = np.diff(send_times) * 1000  # ms
        jitter = intervals - target_interval_ms  # signed deviation
        abs_jitter = np.abs(jitter)
        actual_rps = (len(send_times) - 1) / (send_times[-1] - send_times[0])
        print("  Rate Control:")
        print(f"    Target:    {target_rps:.1f} RPS ({target_interval_ms:.1f}ms interval)")
        print(f"    Actual:    {actual_rps:.1f} RPS")
        print("    Send Jitter (ms, signed):")
        print(f"      mean:  {jitter.mean():+.2f}")
        print(f"      P50:   {np.percentile(abs_jitter, 50):.2f}")
        print(f"      P90:   {np.percentile(abs_jitter, 90):.2f}")
        print(f"      P99:   {np.percentile(abs_jitter, 99):.2f}")
        print(f"      max:   {abs_jitter.max():.2f}")

    if inflight_tracker is not None:
        print("  In-flight:")
        print(f"    Peak:      {inflight_tracker.peak}")
        if inflight_tracker.samples:
            counts = np.array([c for _, c in inflight_tracker.samples])
            print(f"    Mean:      {counts.mean():.1f}")
            print(f"    P50:       {np.percentile(counts, 50):.0f}")
            print(f"    P99:       {np.percentile(counts, 99):.0f}")

    # Streaming stats: TTFB, streaming duration (first→last byte), chunks
    if is_streaming:
        ttfbs = np.array([r.ttfb_ms for r in ok])
        stream_durations = np.array([r.latency_ms - r.ttfb_ms for r in ok])
        chunks = np.array([r.n_chunks for r in ok])
        print("  TTFB (ms):")
        print(f"    mean:  {ttfbs.mean():.1f}")
        print(f"    P50:   {np.percentile(ttfbs, 50):.1f}")
        print(f"    P90:   {np.percentile(ttfbs, 90):.1f}")
        print(f"    P95:   {np.percentile(ttfbs, 95):.1f}")
        print(f"    P99:   {np.percentile(ttfbs, 99):.1f}")
        print(f"    min:   {ttfbs.min():.1f}")
        print(f"    max:   {ttfbs.max():.1f}")
        print("  Streaming Duration (first→last byte, ms):")
        print(f"    mean:  {stream_durations.mean():.1f}")
        print(f"    P50:   {np.percentile(stream_durations, 50):.1f}")
        print(f"    P90:   {np.percentile(stream_durations, 90):.1f}")
        print(f"    P95:   {np.percentile(stream_durations, 95):.1f}")
        print(f"    P99:   {np.percentile(stream_durations, 99):.1f}")
        print(f"    min:   {stream_durations.min():.1f}")
        print(f"    max:   {stream_durations.max():.1f}")
        print(f"  Chunks:   mean={chunks.mean():.1f}, min={chunks.min()}, max={chunks.max()}")

    print("  Total Latency (ms):")
    print(f"    mean:  {latencies.mean():.1f}")
    print(f"    P50:   {np.percentile(latencies, 50):.1f}")
    print(f"    P90:   {np.percentile(latencies, 90):.1f}")
    print(f"    P95:   {np.percentile(latencies, 95):.1f}")
    print(f"    P99:   {np.percentile(latencies, 99):.1f}")
    print(f"    min:   {latencies.min():.1f}")
    print(f"    max:   {latencies.max():.1f}")

    if fail:
        print("\n  Failed requests:")
        for r in fail[:5]:
            error_suffix = f", error={r.error}" if r.error else ""
            print(f"    status={r.status}, text_len={r.text_len}{error_suffix}")

    print(f"{'=' * 60}\n")


def write_csv(results: list[BenchResult], path: Path, is_streaming: bool = False) -> None:
    """Write per-request results to CSV for analysis."""
    fields = [
        "text_len",
        "audio_bytes",
        "audio_duration",
        "latency_ms",
        "status",
        "model_id",
        "speaker_id",
        "style",
        "error",
    ]
    if is_streaming:
        fields.extend(["ttfb_ms", "n_chunks"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                "text_len": r.text_len,
                "audio_bytes": r.audio_bytes,
                "audio_duration": round(r.audio_duration, 4),
                "latency_ms": round(r.latency_ms, 2),
                "status": r.status,
                "model_id": r.model_id,
                "speaker_id": r.speaker_id,
                "style": r.style,
                "error": r.error,
            }
            if is_streaming:
                row["ttfb_ms"] = round(r.ttfb_ms, 2)
                row["n_chunks"] = r.n_chunks
            writer.writerow(row)
    logger.info(f"Per-request CSV written to {path} ({len(results)} rows)")


def _print_cluster_stats(status_data: dict[str, object]) -> None:
    """Print /cluster-status summary when the status endpoint is an aggregate route."""
    if "backends" not in status_data and "per_backend" not in status_data:
        return

    print("\n  Cluster:")
    print(f"    Backends:       {status_data.get('backends', 0)}")
    print(f"    Healthy:        {status_data.get('backends_healthy', 0)}")
    print(f"    In flight:      {status_data.get('total_in_flight', 0)}")
    print(f"    Total requests: {status_data.get('total_requests', 0)}")
    loaded_model_keys = status_data.get("loaded_model_keys", [])
    if loaded_model_keys:
        print(f"    Models:         {loaded_model_keys}")
    gpu_names = status_data.get("gpu_names", [])
    if gpu_names:
        print(f"    GPUs:           {gpu_names}")

    per_backend = status_data.get("per_backend", [])
    if isinstance(per_backend, list) and per_backend:
        print("  Cluster backends:")
        for backend in per_backend:
            if not isinstance(backend, dict):
                continue
            worker = backend.get("worker", {})
            if not isinstance(worker, dict):
                worker = {}
            print(
                "    "
                f"{backend.get('host', '')}: "
                f"healthy={backend.get('healthy', False)}, "
                f"in_flight={backend.get('in_flight', 0)}, "
                f"requests={backend.get('total_requests', 0)}, "
                f"worker_id={worker.get('worker_id', '')}"
            )


async def _print_server_stats(session: aiohttp.ClientSession, base_url: str, status_url: str = "") -> None:
    """Fetch and print batcher stats + VRAM breakdown from server."""
    target = status_url or f"{base_url}/status"
    async with session.get(target) as resp:
        if resp.status != 200:
            body = await resp.read()
            logger.warning("Failed to get status stats: HTTP %s: %s", resp.status, _response_error_text(body))
            return
        try:
            status_data = await resp.json()
        except Exception as exc:
            logger.warning("Failed to parse status stats: %s", exc)
            return
        _print_cluster_stats(status_data)
        batchers = status_data.get("batchers", {})
        for name, stats in batchers.items():
            print(f"  Batcher '{name}': {stats}")

        vram_breakdown = status_data.get("vram_breakdown", {})
        for name, bd in vram_breakdown.items():
            print(f"\n  VRAM Breakdown (model '{name}'):")
            print(f"    PyTorch model:   {bd.get('pytorch_model', 0):.0f} MB")
            print(f"    Decoder (TRT):   {bd.get('decoder_session', 0):.0f} MB")
            print(f"    Enc+DP (TRT):    {bd.get('enc_dp_session', 0):.0f} MB")
            print(f"    Flow (TRT):      {bd.get('flow_session', 0):.0f} MB")
            print(f"    Total:           {bd.get('total', 0):.0f} MB")
            print(f"    Baseline free:   {bd.get('baseline_free', 0):.0f} MB")


async def _run_warmup(
    session: aiohttp.ClientSession,
    base_url: str,
    model_specs: list[ModelSpec],
    n_warmup: int,
    send_fn: object,
    sem: asyncio.Semaphore,
) -> None:
    """Run warmup requests."""
    if n_warmup <= 0:
        return
    logger.info(f"Warmup: {n_warmup} requests...")
    warmup_tasks = [send_fn(session, base_url, model_specs, sem) for _ in range(n_warmup)]  # type: ignore[operator]
    t0 = time.monotonic()
    warmup_results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
    t1 = time.monotonic()
    warmup_ok = [r for r in warmup_results if isinstance(r, BenchResult)]
    warmup_fail = [r for r in warmup_results if not isinstance(r, BenchResult)]
    logger.info(f"Warmup done: {len(warmup_ok)} OK, {len(warmup_fail)} errors in {t1 - t0:.1f}s")
    if warmup_fail:
        for e in warmup_fail[:3]:
            logger.warning(f"  Warmup error: {e}")


async def run_benchmark(
    base_url: str,
    concurrency: int,
    n_requests: int,
    n_warmup: int,
    streaming: bool = False,
    audio_format: str = "wav",
    headers: dict[str, str] | None = None,
    timeout_s: float = 120,
    insecure_tls: bool = False,
    status_url: str = "",
) -> list[BenchResult]:
    """Run the benchmark (concurrency mode)."""
    base_url = _normalize_base_url(base_url)
    headers = headers or {}
    try:
        models_info = await _fetch_models_info(
            base_url,
            headers=headers,
            timeout_s=timeout_s,
            insecure_tls=insecure_tls,
        )
    except Exception as exc:
        logger.error("%s", exc)
        return []

    # Build per-model specs from all loaded models
    model_specs = _build_model_specs(models_info)
    if not model_specs:
        logger.error("No models matched --model-ids")
        return []

    mode_str = "streaming" if streaming else "standard"
    logger.info(
        f"Models: {len(model_specs)}, Concurrency: {concurrency}, "
        f"Requests: {n_requests}, Warmup: {n_warmup}, Mode: {mode_str}"
    )

    sem = asyncio.Semaphore(concurrency)
    send_fn = (
        partial(send_streaming_request, audio_format=audio_format)
        if streaming
        else partial(send_request, audio_format=audio_format)
    )

    async with _make_client_session(headers=headers, timeout_s=timeout_s, insecure_tls=insecure_tls) as session:
        await _run_warmup(session, base_url, model_specs, n_warmup, send_fn, sem)

        # Measurement phase
        logger.info(f"Benchmark: {n_requests} requests at concurrency={concurrency} ({mode_str})...")
        bench_tasks = [send_fn(session, base_url, model_specs, sem) for _ in range(n_requests)]
        t0 = time.monotonic()
        bench_results = await asyncio.gather(*bench_tasks, return_exceptions=True)
        t1 = time.monotonic()

        ok_results = [r for r in bench_results if isinstance(r, BenchResult)]
        err_results = [r for r in bench_results if not isinstance(r, BenchResult)]
        if err_results:
            for e in err_results[:5]:
                logger.error(f"  Error: {e}")

        label = f"Benchmark (C={concurrency}, N={n_requests}, {mode_str})"
        print_stats(ok_results, label, t1 - t0, is_streaming=streaming)
        await _print_server_stats(session, base_url, status_url=status_url)

    return ok_results


async def run_rate_benchmark(
    base_url: str,
    rps: float,
    n_requests: int,
    n_warmup: int,
    max_inflight: int,
    streaming: bool = False,
    audio_format: str = "wav",
    headers: dict[str, str] | None = None,
    timeout_s: float = 120,
    insecure_tls: bool = False,
    status_url: str = "",
) -> list[BenchResult]:
    """Run the benchmark (rate-limited mode).

    Sends requests at a fixed rate using drift-compensated scheduling.
    Each request is fire-and-forget; a semaphore limits max in-flight.
    """
    base_url = _normalize_base_url(base_url)
    headers = headers or {}
    try:
        models_info = await _fetch_models_info(
            base_url,
            headers=headers,
            timeout_s=timeout_s,
            insecure_tls=insecure_tls,
        )
    except Exception as exc:
        logger.error("%s", exc)
        return []

    model_specs = _build_model_specs(models_info)
    if not model_specs:
        logger.error("No models matched --model-ids")
        return []

    mode_str = "streaming" if streaming else "standard"
    interval_ms = 1000.0 / rps
    logger.info(
        f"Models: {len(model_specs)}, RPS: {rps:.1f} ({interval_ms:.1f}ms), "
        f"Requests: {n_requests}, MaxInflight: {max_inflight}, "
        f"Warmup: {n_warmup}, Mode: {mode_str}"
    )

    sem = asyncio.Semaphore(max_inflight)
    send_fn = (
        partial(send_streaming_request, audio_format=audio_format)
        if streaming
        else partial(send_request, audio_format=audio_format)
    )

    async with _make_client_session(headers=headers, timeout_s=timeout_s, insecure_tls=insecure_tls) as session:
        await _run_warmup(session, base_url, model_specs, n_warmup, send_fn, sem)

        # Rate-limited measurement phase
        logger.info(f"Benchmark: {n_requests} requests at {rps:.1f} RPS ({mode_str})...")
        interval = 1.0 / rps
        tracker = _InflightTracker()

        async def tracked_send() -> BenchResult:
            tracker.current += 1
            tracker.peak = max(tracker.peak, tracker.current)
            tracker.samples.append((time.monotonic(), tracker.current))
            try:
                return await send_fn(session, base_url, model_specs, sem)
            finally:
                tracker.current -= 1

        tasks: list[asyncio.Task[BenchResult]] = []
        send_times: list[float] = []

        t_start = time.monotonic()
        for i in range(n_requests):
            target = t_start + i * interval
            now = time.monotonic()
            if target > now:
                await asyncio.sleep(target - now)

            send_times.append(time.monotonic())
            tasks.append(asyncio.create_task(tracked_send()))

        # Wait for all in-flight requests to complete
        bench_results = await asyncio.gather(*tasks, return_exceptions=True)
        t_end = time.monotonic()

        ok_results = [r for r in bench_results if isinstance(r, BenchResult)]
        err_results = [r for r in bench_results if not isinstance(r, BenchResult)]
        if err_results:
            for e in err_results[:5]:
                logger.error(f"  Error: {e}")

        label = f"Benchmark (RPS={rps:.1f}, N={n_requests}, {mode_str})"
        print_stats(
            ok_results,
            label,
            t_end - t_start,
            is_streaming=streaming,
            send_times=send_times,
            target_rps=rps,
            inflight_tracker=tracker,
        )
        await _print_server_stats(session, base_url, status_url=status_url)

    return ok_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP benchmark for CHARAVOICE Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", type=str, default="http://localhost:5000", help="Server URL")
    parser.add_argument("--concurrency", "-c", type=int, default=8, help="Concurrent requests (concurrency mode)")
    parser.add_argument("--n-requests", "-n", type=int, default=200, help="Total requests")
    parser.add_argument("--warmup", "-w", type=int, default=16, help="Warmup requests")
    parser.add_argument("--min-len", type=int, default=3, help="Min text length (chars)")
    parser.add_argument("--max-len", type=int, default=100, help="Max text length (chars)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming response mode")
    parser.add_argument("--format", type=str, default="ogg", choices=["wav", "ogg", "pcm"], help="Audio output format")
    parser.add_argument("--language", type=str, default="JP", choices=["JP", "EN"], help="Request language")
    parser.add_argument(
        "--input-mode", type=str, default="text", choices=["text", "full-control"], help="Request input mode"
    )
    parser.add_argument("--rps", type=float, default=0, help="Target requests/sec (0=concurrency mode)")
    parser.add_argument("--max-inflight", type=int, default=128, help="Max in-flight requests (rate mode)")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key value. When set, it is sent using --api-key-header.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="CHARAVOICE_API_KEY,TTS_API_KEY",
        help="Comma-separated env var names used when --api-key is empty",
    )
    parser.add_argument("--api-key-header", type=str, default="X-API-Key", help="Header name for --api-key")
    parser.add_argument(
        "--header",
        action="append",
        type=_parse_header,
        default=[],
        help="Extra HTTP header in 'Name: value' form. May be repeated.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification. Useful for fixed-IP HTTPS smoke/bench routes.",
    )
    parser.add_argument(
        "--status-url",
        type=str,
        default="",
        help="Optional URL for post-benchmark /status stats. Defaults to <url>/status.",
    )
    parser.add_argument("--csv", type=str, default="", help="Write per-request CSV to this path")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible random request generation")
    parser.add_argument(
        "--model-ids",
        type=str,
        default="",
        help="Comma-separated model IDs to target (default: all loaded models)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    global _text_min_len, _text_max_len, _language, _input_mode, _model_id_filter, _audio_format
    _text_min_len = args.min_len
    _text_max_len = args.max_len
    _language = args.language
    _input_mode = args.input_mode
    _audio_format = args.format
    if args.model_ids:
        _model_id_filter = {int(x.strip()) for x in args.model_ids.split(",") if x.strip()}

    headers = _build_headers(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        api_key_header=args.api_key_header,
        extra_headers=args.header,
    )
    base_url = _normalize_base_url(args.url)

    if args.rps > 0:
        results = asyncio.run(
            run_rate_benchmark(
                base_url=base_url,
                rps=args.rps,
                n_requests=args.n_requests,
                n_warmup=args.warmup,
                max_inflight=args.max_inflight,
                streaming=args.streaming,
                audio_format=args.format,
                headers=headers,
                timeout_s=args.timeout,
                insecure_tls=args.insecure,
                status_url=args.status_url,
            )
        )
    else:
        results = asyncio.run(
            run_benchmark(
                base_url=base_url,
                concurrency=args.concurrency,
                n_requests=args.n_requests,
                n_warmup=args.warmup,
                streaming=args.streaming,
                audio_format=args.format,
                headers=headers,
                timeout_s=args.timeout,
                insecure_tls=args.insecure,
                status_url=args.status_url,
            )
        )

    if args.csv and results:
        write_csv(results, Path(args.csv), is_streaming=args.streaming)


if __name__ == "__main__":
    main()
