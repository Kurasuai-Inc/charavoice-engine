#!/usr/bin/env python3
"""One-time, bounded calibration of TTS inference, watermark and worker batching.

Run on an otherwise idle GPU with the actual model and TensorRT cache. The
command starts candidate servers sequentially, benchmarks the complete HTTP
OGG path, writes every result, and stops all child processes before exiting.
It does not change the settings of an already running server.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

LOG = logging.getLogger("charavoice.calibrate")
_WALL_RE = re.compile(r"Wall time:\s*([\d.]+)s")


@dataclass(frozen=True)
class Candidate:
    inference_batch_max: int
    watermark_batch_max: int
    workers: int

    @property
    def key(self) -> str:
        return f"infer{self.inference_batch_max}-wm{self.watermark_batch_max}-workers{self.workers}"


@dataclass(frozen=True)
class Scenario:
    name: str
    language: str
    streaming: bool
    weight: float


SCENARIOS = (
    Scenario("jp-normal", "JP", False, 0.20),
    Scenario("jp-stream", "JP", True, 0.40),
    Scenario("en-normal", "EN", False, 0.10),
    Scenario("en-stream", "EN", True, 0.30),
)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return parsed


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from no values")
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_csv(csv_path: Path, output: str, expected: int, wall_seconds: float, streaming: bool) -> dict:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    success = [row for row in rows if row["status"] == "200"]
    if len(rows) != expected or len(success) != expected or wall_seconds <= 0:
        raise RuntimeError(f"Expected {expected} successful requests, got {len(success)}/{len(rows)}")
    audio_seconds = sum(float(row["audio_duration"]) for row in success)
    if audio_seconds <= 0:
        raise RuntimeError("Benchmark did not report positive audio durations")
    result = {
        "requests": expected,
        "wall_seconds": wall_seconds,
        "rps": expected / wall_seconds,
        "audio_seconds_per_second": audio_seconds / wall_seconds,
        "latency_p95_ms": percentile([float(row["latency_ms"]) for row in success], 95),
        "csv": str(csv_path),
        "log": output,
    }
    if streaming:
        result["ttfb_p95_ms"] = percentile([float(row["ttfb_ms"]) for row in success], 95)
    return result


def score_candidate(
    measurements: dict[str, dict],
    *,
    max_p95_latency_ms: float,
    max_p95_ttfb_ms: float,
    min_free_vram_mib: int,
    free_vram_mib: int | None,
) -> tuple[float | None, str | None]:
    if free_vram_mib is None:
        return None, "free VRAM could not be measured"
    if free_vram_mib < min_free_vram_mib:
        return None, f"free VRAM {free_vram_mib} MiB is below {min_free_vram_mib} MiB"
    score = 0.0
    for scenario in SCENARIOS:
        item = measurements.get(scenario.name)
        if item is None:
            return None, f"missing {scenario.name} measurement"
        if item["latency_p95_ms"] > max_p95_latency_ms:
            return None, f"{scenario.name} P95 latency exceeds {max_p95_latency_ms} ms"
        if scenario.streaming and item["ttfb_p95_ms"] > max_p95_ttfb_ms:
            return None, f"{scenario.name} P95 TTFB exceeds {max_p95_ttfb_ms} ms"
        rate = item["audio_seconds_per_second"]
        if rate <= 0:
            return None, f"{scenario.name} produced no audio"
        score += scenario.weight * math.log(rate)
    return math.exp(score), None


def choose_best(results: list[dict]) -> dict | None:
    eligible = [result for result in results if result.get("score") is not None]
    return max(eligible, key=lambda result: result["score"], default=None)


def _hash_model_assets(model_dir: Path, names: tuple[str, ...] | None) -> str:
    digest = hashlib.sha256()
    model_dirs = (
        [model_dir / name for name in names]
        if names
        else sorted(path for path in model_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    )
    files = [
        path
        for directory in model_dirs
        for path in sorted(directory.iterdir())
        if path.name == "config.json" or path.name == "style_vectors.npy" or path.suffix in (".safetensors", ".pth")
    ]
    if not files:
        raise ValueError(f"No model assets found in {model_dir}")
    for path in files:
        digest.update(str(path.relative_to(model_dir)).encode())
        with path.open("rb") as handle:
            while block := handle.read(8 << 20):
                digest.update(block)
    return digest.hexdigest()


def _gpu_info() -> dict:
    command = [
        _nvidia_smi(),
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi could not identify the GPU: {result.stderr.strip()}")
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1:
        raise RuntimeError("Calibration requires exactly one GPU visible to nvidia-smi")
    first = rows[0]
    if len(first) != 4:
        raise RuntimeError(f"Unexpected nvidia-smi GPU identity: {result.stdout!r}")
    return {"name": first[0], "memory_mib": int(first[1]), "driver": first[2], "compute_capability": first[3]}


def _free_vram_mib() -> int | None:
    result = subprocess.run(
        [_nvidia_smi(), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    rows = result.stdout.splitlines()
    return int(rows[0].strip()) if result.returncode == 0 and len(rows) == 1 else None


def _nvidia_smi() -> str:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    wsl_path = Path("/usr/lib/wsl/lib/nvidia-smi")
    if wsl_path.is_file():
        return str(wsl_path)
    raise RuntimeError("nvidia-smi is unavailable")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _effective_cpu_count() -> int:
    counts = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        counts.append(len(os.sched_getaffinity(0)))
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        parts = quota.read_text().split()
        if len(parts) == 2 and parts[0] != "max":
            counts.append(max(1, math.ceil(int(parts[0]) / int(parts[1]))))
    return min(counts)


def _effective_ram_mib() -> int:
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1 << 20)
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.is_file():
        value = limit.read_text().strip()
        if value.isdecimal():
            return min(physical, int(value) // (1 << 20))
    return physical


def _json_get(url: str, timeout: float = 3) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.load(response)


def _wait_ready(url: str, child: subprocess.Popen, deadline: float) -> None:
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Process exited during startup: {child.returncode}")
        try:
            _json_get(url)
            return
        except (OSError, urllib.error.URLError, ValueError):
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}")


def _terminate(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _redact_worker_log(path: Path) -> None:
    """Remove core values logged by older run_server versions in base images."""
    if not path.exists():
        return
    original = path.read_text(errors="replace")
    redacted = re.sub(r"(default_core16=)\d+", r"\1<configured>", original)
    if redacted != original:
        path.write_text(redacted)


def _run_candidate(candidate: Candidate, args: argparse.Namespace, deadline: float, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    children: list[subprocess.Popen] = []
    logs = []
    started = time.monotonic()
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        backends = []
        for index in range(candidate.workers):
            port = args.port + 1 + index
            command = [
                sys.executable,
                "-m",
                "charavoice_engine.scripts.run_server",
                "--dir",
                str(args.model_dir),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--trt-cache",
                str(args.trt_cache),
                "--max-batch-size",
                str(candidate.inference_batch_max),
                "--watermark",
                "audioseal_trt",
                "--watermark-core16",
                str(args.watermark_core16),
                "--watermark-batch-max",
                str(candidate.watermark_batch_max),
                "--g2p-workers",
                str(args.g2p_workers),
            ]
            if args.models:
                command += ["--models", args.models]
            output = (run_dir / f"worker-{index + 1}.log").open("w")
            logs.append(output)
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            backend = f"http://127.0.0.1:{port}"
            _wait_ready(backend + "/status", child, min(deadline, time.monotonic() + args.startup_timeout))
            backends.append(backend)
        proxy_output = (run_dir / "proxy.log").open("w")
        logs.append(proxy_output)
        proxy = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("calibration_proxy.py")),
                "--port",
                str(args.port),
                "--backends",
                *backends,
            ],
            stdout=proxy_output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(proxy)
        _wait_ready(base_url + "/cluster-status", proxy, min(deadline, time.monotonic() + 30))
        if not _json_get(base_url + "/cluster-status").get("healthy"):
            raise RuntimeError("A worker is not healthy")
        startup_seconds = time.monotonic() - started
        measurements = {}
        for scenario in SCENARIOS:
            concurrency = args.stream_concurrency if scenario.streaming else args.normal_concurrency
            csv_path = run_dir / f"{scenario.name}.csv"
            benchmark = (
                [sys.executable, str(args.benchmark_script)]
                if args.benchmark_script
                else [sys.executable, "-m", "charavoice_engine.scripts.bench_server"]
            )
            command = [
                *benchmark,
                "--url",
                base_url,
                "--language",
                scenario.language,
                "--format",
                "ogg",
                "--input-mode",
                "text",
                "--min-len",
                str(args.min_len),
                "--max-len",
                str(args.max_len),
                "--concurrency",
                str(concurrency),
                "--n-requests",
                str(args.requests),
                "--warmup",
                str(args.warmup),
                "--seed",
                str(args.seed),
                "--model-ids",
                "0",
                "--csv",
                str(csv_path),
            ]
            if scenario.streaming:
                command.append("--streaming")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Calibration time limit reached")
            result = subprocess.run(command, capture_output=True, text=True, timeout=remaining, check=False)
            output = result.stdout + "\n" + result.stderr
            (run_dir / f"{scenario.name}.log").write_text(output)
            match = _WALL_RE.search(output)
            if result.returncode != 0 or match is None or not csv_path.exists():
                raise RuntimeError(f"{scenario.name} benchmark failed; inspect {run_dir / (scenario.name + '.log')}")
            measurements[scenario.name] = summarize_csv(
                csv_path,
                str(run_dir / f"{scenario.name}.log"),
                args.requests,
                float(match.group(1)),
                scenario.streaming,
            )
        free_vram = _free_vram_mib()
        score, rejection = score_candidate(
            measurements,
            max_p95_latency_ms=args.max_p95_latency_ms,
            max_p95_ttfb_ms=args.max_p95_ttfb_ms,
            min_free_vram_mib=args.min_free_vram_mib,
            free_vram_mib=free_vram,
        )
        return {
            "candidate": asdict(candidate),
            "startup_seconds": startup_seconds,
            "measurements": measurements,
            "free_vram_mib": free_vram,
            "score": score,
            "rejection": rejection,
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        for child in reversed(children):
            _terminate(child)
        for output in logs:
            output.close()
        for path in run_dir.glob("worker-*.log"):
            _redact_worker_log(path)


def _next_quick_candidates(args: argparse.Namespace, results: list[dict], stage: str) -> list[Candidate]:
    if stage == "watermark":
        return [Candidate(args.inference_batches[0], value, 1) for value in args.watermark_batches]
    best = choose_best([result for result in results if result["candidate"]["workers"] == 1])
    wm = int(best["candidate"]["watermark_batch_max"]) if best else args.watermark_batches[0]
    if stage == "inference":
        return [Candidate(value, wm, 1) for value in args.inference_batches]
    infer = int(best["candidate"]["inference_batch_max"]) if best else args.inference_batches[0]
    return [Candidate(infer, wm, value) for value in args.worker_counts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--models", required=True, help="Comma-separated model directories to load; first must be the target model"
    )
    parser.add_argument("--trt-cache", type=Path, required=True)
    parser.add_argument("--benchmark-script", type=Path, help="Path to a compatible patched bench_server.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--watermark-core16", type=int, required=True, help="Actual watermark value to apply during trials"
    )
    parser.add_argument("--inference-batches", type=parse_positive_ints, default=(8, 16))
    parser.add_argument("--watermark-batches", type=parse_positive_ints, default=(16, 32))
    parser.add_argument("--worker-counts", type=parse_positive_ints, default=(1, 2, 3))
    parser.add_argument("--search", choices=("quick", "full"), default="quick")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--normal-concurrency", type=int, default=16)
    parser.add_argument("--stream-concurrency", type=int, default=24)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--g2p-workers", type=int, default=4)
    parser.add_argument("--allocated-cpus", type=int, help="Declared vCPU allocation when cgroups do not expose it")
    parser.add_argument(
        "--allocated-memory-mib", type=int, help="Declared RAM allocation when cgroups do not expose it"
    )
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument("--max-p95-latency-ms", type=float, default=2000)
    parser.add_argument("--max-p95-ttfb-ms", type=float, default=1000)
    parser.add_argument("--min-free-vram-mib", type=int, default=2048)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.model_dir.is_dir() or not 0 <= args.watermark_core16 <= 0xFFFF:
        parser.error("Model directory and a 16-bit watermark core are required")
    if (
        args.requests < 1
        or args.warmup < 0
        or args.max_wall_seconds <= 0
        or args.normal_concurrency < 1
        or args.stream_concurrency < 1
        or args.g2p_workers < 1
        or args.min_len < 1
        or args.min_len > args.max_len
        or args.startup_timeout <= 0
        or args.port < 1
        or args.port + max(args.worker_counts) > 65535
        or (args.allocated_cpus is not None and args.allocated_cpus < 1)
        or (args.allocated_memory_mib is not None and args.allocated_memory_mib < 1)
    ):
        parser.error("Invalid requests, concurrency, text length, worker count, port or time limit")
    if (args.allocated_cpus is None) != (args.allocated_memory_mib is None):
        parser.error("Pass both --allocated-cpus and --allocated-memory-mib together")
    if args.output.exists():
        parser.error("Output already exists; use a new path for each calibration")
    if args.benchmark_script and not args.benchmark_script.is_file():
        parser.error("--benchmark-script must name an existing file")
    model_names = tuple(item.strip() for item in args.models.split(",") if item.strip())
    if not model_names:
        parser.error("--models must include at least one model directory")
    fingerprint = {
        "gpu": _gpu_info(),
        "cpu_count": _effective_cpu_count(),
        "memory_mib": _effective_ram_mib(),
        "declared_allocation": (
            {"cpus": args.allocated_cpus, "memory_mib": args.allocated_memory_mib}
            if args.allocated_cpus is not None
            else None
        ),
        "model_sha256": _hash_model_assets(args.model_dir, model_names),
        "models": list(model_names),
        "versions": {
            name: _version(name) for name in ("charavoice-engine", "torch", "onnxruntime-gpu", "tensorrt-cu12-libs")
        },
        "workload": {
            "format": "ogg",
            "watermark": "audioseal_trt",
            "scenarios": [asdict(item) for item in SCENARIOS],
            "min_text_len": args.min_len,
            "max_text_len": args.max_len,
        },
    }
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint,
        "search": args.search,
        "requests_per_scenario": args.requests,
        "server_settings": {"g2p_workers": args.g2p_workers},
        "benchmark_script_sha256": (
            hashlib.sha256(args.benchmark_script.read_bytes()).hexdigest() if args.benchmark_script else None
        ),
        "constraints": {
            "max_p95_latency_ms": args.max_p95_latency_ms,
            "max_p95_ttfb_ms": args.max_p95_ttfb_ms,
            "min_free_vram_mib": args.min_free_vram_mib,
        },
        "candidates": [],
        "selected": None,
        "complete": False,
    }
    run_root = args.output.with_suffix("")
    seen: set[Candidate] = set()
    try:
        stages = ("full",) if args.search == "full" else ("watermark", "inference", "workers")
        for stage in stages:
            if stage == "full":
                candidates = [
                    Candidate(infer, wm, workers)
                    for infer in args.inference_batches
                    for wm in args.watermark_batches
                    for workers in args.worker_counts
                ]
            else:
                candidates = _next_quick_candidates(args, report["candidates"], stage)
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                if time.monotonic() >= deadline:
                    LOG.warning("Calibration time limit reached before %s", candidate.key)
                    break
                LOG.info("Calibrating %s", candidate.key)
                try:
                    result = _run_candidate(candidate, args, deadline, run_root / candidate.key)
                except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
                    LOG.warning("Candidate %s failed: %s", candidate.key, exc)
                    result = {"candidate": asdict(candidate), "score": None, "rejection": str(exc)}
                report["candidates"].append(result)
                _write_json(args.output, report)
            if time.monotonic() >= deadline:
                break
    finally:
        best = choose_best(report["candidates"])
        report["selected"] = best["candidate"] if best else None
        report["elapsed_seconds"] = time.monotonic() - started
        report["complete"] = time.monotonic() < deadline
        _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": report["selected"],
                "elapsed_seconds": report["elapsed_seconds"],
                "complete": report["complete"],
            }
        )
    )
    if report["selected"] is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
