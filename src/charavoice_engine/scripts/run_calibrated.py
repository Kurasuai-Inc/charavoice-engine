#!/usr/bin/env python3
"""Run a calibrated CHARAVOICE worker cluster from a hardware-matched report."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

from charavoice_engine.scripts.calibrate import (
    _effective_cpu_count,
    _effective_ram_mib,
    _gpu_info,
    _hash_model_assets,
    _terminate,
    _version,
    _wait_ready,
)

LOG = logging.getLogger("charavoice.run_calibrated")


def check_fingerprint(
    profile: dict,
    model_dir: Path,
    models: str,
    *,
    allocated_cpus: int | None = None,
    allocated_memory_mib: int | None = None,
) -> None:
    original = profile["fingerprint"]
    requested = [item.strip() for item in models.split(",") if item.strip()]
    if requested != original["models"]:
        raise ValueError("Model list differs from the calibrated profile")
    if _hash_model_assets(model_dir, tuple(requested)) != original["model_sha256"]:
        raise ValueError("Model assets differ from the calibrated profile")
    gpu = _gpu_info()
    for field in ("name", "memory_mib", "compute_capability"):
        if gpu[field] != original["gpu"][field]:
            raise ValueError(f"GPU {field} differs from the calibrated profile")
    if _effective_cpu_count() < original["cpu_count"]:
        raise ValueError("Available CPU count is below the calibrated profile")
    if _effective_ram_mib() < original["memory_mib"] * 0.95:
        raise ValueError("Available RAM is below the calibrated profile")
    allocation = (
        {"cpus": allocated_cpus, "memory_mib": allocated_memory_mib}
        if allocated_cpus is not None and allocated_memory_mib is not None
        else None
    )
    if allocation != original.get("declared_allocation"):
        raise ValueError("Declared CPU/RAM allocation differs from the calibrated profile")
    for package, calibrated in original["versions"].items():
        if calibrated is not None and _version(package) != calibrated:
            raise ValueError(f"{package} version differs from the calibrated profile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--trt-cache", type=Path, required=True)
    parser.add_argument("--watermark-core16", type=int, required=True)
    parser.add_argument("--allocated-cpus", type=int, help="Declared vCPU allocation from the hosting platform")
    parser.add_argument("--allocated-memory-mib", type=int, help="Declared RAM allocation from the hosting platform")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--log-dir", type=Path, default=Path("calibrated-logs"))
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow a time-limited partial search result")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 0 <= args.watermark_core16 <= 0xFFFF:
        parser.error("--watermark-core16 must be a 16-bit integer")
    if (args.allocated_cpus is None) != (args.allocated_memory_mib is None):
        parser.error("Pass both allocation arguments together")
    profile = json.loads(args.profile.read_text())
    if profile.get("schema_version") != 1 or profile.get("selected") is None:
        parser.error("Profile does not contain a selected calibration result")
    if not profile.get("complete") and not args.allow_incomplete:
        parser.error("Calibration stopped before all candidates; pass --allow-incomplete to use its partial result")
    check_fingerprint(
        profile,
        args.model_dir,
        args.models,
        allocated_cpus=args.allocated_cpus,
        allocated_memory_mib=args.allocated_memory_mib,
    )
    selected = profile["selected"]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    children: list[subprocess.Popen] = []
    outputs = []

    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        backends = []
        for index in range(selected["workers"]):
            port = args.port + index + 1
            output = (args.log_dir / f"worker-{index + 1}.log").open("w")
            outputs.append(output)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "charavoice_engine.scripts.run_server",
                    "--dir",
                    str(args.model_dir),
                    "--models",
                    args.models,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--trt-cache",
                    str(args.trt_cache),
                    "--max-batch-size",
                    str(selected["inference_batch_max"]),
                    "--watermark",
                    "audioseal_trt",
                    "--watermark-core16",
                    str(args.watermark_core16),
                    "--watermark-batch-max",
                    str(selected["watermark_batch_max"]),
                    "--g2p-workers",
                    str(profile["server_settings"]["g2p_workers"]),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(child)
            backend = f"http://127.0.0.1:{port}"
            _wait_ready(backend + "/status", child, time.monotonic() + args.startup_timeout)
            backends.append(backend)
        proxy = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("calibration_proxy.py")),
                "--host",
                "0.0.0.0,::",
                "--port",
                str(args.port),
                "--backends",
                *backends,
            ],
            start_new_session=True,
        )
        children.append(proxy)
        _wait_ready(f"http://127.0.0.1:{args.port}/cluster-status", proxy, time.monotonic() + 30)
        LOG.info("Serving calibrated profile on port %s: %s", args.port, selected)
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError("A calibrated worker or proxy exited; inspect worker logs")
    except KeyboardInterrupt:
        LOG.info("Stopping calibrated cluster")
    finally:
        for child in reversed(children):
            _terminate(child)
        for output in outputs:
            output.close()


if __name__ == "__main__":
    main()
