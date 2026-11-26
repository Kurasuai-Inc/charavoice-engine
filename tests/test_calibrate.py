"""Pure calibration decisions and benchmark parsing."""

import argparse
import csv

import pytest

from charavoice_engine.scripts import run_calibrated
from charavoice_engine.scripts.calibrate import (
    Candidate,
    _next_quick_candidates,
    _redact_worker_log,
    choose_best,
    score_candidate,
    summarize_csv,
)


def test_quick_search_keeps_inference_and_watermark_from_same_trial():
    args = argparse.Namespace(inference_batches=(8, 16), watermark_batches=(16, 32), worker_counts=(1, 2, 3))
    trials = [
        {"candidate": {"inference_batch_max": 8, "watermark_batch_max": 32, "workers": 1}, "score": 100},
        {"candidate": {"inference_batch_max": 16, "watermark_batch_max": 16, "workers": 1}, "score": 90},
    ]
    assert _next_quick_candidates(args, trials, "workers") == [
        Candidate(8, 32, 1),
        Candidate(8, 32, 2),
        Candidate(8, 32, 3),
    ]


def test_score_rejects_latency_and_vram_constraints():
    metrics = {
        name: {"latency_p95_ms": 300, "ttfb_p95_ms": 200, "audio_seconds_per_second": 50}
        for name in ("jp-normal", "jp-stream", "en-normal", "en-stream")
    }
    score, reason = score_candidate(
        metrics, max_p95_latency_ms=1000, max_p95_ttfb_ms=600, min_free_vram_mib=2048, free_vram_mib=4096
    )
    assert score == pytest.approx(50) and reason is None
    metrics["jp-stream"]["ttfb_p95_ms"] = 700
    assert (
        score_candidate(
            metrics, max_p95_latency_ms=1000, max_p95_ttfb_ms=600, min_free_vram_mib=2048, free_vram_mib=4096
        )[0]
        is None
    )
    assert (
        score_candidate(
            metrics, max_p95_latency_ms=1000, max_p95_ttfb_ms=1000, min_free_vram_mib=2048, free_vram_mib=1024
        )[0]
        is None
    )


def test_summarize_csv_requires_every_request_to_succeed(tmp_path):
    path = tmp_path / "requests.csv"
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("status", "audio_duration", "latency_ms", "ttfb_ms"))
        writer.writeheader()
        writer.writerow({"status": 200, "audio_duration": 4, "latency_ms": 100, "ttfb_ms": 50})
        writer.writerow({"status": 200, "audio_duration": 6, "latency_ms": 200, "ttfb_ms": 80})
    result = summarize_csv(path, "bench.log", 2, 1.0, True)
    assert result["audio_seconds_per_second"] == 10
    assert result["ttfb_p95_ms"] > 50
    assert choose_best([{"score": None}, {"score": 4}])["score"] == 4
    with path.open("a", newline="") as output:
        csv.writer(output).writerow((500, 0, 0, 0))
    try:
        summarize_csv(path, "bench.log", 3, 1.0, True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("A failed request must invalidate the candidate")


def test_calibrated_runner_rejects_hardware_or_model_mismatch(monkeypatch, tmp_path):
    profile = {
        "fingerprint": {
            "models": ["voice-24"],
            "model_sha256": "model-a",
            "cpu_count": 8,
            "memory_mib": 65536,
            "gpu": {"name": "RTX 5090", "memory_mib": 32607, "compute_capability": "12.0"},
            "versions": {"charavoice-engine": "0.2.0"},
        }
    }
    monkeypatch.setattr(run_calibrated, "_hash_model_assets", lambda *_: "model-a")
    monkeypatch.setattr(run_calibrated, "_gpu_info", lambda: profile["fingerprint"]["gpu"].copy())
    monkeypatch.setattr(run_calibrated, "_effective_cpu_count", lambda: 8)
    monkeypatch.setattr(run_calibrated, "_effective_ram_mib", lambda: 65536)
    monkeypatch.setattr(run_calibrated, "_version", lambda _name: "0.2.0")
    run_calibrated.check_fingerprint(profile, tmp_path, "voice-24")
    profile["fingerprint"]["declared_allocation"] = {"cpus": 8, "memory_mib": 65536}
    with pytest.raises(ValueError, match="Declared"):
        run_calibrated.check_fingerprint(profile, tmp_path, "voice-24")
    run_calibrated.check_fingerprint(profile, tmp_path, "voice-24", allocated_cpus=8, allocated_memory_mib=65536)
    with pytest.raises(ValueError, match="Model list"):
        run_calibrated.check_fingerprint(profile, tmp_path, "voice-1", allocated_cpus=8, allocated_memory_mib=65536)
    monkeypatch.setattr(run_calibrated, "_effective_cpu_count", lambda: 4)
    with pytest.raises(ValueError, match="CPU count"):
        run_calibrated.check_fingerprint(profile, tmp_path, "voice-24", allocated_cpus=8, allocated_memory_mib=65536)
    monkeypatch.setattr(run_calibrated, "_effective_cpu_count", lambda: 8)
    monkeypatch.setattr(run_calibrated, "_effective_ram_mib", lambda: 16384)
    with pytest.raises(ValueError, match="RAM"):
        run_calibrated.check_fingerprint(profile, tmp_path, "voice-24", allocated_cpus=8, allocated_memory_mib=65536)


def test_legacy_worker_log_does_not_keep_watermark_core(tmp_path):
    log = tmp_path / "worker-1.log"
    log.write_text("AudioSeal TRT initialized (default_core16=4660, batch_max=16)\n")
    _redact_worker_log(log)
    assert "4660" not in log.read_text()
