# GPU calibration for OGG + AudioSeal TensorRT

`calibrate` measures the complete HTTP synthesis path with the existing
`bench_server` workload, including OGG encoding and actual AudioSeal TRT
watermarking. It searches inference batch size, watermark microbatch size, and
worker count on an otherwise idle GPU. No GPU model names or batch sizes are
hard-coded into the selection logic.
Expose one GPU to `nvidia-smi` for each calibration run.
For streaming OGG, the benchmark derives audio duration from Opus granules;
compressed byte count is not a duration measure.
When testing an older image that lacks this parser, copy the updated
`bench_server.py` onto it and pass `--benchmark-script /path/to/bench_server.py`.

For the 24-speaker JP/EN model, after the model and TensorRT cache have been
installed on the target machine:

```bash
python -m charavoice_engine.scripts.calibrate \
  --model-dir /models \
  --models voice_24spk_nebby_stella_22chars_jp_en_20260626_100e \
  --trt-cache /app/trt_cache \
  --allocated-cpus 8 --allocated-memory-mib 65536 \
  --watermark-core16 4660 \
  --output /app/runs/calibration-$(date +%Y%m%d%H%M%S).json \
  --max-wall-seconds 900
```

The core value above is a test watermark. Pass the actual production core when
calibrating a production workload; the core is not written into the report,
and older worker logs are redacted after each trial.
The default quick search compares watermark batches 16/32, inference batches
8/16, then workers 1/2/3, keeping the best single-worker setting together.
This is at most five distinct trials. `--search full` tests every combination.
Each trial runs JP and EN, normal and streaming, with 200 measured requests
plus 16 warmups per scenario. Normal concurrency is 16 and streaming
concurrency is 24; set these to the traffic level you intend to serve. The
default weighted score favors JP/EN streaming, while rejecting a trial if any
request fails, if P95 latency exceeds 2000 ms, if stream P95 time to first
byte exceeds 1000 ms, or if less than 2 GiB of VRAM remains. The full report
contains per-request CSVs and logs beside the JSON file. Review the scenario
rates and constraints, not only the chosen score.

The default ten-minute limit covers candidate startup and benchmark work. The
model fingerprint is calculated before that timer starts. Provisioning an
instance or pulling its image is also outside the timer. Cold TensorRT engine
builds can use the entire limit; prebuild/warm the cache, or raise
`--max-wall-seconds`. A timed-out search writes `complete: false` and may have
no selection. Re-run with a fresh output path after warming the cache. The
calibrator stops all worker and proxy processes after every candidate.
On hosts such as Salad where CPU/RAM cgroup limits are not visible in the
container, pass both declared allocation flags. They become part of the
profile and must match at serving time. The report also records the observed
CPU affinity and physical/cgroup RAM for diagnostics.

To serve with a complete profile on equivalent hardware:

```bash
python -m charavoice_engine.scripts.run_calibrated \
  --profile /app/runs/calibration-YYYYMMDDHHMMSS.json \
  --model-dir /models \
  --models voice_24spk_nebby_stella_22chars_jp_en_20260626_100e \
  --trt-cache /app/trt_cache \
  --allocated-cpus 8 --allocated-memory-mib 65536 \
  --watermark-core16 "$CHARAVOICE_WATERMARK_CORE16" \
  --port 5000
```

The runner verifies the GPU name, VRAM, compute capability, model file hash,
software versions, and available CPU and RAM before starting workers. It rejects
incomplete searches unless `--allow-incomplete` is explicit. A profile is
specific to its model, image/software, and hardware allocation; calibrate
again when those change. The report does not contain the watermark core.

`Dockerfile.calibration` overlays these commands on a model-bundled image and
starts an idle `/status` endpoint and writes a startup record so the GPU is free
for trial servers while the instance can still pass readiness checks. Build
with `--build-arg BASE_IMAGE=<digest-pinned compatible image>`. The idle image
is only for measurement; production deployments should start
`run_calibrated` with the selected profile and inject the secret core at
runtime.
