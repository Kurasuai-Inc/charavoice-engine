# syntax=docker/dockerfile:1
# ==============================================================================
# CHARAVOICE Engine Docker image
#
# Multi-stage build:
#   1. builder: install uv, Python 3.13, compile dependencies
#   2. cleanup: strip Windows TRT builder resources + NCCL (~3GB)
#   3. runtime: copy cleaned venv, minimal runtime image
#
# Usage:
#   docker build --build-arg CHARAVOICE_SOURCE_COMMIT=$(git rev-parse HEAD) -t charavoice-engine .
#   docker run --gpus all -v ./model_assets:/models -v ./bert:/app/bert -p 5000:5000 charavoice-engine
# ==============================================================================

# --------------- builder stage ------------------------------------------------
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.13

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates \
        build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python 3.13 via uv
RUN uv python install 3.13

# Install dependencies first (layer cache)
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra server --extra tensorrt --extra watermark --no-install-project --no-dev

# Copy source and install project
COPY src/ src/
COPY frontend/ frontend/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra server --extra tensorrt --extra watermark --no-dev

# --------------- cleanup stage (strip unused files before final copy) ---------
FROM builder AS cleanup
# Windows TRT builder resources: never used on Linux (~2.8GB)
RUN rm -f /app/.venv/lib/python3.*/site-packages/tensorrt_libs/libnvinfer_builder_resource_win_*.so*

# --------------- runtime stage ------------------------------------------------
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire venv + project from cleanup (Windows TRT + NCCL stripped)
COPY --from=cleanup /app /app

# Copy additional project files needed at runtime
COPY LICENSE NOTICE THIRD_PARTY_LICENSES.md docker-entrypoint.sh ./

# AGPL-3.0 §13: GET /source reports the commit this image was built from.
# Build with: docker build --build-arg CHARAVOICE_SOURCE_COMMIT=$(git rev-parse HEAD) .
# The public source URL defaults to the upstream repository; a fork overrides CHARAVOICE_SOURCE_URL.
# Required: the build stops here without it (the server would refuse to start anyway).
ARG CHARAVOICE_SOURCE_COMMIT
RUN test -n "${CHARAVOICE_SOURCE_COMMIT}" \
    || { echo "ERROR: build with --build-arg CHARAVOICE_SOURCE_COMMIT=\$(git rev-parse HEAD)" >&2; exit 1; }
ENV CHARAVOICE_SOURCE_COMMIT=${CHARAVOICE_SOURCE_COMMIT}
ENV CHARAVOICE_SOURCE_URL=https://github.com/Kurasuai-Inc/charavoice-engine
# NOTICE is read from this one place (GET /licenses); the server refuses to start without it.
ENV CHARAVOICE_NOTICE_PATH=/app/NOTICE
RUN chmod +x docker-entrypoint.sh

# Ensure uv + managed Python are available in runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /root/.local/share/uv /root/.local/share/uv

# dict_data/ for pyopenjtalk user dictionary (written at runtime)
RUN mkdir -p /app/dict_data

# Default volume mount points
# model_assets/ -> /models (safetensors, config.json, style_vectors.npy)
# bert/         -> /app/bert (ONNX BERT models)
# trt_cache/    -> /app/trt_cache (TensorRT engine cache, persisted)
VOLUME ["/models", "/app/bert", "/app/trt_cache"]

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:5000/status || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--dir", "/models", "--host", "0.0.0.0", "--port", "5000", "--trt-cache", "/app/trt_cache"]
