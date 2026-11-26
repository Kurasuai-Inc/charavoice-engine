"""FastAPI application for the CHARAVOICE Engine TTS server."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from urllib.request import urlopen

import numpy as np
import psutil
from fastapi import Body, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from numpy.typing import NDArray
from pydantic import ValidationError

from charavoice_engine.constants import (
    DEFAULT_STYLE,
    Languages,
)
from charavoice_engine.inference.engine_base import InferenceEngine, InferenceRequest, InferenceResult
from charavoice_engine.inference.text_processing import (
    FullControlChunk,
    G2pResult,
    InputChunk,
    TextChunk,
    check_split_and_run_g2p,
)
from charavoice_engine.server import legal
from charavoice_engine.server import metrics as _metrics
from charavoice_engine.server.batcher import DynamicBatcher
from charavoice_engine.server.responses import ClosingStreamingResponse as StreamingResponse
from charavoice_engine.server.schemas import (
    Chunk,
    FullControlConvertRequest,
    FullControlConvertResponse,
    TTSRequest,
)
from charavoice_engine.server.streaming import (
    SAMPLE_RATE,
    audio_chunk_to_pcm_int16_clamped,
    audio_to_ogg_opus_bytes,
    audio_to_wav_bytes,
    buffer_streaming_chunks,
    ogg_streaming_chunks,
)

logger = logging.getLogger(__name__)


def _get_g2p_executor() -> Executor:
    """Return the initialized G2P process pool."""
    from charavoice_engine.nlp.japanese.g2p_pool import get_pool

    pool = get_pool()
    if pool is None:
        raise RuntimeError("G2P process pool is not initialized")
    return pool


@dataclass
class LoadedModel:
    """A loaded model with its engine, batcher, and metadata."""

    name: str
    engine: InferenceEngine
    batcher: DynamicBatcher
    style_vectors: NDArray[Any]  # [num_styles, 256]
    spk2id: dict[str, int]
    id2spk: dict[int, str]
    style2id: dict[str, int]
    config_path: str
    model_path: str
    device: str = "cuda"


# Global state populated by run_server.py before app starts
_models: dict[str, LoadedModel] = {}  # name -> model
_models_list: list[LoadedModel] = []  # model_id (int) access
_in_flight: int = 0
_ogg_pool: ProcessPoolExecutor | None = None
_stream_buffer_chunks: int = 10
_ogg_page_duration_ms: float | None = None
_worker_metadata: dict[str, Any] = {
    "worker_id": "",
    "worker_site": "",
    "capacity_audio_sec_per_sec": 0.0,
    "requested_model_keys": [],
    "bert_languages": [],
}


def set_models(models: dict[str, LoadedModel], models_list: list[LoadedModel]) -> None:
    """Set loaded models (called from run_server before uvicorn starts)."""
    global _models, _models_list
    _models = models
    _models_list = models_list


def set_streaming_config(*, stream_buffer_chunks: int, ogg_page_duration_ms: float | None = None) -> None:
    """Set streaming response configuration before uvicorn starts."""
    if stream_buffer_chunks < 1:
        raise ValueError("stream_buffer_chunks must be >= 1")
    if ogg_page_duration_ms is not None and ogg_page_duration_ms <= 0:
        raise ValueError("ogg_page_duration_ms must be positive")
    global _stream_buffer_chunks, _ogg_page_duration_ms
    _stream_buffer_chunks = stream_buffer_chunks
    _ogg_page_duration_ms = ogg_page_duration_ms


def set_worker_metadata(metadata: dict[str, Any]) -> None:
    """Set worker metadata exposed through /status and /cluster-status."""
    _worker_metadata.update(metadata)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop DynamicBatcher dispatch loops and metrics gauge updater."""
    global _ogg_pool
    legal.validate_startup()  # refuse to serve without the AGPL-3.0 §13 source information
    for model in _models_list:
        await model.batcher.start()
    logger.info(f"Started {len(_models_list)} model batcher(s)")
    _ogg_pool = ProcessPoolExecutor(max_workers=4)
    logger.info("OGG ProcessPool started (4 workers)")
    gauge_task = asyncio.create_task(_metrics.update_gauges_loop(_models))
    yield
    gauge_task.cancel()
    for model in _models_list:
        await model.batcher.stop()
    if _ogg_pool is not None:
        _ogg_pool.shutdown(wait=False)
    logger.info("All batchers stopped")


app = FastAPI(title="CHARAVOICE Engine TTS Server", lifespan=lifespan)


@app.middleware("http")
async def track_in_flight(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Track number of in-flight /voice requests for KEDA scaling metrics."""
    global _in_flight
    is_voice = request.url.path == "/voice"
    if is_voice:
        _in_flight += 1
        _metrics.IN_FLIGHT.inc()
    try:
        import time as _time

        start = _time.monotonic()
        response = await call_next(request)
        if is_voice:
            _metrics.REQUEST_LATENCY.observe(_time.monotonic() - start)
            _metrics.REQUEST_COUNT.labels(status=str(response.status_code)).inc()
        return response
    except Exception:
        if is_voice:
            _metrics.REQUEST_COUNT.labels(status="500").inc()
        raise
    finally:
        if is_voice:
            _in_flight -= 1
            _metrics.IN_FLIGHT.dec()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Audio-Sample-Rate", "X-Audio-Channels", "X-Audio-Format"],
)

# Mount demo frontend (static files) if directory exists
_demo_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "demo"
if _demo_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")
    logger.info(f"Demo frontend mounted at /demo from {_demo_dir}")


def _raise_validation_error(msg: str, param: str):
    logger.warning(f"Validation error: {msg}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["body", param])],
    )


def _resolve_model(
    model_name: str | None,
    model_id: int,
) -> tuple[LoadedModel, int]:
    """Resolve model_name/model_id to a LoadedModel."""
    if model_name is not None:
        if model_name not in _models:
            _raise_validation_error(f"model_name={model_name} not found", "model_name")
        model = _models[model_name]
        resolved_id = _models_list.index(model)
        return model, resolved_id

    if model_id >= len(_models_list) or model_id < 0:
        _raise_validation_error(f"model_id={model_id} not found", "model_id")
    return _models_list[model_id], model_id


def _resolve_speaker(
    model: LoadedModel,
    speaker_name: str | None,
    speaker_id: int,
) -> int:
    """Resolve speaker_name/speaker_id."""
    if speaker_name is not None:
        if speaker_name not in model.spk2id:
            _raise_validation_error(
                f"speaker_name={speaker_name} not found",
                "speaker_name",
            )
        return model.spk2id[speaker_name]

    if speaker_id not in model.id2spk:
        _raise_validation_error(f"speaker_id={speaker_id} not found", "speaker_id")
    return speaker_id


def _compute_style_vector(
    model: LoadedModel,
    style: str,
    style_weight: float,
) -> NDArray[Any]:
    """Compute style vector: mean + (style_vec - mean) * weight."""
    if style not in model.style2id:
        _raise_validation_error(f"style={style} not found", "style")
    style_id = model.style2id[style]
    mean = model.style_vectors[0]
    vec = model.style_vectors[style_id]
    return mean + (vec - mean) * style_weight


@app.post("/full-control/convert", response_model=FullControlConvertResponse)
async def convert_full_control(request: FullControlConvertRequest) -> FullControlConvertResponse:
    """Convert text chunks into /voice-compatible full-control chunks."""

    if request.text is not None:
        source_chunks = [Chunk(language=request.language, text=request.text)]
    else:
        assert request.chunks is not None
        source_chunks = request.chunks

    convert_languages = set(request.convert_languages)
    converted: list[Chunk] = []
    warnings: list[str] = []
    for index, chunk in enumerate(source_chunks):
        if chunk.text is None or chunk.language not in convert_languages:
            converted.append(chunk)
            continue
        full_control, warning = await _convert_text_chunk_to_full_control(chunk, index)
        converted.append(full_control)
        if warning is not None:
            warnings.append(warning)
    return FullControlConvertResponse(chunks=converted, warnings=warnings)


def _clean_text_for_full_control(text: str, language: Languages) -> tuple[str, list[str], list[int], list[int]]:
    from charavoice_engine.nlp import clean_text

    return clean_text(text, language, raise_yomi_error=False)


async def _clean_text_for_full_control_async(
    text: str, language: Languages
) -> tuple[str, list[str], list[int], list[int]]:
    try:
        executor = _get_g2p_executor()
    except RuntimeError:
        return _clean_text_for_full_control(text, language)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _clean_text_for_full_control, text, language)


async def _convert_text_chunk_to_full_control(chunk: Chunk, index: int) -> tuple[Chunk, str | None]:
    """Convert one text chunk to full-control, falling back to text if alignment is not representable."""

    if chunk.text is None:
        raise ValueError("chunk.text is required")

    try:
        surface_text, phones, tones, word2ph = await _clean_text_for_full_control_async(chunk.text, chunk.language)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"chunks[{index}] G2P failed: {exc}",
        ) from exc

    if len(phones) < 2 or len(tones) < 2 or len(word2ph) < 2:
        return chunk, f"chunks[{index}] was left as text because G2P returned an incomplete alignment."

    control_phones = phones[1:-1]
    control_tones = tones[1:-1]
    control_word2ph = word2ph[1:-1]
    if len(control_word2ph) != len(surface_text):
        return (
            chunk,
            f"chunks[{index}] {chunk.language.value} was left as text because its word2ph length "
            f"({len(control_word2ph)}) does not match surface_text length ({len(surface_text)}).",
        )

    try:
        return (
            Chunk(
                language=chunk.language,
                surface_text=surface_text,
                phones=control_phones,
                tones=control_tones,
                word2ph=control_word2ph,
            ),
            None,
        )
    except ValidationError as exc:
        return chunk, f"chunks[{index}] was left as text because full-control validation failed: {exc}"


@app.api_route("/voice", methods=["GET", "POST"], response_model=None)
async def voice(
    request: Request,
    tts_req: TTSRequest | None = Body(default=None),
    text: str | None = Query(None, description="Text to synthesize"),
    encoding: str | None = Query(None, description="URL-decode text with this encoding"),
    model_name: str | None = Query(None, description="Model name (priority over model_id)"),
    model_id: int = Query(0, description="Model ID"),
    speaker_name: str | None = Query(None, description="Speaker name (priority over speaker_id)"),
    speaker_id: int = Query(0, description="Speaker ID"),
    noise: float = Query(0.667, ge=0.0),
    length: float = Query(1.0, gt=0.0),
    language: Languages = Query(Languages.JP),
    auto_split: bool = Query(True),
    split_interval: float = Query(0.5, ge=0.0),
    assist_text: str | None = Query(None),
    assist_text_weight: float = Query(0.7, ge=0.0, le=1.0),
    style: str | None = Query(DEFAULT_STYLE),
    style_weight: float = Query(1.0, ge=0.0),
    streaming: bool = Query(False),
    format: str = Query("ogg", description="Output format: ogg, wav, or streaming-only pcm"),
    watermark_core16: int | None = Query(None, ge=0, le=65535),
    watermark_extension: str | None = Query(None),
) -> Response | StreamingResponse:
    """Synthesize speech from JSON body or production-compatible query params."""
    if tts_req is None:
        if text is not None and encoding is not None:
            text = unquote(text, encoding=encoding)
        try:
            tts_req = TTSRequest(
                text=text,
                model_name=model_name,
                model_id=model_id,
                speaker_name=speaker_name,
                speaker_id=speaker_id,
                noise=noise,
                length=length,
                language=language,
                auto_split=auto_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                style=style,
                style_weight=style_weight,
                streaming=streaming,
                format=format,
                watermark_core16=watermark_core16,
                watermark_extension=watermark_extension,
            )
        except ValidationError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors()) from e
    return await _handle_tts_request(tts_req)


async def _handle_tts_request(tts_req: TTSRequest) -> Response | StreamingResponse:
    """Synthesize speech from a validated request."""
    if tts_req.text is not None and not tts_req.text.strip():
        _raise_validation_error("Text is empty or whitespace-only", "text")

    model, _ = _resolve_model(tts_req.model_name, tts_req.model_id)
    sid = _resolve_speaker(model, tts_req.speaker_name, tts_req.speaker_id)
    style = tts_req.style or DEFAULT_STYLE
    style_vec = _compute_style_vector(model, style, tts_req.style_weight)

    chunks = _schema_chunks_to_inference_chunks(tts_req) if tts_req.chunks is not None else None

    def _make_request(
        line_text: str,
        seq_len_hint: int = 0,
        g2p_result: G2pResult | None = None,
        request_chunks: list[InputChunk] | None = None,
    ) -> InferenceRequest:
        return InferenceRequest(
            text=line_text,
            language=request_chunks[0].language if request_chunks else tts_req.language,
            speaker_id=sid,
            style_vec=style_vec,
            noise_scale=tts_req.noise,
            length_scale=tts_req.length,
            assist_text=tts_req.assist_text if tts_req.assist_text else None,
            assist_text_weight=tts_req.assist_text_weight,
            watermark_core16=tts_req.watermark_core16,
            watermark_extension=tts_req.watermark_extension,
            seq_len_hint=seq_len_hint,
            g2p_result=g2p_result,
            chunks=request_chunks,
        )

    if tts_req.streaming:
        if chunks is not None:
            return await _handle_streaming_chunk_request(
                model,
                _make_request(_chunks_display_text(chunks), request_chunks=chunks),
                tts_req.format,
            )
        assert tts_req.text is not None
        return await _handle_streaming_request(
            model,
            tts_req.text,
            tts_req.language,
            tts_req.auto_split,
            tts_req.split_interval,
            _make_request,
            tts_req.format,
        )

    try:
        if chunks is not None:
            result = await model.batcher.submit(_make_request(_chunks_display_text(chunks), request_chunks=chunks))
            audio = result.audio
        else:
            line_groups = await _prepare_line_groups(model, tts_req)

            all_chunks = [chunk for group in line_groups for chunk in group]
            all_results: list[InferenceResult] = await asyncio.gather(
                *[model.batcher.submit(_make_request(txt, seq_len, g2p)) for txt, seq_len, g2p in all_chunks]
            )

            silence_samples = int(SAMPLE_RATE * tts_req.split_interval)
            silence = np.zeros(silence_samples, dtype=np.float32)
            audio_parts: list[NDArray] = []
            idx = 0
            for group_idx, group in enumerate(line_groups):
                for chunk_result in all_results[idx : idx + len(group)]:
                    audio_parts.append(chunk_result.audio)
                idx += len(group)
                if group_idx < len(line_groups) - 1 and tts_req.split_interval > 0:
                    audio_parts.append(silence)
            audio = np.concatenate(audio_parts)

    except TimeoutError as e:
        logger.error(f"Inference timed out: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        ) from e
    except ValueError as e:
        logger.warning(f"Inference input error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    duration = len(audio) / SAMPLE_RATE
    duration_header = {"X-Audio-Duration": f"{duration:.4f}"}

    if tts_req.format == "ogg":
        loop = asyncio.get_running_loop()
        ogg_bytes = await loop.run_in_executor(_ogg_pool, audio_to_ogg_opus_bytes, audio, SAMPLE_RATE)
        return Response(content=ogg_bytes, media_type="audio/ogg", headers=duration_header)

    wav_bytes = audio_to_wav_bytes(audio, SAMPLE_RATE)
    return Response(content=wav_bytes, media_type="audio/wav", headers=duration_header)


def _schema_chunks_to_inference_chunks(tts_req: TTSRequest) -> list[InputChunk]:
    assert tts_req.chunks is not None
    chunks: list[InputChunk] = []
    for index, chunk in enumerate(tts_req.chunks):
        if chunk.text is not None:
            if not chunk.text.strip():
                _raise_validation_error(f"chunks[{index}].text is empty or whitespace-only", "chunks")
            chunks.append(TextChunk(language=chunk.language, text=chunk.text))
            continue
        assert chunk.surface_text is not None
        assert chunk.phones is not None
        assert chunk.tones is not None
        assert chunk.word2ph is not None
        chunks.append(
            FullControlChunk(
                language=chunk.language,
                surface_text=chunk.surface_text,
                phones=chunk.phones,
                tones=chunk.tones,
                word2ph=chunk.word2ph,
            )
        )
    return chunks


def _chunks_display_text(chunks: list[InputChunk]) -> str:
    return "".join(chunk.text if isinstance(chunk, TextChunk) else chunk.surface_text for chunk in chunks)


async def _prepare_line_groups(
    model: LoadedModel,
    tts_req: TTSRequest,
) -> list[list[tuple[str, int, G2pResult | None]]]:
    text_config = model.engine.text_config
    if text_config is None:
        raise RuntimeError("Loaded engine does not expose text preprocessing config")

    assert tts_req.text is not None
    if tts_req.auto_split:
        lines = [line for line in tts_req.text.split("\n") if line.strip()]
        if not lines:
            _raise_validation_error("Text is empty after splitting", "text")
    else:
        lines = [tts_req.text]

    loop = asyncio.get_running_loop()
    line_results = await asyncio.gather(
        *[
            loop.run_in_executor(_get_g2p_executor(), check_split_and_run_g2p, line, tts_req.language, text_config)
            for line in lines
        ]
    )
    return line_results  # type: ignore[return-value]


async def _handle_streaming_request(
    model: LoadedModel,
    text: str,
    language: Languages,
    auto_split: bool,
    split_interval: float,
    make_request: Any,
    format: str = "pcm",
) -> StreamingResponse:
    """Handle streaming=true: yield audio chunks via StreamingResponse.

    When format='ogg', chunks are encoded to OGG/Opus incrementally.
    Otherwise, raw PCM int16le chunks are returned.

    Each line is submitted via submit_streaming() so decoder chunks are
    streamed as they're produced. Lines are processed sequentially so
    each line's chunks can be batched with other concurrent requests.
    """
    # Split text into lines
    if auto_split:
        lines = [line for line in text.split("\n") if line.strip()]
        if not lines:
            _raise_validation_error("Text is empty after splitting", "text")
    else:
        lines = [text]

    text_config = model.engine.text_config
    line_groups: list[list[tuple[str, int, G2pResult | None]]]
    if text_config is None:
        raise RuntimeError("Loaded engine does not expose text preprocessing config")
    loop = asyncio.get_running_loop()
    line_results = await asyncio.gather(
        *[
            loop.run_in_executor(_get_g2p_executor(), check_split_and_run_g2p, line, language, text_config)
            for line in lines
        ]
    )
    line_groups = line_results  # type: ignore[assignment]

    silence_samples = int(SAMPLE_RATE * split_interval)

    async def _generate_raw():
        """Yield raw float32 NDArray chunks."""
        silence = np.zeros(silence_samples, dtype=np.float32) if silence_samples > 0 else None
        for group_idx, group in enumerate(line_groups):
            for chunk_text, seq_len, g2p in group:
                req = make_request(chunk_text, seq_len, g2p)
                async with aclosing(model.batcher.submit_streaming(req)) as synthesis:
                    async for audio_chunk in synthesis:
                        yield audio_chunk
            if group_idx < len(line_groups) - 1 and silence is not None:
                yield silence

    if format == "ogg":
        return StreamingResponse(
            ogg_streaming_chunks(_generate_raw(), SAMPLE_RATE, _ogg_page_duration_ms),
            media_type="audio/ogg",
        )

    async def _generate_pcm():
        silence_bytes = (np.zeros(silence_samples, dtype=np.int16)).tobytes() if silence_samples > 0 else b""
        for group_idx, group in enumerate(line_groups):
            for chunk_text, seq_len, g2p in group:
                req = make_request(chunk_text, seq_len, g2p)
                async with aclosing(model.batcher.submit_streaming(req)) as synthesis:
                    async for audio_chunk in synthesis:
                        yield audio_chunk_to_pcm_int16_clamped(audio_chunk)
            if group_idx < len(line_groups) - 1 and silence_bytes:
                yield silence_bytes

    return StreamingResponse(
        buffer_streaming_chunks(_generate_pcm(), min_chunks=_stream_buffer_chunks),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Channels": "1",
            "X-Audio-Format": "int16le",
        },
    )


async def _handle_streaming_chunk_request(
    model: LoadedModel,
    request: InferenceRequest,
    format: str = "pcm",
) -> StreamingResponse:
    """Handle streaming=true with mixed-language chunks input."""

    async def _generate_raw():
        async with aclosing(model.batcher.submit_streaming(request)) as synthesis:
            async for audio_chunk in synthesis:
                yield audio_chunk

    if format == "ogg":
        return StreamingResponse(
            ogg_streaming_chunks(_generate_raw(), SAMPLE_RATE, _ogg_page_duration_ms),
            media_type="audio/ogg",
        )

    async def _generate_pcm():
        async with aclosing(model.batcher.submit_streaming(request)) as synthesis:
            async for audio_chunk in synthesis:
                yield audio_chunk_to_pcm_int16_clamped(audio_chunk)

    return StreamingResponse(
        buffer_streaming_chunks(_generate_pcm(), min_chunks=_stream_buffer_chunks),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Channels": "1",
            "X-Audio-Format": "int16le",
        },
    )


def _supported_languages_for_worker() -> list[str]:
    """Return BERT languages configured for this worker."""
    allowed_languages = {"JP", "EN"}
    metadata_langs = _worker_metadata.get("bert_languages") or []
    if metadata_langs:
        supported_from_metadata = [
            str(lang).upper() for lang in metadata_langs if str(lang).upper() in allowed_languages
        ]
        if supported_from_metadata:
            return supported_from_metadata

    env_value = os.environ.get("CHARAVOICE_LANGS")
    if not env_value:
        return ["JP", "EN"]

    supported: list[str] = []
    for raw_name in env_value.split(","):
        name = raw_name.strip().upper()
        if not name or name not in allowed_languages:
            continue
        if name not in supported:
            supported.append(name)
    return supported or ["JP", "EN"]


def _model_config_metadata(model: LoadedModel) -> dict[str, Any]:
    """Read lightweight provider metadata from config.json."""
    metadata: dict[str, Any] = {
        "provider": "CHARAVOICE Engine",
        "config_path": model.config_path,
        "model_path": model.model_path,
        "device": model.device,
    }
    try:
        data = _json.loads(Path(model.config_path).read_text(encoding="utf-8"))
    except Exception as exc:
        metadata["config_error"] = str(exc)
        return metadata

    data_section = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
    metadata["model_version"] = data.get("version", "")
    metadata["config_model_name"] = data.get("model_name") or data_section.get("model_name", "")
    metadata["sampling_rate"] = data_section.get("sampling_rate", 0)
    return metadata


def _model_metadata(model: LoadedModel) -> dict[str, Any]:
    """Return compact metadata for routing and operational dashboards."""
    provider_metadata = _model_config_metadata(model)
    return {
        "name": model.name,
        "model_version": provider_metadata.get("model_version", ""),
        "config_model_name": provider_metadata.get("config_model_name", ""),
        "sampling_rate": provider_metadata.get("sampling_rate", 0),
        "device": model.device,
        "config_file": Path(model.config_path).name,
        "model_file": Path(model.model_path).name,
        "speaker_count": len(model.spk2id),
        "style_count": len(model.style2id),
        "speakers": list(model.spk2id.keys()),
        "styles": list(model.style2id.keys()),
    }


def _worker_status() -> dict[str, Any]:
    """Return normalized worker metadata with currently loaded model keys."""
    loaded_model_keys = list(_models.keys())
    requested_model_keys = _worker_metadata.get("requested_model_keys") or loaded_model_keys
    return {
        "worker_id": _worker_metadata.get("worker_id", ""),
        "worker_site": _worker_metadata.get("worker_site", ""),
        "capacity_audio_sec_per_sec": float(_worker_metadata.get("capacity_audio_sec_per_sec") or 0.0),
        "requested_model_keys": list(requested_model_keys),
        "loaded_model_keys": loaded_model_keys,
        "bert_languages": _supported_languages_for_worker(),
        "model_count": len(loaded_model_keys),
    }


@app.get("/licenses")
def get_licenses():
    """Return the engine license, third-party credits (incl. the CC-BY-SA-4.0 Japanese BERT) and NOTICE."""
    return legal.licenses_info()


@app.get("/source")
def get_source():
    """AGPL-3.0 §13 source offer: the public source URL (null until configured) and this build's commit."""
    return legal.source_info()


@app.get("/models/info")
def get_models_info():
    """Return loaded model information."""
    result: dict[str, dict[str, Any]] = {}
    supported_languages = _supported_languages_for_worker()
    for i, model in enumerate(_models_list):
        provider_metadata = _model_config_metadata(model)
        speakers = [
            {
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "styles": [
                    {"style_id": style_id, "style_name": style_name}
                    for style_name, style_id in sorted(model.style2id.items(), key=lambda item: item[1])
                ],
            }
            for speaker_name, speaker_id in sorted(model.spk2id.items(), key=lambda item: item[1])
        ]
        result[str(i)] = {
            "model_id": i,
            "model_name": model.name,
            "config_file": Path(model.config_path).name,
            "model_file": Path(model.model_path).name,
            "device": model.device,
            "spk2id": model.spk2id,
            "id2spk": model.id2spk,
            "style2id": model.style2id,
            "supported_languages": supported_languages,
            "speakers": speakers,
            "provider_metadata": provider_metadata,
        }
    return result


@app.get("/status")
def get_status():
    """Return server status."""
    import torch

    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    devices = ["cpu"]
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")

    gpu_info: list[dict[str, Any]] = []
    try:
        import GPUtil  # type: ignore[reportMissingImports]

        for gpu in GPUtil.getGPUs():
            gpu_info.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_name": gpu.name,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
    except ImportError:
        # GPUtil is optional; fall back to torch CUDA info
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i)
            gpu_info.append(
                {
                    "gpu_id": i,
                    "gpu_name": props.name,
                    "gpu_load": None,
                    "gpu_memory": {
                        "total": props.total_memory // (1024 * 1024),
                        "used": allocated // (1024 * 1024),
                        "free": (props.total_memory - reserved) // (1024 * 1024),
                    },
                }
            )

    # Batcher stats
    batcher_stats = {}
    for name, model in _models.items():
        batcher_stats[name] = model.batcher.stats()

    queue_per_model = {
        name: {
            "queue_size": stats.get("queue_size", 0),
            "max_queue_size": getattr(_models[name].batcher, "max_queue_size", stats.get("max_queue_size", 0)),
        }
        for name, stats in batcher_stats.items()
    }

    # VRAM breakdown per model
    vram_breakdown: dict[str, dict[str, float]] = {}
    for name, model in _models.items():
        bd = model.engine.vram_breakdown
        if bd is not None:
            vram_breakdown[name] = bd.to_dict()

    return {
        "in_flight": _in_flight,
        "devices": devices,
        "cpu_percent": cpu_percent,
        "memory_total": memory_info.total,
        "memory_available": memory_info.available,
        "memory_used": memory_info.used,
        "memory_percent": memory_info.percent,
        "gpu": gpu_info,
        "batchers": batcher_stats,
        "queue": {
            "total_queue_size": sum(item["queue_size"] for item in queue_per_model.values()),
            "per_model": queue_per_model,
        },
        "vram_breakdown": vram_breakdown,
        "worker": _worker_status(),
        "loaded_model_keys": list(_models.keys()),
        "model_metadata": {name: _model_metadata(model) for name, model in _models.items()},
    }


def _fetch_peer_status(host: str) -> dict[str, Any]:
    """Fetch /status from a peer backend. Returns in_flight, health, and telemetry."""
    try:
        resp = urlopen(f"http://{host}/status", timeout=3)
        data = _json.loads(resp.read())
        total_requests = sum(b.get("total_requests", 0) for b in data.get("batchers", {}).values())
        total_text_length = sum(b.get("total_text_length", 0) for b in data.get("batchers", {}).values())
        gpu_names = [g.get("gpu_name", "unknown") for g in data.get("gpu", [])]
        return {
            "host": host,
            "in_flight": data.get("in_flight", 0),
            "healthy": True,
            "total_requests": total_requests,
            "total_text_length": total_text_length,
            "gpu_names": gpu_names,
            "worker": data.get("worker", {}),
            "loaded_model_keys": data.get("loaded_model_keys", []),
            "model_metadata": data.get("model_metadata", {}),
        }
    except Exception:
        return {
            "host": host,
            "in_flight": 0,
            "healthy": False,
            "total_requests": 0,
            "total_text_length": 0,
            "gpu_names": [],
            "worker": {},
            "loaded_model_keys": [],
            "model_metadata": {},
        }


@app.get("/cluster-status")
async def cluster_status():
    """Aggregate in_flight, health, and telemetry across all cluster backends."""
    backends_str = os.environ.get("CHARAVOICE_CLUSTER_BACKENDS", "")
    if not backends_str:
        total_requests = sum(m.batcher.stats().get("total_requests", 0) for m in _models.values())
        total_text_length = sum(m.batcher.stats().get("total_text_length", 0) for m in _models.values())
        return {
            "total_in_flight": _in_flight,
            "backends": 1,
            "backends_healthy": 1,
            "total_requests": total_requests,
            "total_text_length": total_text_length,
            "gpu_names": [],
            "worker": _worker_status(),
            "loaded_model_keys": list(_models.keys()),
            "model_metadata": {name: _model_metadata(model) for name, model in _models.items()},
        }

    backends = [b.strip() for b in backends_str.split(",") if b.strip()]
    loop = asyncio.get_running_loop()
    results = list(await asyncio.gather(*[loop.run_in_executor(None, _fetch_peer_status, b) for b in backends]))

    healthy = [r for r in results if r["healthy"]]
    all_gpu_names: list[str] = []
    for r in results:
        all_gpu_names.extend(r.get("gpu_names", []))
    loaded_model_keys = sorted({key for r in results for key in r.get("loaded_model_keys", [])})
    model_metadata: dict[str, Any] = {}
    for r in results:
        model_metadata.update(r.get("model_metadata", {}))

    return {
        "total_in_flight": sum(r["in_flight"] for r in results),
        "backends": len(backends),
        "backends_healthy": len(healthy),
        "total_requests": sum(r["total_requests"] for r in results),
        "total_text_length": sum(r["total_text_length"] for r in results),
        "gpu_names": all_gpu_names,
        "loaded_model_keys": loaded_model_keys,
        "model_metadata": model_metadata,
        "per_backend": results,
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Expose Prometheus metrics."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
