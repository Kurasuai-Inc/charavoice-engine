"""Pydantic v2 request/response schemas for CHARAVOICE Engine."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from charavoice_engine.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from charavoice_engine.watermark.protocol import normalize_extension_hex


class Chunk(BaseModel):
    """One text or full-control unit inside a mixed-language request.

    Text chunks use language + text.
    Full-control chunks use language + surface_text + phones + tones + word2ph.
    Full-control phones/tones/word2ph are provided before adding leading/trailing
    "_" and before add_blank intersperse.
    """

    model_config = ConfigDict(extra="forbid")

    language: Languages = Field(..., description="Chunk language")
    text: str | None = Field(default=None, min_length=1, description="Text to synthesize for this chunk")
    surface_text: str | None = Field(default=None, min_length=1, description="Written text used for BERT alignment")
    phones: list[str] | None = Field(
        default=None, min_length=1, description="Phone symbols without leading/trailing '_'"
    )
    tones: list[int] | None = Field(
        default=None, min_length=1, description="Tone values without leading/trailing silence"
    )
    word2ph: list[int] | None = Field(default=None, min_length=1, description="Phone counts per surface_text character")

    @model_validator(mode="after")
    def _check_mode_and_alignment(self) -> Chunk:
        has_text = self.text is not None
        full_fields = (self.surface_text, self.phones, self.tones, self.word2ph)
        has_full_control = all(value is not None for value in full_fields)
        has_partial_full_control = any(value is not None for value in full_fields)
        if has_partial_full_control and not has_full_control:
            raise ValueError("Full-control chunks require surface_text, phones, tones, and word2ph")
        if has_text == has_full_control or (has_text and has_partial_full_control):
            raise ValueError("Each chunk must provide exactly one of 'text' or full-control fields")
        if self.language not in (Languages.JP, Languages.EN):
            raise ValueError("chunk language must be JP or EN")
        if not has_full_control:
            return self
        assert self.surface_text is not None
        assert self.phones is not None
        assert self.tones is not None
        assert self.word2ph is not None
        if len(self.phones) != len(self.tones):
            raise ValueError("len(phones) must equal len(tones)")
        if len(self.word2ph) != len(self.surface_text):
            raise ValueError("len(word2ph) must equal len(surface_text)")
        if any(value < 0 for value in self.word2ph):
            raise ValueError("word2ph values must be non-negative")
        if sum(self.word2ph) != len(self.phones):
            raise ValueError("sum(word2ph) must equal len(phones)")
        if any(phone == "_" for phone in self.phones):
            raise ValueError("phones must not include leading/trailing '_'")
        return self


class TTSRequest(BaseModel):
    """TTS synthesis request."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, min_length=1, description="Text to synthesize")
    chunks: list[Chunk] | None = Field(
        default=None,
        min_length=1,
        description="Mixed-language text/full-control chunks for one inference request",
    )
    model_name: str | None = Field(None, description="Model name (priority over model_id)")
    model_id: int = Field(0, description="Model ID from /models/info")
    speaker_name: str | None = Field(None, description="Speaker name (priority over speaker_id)")
    speaker_id: int = Field(0, description="Speaker ID from config.json spk2id")
    noise: float = Field(DEFAULT_NOISE, ge=0.0, description="Sample noise scale")
    length: float = Field(DEFAULT_LENGTH, gt=0.0, description="Speech speed (1.0 = normal)")
    language: Languages = Field(Languages.JP, description="Text language")
    auto_split: bool = Field(DEFAULT_LINE_SPLIT, description="Split by newlines")
    split_interval: float = Field(DEFAULT_SPLIT_INTERVAL, ge=0.0, description="Silence between splits (sec)")
    assist_text: str | None = Field(None, description="Assist text for emotion reference")
    assist_text_weight: float = Field(DEFAULT_ASSIST_TEXT_WEIGHT, ge=0.0, le=1.0)
    style: str | None = Field(DEFAULT_STYLE, description="Voice style name")
    style_weight: float = Field(DEFAULT_STYLE_WEIGHT, ge=0.0)
    streaming: bool = Field(False, description="Return a streaming audio response")
    format: str = Field(
        "ogg",
        pattern="^(ogg|wav|pcm)$",
        description="Output format: ogg (Opus), wav, or streaming-only pcm",
    )
    watermark_core16: int | None = Field(None, ge=0, le=65535, description="Optional 16-bit core watermark word")
    watermark_extension: str | None = Field(
        None,
        description="Optional extension payload as hex string. Excess bytes are dropped to fit the audio length.",
    )

    @model_validator(mode="after")
    def _check_text_or_chunks(self) -> TTSRequest:
        has_text = self.text is not None
        has_chunks = self.chunks is not None
        if has_text == has_chunks:
            raise ValueError("Exactly one of 'text' or 'chunks' must be provided")
        if self.language not in (Languages.JP, Languages.EN):
            raise ValueError("language must be JP or EN")
        if self.streaming and self.format == "wav":
            raise ValueError("streaming=true supports format='ogg' or format='pcm'")
        if not self.streaming and self.format == "pcm":
            raise ValueError("format='pcm' requires streaming=true")
        if self.watermark_extension is not None:
            self.watermark_extension = normalize_extension_hex(self.watermark_extension)
        return self


class TTSBatchRequest(BaseModel):
    """Batch TTS request containing multiple synthesis items."""

    items: list[TTSRequest] = Field(..., min_length=1, description="List of TTS requests")


class FullControlConvertRequest(BaseModel):
    """Request to convert text chunks into full-control chunks."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(None, min_length=1, description="Text to convert")
    chunks: list[Chunk] | None = Field(
        None,
        min_length=1,
        description="Chunks to convert; existing full-control chunks are preserved",
    )
    language: Languages = Field(Languages.JP, description="Language for top-level text")
    convert_languages: list[Languages] = Field(
        default_factory=lambda: [Languages.JP],
        description="Text chunk languages to convert to full-control",
    )

    @model_validator(mode="after")
    def _check_text_or_chunks(self) -> FullControlConvertRequest:
        has_text = self.text is not None
        has_chunks = self.chunks is not None
        if has_text == has_chunks:
            raise ValueError("Exactly one of 'text' or 'chunks' must be provided")
        if self.language not in (Languages.JP, Languages.EN):
            raise ValueError("language must be JP or EN")
        return self


class FullControlConvertResponse(BaseModel):
    """Converted chunks that can be sent to /voice."""

    chunks: list[Chunk] = Field(..., min_length=1)
    warnings: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    model_id: int | None = None
    model_name: str | None = None
    config_file: str | None = None
    model_file: str | None = None
    device: str
    spk2id: dict[str, int]
    id2spk: dict[int, str]
    style2id: dict[str, int]
    supported_languages: list[str] = Field(default_factory=list)
    speakers: list[dict[str, Any]] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class ModelsInfoResponse(BaseModel):
    """Response for /models/info endpoint."""

    models: dict[str, ModelInfo] = Field(default_factory=dict)


class ServerStatus(BaseModel):
    """Server status response."""

    devices: list[str]
    cpu_percent: float
    memory_total: int
    memory_available: int
    memory_used: int
    memory_percent: float
    gpu: list[dict[str, Any]] = Field(default_factory=list)


class EngineInfo(BaseModel):
    """Inference engine information."""

    backend: str
    bert_cache_enabled: bool
    loaded_models: int
