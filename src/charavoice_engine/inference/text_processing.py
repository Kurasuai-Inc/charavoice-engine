"""Text preprocessing pipeline for TTS inference.

Standalone functions for G2P, BERT feature extraction, and numpy conversion.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from charavoice_engine.config import BertBackend, TensorRtConfig
from charavoice_engine.constants import Languages
from charavoice_engine.inference.engine_base import InferenceRequest
from charavoice_engine.nlp.bert_cache import BertFeatureCache
from charavoice_engine.nlp.fast_bert import (
    BertExtractItem,
    extract_bert_feature_fast,
    extract_bert_features_batch,
)

logger = logging.getLogger(__name__)

_BERT_SLOT_MAP: dict[Languages, str] = {
    Languages.JP: "ja_bert",
    Languages.EN: "en_bert",
}


@dataclass(frozen=True)
class G2pResult:
    """Pre-computed G2P result (phone/tone/language_ids/word2ph after intersperse)."""

    norm_text: str
    phone: list[int]
    tone: list[int]
    language_ids: list[int]
    word2ph: list[int]
    bert_text_override: str | None = None  # When set, use this for BERT instead of preprocess_text_for_bert(norm_text)

    def as_tuple(self) -> tuple[str, list[int], list[int], list[int], list[int]]:
        """Convert to tuple format expected by g2p_results_to_preprocessed."""
        return (self.norm_text, self.phone, self.tone, self.language_ids, self.word2ph)


@dataclass(frozen=True)
class TextChunk:
    """Text-only chunk in a mixed-language request."""

    language: Languages
    text: str


@dataclass(frozen=True)
class FullControlChunk:
    """Full-control chunk with explicit phone-level labels."""

    language: Languages
    surface_text: str
    phones: list[str]
    tones: list[int]
    word2ph: list[int]


InputChunk = TextChunk | FullControlChunk


def _route_bert_to_slots(
    bert_feature: NDArray[Any],
    language: Languages,
    n_phones: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Route BERT feature into bilingual JP-Extra (ja_bert, en_bert) slots."""
    target = _BERT_SLOT_MAP.get(language)
    if target is None:
        raise ValueError(f"Unsupported language: {language}")
    zeros = np.zeros((1024, n_phones), dtype=np.float32)
    return (
        bert_feature if target == "ja_bert" else zeros,
        bert_feature if target == "en_bert" else zeros,
    )


@dataclass(frozen=True)
class TextProcessingConfig:
    """Immutable config bundle for text preprocessing functions."""

    add_blank: bool
    max_seq_len: int
    device: str
    bert_backend: BertBackend
    onnx_providers: tuple[str | tuple[str, dict[str, Any]], ...]
    trt_config: TensorRtConfig | None = None


def run_g2p(
    text: str,
    language: Languages,
    config: TextProcessingConfig,
) -> tuple[str, list[int], list[int], list[int], list[int]]:
    """Run G2P only (no BERT extraction).

    Returns:
        (norm_text, phone, tone, language_ids, word2ph)
        phone/tone/language_ids are interspersed if add_blank is set.
    """
    from charavoice_engine.models.commons import intersperse
    from charavoice_engine.nlp import clean_text, cleaned_text_to_sequence

    norm_text, phone, tone, word2ph = clean_text(
        text,
        language,
        raise_yomi_error=False,
    )
    phone, tone, language_ids = cleaned_text_to_sequence(phone, tone, language)

    if config.add_blank:
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language_ids = intersperse(language_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    return norm_text, phone, tone, language_ids, word2ph


def check_split_and_run_g2p(
    text: str,
    language: Languages,
    config: TextProcessingConfig,
) -> list[tuple[str, int, G2pResult]]:
    """Check text length, split if needed, and run G2P in one pass.

    Combines check_and_split_text_with_len and run_g2p so that the
    Japanese frontend is called only once per text, eliminating the
    redundant second call in the engine.

    Returns:
        List of (original_text, seq_len_estimate, G2pResult) tuples.
    """
    from charavoice_engine.models.commons import intersperse
    from charavoice_engine.nlp import clean_text, cleaned_text_to_sequence
    from charavoice_engine.nlp.text_splitter import (
        estimate_seq_len,
        split_text_by_seq_len,
    )

    norm_text, phone_strs, tone_raw, word2ph = clean_text(
        text,
        language,
        raise_yomi_error=False,
    )

    seq_len = estimate_seq_len(word2ph, config.add_blank)

    if seq_len <= config.max_seq_len:
        # Text fits — complete G2P inline (no re-run needed)
        phone, tone, language_ids = cleaned_text_to_sequence(phone_strs, tone_raw, language)
        if config.add_blank:
            phone = intersperse(phone, 0)
            tone = intersperse(tone, 0)
            language_ids = intersperse(language_ids, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        return [(text, seq_len, G2pResult(norm_text, phone, tone, language_ids, word2ph))]

    # Text too long — split at word2ph boundaries, then run_g2p per chunk.
    # Re-G2P on a chunk may produce a slightly different seq_len than estimated
    # from the full-text word2ph because pyopenjtalk word boundaries are context-
    # dependent. Use a margin so re-G2P results still fit within max_seq_len.
    _SPLIT_MARGIN = 32  # phones of headroom for G2P divergence
    split_limit = config.max_seq_len - _SPLIT_MARGIN
    chunks = split_text_by_seq_len(
        norm_text,
        word2ph,
        max_seq_len=split_limit,
        add_blank=config.add_blank,
    )
    return [(c.text, 0, G2pResult(*run_g2p(c.text, language, config))) for c in chunks]


def check_split_and_prepare_full_control_chunks(
    chunks: list[FullControlChunk],
    config: TextProcessingConfig,
) -> list[tuple[str, int, G2pResult]]:
    """Build G2pResult objects from full-control chunks and split if needed.

    Each chunk provides surface_text for BERT/alignment and explicit
    phones/tones/word2ph labels. Input phones/tones/word2ph must not include
    leading/trailing "_" markers and must be pre-add_blank.

    Returns:
        List of (surface_text_concat, seq_len_estimate, G2pResult) tuples.
    """
    from charavoice_engine.nlp.text_splitter import estimate_seq_len

    if not chunks:
        raise ValueError("chunks must not be empty")
    language = chunks[0].language
    if any(chunk.language != language for chunk in chunks):
        raise ValueError("full-control chunk splitting requires a single language")

    g2p_result = _build_full_control_group(chunks, language, config)
    surface_text_concat = g2p_result.bert_text_override or g2p_result.norm_text

    seq_len = estimate_seq_len(g2p_result.word2ph, add_blank=False)  # already interspersed

    if seq_len <= config.max_seq_len:
        return [(surface_text_concat, seq_len, g2p_result)]

    # Too long — split at chunk boundaries.
    _SPLIT_MARGIN = 32
    split_limit = config.max_seq_len - _SPLIT_MARGIN

    # Compute per-chunk phone counts from the interspersed word2ph.
    # word2ph layout: [leading_"_", ...chars..., trailing_"_"]
    char_offset = 0
    chunk_seq_lens: list[int] = []
    for chunk in chunks:
        n_char = len(chunk.surface_text)
        # Sum phone counts for this chunk's characters in the interspersed word2ph
        # word2ph layout: [leading_"_", ...chars..., trailing_"_"]
        chunk_phones = sum(g2p_result.word2ph[1 + char_offset : 1 + char_offset + n_char])
        chunk_seq_lens.append(chunk_phones)
        char_offset += n_char

    # Greedy accumulation
    result: list[tuple[str, int, G2pResult]] = []
    current_group: list[FullControlChunk] = []
    current_phone_sum = 0
    # Account for leading/trailing "_" overhead per chunk
    overhead = g2p_result.word2ph[0] + g2p_result.word2ph[-1]  # typically 1+1=2, or 3+1=4 with intersperse

    for i, chunk in enumerate(chunks):
        chunk_cost = chunk_seq_lens[i]
        if current_group and (current_phone_sum + chunk_cost + overhead) > split_limit:
            # Flush current group
            g2p_chunk = _build_full_control_group(current_group, language, config)
            surface_chunk = "".join(s.surface_text for s in current_group)
            result.append((surface_chunk, 0, g2p_chunk))
            current_group = []
            current_phone_sum = 0
        current_group.append(chunk)
        current_phone_sum += chunk_cost

    # Flush remaining
    if current_group:
        g2p_chunk = _build_full_control_group(current_group, language, config)
        surface_chunk = "".join(s.surface_text for s in current_group)
        result.append((surface_chunk, 0, g2p_chunk))

    return result


def _build_full_control_group(
    chunks: list[FullControlChunk],
    language: Languages,
    config: TextProcessingConfig,
) -> G2pResult:
    """Build a G2pResult from a full-control chunk group."""
    from charavoice_engine.models.commons import intersperse
    from charavoice_engine.nlp import cleaned_text_to_sequence
    from charavoice_engine.nlp.symbols import EN_SYMBOLS, JP_SYMBOLS, NUM_EN_TONES, NUM_JP_TONES, PUNCTUATIONS

    if language == Languages.JP:
        allowed_phones = set(JP_SYMBOLS) | set(PUNCTUATIONS)
        max_tone = NUM_JP_TONES - 1
    elif language == Languages.EN:
        allowed_phones = set(EN_SYMBOLS) | set(PUNCTUATIONS)
        max_tone = NUM_EN_TONES - 1
    else:
        raise ValueError(f"Full-control input is not supported for {language}")

    surface_text_concat = "".join(chunk.surface_text for chunk in chunks)
    phone_strs: list[str] = []
    tone_raw: list[int] = []
    word2ph: list[int] = []

    for index, chunk in enumerate(chunks):
        _validate_full_control_chunk(chunk, index, allowed_phones, max_tone)
        phone_strs.extend(chunk.phones)
        tone_raw.extend(chunk.tones)
        word2ph.extend(chunk.word2ph)

    phone_strs = ["_", *phone_strs, "_"]
    tone_raw = [0, *tone_raw, 0]
    word2ph = [1, *word2ph, 1]

    phone, tone, language_ids = cleaned_text_to_sequence(phone_strs, tone_raw, language)

    if config.add_blank:
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language_ids = intersperse(language_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    return G2pResult(
        surface_text_concat,
        phone,
        tone,
        language_ids,
        word2ph,
        bert_text_override=surface_text_concat,
    )


def _validate_full_control_chunk(
    chunk: FullControlChunk,
    index: int,
    allowed_phones: set[str],
    max_tone: int,
) -> None:
    """Validate one full-control chunk before converting it to model IDs."""
    if len(chunk.phones) != len(chunk.tones):
        raise ValueError(f"chunks[{index}]: len(phones) must equal len(tones)")
    if len(chunk.word2ph) != len(chunk.surface_text):
        raise ValueError(f"chunks[{index}]: len(word2ph) must equal len(surface_text)")
    if any(value < 0 for value in chunk.word2ph):
        raise ValueError(f"chunks[{index}]: word2ph values must be non-negative")
    if sum(chunk.word2ph) != len(chunk.phones):
        raise ValueError(f"chunks[{index}]: sum(word2ph) must equal len(phones)")
    invalid_phones = sorted({phone for phone in chunk.phones if phone not in allowed_phones})
    if invalid_phones:
        raise ValueError(f"chunks[{index}]: unsupported phones: {invalid_phones}")
    invalid_tones = [tone for tone in chunk.tones if tone < 0 or tone > max_tone]
    if invalid_tones:
        raise ValueError(f"chunks[{index}]: tones must be in [0, {max_tone}]")


def check_and_split_text_with_len(
    text: str,
    language: Languages,
    config: TextProcessingConfig,
) -> list[tuple[str, int]]:
    """Check if text exceeds max_seq_len and split if needed.

    Runs G2P to get word2ph, estimates post-intersperse seq_len,
    and splits at word2ph character boundaries if it exceeds the limit.

    Returns:
        List of (chunk_text, seq_len_estimate) tuples.
    """
    from charavoice_engine.nlp import clean_text
    from charavoice_engine.nlp.text_splitter import (
        estimate_seq_len,
        split_text_by_seq_len,
    )

    norm_text, _phone_strs, _tone, word2ph = clean_text(
        text,
        language,
        raise_yomi_error=False,
    )

    seq_len = estimate_seq_len(word2ph, config.add_blank)
    if seq_len <= config.max_seq_len:
        return [(text, seq_len)]

    chunks = split_text_by_seq_len(
        norm_text,
        word2ph,
        max_seq_len=config.max_seq_len,
        add_blank=config.add_blank,
    )
    # Split chunks don't carry word2ph, so seq_len is unknown (0).
    # This is rare (text > max_seq_len) and the engine re-sorts anyway.
    return [(c.text, 0) for c in chunks]


def check_and_split_text(
    text: str,
    language: Languages,
    config: TextProcessingConfig,
) -> list[str]:
    """Check if text exceeds max_seq_len and split if needed.

    Wrapper around check_and_split_text_with_len that returns only text chunks.
    """
    return [chunk_text for chunk_text, _ in check_and_split_text_with_len(text, language, config)]


def get_text(
    text: str,
    language: Languages,
    config: TextProcessingConfig,
    bert_cache: BertFeatureCache | None,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    *,
    profile: bool = False,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], tuple[float, float]]:
    """G2P + BERT -> (ja_bert, en_bert, phones, tones, lang_ids, timings) numpy.

    Returns:
        Tuple of numpy arrays + timing:
            ja_bert: [1024, seq_len] float32
            en_bert: [1024, seq_len] float32
            phones: [seq_len] int64
            tones: [seq_len] int64
            lang_ids: [seq_len] int64
            timings: (g2p_ms, bert_ms) — zero if profile=False
    """
    import time

    from charavoice_engine.nlp import clean_text, cleaned_text_to_sequence

    t0 = time.perf_counter() if profile else 0.0

    norm_text, phone, tone, word2ph = clean_text(
        text,
        language,
        raise_yomi_error=False,
    )
    phone, tone, language_ids = cleaned_text_to_sequence(phone, tone, language)

    if config.add_blank:
        from charavoice_engine.models.commons import intersperse

        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language_ids = intersperse(language_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    t_g2p = time.perf_counter() if profile else 0.0

    # Check seq_len against max before expensive BERT extraction
    seq_len = len(phone)
    if seq_len > config.max_seq_len:
        raise ValueError(
            f"Text too long: seq_len={seq_len} exceeds max={config.max_seq_len} "
            f"(~{config.max_seq_len * 10 // 37} chars). "
            f"Use auto_split=true or shorten the text."
        )

    # BERT extraction (with cache, returns torch tensor)
    bert_ori = extract_bert_feature_fast(
        norm_text=norm_text,
        word2ph=word2ph,
        language=language,
        device=config.device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        cache=bert_cache,
        bert_backend=config.bert_backend,
        onnx_providers=config.onnx_providers,
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone)

    t_bert = time.perf_counter() if profile else 0.0

    # Convert to numpy for ONNX
    bert_ori_np = bert_ori.cpu().numpy().astype(np.float32)
    ja_bert, en_bert = _route_bert_to_slots(bert_ori_np, language, len(phone))

    phone_np = np.array(phone, dtype=np.int64)
    tone_np = np.array(tone, dtype=np.int64)
    language_ids_np = np.array(language_ids, dtype=np.int64)

    timings = ((t_g2p - t0) * 1000, (t_bert - t_g2p) * 1000) if profile else (0.0, 0.0)

    return ja_bert, en_bert, phone_np, tone_np, language_ids_np, timings


def _chunk_boundary_slice(index: int, total: int, *, add_blank: bool) -> slice:
    """Return the phone-axis slice that removes only internal chunk boundaries."""
    if total == 1:
        return slice(None)
    leading = 3 if add_blank else 1
    trailing = 2 if add_blank else 1
    start = 0 if index == 0 else leading
    stop = None if index == total - 1 else -trailing
    return slice(start, stop)


def _chunk_to_g2p_result(chunk: InputChunk, config: TextProcessingConfig) -> G2pResult:
    if isinstance(chunk, TextChunk):
        return G2pResult(*run_g2p(chunk.text, chunk.language, config))
    return _build_full_control_group([chunk], chunk.language, config)


def prepare_chunks_to_preprocessed(
    request: InferenceRequest,
    config: TextProcessingConfig,
    bert_cache: BertFeatureCache | None,
) -> dict[str, Any]:
    """Convert mixed-language chunks to one preprocessed model input.

    This keeps the synthesis path as a single model inference: per-chunk
    G2P/BERT features are concatenated along the phone axis without repeating
    chunk-boundary silence markers.
    """
    if not request.chunks:
        raise ValueError("chunks must not be empty")

    g2p_results = [_chunk_to_g2p_result(chunk, config) for chunk in request.chunks]
    bert_items = [
        BertExtractItem(
            norm_text=g2p.norm_text,
            word2ph=g2p.word2ph,
            language=chunk.language,
            assist_text=request.assist_text,
            assist_text_weight=request.assist_text_weight,
            bert_text=g2p.bert_text_override,
        )
        for chunk, g2p in zip(request.chunks, g2p_results, strict=True)
    ]
    bert_features = extract_bert_features_batch(
        items=bert_items,
        cache=bert_cache,
        onnx_providers=config.onnx_providers,
        trt_config=config.trt_config,
    )

    ja_parts: list[NDArray[Any]] = []
    en_parts: list[NDArray[Any]] = []
    phone_parts: list[NDArray[Any]] = []
    tone_parts: list[NDArray[Any]] = []
    lang_parts: list[NDArray[Any]] = []

    total = len(request.chunks)
    for index, (chunk, g2p, bert_ori) in enumerate(zip(request.chunks, g2p_results, bert_features, strict=True)):
        n_phones = len(g2p.phone)
        if bert_ori.shape[-1] != n_phones:
            raise ValueError(f"chunks[{index}]: BERT feature length {bert_ori.shape[-1]} != phone length {n_phones}")

        bert_ori_np = bert_ori.cpu().numpy().astype(np.float32)
        ja_bert, en_bert = _route_bert_to_slots(bert_ori_np, chunk.language, n_phones)
        phone_np = np.array(g2p.phone, dtype=np.int64)
        tone_np = np.array(g2p.tone, dtype=np.int64)
        lang_np = np.array(g2p.language_ids, dtype=np.int64)

        boundary = _chunk_boundary_slice(index, total, add_blank=config.add_blank)
        phone_parts.append(phone_np[boundary])
        tone_parts.append(tone_np[boundary])
        lang_parts.append(lang_np[boundary])
        ja_parts.append(ja_bert[:, boundary])
        en_parts.append(en_bert[:, boundary])

    phones = np.concatenate(phone_parts)
    if len(phones) > config.max_seq_len:
        raise ValueError(
            f"Chunks too long: seq_len={len(phones)} exceeds max={config.max_seq_len}. "
            "Split the request into smaller chunk groups."
        )

    return {
        "phones": phones,
        "tones": np.concatenate(tone_parts),
        "lang_ids": np.concatenate(lang_parts),
        "ja_bert": np.concatenate(ja_parts, axis=1),
        "en_bert": np.concatenate(en_parts, axis=1),
        "sid": request.speaker_id,
        "style_vec": request.style_vec.astype(np.float32),
        "length_scale": request.length_scale,
        "noise_scale": request.noise_scale,
    }


def preprocess_request_to_numpy(
    request: InferenceRequest,
    config: TextProcessingConfig,
    bert_cache: BertFeatureCache | None,
    *,
    profile: bool = False,
) -> dict[str, Any]:
    """InferenceRequest -> numpy dict (phones, tones, bilingual BERT, sid, style_vec, ...).

    Returns dict with keys:
        phones, tones, lang_ids: [seq_len] int64
        ja_bert, en_bert: [1024, seq_len] float32
        sid: int
        style_vec: [256] float32
        _g2p_ms, _bert_ms: float (only when profile=True)
    """
    ja_bert, en_bert, phones, tones, lang_ids, timings = get_text(
        text=request.text,
        language=request.language,
        config=config,
        bert_cache=bert_cache,
        assist_text=request.assist_text,
        assist_text_weight=request.assist_text_weight,
        profile=profile,
    )

    if request.skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if request.skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]

    result: dict[str, Any] = {
        "phones": phones,
        "tones": tones,
        "lang_ids": lang_ids,
        "ja_bert": ja_bert,
        "en_bert": en_bert,
        "sid": request.speaker_id,
        "style_vec": request.style_vec.astype(np.float32),
    }
    if profile:
        result["_g2p_ms"] = timings[0]
        result["_bert_ms"] = timings[1]
    return result


def g2p_results_to_preprocessed(
    requests: list[InferenceRequest],
    g2p_results: list[tuple[str, list[int], list[int], list[int], list[int]]],
    bert_cache: BertFeatureCache | None,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    trt_config: TensorRtConfig | None = None,
) -> list[dict[str, Any]]:
    """Run batched BERT extraction on pre-computed G2P results.

    This is the second half of batch preprocessing, separated so that
    G2P can run upfront (for seq_len sorting) while BERT extraction can be
    pipelined with net_g forward.
    """
    # 1. Build BertExtractItem list for batch BERT
    bert_items: list[BertExtractItem] = []
    for g2p, r in zip(g2p_results, requests, strict=True):
        # If the request has a G2pResult with bert_text_override, use it for BERT input
        bert_text_override = None
        if r.g2p_result is not None and r.g2p_result.bert_text_override is not None:
            bert_text_override = r.g2p_result.bert_text_override
        bert_items.append(
            BertExtractItem(
                norm_text=g2p[0],
                word2ph=g2p[4],
                language=r.language,
                assist_text=r.assist_text,
                assist_text_weight=r.assist_text_weight,
                bert_text=bert_text_override,
            )
        )

    # 2. Batched BERT extraction (single ONNX forward per language)
    bert_features = extract_bert_features_batch(
        items=bert_items,
        cache=bert_cache,
        onnx_providers=onnx_providers,
        trt_config=trt_config,
    )

    # 3. Assemble numpy dicts
    preprocessed: list[dict[str, Any]] = []
    for r, g2p, bert_ori in zip(requests, g2p_results, bert_features, strict=True):
        _norm_text, phone, tone, lang_ids, _word2ph = g2p
        n_phones = len(phone)

        assert bert_ori.shape[-1] == n_phones, f"BERT feature length {bert_ori.shape[-1]} != phone length {n_phones}"

        bert_ori_np = bert_ori.cpu().numpy().astype(np.float32)
        ja_bert, en_bert = _route_bert_to_slots(bert_ori_np, r.language, n_phones)

        phones_np = np.array(phone, dtype=np.int64)
        tones_np = np.array(tone, dtype=np.int64)
        lang_ids_np = np.array(lang_ids, dtype=np.int64)

        if r.skip_start:
            phones_np = phones_np[3:]
            tones_np = tones_np[3:]
            lang_ids_np = lang_ids_np[3:]
            ja_bert = ja_bert[:, 3:]
            en_bert = en_bert[:, 3:]
        if r.skip_end:
            phones_np = phones_np[:-2]
            tones_np = tones_np[:-2]
            lang_ids_np = lang_ids_np[:-2]
            ja_bert = ja_bert[:, :-2]
            en_bert = en_bert[:, :-2]

        preprocessed.append(
            {
                "phones": phones_np,
                "tones": tones_np,
                "lang_ids": lang_ids_np,
                "ja_bert": ja_bert,
                "en_bert": en_bert,
                "sid": r.speaker_id,
                "style_vec": r.style_vec.astype(np.float32),
                "length_scale": r.length_scale,
                "noise_scale": r.noise_scale,
            }
        )

    return preprocessed
