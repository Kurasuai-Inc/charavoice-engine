"""Optimized BERT feature extraction with caching and AMP.

Wraps the original extract_bert_feature() from style_bert_vits2.nlp.
Adds:
1. LRU cache lookup (eliminates ~22ms BERT forward on cache hit)
2. FP16 autocast for BERT forward pass (minor speedup on GPU)
3. ONNX BERT backend (CUDA EP FP16: ~5-10ms vs PyTorch ~22ms)
4. Batch ONNX BERT extraction (single forward for N items)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from charavoice_engine.config import BertBackend, TensorRtConfig
from charavoice_engine.constants import Languages
from charavoice_engine.nlp.bert_cache import BertFeatureCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class for batch BERT extraction
# ---------------------------------------------------------------------------


@dataclass
class BertExtractItem:
    """Input item for batched BERT feature extraction."""

    norm_text: str
    word2ph: list[int]
    language: Languages
    assist_text: str | None = None
    assist_text_weight: float = 0.7
    bert_text: str | None = None  # Pre-constructed BERT text (skip preprocess_text_for_bert when set)


def extract_bert_feature_fast(
    norm_text: str,
    word2ph: list[int],
    language: Languages,
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    cache: BertFeatureCache | None = None,
    bert_backend: BertBackend | None = None,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
) -> torch.Tensor:
    """Extract BERT features via ONNX with optional cache.

    Args:
        norm_text: Normalized text (output of clean_text).
        word2ph: Phone-per-character mapping.
        language: Text language.
        device: Target device.
        assist_text: Optional assist text for emotion guidance.
        assist_text_weight: Weight for assist text blending.
        cache: Optional BertFeatureCache instance.
        bert_backend: BERT inference backend.
        onnx_providers: ONNX Runtime providers.

    Returns:
        BERT feature tensor of shape [1024, phone_length].
    """
    # 1. Check cache
    if cache is not None:
        cached = cache.get(norm_text, str(language), assist_text, word2ph)
        if cached is not None:
            return cached

    # 2. Run ONNX BERT feature extraction
    feature = _extract_onnx(
        norm_text,
        word2ph,
        language,
        onnx_providers,
        assist_text,
        assist_text_weight,
    )

    # Ensure output is float32
    feature = feature.float()

    # 3. Store in cache
    if cache is not None:
        cache.put(norm_text, str(language), feature, assist_text, word2ph)

    return feature


def _extract_onnx(
    norm_text: str,
    word2ph: list[int],
    language: Languages,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]] | None,
    assist_text: str | None,
    assist_text_weight: float,
    skip_preprocess: bool = False,
) -> torch.Tensor:
    """Call the existing ONNX BERT feature extraction pipeline.

    Uses the local ONNX BERT feature extraction pipeline, which returns
    an ndarray; we convert to torch.Tensor for the cache storage format.
    """
    from charavoice_engine.nlp import extract_bert_feature_onnx

    if onnx_providers is None:
        onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    result = extract_bert_feature_onnx(
        norm_text,
        word2ph,
        language,
        onnx_providers,
        assist_text,
        assist_text_weight,
        skip_preprocess=skip_preprocess,
    )
    return torch.from_numpy(result)


# ---------------------------------------------------------------------------
# Batch BERT session cache (model_fp16_batch.onnx)
# ---------------------------------------------------------------------------

# Cached batch ONNX sessions: {language: InferenceSession}
_batch_bert_sessions: dict[Languages, Any] = {}
_batch_bert_lock = threading.Lock()


def _build_bert_trt_providers(
    trt_config: TensorRtConfig,
    language: Languages,
) -> list[str | tuple[str, dict[str, Any]]]:
    """Build TRT EP provider list for batch BERT model.

    Only includes input_ids and attention_mask in profile shapes.
    token_type_ids must NOT be included: even though TRT EP ignores
    non-existent inputs at runtime, the cache validation compares profile
    shapes and causes a full engine rebuild every startup if extra entries
    are present.
    """
    from pathlib import Path

    cache_dir = Path(trt_config.engine_cache_path) / f"bert_{language.name.lower()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = str(cache_dir)

    b_min, b_opt, b_max = trt_config.bert_batch_min, trt_config.bert_batch_opt, trt_config.bert_batch_max
    s_min, s_opt, s_max = trt_config.bert_seq_min, trt_config.bert_seq_opt, trt_config.bert_seq_max

    shapes_base = f"input_ids:{b_min}x{s_min},attention_mask:{b_min}x{s_min}"
    shapes_opt = f"input_ids:{b_opt}x{s_opt},attention_mask:{b_opt}x{s_opt}"
    shapes_max = f"input_ids:{b_max}x{s_max},attention_mask:{b_max}x{s_max}"

    trt_opts: dict[str, Any] = {
        "trt_fp16_enable": trt_config.fp16,
        "trt_max_workspace_size": trt_config.max_workspace_size,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_path,
        "trt_builder_optimization_level": 3,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache_path,
        "trt_profile_min_shapes": shapes_base,
        "trt_profile_opt_shapes": shapes_opt,
        "trt_profile_max_shapes": shapes_max,
    }

    return [("TensorrtExecutionProvider", trt_opts)]


def _load_batch_bert_session(
    language: Languages,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    trt_config: TensorRtConfig | None = None,
) -> Any:
    """Load or return cached batch ONNX BERT session (model_fp16_batch.onnx).

    If trt_config is provided and bert_trt_enabled, loads with TRT EP.
    Raises FileNotFoundError if the batch model file does not exist.
    """
    if language in _batch_bert_sessions:
        return _batch_bert_sessions[language]

    with _batch_bert_lock:
        # Double-check after acquiring lock (another thread may have loaded it)
        if language in _batch_bert_sessions:
            return _batch_bert_sessions[language]

        import onnxruntime

        from charavoice_engine.constants import DEFAULT_ONNX_BERT_MODEL_PATHS

        batch_model_path = DEFAULT_ONNX_BERT_MODEL_PATHS[language] / "model_fp16_batch.onnx"
        if not batch_model_path.exists():
            raise FileNotFoundError(
                f"Batch BERT model not found for {language.name}: {batch_model_path}. "
                "Run the BERT ONNX export step before starting CHARAVOICE Engine."
            )

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.log_severity_level = 3
        onnxruntime.set_default_logger_severity(3)

        # Determine if TRT EP should be used
        use_trt = trt_config is not None and trt_config.bert_trt_enabled
        if use_trt:
            assert trt_config is not None
            trt_providers = _build_bert_trt_providers(trt_config, language)
            logger.info("Loading batch BERT TRT session for %s from %s", language.name, batch_model_path)
            session = onnxruntime.InferenceSession(
                str(batch_model_path),
                sess_options=sess_options,
                providers=trt_providers,
            )
            actual_ep = session.get_providers()[0]
            if actual_ep != "TensorrtExecutionProvider":
                raise RuntimeError(
                    f"Batch BERT session provider mismatch for {language.name}: "
                    f"expected TensorrtExecutionProvider, got {session.get_providers()}"
                )
            logger.info("Batch BERT session for %s using %s", language.name, actual_ep)
        else:
            session = onnxruntime.InferenceSession(
                str(batch_model_path),
                sess_options=sess_options,
                providers=onnx_providers,
            )

        _batch_bert_sessions[language] = session
        logger.info("Loaded batch BERT session for %s from %s", language.name, batch_model_path)
        return session


def preload_batch_bert_session(
    language: Languages,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    trt_config: TensorRtConfig | None = None,
) -> Any:
    """Preload batch BERT session for the given language (public API for engine startup)."""
    return _load_batch_bert_session(language, onnx_providers, trt_config)


def warmup_batch_bert_sessions(trt_config: TensorRtConfig) -> None:
    """Warmup all loaded batch BERT TRT sessions with representative inputs.

    Public API for streaming_engine to call without accessing private state.
    """
    for lang, session in _batch_bert_sessions.items():
        if session is None:
            continue
        if session.get_providers()[0] != "TensorrtExecutionProvider":
            continue
        input_names = {inp.name for inp in session.get_inputs()}
        has_token_type_ids = "token_type_ids" in input_names
        for B in (1, trt_config.bert_batch_opt):
            S = trt_config.bert_seq_opt
            feeds: dict[str, Any] = {
                "input_ids": np.zeros((B, S), dtype=np.int64),
                "attention_mask": np.ones((B, S), dtype=np.int64),
            }
            if has_token_type_ids:
                feeds["token_type_ids"] = np.zeros((B, S), dtype=np.int64)
            session.run(None, feeds)
            logger.info("BERT TRT warmup complete for %s (B=%d, S=%d)", lang.name, B, S)


# ---------------------------------------------------------------------------
# Batch BERT feature extraction (ONNX only)
# ---------------------------------------------------------------------------


def extract_bert_features_batch(
    items: list[BertExtractItem],
    cache: BertFeatureCache | None = None,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
    trt_config: TensorRtConfig | None = None,
) -> list[torch.Tensor]:
    """Extract BERT features for multiple items via batched ONNX forward.

    Items with assist_text are extracted individually (rare in practice).
    Items without assist_text are grouped by language and batched together.

    Args:
        items: List of BertExtractItem, one per request.
        cache: Optional BertFeatureCache instance.
        onnx_providers: ONNX Runtime execution providers.

    Returns:
        List of BERT feature tensors, each [1024, phone_length], in the
        same order as the input items.
    """
    if not items:
        return []
    if onnx_providers is None:
        onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    n = len(items)
    results: list[torch.Tensor | None] = [None] * n

    # --- Phase 1: Cache lookup ---
    miss_indices: list[int] = []
    for i, item in enumerate(items):
        if cache is not None:
            cache_text = item.bert_text or item.norm_text
            cached = cache.get(cache_text, str(item.language), item.assist_text, item.word2ph)
            if cached is not None:
                results[i] = cached
                continue
        miss_indices.append(i)

    if not miss_indices:
        return results  # type: ignore[return-value]

    # --- Phase 2: Separate assist_text items (individual extraction) ---
    batch_indices: list[int] = []  # indices that can be batched
    for i in miss_indices:
        if items[i].assist_text is not None:
            has_bert_override = items[i].bert_text is not None
            bert_input: str = items[i].bert_text if has_bert_override else items[i].norm_text  # type: ignore[assignment]
            feature = _extract_onnx(
                bert_input,
                items[i].word2ph,
                items[i].language,
                onnx_providers,
                items[i].assist_text,
                items[i].assist_text_weight,
                skip_preprocess=has_bert_override,
            ).float()
            results[i] = feature
            if cache is not None:
                cache_text = items[i].bert_text or items[i].norm_text
                cache.put(
                    cache_text,
                    str(items[i].language),
                    feature,
                    items[i].assist_text,
                    items[i].word2ph,
                )
        else:
            batch_indices.append(i)

    if not batch_indices:
        return results  # type: ignore[return-value]

    # --- Phase 3: Group by language, then batch ONNX forward per group ---
    lang_groups: dict[Languages, list[int]] = {}
    for i in batch_indices:
        lang = items[i].language
        if lang not in lang_groups:
            lang_groups[lang] = []
        lang_groups[lang].append(i)

    for _lang, indices in lang_groups.items():
        group_items = [items[i] for i in indices]
        group_features = _extract_onnx_batch(group_items, onnx_providers, trt_config)

        for i, feature in zip(indices, group_features, strict=True):
            feature = feature.float()
            results[i] = feature
            if cache is not None:
                cache_text = items[i].bert_text or items[i].norm_text
                cache.put(cache_text, str(items[i].language), feature, word2ph=items[i].word2ph)

    return results  # type: ignore[return-value]


def _extract_onnx_batch(
    items: list[BertExtractItem],
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    trt_config: TensorRtConfig | None = None,
) -> list[torch.Tensor]:
    """Run ONNX BERT inference for a group of same-language items.

    If model_fp16_batch.onnx exists: true batched forward (single session.run
    with [N, max_seq, 1024] output), then slice per-item actual_len.

    All items must share the same language.

    Returns:
        List of torch.Tensor, each [1024, phone_length].
    """

    from charavoice_engine.nlp import onnx_bert_models

    language = items[0].language

    # --- 1. Preprocess texts ---
    preprocessed_texts: list[str] = []
    for item in items:
        if item.bert_text is not None:
            preprocessed_texts.append(item.bert_text)
        else:
            preprocessed_texts.append(_preprocess_text_for_bert(item.norm_text, item.language))

    # --- 2. Load tokenizer (shared for both batch and loop paths) ---
    tokenizer = onnx_bert_models.load_tokenizer(language)

    # --- 3. Try batch session ---
    batch_session = _load_batch_bert_session(language, onnx_providers, trt_config)

    return _run_batch_forward(
        items,
        preprocessed_texts,
        tokenizer,
        batch_session,
        onnx_providers,
        trt_config,
    )


def _run_batch_forward(
    items: list[BertExtractItem],
    preprocessed_texts: list[str],
    tokenizer: Any,
    session: Any,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    trt_config: TensorRtConfig | None = None,
) -> list[torch.Tensor]:
    """True batched BERT forward: one session.run for N items.

    Uses model_fp16_batch.onnx which outputs [batch, seq_len, 1024].
    """
    import onnxruntime

    from charavoice_engine.nlp.onnx_utils import get_onnx_device_options

    input_names = {inp.name for inp in session.get_inputs()}
    output_name = session.get_outputs()[0].name
    device_type, device_id, run_options = get_onnx_device_options(session)
    has_token_type_ids = "token_type_ids" in input_names

    from charavoice_engine.nlp import onnx_bert_models

    # Batch tokenize with padding. Fast tokenizers are not thread-safe when
    # padding/truncation state is changed, so serialize tokenizer calls.
    with onnx_bert_models.tokenizer_lock(items[0].language):
        encoded = tokenizer(
            preprocessed_texts,
            padding=True,
            return_tensors="np",
            return_attention_mask=True,
        )

        # Record actual (pre-padding) lengths per item for later slicing
        actual_lengths: list[int] = []
        for text in preprocessed_texts:
            single_enc = tokenizer(text, return_tensors=None)
            actual_lengths.append(len(single_enc["input_ids"]))

    # Build IO binding for batched forward
    ids = encoded["input_ids"].astype(np.int64)
    mask = encoded["attention_mask"].astype(np.int64)

    # Ensure minimum sequence length for TRT profile (bert_seq_min)
    min_seq = trt_config.bert_seq_min if trt_config is not None else 0
    if min_seq > 0 and ids.shape[1] < min_seq:
        pad_width = min_seq - ids.shape[1]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids = np.pad(ids, ((0, 0), (0, pad_width)), constant_values=pad_id)
        mask = np.pad(mask, ((0, 0), (0, pad_width)), constant_values=0)

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input(
        "input_ids",
        onnxruntime.OrtValue.ortvalue_from_numpy(ids, device_type, device_id),
    )
    io_binding.bind_ortvalue_input(
        "attention_mask",
        onnxruntime.OrtValue.ortvalue_from_numpy(mask, device_type, device_id),
    )
    if has_token_type_ids:
        tids_raw = encoded.get("token_type_ids")
        if tids_raw is not None:
            tids = tids_raw.astype(np.int64)
            # Pad token_type_ids to match ids shape (after min_seq padding)
            if tids.shape[1] < ids.shape[1]:
                tids = np.pad(tids, ((0, 0), (0, ids.shape[1] - tids.shape[1])), constant_values=0)
        else:
            tids = np.zeros_like(ids)
        io_binding.bind_ortvalue_input(
            "token_type_ids",
            onnxruntime.OrtValue.ortvalue_from_numpy(tids, device_type, device_id),
        )

    io_binding.bind_output(output_name, device_type)
    session.run_with_iobinding(io_binding, run_options=run_options)

    # Output: [batch, max_seq, 1024]
    hidden_batch = io_binding.get_outputs()[0].numpy()

    # Slice per item and expand via word2ph
    result_features: list[torch.Tensor] = []
    for idx, item in enumerate(items):
        actual_len = actual_lengths[idx]
        hidden = hidden_batch[idx, :actual_len, :]  # [actual_len, 1024]

        # Handle token/char mismatch (rare: special chars producing multiple tokens)
        if len(item.word2ph) != actual_len:
            from charavoice_engine.nlp import _align_hidden_to_chars

            with onnx_bert_models.tokenizer_lock(item.language):
                hidden = _align_hidden_to_chars(hidden, tokenizer, preprocessed_texts[idx], len(item.word2ph))

        phone_level_feature = []
        for i in range(len(item.word2ph)):
            repeat_feature = np.tile(hidden[i], (item.word2ph[i], 1))
            phone_level_feature.append(repeat_feature)

        phone_level_feature = np.concatenate(phone_level_feature, axis=0)
        result_features.append(torch.from_numpy(phone_level_feature.T.copy()))

    return result_features


def _preprocess_text_for_bert(norm_text: str, language: Languages) -> str:
    """Preprocess text for BERT tokenization (language-specific).

    Delegates to the canonical implementation in nlp.__init__.
    """
    from charavoice_engine.nlp import preprocess_text_for_bert

    return preprocess_text_for_bert(norm_text, language)
