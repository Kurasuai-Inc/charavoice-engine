"""NLP dispatch layer for charavoice_engine.

Provides language-dispatched text cleaning, phoneme conversion,
and BERT feature extraction (ONNX only).

Ported from style_bert_vits2.nlp.__init__
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from charavoice_engine.constants import Languages
from charavoice_engine.nlp import onnx_bert_models
from charavoice_engine.nlp.onnx_utils import get_onnx_device_options
from charavoice_engine.nlp.symbols import (
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    SYMBOLS,
)

_logger = logging.getLogger(__name__)

__symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}


def _align_hidden_to_chars(
    hidden: NDArray[Any],
    tokenizer: Any,
    text: str,
    n_word2ph: int,
) -> NDArray[Any]:
    """Aggregate token-level BERT hidden states to character-level using offset mapping.

    When BERT produces more tokens than characters (e.g. a rare character splits
    into multiple tokens), this function merges those hidden states so the output
    has exactly n_word2ph entries (= len(norm_text) + 2).

    Args:
        hidden: [n_tokens, dim] BERT hidden states (including [CLS]/[SEP]).
        tokenizer: The BERT tokenizer (must support return_offsets_mapping).
        text: The preprocessed text that was fed to BERT.
        n_word2ph: Expected output length (len(word2ph)).

    Returns:
        [n_word2ph, dim] character-level hidden states.
    """
    n_tokens = hidden.shape[0]
    _logger.warning(
        "BERT token/char mismatch: %d tokens vs %d word2ph for text '%.50s', using offset mapping to align",
        n_tokens,
        n_word2ph,
        text,
    )

    try:
        encoded = tokenizer(text, return_offsets_mapping=True, return_tensors=None)
        offsets: list[tuple[int, int]] = encoded["offset_mapping"]
    except Exception as exc:
        _logger.warning(
            "BERT tokenizer offset mapping unavailable for text '%.50s'; using proportional hidden-state alignment: %s",
            text,
            exc,
        )
        return _align_hidden_to_chars_proportionally(hidden, n_word2ph)

    dim = hidden.shape[1]
    char_hidden = np.zeros((n_word2ph, dim), dtype=hidden.dtype)
    char_counts = np.zeros(n_word2ph, dtype=np.int32)

    for tok_idx, (start, end) in enumerate(offsets):
        if tok_idx >= n_tokens:
            break
        if start == end:
            # Special token ([CLS] or [SEP])
            if tok_idx == 0:
                char_hidden[0] += hidden[tok_idx]
                char_counts[0] += 1
            else:
                char_hidden[-1] += hidden[tok_idx]
                char_counts[-1] += 1
        else:
            # Content token: map to character position (+1 for start marker)
            char_idx = start + 1
            if char_idx < n_word2ph - 1:
                char_hidden[char_idx] += hidden[tok_idx]
                char_counts[char_idx] += 1

    # Average multi-token characters, fill gaps
    for i in range(n_word2ph):
        if char_counts[i] > 1:
            char_hidden[i] /= char_counts[i]
        elif char_counts[i] == 0:
            char_hidden[i] = char_hidden[max(0, i - 1)]

    return char_hidden


def _align_hidden_to_chars_proportionally(
    hidden: NDArray[Any],
    n_word2ph: int,
) -> NDArray[Any]:
    """Fallback token-to-character alignment for tokenizers without offset mapping."""
    n_tokens = hidden.shape[0]
    dim = hidden.shape[1]
    if n_word2ph <= 0:
        return np.zeros((0, dim), dtype=hidden.dtype)
    if n_tokens <= 0:
        return np.zeros((n_word2ph, dim), dtype=hidden.dtype)
    if n_word2ph == 1:
        return hidden[:1].copy()

    char_hidden = np.zeros((n_word2ph, dim), dtype=hidden.dtype)
    char_hidden[0] = hidden[0]
    char_hidden[-1] = hidden[-1]

    content_target_count = max(0, n_word2ph - 2)
    if content_target_count == 0:
        return char_hidden

    content_hidden = hidden[1:-1] if n_tokens > 2 else hidden
    source_count = content_hidden.shape[0]
    if source_count <= 0:
        char_hidden[1:-1] = hidden[min(1, n_tokens - 1)]
        return char_hidden

    if content_target_count == 1:
        char_hidden[1] = content_hidden[source_count // 2]
        return char_hidden

    for target_index in range(content_target_count):
        source_index = round(target_index * (source_count - 1) / (content_target_count - 1))
        char_hidden[target_index + 1] = content_hidden[source_index]
    return char_hidden


def _run_bert_onnx(
    text: str,
    language: Languages,
    session: onnxruntime.InferenceSession,
    tokenizer: Any,
    input_names: list[str],
    output_name: str,
    device_type: str,
    device_id: int,
    run_options: onnxruntime.RunOptions | None,
) -> NDArray[Any]:
    """Run a single BERT forward pass via IO binding and return the output."""
    with onnx_bert_models.tokenizer_lock(language):
        inputs = tokenizer(text, return_tensors="np")
    input_tensor = [inputs[name].astype(np.int64) for name in input_names]  # type: ignore[union-attr]

    io_binding = session.io_binding()
    for name, value in zip(input_names, input_tensor, strict=True):
        gpu_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(value, device_type, device_id)
        io_binding.bind_ortvalue_input(name, gpu_tensor)
    io_binding.bind_output(output_name, device_type)
    session.run_with_iobinding(io_binding, run_options=run_options)
    return io_binding.get_outputs()[0].numpy()  # type: ignore[no-any-return]


def preprocess_text_for_bert(text: str, language: Languages) -> str:
    """Preprocess text for BERT tokenization (language-specific).

    For Japanese: runs text_to_sep_kata() to convert to pronounceable form.
    For other languages: returns text as-is (tokenizer handles it directly).
    """
    if language == Languages.JP:
        from charavoice_engine.nlp.japanese.g2p import text_to_sep_kata

        return "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if language == Languages.EN:
        return text
    raise ValueError(f"Language {language} not supported")


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    language: Languages,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    skip_preprocess: bool = False,
) -> NDArray[Any]:
    """
    テキストから BERT の特徴量を抽出する (ONNX 推論)

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語
        onnx_providers: ONNX 推論で利用する ExecutionProvider
        assist_text: 補助テキスト
        assist_text_weight: 補助テキストの重み
        skip_preprocess: True の場合、text の前処理 (text_to_sep_kata) をスキップする。
            full-control API で surface_text が既に構築済みの場合に使用。

    Returns:
        NDArray[Any]: BERT の特徴量
    """

    if language not in (Languages.JP, Languages.EN):
        raise ValueError(f"Language {language} not supported")

    # JP のみ text_to_sep_kata で前処理 (word2ph との整合性を保つため)
    if not skip_preprocess:
        text = preprocess_text_for_bert(text, language)
    if assist_text:
        assist_text = preprocess_text_for_bert(assist_text, language)

    # トークナイザーとモデルの読み込み
    tokenizer = onnx_bert_models.load_tokenizer(language)
    session = onnx_bert_models.load_model(language=language, onnx_providers=onnx_providers)
    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    # 入力テンソルの転送に使用するデバイス種別, デバイス ID, 実行オプションを取得
    device_type, device_id, run_options = get_onnx_device_options(session)

    # text から BERT 特徴量を抽出
    res = _run_bert_onnx(
        text, language, session, tokenizer, input_names, output_name, device_type, device_id, run_options
    )

    # assist_text がある場合はその特徴量の平均も抽出
    style_res_mean = None
    if assist_text:
        style_res = _run_bert_onnx(
            assist_text, language, session, tokenizer, input_names, output_name, device_type, device_id, run_options
        )
        style_res_mean = np.mean(style_res, axis=0)

    # トークン数と word2ph 長が異なる場合 (特殊文字で 1文字→複数トークン)、
    # offset mapping を使ってトークンレベル→文字レベルに集約する
    if len(word2ph) != res.shape[0]:
        with onnx_bert_models.tokenizer_lock(language):
            res = _align_hidden_to_chars(res, tokenizer, text, len(word2ph))

    # 各 word の BERT 特徴量を phone 数ぶん複製して展開
    phone_level_feature = []
    for i in range(len(word2ph)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                np.tile(res[i], (word2ph[i], 1)) * (1 - assist_text_weight)
                + np.tile(style_res_mean, (word2ph[i], 1)) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(res[i], (word2ph[i], 1))
        phone_level_feature.append(repeat_feature)

    return np.concatenate(phone_level_feature, axis=0).T  # type: ignore[no-any-return]


def clean_text(
    text: str,
    language: Languages,
    raise_yomi_error: bool = False,
) -> tuple[str, list[str], list[int], list[int]]:
    """
    テキストをクリーニングし、音素に変換する

    Args:
        text: クリーニングするテキスト
        language: テキストの言語
        raise_yomi_error: False の場合、読めない文字が消えたような扱いとして処理される

    Returns:
        tuple[str, list[str], list[int], list[int]]: クリーニングされたテキストと、音素・アクセント・word2ph のリスト
    """

    if language == Languages.JP:
        from charavoice_engine.nlp.japanese.g2p import g2p
        from charavoice_engine.nlp.japanese.normalizer import normalize_text

        norm_text = normalize_text(text)
        phones, tones, word2ph, norm_text = g2p(norm_text, raise_yomi_error=raise_yomi_error)
    elif language == Languages.EN:
        from charavoice_engine.nlp.english.g2p import g2p
        from charavoice_engine.nlp.english.normalizer import normalize_text

        norm_text = normalize_text(text)
        phones, tones, word2ph = g2p(norm_text)
    else:
        raise ValueError(f"Language {language} not supported")

    return norm_text, phones, tones, word2ph


def cleaned_text_to_sequence(
    cleaned_phones: list[str], tones: list[int], language: Languages
) -> tuple[list[int], list[int], list[int]]:
    """
    音素リスト・アクセントリスト・言語を、テキスト内の対応する ID に変換する
    """

    phones = [__symbol_to_id[symbol] for symbol in cleaned_phones]
    tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]
    lang_id = LANGUAGE_ID_MAP[language]
    lang_ids = [lang_id for i in phones]

    return phones, tones, lang_ids
