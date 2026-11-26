"""ONNX 版 BERT モデルをロード/取得するためのモジュール。

style_bert_vits2.nlp.onnx_bert_models から移植。
loguru → logging、style_bert_vits2.constants → charavoice_engine.constants に変換。
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import onnxruntime
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    DebertaV2TokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from charavoice_engine.constants import DEFAULT_ONNX_BERT_MODEL_PATHS, Languages

logger = logging.getLogger(__name__)


# 各言語ごとのロード済みの BERT モデルを格納する辞書
__loaded_models: dict[Languages, onnxruntime.InferenceSession] = {}

# 各言語ごとのロード済みの BERT トークナイザーを格納する辞書
__loaded_tokenizers: dict[
    Languages,
    PreTrainedTokenizer | PreTrainedTokenizerFast | DebertaV2TokenizerFast,
] = {}

# Hugging Face fast tokenizers mutate padding/truncation state during calls and
# can raise "Already borrowed" when shared across request threads. Keep the
# tokenizer objects cached, but serialize actual tokenizer calls per language.
__tokenizer_locks: dict[Languages, threading.RLock] = {}


def tokenizer_lock(language: Languages) -> threading.RLock:
    """Return the per-language lock guarding tokenizer calls."""
    if language not in __tokenizer_locks:
        __tokenizer_locks[language] = threading.RLock()
    return __tokenizer_locks[language]


def load_model(
    language: Languages,
    pretrained_model_name_or_path: str | None = None,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]] = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
    cache_dir: str | None = None,
    revision: str = "main",
    enable_cpu_mem_arena: bool | None = None,
) -> onnxruntime.InferenceSession:  # fmt: skip
    """
    指定された言語の ONNX 版 BERT モデルをロードし、ロード済みの ONNX 版 BERT モデルを返す。
    一度ロードされていれば、ロード済みの ONNX 版 BERT モデルを即座に返す。
    """

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_models:
        return __loaded_models[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_ONNX_BERT_MODEL_PATHS[language].exists(), \
            f"The default {language.name} ONNX BERT model does not exist on the file system. Please specify the path to the pre-trained model."  # fmt: skip
        pretrained_model_name_or_path = str(DEFAULT_ONNX_BERT_MODEL_PATHS[language])

    # pretrained_model_name_or_path に Hugging Face のリポジトリ名が指定された場合 (aaaa/bbbb のフォーマットを想定):
    if len(pretrained_model_name_or_path.split("/")) == 2:
        model_path = Path(
            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model_fp16.onnx",
                cache_dir=cache_dir,
                revision=revision,
            )
        )
        # 英語用 BERT のみ、spm.model もダウンロードする
        if language == Languages.EN:
            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="spm.model",
                cache_dir=cache_dir,
                revision=revision,
            )
    else:
        model_path = Path(pretrained_model_name_or_path).resolve() / "model_fp16.onnx"

    # 推論時に一番優先される ExecutionProvider の名前を取得
    assert len(onnx_providers) > 0
    first_provider_name = onnx_providers[0] if type(onnx_providers[0]) is str else onnx_providers[0][0]

    # 推論セッションの設定
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL  # fmt: skip
    sess_options.log_severity_level = 3
    onnxruntime.set_default_logger_severity(3)

    # CPU 推論時のみ enable_cpu_mem_arena を無効化
    if enable_cpu_mem_arena is not None:
        sess_options.enable_cpu_mem_arena = enable_cpu_mem_arena
    elif first_provider_name == "CPUExecutionProvider":
        sess_options.enable_cpu_mem_arena = False

    # BERT モデルをロードし、辞書に格納して返す
    start_time = time.time()
    __loaded_models[language] = onnxruntime.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=onnx_providers,
    )
    logger.info(
        f"Loaded the {language.name} ONNX BERT model from {pretrained_model_name_or_path} ({time.time() - start_time:.2f}s)"
    )

    return __loaded_models[language]


def load_tokenizer(
    language: Languages,
    pretrained_model_name_or_path: str | None = None,
    cache_dir: str | None = None,
    revision: str = "main",
) -> PreTrainedTokenizer | PreTrainedTokenizerFast | DebertaV2TokenizerFast:
    """
    指定された言語の ONNX 版 BERT トークナイザーをロードし、ロード済みの ONNX 版 BERT トークナイザーを返す。
    一度ロードされていれば、ロード済みの ONNX 版 BERT トークナイザーを即座に返す。
    """

    with tokenizer_lock(language):
        # すでにロード済みの場合はそのまま返す
        if language in __loaded_tokenizers:
            return __loaded_tokenizers[language]

        # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
        if pretrained_model_name_or_path is None:
            assert DEFAULT_ONNX_BERT_MODEL_PATHS[language].exists(), \
                f"The default {language.name} BERT tokenizer does not exist on the file system. Please specify the path to the pre-trained model."  # fmt: skip
            pretrained_model_name_or_path = str(DEFAULT_ONNX_BERT_MODEL_PATHS[language])

        # BERT トークナイザーをロードし、辞書に格納して返す
        if language == Languages.EN:
            __loaded_tokenizers[language] = DebertaV2TokenizerFast.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                revision=revision,
                fix_mistral_regex=True,
            )
        else:
            __loaded_tokenizers[language] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                revision=revision,
                use_fast=True,
            )
        logger.info(f"Loaded the {language.name} ONNX BERT tokenizer from {pretrained_model_name_or_path}")

    return __loaded_tokenizers[language]


def is_model_loaded(language: Languages) -> bool:
    """指定された言語の ONNX 版 BERT モデルがロード済みかどうかを返す。"""
    return language in __loaded_models


def is_tokenizer_loaded(language: Languages) -> bool:
    """指定された言語の ONNX 版 BERT トークナイザーがロード済みかどうかを返す。"""
    return language in __loaded_tokenizers


def unload_model(language: Languages) -> None:
    """指定された言語の ONNX 版 BERT モデルをアンロードする。"""
    if language in __loaded_models:
        del __loaded_models[language]
        gc.collect()
        logger.info(f"Unloaded the {language.name} ONNX BERT model")


def unload_tokenizer(language: Languages) -> None:
    """指定された言語の ONNX 版 BERT トークナイザーをアンロードする。"""
    if language in __loaded_tokenizers:
        del __loaded_tokenizers[language]
        gc.collect()
        logger.info(f"Unloaded the {language.name} ONNX BERT tokenizer")


def unload_all_models() -> None:
    """すべての ONNX 版 BERT モデルをアンロードする。"""
    for language in list(__loaded_models.keys()):
        unload_model(language)
    logger.info("Unloaded all ONNX BERT models")


def unload_all_tokenizers() -> None:
    """すべての ONNX 版 BERT トークナイザーをアンロードする。"""
    for language in list(__loaded_tokenizers.keys()):
        unload_tokenizer(language)
    logger.info("Unloaded all ONNX BERT tokenizers")
