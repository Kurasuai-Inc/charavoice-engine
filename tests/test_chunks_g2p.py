from typing import Any, cast

import numpy as np
import pytest
import torch

from charavoice_engine.config import BertBackend
from charavoice_engine.constants import Languages
from charavoice_engine.inference.engine_base import InferenceRequest
from charavoice_engine.inference.text_processing import (
    FullControlChunk,
    TextProcessingConfig,
    check_split_and_prepare_full_control_chunks,
    prepare_chunks_to_preprocessed,
)


def _config(*, add_blank: bool = False, max_seq_len: int = 1024) -> TextProcessingConfig:
    return TextProcessingConfig(
        add_blank=add_blank,
        max_seq_len=max_seq_len,
        device="cpu",
        bert_backend=BertBackend.ONNX_CUDA,
        onnx_providers=("CPUExecutionProvider",),
    )


def test_jp_full_control_uses_surface_text_for_bert_override() -> None:
    result = check_split_and_prepare_full_control_chunks(
        [
            FullControlChunk(
                language=Languages.JP,
                surface_text="今日",
                phones=["ky", "o:"],
                tones=[1, 0],
                word2ph=[1, 1],
            )
        ],
        _config(),
    )

    assert len(result) == 1
    text, seq_len, g2p_result = result[0]
    assert text == "今日"
    assert seq_len == len(g2p_result.phone)
    assert g2p_result.norm_text == "今日"
    assert g2p_result.bert_text_override == "今日"
    assert len(g2p_result.word2ph) == len("今日") + 2
    assert sum(g2p_result.word2ph) == len(g2p_result.phone)


def test_full_control_accepts_zero_word2ph_entries() -> None:
    result = check_split_and_prepare_full_control_chunks(
        [
            FullControlChunk(
                language=Languages.JP,
                surface_text="今日",
                phones=["ky", "o:"],
                tones=[1, 0],
                word2ph=[2, 0],
            )
        ],
        _config(),
    )

    text, seq_len, g2p_result = result[0]
    assert text == "今日"
    assert seq_len == len(g2p_result.phone)
    assert g2p_result.word2ph == [1, 2, 0, 1]
    assert sum(g2p_result.word2ph) == len(g2p_result.phone)


def test_full_control_preserves_explicit_english_pronunciation() -> None:
    sequel = check_split_and_prepare_full_control_chunks(
        [
            FullControlChunk(
                language=Languages.EN,
                surface_text="SQL",
                phones=["s", "iy", "k", "w", "ah", "l"],
                tones=[0, 1, 0, 0, 1, 0],
                word2ph=[2, 2, 2],
            )
        ],
        _config(),
    )[0][2]
    spelled = check_split_and_prepare_full_control_chunks(
        [
            FullControlChunk(
                language=Languages.EN,
                surface_text="SQL",
                phones=["eh", "s", "k", "y", "uw", "eh", "l"],
                tones=[1, 0, 0, 0, 1, 1, 0],
                word2ph=[2, 3, 2],
            )
        ],
        _config(),
    )[0][2]

    assert sequel.bert_text_override == "SQL"
    assert spelled.bert_text_override == "SQL"
    assert sequel.phone != spelled.phone
    assert sum(sequel.word2ph) == len(sequel.phone)
    assert sum(spelled.word2ph) == len(spelled.phone)


def test_full_control_add_blank_updates_alignment() -> None:
    result = check_split_and_prepare_full_control_chunks(
        [
            FullControlChunk(
                language=Languages.JP,
                surface_text="あ",
                phones=["a"],
                tones=[0],
                word2ph=[1],
            )
        ],
        _config(add_blank=True),
    )

    g2p_result = result[0][2]
    assert len(g2p_result.phone) == 7
    assert g2p_result.word2ph == [3, 2, 2]
    assert sum(g2p_result.word2ph) == len(g2p_result.phone)


def test_full_control_rejects_unsupported_language() -> None:
    with pytest.raises(ValueError, match="Full-control input is not supported"):
        check_split_and_prepare_full_control_chunks(
            [
                FullControlChunk(
                    language=cast(Languages, "Z" + "H"),
                    surface_text="你好",
                    phones=["n", "i", "h", "ao"],
                    tones=[0, 0, 0, 0],
                    word2ph=[2, 2],
                )
            ],
            _config(),
        )


def test_full_control_rejects_phones_outside_language_inventory() -> None:
    with pytest.raises(ValueError, match="unsupported phones"):
        check_split_and_prepare_full_control_chunks(
            [
                FullControlChunk(
                    language=Languages.EN,
                    surface_text="x",
                    phones=["not-a-phone"],
                    tones=[0],
                    word2ph=[1],
                )
            ],
            _config(),
        )


def test_full_control_rejects_tones_outside_language_inventory() -> None:
    with pytest.raises(ValueError, match=r"tones must be in \[0, 1\]"):
        check_split_and_prepare_full_control_chunks(
            [
                FullControlChunk(
                    language=Languages.JP,
                    surface_text="あ",
                    phones=["a"],
                    tones=[2],
                    word2ph=[1],
                )
            ],
            _config(),
        )


def test_full_control_rejects_negative_word2ph_entries() -> None:
    with pytest.raises(ValueError, match="word2ph values must be non-negative"):
        check_split_and_prepare_full_control_chunks(
            [
                FullControlChunk(
                    language=Languages.JP,
                    surface_text="今日",
                    phones=["ky"],
                    tones=[1],
                    word2ph=[1, -1],
                )
            ],
            _config(),
        )


def test_mixed_full_control_chunks_are_concatenated_for_one_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_bert_features_batch(
        items: list[Any],
        cache: object | None = None,
        onnx_providers: object | None = None,
        trt_config: object | None = None,
    ) -> list[torch.Tensor]:
        return [torch.full((1024, sum(item.word2ph)), float(index + 1)) for index, item in enumerate(items)]

    monkeypatch.setattr(
        "charavoice_engine.inference.text_processing.extract_bert_features_batch",
        fake_extract_bert_features_batch,
    )

    request = InferenceRequest(
        text="今日AI",
        language=Languages.JP,
        speaker_id=3,
        style_vec=np.zeros(256, dtype=np.float32),
        chunks=[
            FullControlChunk(
                language=Languages.JP,
                surface_text="今日",
                phones=["ky", "o:"],
                tones=[1, 0],
                word2ph=[1, 1],
            ),
            FullControlChunk(
                language=Languages.EN,
                surface_text="AI",
                phones=["ey", "ay"],
                tones=[1, 1],
                word2ph=[1, 1],
            ),
        ],
    )

    preprocessed = prepare_chunks_to_preprocessed(request, _config(add_blank=True), bert_cache=None)

    assert preprocessed["sid"] == 3
    assert preprocessed["phones"].shape == (13,)
    assert preprocessed["tones"].shape == (13,)
    assert preprocessed["lang_ids"].shape == (13,)
    assert 1 in preprocessed["lang_ids"]
    assert 2 in preprocessed["lang_ids"]
    assert preprocessed["ja_bert"].shape == (1024, 13)
    assert preprocessed["en_bert"].shape == (1024, 13)
    assert np.count_nonzero(preprocessed["ja_bert"]) > 0
    assert np.count_nonzero(preprocessed["en_bert"]) > 0
