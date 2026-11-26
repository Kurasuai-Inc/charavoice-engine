"""Runtime ABI tests for the bilingual JP-Extra model path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from charavoice_engine.config import TensorRtConfig
from charavoice_engine.constants import Languages
from charavoice_engine.inference.enc_dp_graph import get_enc_dp_graph_adapter
from charavoice_engine.inference.streaming_engine import StreamingEngine
from charavoice_engine.models.models_bilingual_jp_extra import TextEncoder
from charavoice_engine.models.utils.checkpoints import load_checkpoint
from charavoice_engine.models.utils.safetensors import load_safetensors
from charavoice_engine.nlp.fast_bert import _build_bert_trt_providers
from charavoice_engine.nlp.symbols import SYMBOLS


def test_enc_dp_adapter_exposes_bilingual_bert_inputs_only() -> None:
    adapter = get_enc_dp_graph_adapter()

    assert adapter.input_names == ("x", "x_lengths", "tone", "language", "ja_bert", "en_bert", "style_vec", "g")
    assert "bert" not in adapter.input_names


def test_enc_dp_runtime_feed_pads_bilingual_bert_slots() -> None:
    adapter = get_enc_dp_graph_adapter()
    preprocessed = [
        {
            "phones": np.array([1, 2, 3], dtype=np.int64),
            "tones": np.array([6, 7, 6], dtype=np.int64),
            "lang_ids": np.array([1, 1, 1], dtype=np.int64),
            "ja_bert": np.ones((1024, 3), dtype=np.float32),
            "en_bert": np.zeros((1024, 3), dtype=np.float32),
            "style_vec": np.full(256, 0.25, dtype=np.float32),
        },
        {
            "phones": np.array([4], dtype=np.int64),
            "tones": np.array([8], dtype=np.int64),
            "lang_ids": np.array([2], dtype=np.int64),
            "ja_bert": np.zeros((1024, 1), dtype=np.float32),
            "en_bert": np.full((1024, 1), 2.0, dtype=np.float32),
            "style_vec": np.full(256, -0.5, dtype=np.float32),
        },
    ]
    g = np.zeros((2, 512, 1), dtype=np.float32)

    feed, phone_lengths = adapter.runtime_feed(preprocessed, g)

    assert phone_lengths == [3, 1]
    assert "bert" not in feed
    assert feed["x"].shape == (2, 3)
    assert feed["ja_bert"].shape == (2, 1024, 3)
    assert feed["en_bert"].shape == (2, 1024, 3)
    np.testing.assert_array_equal(feed["x"][1], np.array([4, 0, 0], dtype=np.int64))
    assert feed["ja_bert"][0, :, :3].sum() == 1024 * 3
    assert feed["en_bert"][1, :, 0].sum() == 1024 * 2
    assert feed["en_bert"][1, :, 1:].sum() == 0


def test_text_encoder_uses_language_specific_bert_projections() -> None:
    encoder = TextEncoder(
        n_vocab=len(SYMBOLS),
        out_channels=4,
        hidden_channels=8,
        filter_channels=16,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
    )

    assert not hasattr(encoder, "bert_proj")
    assert hasattr(encoder, "ja_bert_proj")
    assert hasattr(encoder, "en_bert_proj")

    x = torch.randint(0, len(SYMBOLS), (2, 5))
    x_lengths = torch.tensor([5, 3], dtype=torch.long)
    tone = torch.randint(0, 12, (2, 5))
    language = torch.randint(0, 3, (2, 5))
    ja_bert = torch.randn(2, 1024, 5)
    en_bert = torch.randn(2, 1024, 5)
    style_vec = torch.randn(2, 256)

    encoded, m, logs, x_mask = encoder(x, x_lengths, tone, language, ja_bert, en_bert, style_vec)

    assert encoded.shape == (2, 8, 5)
    assert m.shape == (2, 4, 5)
    assert logs.shape == (2, 4, 5)
    assert x_mask.shape == (2, 1, 5)


def test_text_encoder_masks_inactive_language_bert_projection_bias() -> None:
    torch.manual_seed(1234)
    encoder = TextEncoder(
        n_vocab=len(SYMBOLS),
        out_channels=4,
        hidden_channels=8,
        filter_channels=16,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
    )
    encoder.eval()

    x = torch.randint(0, len(SYMBOLS), (1, 5))
    x_lengths = torch.tensor([5], dtype=torch.long)
    tone = torch.randint(0, 12, (1, 5))
    language = torch.ones((1, 5), dtype=torch.long)
    ja_bert = torch.randn(1, 1024, 5)
    en_bert = torch.zeros(1, 1024, 5)
    style_vec = torch.randn(1, 256)

    with torch.inference_mode():
        before = encoder(x, x_lengths, tone, language, ja_bert, en_bert, style_vec)
        encoder.en_bert_proj.bias.add_(100.0)
        after = encoder(x, x_lengths, tone, language, ja_bert, en_bert, style_vec)

    for before_tensor, after_tensor in zip(before, after, strict=True):
        torch.testing.assert_close(before_tensor, after_tensor)


def test_trt_provider_builders_do_not_add_secondary_cuda_or_cpu_providers() -> None:
    trt = TensorRtConfig()

    decoder_providers = StreamingEngine._build_trt_providers(
        trt,
        "decoder",
        inter_channels=192,
        gin_channels=512,
    )
    bert_providers = _build_bert_trt_providers(trt, Languages.JP)

    assert [provider[0] if isinstance(provider, tuple) else provider for provider in decoder_providers] == [
        "TensorrtExecutionProvider"
    ]
    assert [provider[0] if isinstance(provider, tuple) else provider for provider in bert_providers] == [
        "TensorrtExecutionProvider"
    ]


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)


def _checkpoint(path: Path, model_state: dict[str, torch.Tensor]) -> None:
    torch.save(
        {
            "iteration": 123,
            "learning_rate": 1e-4,
            "optimizer": None,
            "model": model_state,
        },
        path,
    )


def test_load_checkpoint_is_strict(tmp_path: Path) -> None:
    model = _TinyModel()
    path = tmp_path / "ok.pth"
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _checkpoint(path, state)

    loaded, optimizer, learning_rate, iteration = load_checkpoint(path, model, skip_optimizer=True)

    assert loaded is model
    assert optimizer is None
    assert learning_rate == 1e-4
    assert iteration == 123


def test_load_checkpoint_rejects_missing_model_key(tmp_path: Path) -> None:
    model = _TinyModel()
    path = tmp_path / "missing.pth"
    state = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.endswith("bias")}
    _checkpoint(path, state)

    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(path, model, skip_optimizer=True)


def test_load_safetensors_is_strict_and_skips_iteration_tensor(tmp_path: Path) -> None:
    model = _TinyModel()
    path = tmp_path / "ok.safetensors"
    tensors = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tensors["iteration"] = torch.tensor(456)
    save_file(tensors, path)

    loaded, iteration = load_safetensors(path, model)

    assert loaded is model
    assert iteration == 456


def test_load_safetensors_rejects_unexpected_key(tmp_path: Path) -> None:
    model = _TinyModel()
    path = tmp_path / "unexpected.safetensors"
    tensors = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tensors["unexpected.weight"] = torch.zeros(1)
    save_file(tensors, path)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_safetensors(path, model)
