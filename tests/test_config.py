"""Tests for charavoice_engine.config."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from charavoice_engine.config import (
    BertBackend,
    FastInferenceConfig,
    HyperParameters,
    HyperParametersData,
    HyperParametersModel,
    HyperParametersTrain,
    ServerConfig,
    TensorRtConfig,
)


class TestBertBackend:
    def test_enum_values(self):
        assert BertBackend.ONNX_CUDA.value == "onnx_cuda"


class TestFastInferenceConfig:
    def test_defaults(self):
        cfg = FastInferenceConfig()
        assert cfg.inference_timeout == 60.0
        assert isinstance(cfg.trt, TensorRtConfig)
        assert cfg.trt.fp16 is True
        assert cfg.trt.max_workspace_size == 4 << 30
        assert cfg.bert_backend == BertBackend.ONNX_CUDA
        assert cfg.bert_cache_enabled is True
        assert cfg.bert_cache_max_size == 10000
        assert cfg.max_batch_size == 8
        assert cfg.device == "cuda"

    def test_custom_values(self):
        cfg = FastInferenceConfig(
            device="cpu",
            max_batch_size=1,
        )
        assert cfg.device == "cpu"
        assert cfg.max_batch_size == 1


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 5000
        assert cfg.workers == 1
        assert cfg.max_concurrent_requests == 32
        assert isinstance(cfg.inference, FastInferenceConfig)


class TestHyperParameters:
    def test_defaults(self):
        hps = HyperParameters()
        assert hps.version == "3.0.0-Bilingual-JP-Extra"
        assert hps.model_name == "Dummy"
        assert hps.data.add_blank is True
        assert hps.data.spk2id == {"Dummy": 0}
        assert hps.data.num_styles == 1
        assert hps.data.sampling_rate == 44100
        assert hps.train.seed == 42
        assert hps.model.hidden_channels == 192
        assert hps.model.gin_channels == 512
        assert hps.model.n_layers_trans_flow == 6
        assert hps.model.slm.hidden == 768

    def test_data_style2id_default(self):
        hps = HyperParameters()
        assert hps.data.style2id == {"Neutral": 0}

    def test_load_from_json(self):
        data = {
            "model_name": "TestModel",
            "version": "2.0",
            "data": {
                "sampling_rate": 22050,
                "spk2id": {"Alice": 0, "Bob": 1},
                "n_speakers": 2,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            hps = HyperParameters.load_from_json(Path(f.name))

        assert hps.model_name == "TestModel"
        assert hps.version == "2.0"
        assert hps.data.sampling_rate == 22050
        assert hps.data.spk2id == {"Alice": 0, "Bob": 1}
        # Unspecified fields get defaults
        assert hps.data.add_blank is True
        assert hps.train.seed == 42

    def test_unknown_fields_ignored(self):
        """Forward compatibility: unknown keys in JSON should not cause errors."""
        data = {
            "model_name": "Test",
            "unknown_top_level": 999,
            "data": {"sampling_rate": 44100, "unknown_nested": "value"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            hps = HyperParameters.load_from_json(Path(f.name))

        assert hps.model_name == "Test"
        assert hps.data.sampling_rate == 44100

    def test_nested_model_defaults(self):
        hps = HyperParameters()
        assert isinstance(hps.train, HyperParametersTrain)
        assert isinstance(hps.data, HyperParametersData)
        assert isinstance(hps.model, HyperParametersModel)
        assert hps.model_dir is None
        assert hps.speedup is False


class TestTensorRtConfig:
    def test_defaults(self):
        trt = TensorRtConfig()
        assert trt.fp16 is True
        assert trt.max_workspace_size == 4 << 30
        assert trt.engine_cache_path == "./trt_cache"
        assert trt.max_seq_len == 768

    def test_batch_defaults(self):
        trt = TensorRtConfig()
        assert trt.enc_dp_batch_min == 1
        assert trt.enc_dp_batch_opt == 8
        assert trt.enc_dp_batch_max == 16
        assert trt.flow_batch_min == 1
        assert trt.flow_batch_opt == 8
        assert trt.flow_batch_max == 16
        assert trt.decoder_batch_min == 1
        assert trt.decoder_batch_opt == 8
        assert trt.decoder_batch_max == 16

    def test_custom_values(self):
        trt = TensorRtConfig(fp16=False, max_workspace_size=2 << 30, max_seq_len=512)
        assert trt.fp16 is False
        assert trt.max_workspace_size == 2 << 30
        assert trt.max_seq_len == 512

    def test_custom_batch_values(self):
        trt = TensorRtConfig(decoder_batch_opt=4, flow_batch_max=32)
        assert trt.decoder_batch_opt == 4
        assert trt.flow_batch_max == 32
        assert trt.enc_dp_batch_opt == 8  # other defaults preserved

    def test_embedded_in_fast_inference_config(self):
        cfg = FastInferenceConfig(trt=TensorRtConfig(fp16=False))
        assert cfg.trt.fp16 is False
        assert cfg.trt.max_seq_len == 768  # other defaults preserved

    def test_inference_timeout_zero_means_no_timeout(self):
        cfg = FastInferenceConfig(inference_timeout=0)
        assert cfg.inference_timeout == 0
