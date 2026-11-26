"""Central configuration for CHARAVOICE Engine streaming inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class BertBackend(Enum):
    ONNX_CUDA = "onnx_cuda"


class InferenceProfile(StrEnum):
    DEFAULT = "default"
    THROUGHPUT128 = "throughput128"


@dataclass
class TensorRtConfig:
    """Streaming engine 用 TRT 設定 (per-component profiles)."""

    fp16: bool = True
    max_workspace_size: int = 4 << 30  # 4GB
    engine_cache_path: str = "./trt_cache"
    max_seq_len: int = 768  # テキスト分割上限 (~200字)

    # Per-component TRT profiles — sequence length
    enc_dp_t_phone_min: int = 1
    enc_dp_t_phone_opt: int = 50
    enc_dp_t_phone_max: int = 768  # ~170 chars x ~4.4 phones/char (with blanks)
    flow_t_mel_min: int = 1
    flow_t_mel_opt: int = 256
    flow_t_mel_max: int = 2048  # ~23秒の音声

    # Per-component TRT profiles — batch dimension
    enc_dp_batch_min: int = 1
    enc_dp_batch_opt: int = 8
    enc_dp_batch_max: int = 16
    flow_batch_min: int = 1
    flow_batch_opt: int = 8
    flow_batch_max: int = 16
    decoder_batch_min: int = 1
    decoder_batch_opt: int = 8
    decoder_batch_max: int = 16

    # BERT TRT profiles (batch BERT model only, JP)
    bert_trt_enabled: bool = True
    bert_batch_min: int = 1
    bert_batch_opt: int = 8
    bert_batch_max: int = 16
    bert_seq_min: int = 2
    bert_seq_opt: int = 30
    bert_seq_max: int = 200


@dataclass
class FastInferenceConfig:
    """All optimization settings in one place."""

    # TensorRT
    trt: TensorRtConfig = field(default_factory=TensorRtConfig)

    # BERT
    bert_backend: BertBackend = BertBackend.ONNX_CUDA
    bert_cache_enabled: bool = True
    bert_cache_max_size: int = 10000

    # Inference
    inference_timeout: float = 60.0  # seconds; 0 = no timeout

    # Batching
    max_batch_size: int = 8
    batch_timeout_ms: float = 50.0  # 固定タイムアウト (adaptive=False 時に使用)
    adaptive_batch_timeout: bool = True
    batch_timeout_min_ms: float = 2.0
    batch_timeout_max_ms: float = 100.0

    # Device
    device: str = "cuda"
    profile: InferenceProfile = InferenceProfile.DEFAULT

    def __post_init__(self) -> None:
        self.profile = InferenceProfile(self.profile)
        if self.throughput_optimized:
            self.adaptive_batch_timeout = False
            self.batch_timeout_ms = 0.0

    @property
    def throughput_optimized(self) -> bool:
        return self.profile == InferenceProfile.THROUGHPUT128

    @property
    def decoder_chunk_frames(self) -> int:
        return 128 if self.throughput_optimized else 32


@dataclass
class ServerConfig:
    """Server-specific configuration."""

    host: str = "0.0.0.0"
    port: int = 5000
    workers: int = 1
    max_concurrent_requests: int = 32
    inference: FastInferenceConfig = field(default_factory=FastInferenceConfig)


# ---------------------------------------------------------------------------
# HyperParameters — config.json loader for Style-Bert-VITS2 models
# Ported from style_bert_vits2.models.hyper_parameters
# ---------------------------------------------------------------------------


class HyperParametersTrain(BaseModel):
    log_interval: int = 200
    eval_interval: int = 1000
    save_interval: int = 1000
    save_epoch_interval: int = 0
    keep_ckpts: int = 5
    seed: int = 42
    epochs: int = 1000
    learning_rate: float = 0.0001
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    batch_size: int = 6
    bf16_run: bool = False
    fp16_run: bool = False
    lr_decay: float = 0.99996
    segment_size: int = 16384
    init_lr_ratio: int = 1
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    c_commit: int = 100
    skip_optimizer: bool = False
    freeze_JP_bert: bool = False
    freeze_EN_bert: bool = False
    freeze_emo: bool = False
    freeze_style: bool = False
    freeze_decoder: bool = False
    balance_training: bool = True
    balance_fields: list[str] = ["speaker", "language", "style"]


class HyperParametersData(BaseModel):
    training_files: str = "Data/Dummy/train.list"
    validation_files: str = "Data/Dummy/val.list"
    max_wav_value: float = 32768.0
    sampling_rate: int = 44100
    filter_length: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    add_blank: bool = True
    n_speakers: int = 1
    cleaned_text: bool = True
    spk2id: dict[str, int] = {"Dummy": 0}
    num_styles: int = 1
    style2id: dict[str, int] = {"Neutral": 0}


class HyperParametersModelSLM(BaseModel):
    model: str = "./slm/wavlm-base-plus"
    sr: int = 16000
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64


class HyperParametersModel(BaseModel):
    use_spk_conditioned_encoder: bool = True
    use_noise_scaled_mas: bool = True
    use_mel_posterior_encoder: bool = False
    use_duration_discriminator: bool = False
    use_wavlm_discriminator: bool = True
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = [3, 7, 11]
    resblock_dilation_sizes: list[list[int]] = [
        [1, 3, 5],
        [1, 3, 5],
        [1, 3, 5],
    ]
    upsample_rates: list[int] = [8, 8, 2, 2, 2]
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = [16, 16, 8, 2, 2]
    n_layers_q: int = 3
    n_flow_layer: int = 4
    n_layers_trans_flow: int = 6
    flow_share_parameter: bool = False
    use_transformer_flow: bool = True
    use_spectral_norm: bool = False
    gin_channels: int = 512
    slm: HyperParametersModelSLM = HyperParametersModelSLM()


class HyperParameters(BaseModel):
    model_name: str = "Dummy"
    version: str = "3.0.0-Bilingual-JP-Extra"
    train: HyperParametersTrain = HyperParametersTrain()
    data: HyperParametersData = HyperParametersData()
    model: HyperParametersModel = HyperParametersModel()

    model_dir: str | None = None
    speedup: bool = False
    repo_id: str | None = None

    model_config = ConfigDict(protected_namespaces=())

    @staticmethod
    def load_from_json(json_path: str | Path) -> HyperParameters:
        """与えられた JSON ファイルからハイパーパラメータを読み込む。"""
        with open(json_path, encoding="utf-8") as f:
            return HyperParameters.model_validate_json(f.read())
