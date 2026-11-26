"""Streaming inference engine: ONNX/TRT Stage 1-4 + ONNX/TRT Decoder Stage 5.

Runs enc_p+dp and flow via ONNX/TRT sessions, with length regulator and
optional SDP in PyTorch. Decodes chunk-by-chunk via ONNX Runtime with overlap-save.

Supports batched inference (B>1) for all ONNX/TRT stages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from charavoice_engine.config import BertBackend, FastInferenceConfig, HyperParameters, TensorRtConfig
from charavoice_engine.constants import Languages
from charavoice_engine.inference.enc_dp_graph import BilingualJPExtraEncDpGraphAdapter, get_enc_dp_graph_adapter
from charavoice_engine.inference.engine_base import InferenceEngine, InferenceRequest
from charavoice_engine.inference.overlap_save import (
    OverlapSaveConfig,
    adjust_duration_for_chunk_boundary,
    split_by_duration_if_needed,
)
from charavoice_engine.inference.text_processing import (
    TextProcessingConfig,
    g2p_results_to_preprocessed,
    prepare_chunks_to_preprocessed,
    run_g2p,
)
from charavoice_engine.models import commons
from charavoice_engine.models.models_bilingual_jp_extra import SynthesizerTrn as BilingualJPExtraSynthesizerTrn
from charavoice_engine.nlp.bert_cache import BertFeatureCache

logger = logging.getLogger(__name__)


@dataclass
class VramBreakdown:
    """Per-component VRAM usage (bytes) measured during load()."""

    pytorch_model: int = 0
    decoder_session: int = 0
    enc_dp_session: int = 0
    flow_session: int = 0
    total: int = 0
    baseline_free: int = 0

    def to_dict(self) -> dict[str, float]:
        """MB 単位の dict (JSON シリアライズ用)."""
        return {f.name: getattr(self, f.name) / (1024 * 1024) for f in fields(self)}


class StreamingEngine(InferenceEngine):
    """ONNX/TRT encoder/flow + ONNX/TRT decoder with overlap-save chunking."""

    def __init__(self, config: FastInferenceConfig | None = None) -> None:
        self.config = config or FastInferenceConfig()
        self._net_g: BilingualJPExtraSynthesizerTrn | None = None
        self._hps: HyperParameters | None = None
        self._device: str = "cuda"
        self._style_vectors: NDArray[Any] | None = None
        self._bert_cache: BertFeatureCache | None = None
        self._text_config: TextProcessingConfig | None = None
        self._decoder_session: Any = None  # onnxruntime.InferenceSession
        self._enc_dp_session: Any = None  # onnxruntime.InferenceSession
        self._flow_session: Any = None  # onnxruntime.InferenceSession
        self._enc_dp_graph: BilingualJPExtraEncDpGraphAdapter | None = None
        self._overlap_config = OverlapSaveConfig(chunk_size=self.config.decoder_chunk_frames)
        self._loaded: bool = False
        self._vram_breakdown: VramBreakdown | None = None

        if self.config.bert_cache_enabled:
            self._bert_cache = BertFeatureCache(
                max_size=self.config.bert_cache_max_size,
                device=self.config.device,
            )

    def load(
        self,
        model_path: str,
        config_path: str,
        style_vec_path: str,
        device: str = "cuda",
        preload_bert: bool = True,
    ) -> None:
        import onnxruntime

        from charavoice_engine.inference.onnx_session import preload_tensorrt_libs
        from charavoice_engine.models.infer import get_net_g

        self._device = device
        self._hps = HyperParameters.load_from_json(config_path)
        if not self._hps.version.endswith("Bilingual-JP-Extra"):
            raise ValueError(f"Unsupported model version for CHARAVOICE Engine bilingual runtime: {self._hps.version}")
        self._enc_dp_graph = get_enc_dp_graph_adapter()
        self._style_vectors = np.load(style_vec_path)

        # --- Enable TF32 for Ampere+ GPUs (matmul/conv ~2x faster, sufficient precision for TTS) ---
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        use_cuda = device.startswith("cuda")
        vram = VramBreakdown() if use_cuda else None

        # --- VRAM baseline ---
        if vram is not None:
            vram.baseline_free = self._get_vram_free()
            free_before = vram.baseline_free

        # --- Load PyTorch model (needed for emb_g, sdp, length regulator) ---
        logger.info("Loading PyTorch model from %s", model_path)
        self._net_g = get_net_g(model_path, self._hps.version, device, self._hps)
        self._net_g.dec.remove_weight_norm()

        if vram is not None:
            free_after = self._get_vram_free()
            vram.pytorch_model = free_before - free_after
            free_before = free_after

        # --- TRT provider config (per-component profiles) ---
        preload_tensorrt_libs()

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        trt_config = self.config.trt
        config_dir = Path(config_path).parent
        model_name = config_dir.name

        decoder_providers = self._build_trt_providers(
            trt_config,
            "decoder",
            model_name,
            inter_channels=self._hps.model.inter_channels,
            gin_channels=self._hps.model.gin_channels,
            decoder_input_frames=self._overlap_config.input_frames,
        )
        enc_dp_providers = self._build_trt_providers(
            trt_config,
            "enc_p_dp",
            model_name,
            inter_channels=self._hps.model.inter_channels,
            gin_channels=self._hps.model.gin_channels,
            enc_dp_graph=self._enc_dp_graph,
        )
        flow_providers = self._build_trt_providers(
            trt_config,
            "flow",
            model_name,
            inter_channels=self._hps.model.inter_channels,
            gin_channels=self._hps.model.gin_channels,
        )

        logger.info(
            "TRT profiles: enc_p_dp B=[%d,%d,%d] T=[%d,%d,%d], flow B=[%d,%d,%d] T=[%d,%d,%d], decoder B=[%d,%d,%d]",
            trt_config.enc_dp_batch_min,
            trt_config.enc_dp_batch_opt,
            trt_config.enc_dp_batch_max,
            trt_config.enc_dp_t_phone_min,
            trt_config.enc_dp_t_phone_opt,
            trt_config.enc_dp_t_phone_max,
            trt_config.flow_batch_min,
            trt_config.flow_batch_opt,
            trt_config.flow_batch_max,
            trt_config.flow_t_mel_min,
            trt_config.flow_t_mel_opt,
            trt_config.flow_t_mel_max,
            trt_config.decoder_batch_min,
            trt_config.decoder_batch_opt,
            trt_config.decoder_batch_max,
        )

        # --- Find or auto-export ONNX files ---
        decoder_onnx = config_dir / "decoder.onnx"
        if not decoder_onnx.exists():
            logger.info("Decoder ONNX not found, auto-exporting to %s", decoder_onnx)
            from charavoice_engine.convert.export_decoder import export_decoder_onnx

            export_decoder_onnx(
                model_path=model_path,
                config_path=config_path,
                output_path=str(decoder_onnx),
                device="cpu",
                input_frames=OverlapSaveConfig().input_frames,
            )

        if self.config.throughput_optimized:
            from charavoice_engine.inference.decoder_graph import resize_decoder_graph

            decoder_onnx = resize_decoder_graph(
                decoder_onnx,
                self._overlap_config.input_frames,
                self._overlap_config.hop_length,
                Path(trt_config.engine_cache_path),
            )

        enc_dp_onnx = config_dir / "enc_p_dp.onnx"
        if not enc_dp_onnx.exists():
            logger.info("enc_p+dp ONNX not found, auto-exporting to %s", enc_dp_onnx)
            from charavoice_engine.convert.export_components import export_enc_p_dp_onnx

            export_enc_p_dp_onnx(
                net_g=self._net_g,
                output_path=str(enc_dp_onnx),
                device=device,
                gin_channels=self._hps.model.gin_channels,
            )

        flow_onnx = config_dir / "flow.onnx"
        if not flow_onnx.exists():
            logger.info("flow ONNX not found, auto-exporting to %s", flow_onnx)
            from charavoice_engine.convert.export_components import export_flow_onnx

            export_flow_onnx(
                net_g=self._net_g,
                output_path=str(flow_onnx),
                device=device,
                inter_channels=self._hps.model.inter_channels,
                gin_channels=self._hps.model.gin_channels,
            )

        # --- Create + warmup ONNX sessions (one component at a time for VRAM measurement) ---
        self._decoder_session = self._create_onnx_session(
            onnxruntime, str(decoder_onnx), sess_options, decoder_providers, "decoder"
        )
        self._warmup_decoder()
        if vram is not None:
            free_after = self._get_vram_free()
            vram.decoder_session = free_before - free_after
            free_before = free_after

        self._enc_dp_session = self._create_onnx_session(
            onnxruntime, str(enc_dp_onnx), sess_options, enc_dp_providers, "enc_p+dp"
        )
        self._warmup_enc_dp()
        if vram is not None:
            free_after = self._get_vram_free()
            vram.enc_dp_session = free_before - free_after
            free_before = free_after

        self._flow_session = self._create_onnx_session(
            onnxruntime, str(flow_onnx), sess_options, flow_providers, "flow"
        )
        self._warmup_flow()
        if vram is not None:
            free_after = self._get_vram_free()
            vram.flow_session = free_before - free_after

        logger.info("All ONNX session warmups complete")

        # --- Text processing config ---
        bert_providers: tuple[str | tuple[str, dict[str, Any]], ...] = (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        )
        self._text_config = TextProcessingConfig(
            add_blank=self._hps.data.add_blank,
            max_seq_len=self.config.trt.max_seq_len,
            device=device,
            bert_backend=BertBackend.ONNX_CUDA,
            onnx_providers=bert_providers,
            trt_config=trt_config if trt_config.bert_trt_enabled else None,
        )

        # --- Preload and warmup BERT TRT sessions ---
        if preload_bert:
            self._preload_bert_sessions()
            self._warmup_bert_sessions()

        # --- VRAM total and log ---
        if vram is not None:
            free_final = self._get_vram_free()
            vram.total = vram.baseline_free - free_final
            self._vram_breakdown = vram
            logger.info(
                "VRAM breakdown: pytorch=%dMB decoder=%dMB enc+dp=%dMB flow=%dMB total=%dMB (free=%dMB)",
                vram.pytorch_model // (1024 * 1024),
                vram.decoder_session // (1024 * 1024),
                vram.enc_dp_session // (1024 * 1024),
                vram.flow_session // (1024 * 1024),
                vram.total // (1024 * 1024),
                free_final // (1024 * 1024),
            )

        self._loaded = True
        logger.info("StreamingEngine loaded (device=%s, version=%s)", device, self._hps.version)

    @staticmethod
    def _build_trt_providers(
        trt_config: TensorRtConfig,
        component: str,
        model_name: str = "",
        inter_channels: int = 192,
        gin_channels: int = 512,
        enc_dp_graph: BilingualJPExtraEncDpGraphAdapter | None = None,
        decoder_input_frames: int = 58,
    ) -> list[str | tuple[str, dict[str, Any]]]:
        """Build TRT EP provider list with component-specific profile shapes.

        Cache is keyed by model_name/component to prevent different models from
        sharing TRT engines (which would silently use wrong weights).
        """
        cache_component = (
            f"decoder_f{decoder_input_frames}" if component == "decoder" and decoder_input_frames != 58 else component
        )
        cache_dir = Path(trt_config.engine_cache_path) / model_name / cache_component
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir)

        trt_opts: dict[str, Any] = {
            "trt_fp16_enable": trt_config.fp16,
            "trt_max_workspace_size": trt_config.max_workspace_size,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_path,
            "trt_builder_optimization_level": 3,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache_path,
        }

        if component == "enc_p_dp":
            # enc_p produces NaN under FP16 due to BERT feature precision loss.
            # Force FP32 for this component regardless of global config.
            trt_opts["trt_fp16_enable"] = False
            b_min = trt_config.enc_dp_batch_min
            b_opt = trt_config.enc_dp_batch_opt
            b_max = trt_config.enc_dp_batch_max
            t_min = trt_config.enc_dp_t_phone_min
            t_opt = trt_config.enc_dp_t_phone_opt
            t_max = trt_config.enc_dp_t_phone_max
            if enc_dp_graph is None:
                raise ValueError("enc_dp_graph is required for enc_p_dp TRT profiles")
            trt_opts["trt_profile_min_shapes"] = enc_dp_graph.trt_profile_shapes(trt_config, gin_channels, b_min, t_min)
            trt_opts["trt_profile_opt_shapes"] = enc_dp_graph.trt_profile_shapes(trt_config, gin_channels, b_opt, t_opt)
            trt_opts["trt_profile_max_shapes"] = enc_dp_graph.trt_profile_shapes(trt_config, gin_channels, b_max, t_max)
        elif component == "flow":
            b_min = trt_config.flow_batch_min
            b_opt = trt_config.flow_batch_opt
            b_max = trt_config.flow_batch_max
            t_min = trt_config.flow_t_mel_min
            t_opt = trt_config.flow_t_mel_opt
            t_max = trt_config.flow_t_mel_max
            trt_opts["trt_profile_min_shapes"] = (
                f"z_p:{b_min}x{inter_channels}x{t_min},y_mask:{b_min}x1x{t_min},g:{b_min}x{gin_channels}x1"
            )
            trt_opts["trt_profile_opt_shapes"] = (
                f"z_p:{b_opt}x{inter_channels}x{t_opt},y_mask:{b_opt}x1x{t_opt},g:{b_opt}x{gin_channels}x1"
            )
            trt_opts["trt_profile_max_shapes"] = (
                f"z_p:{b_max}x{inter_channels}x{t_max},y_mask:{b_max}x1x{t_max},g:{b_max}x{gin_channels}x1"
            )
        elif component == "decoder":
            b_min = trt_config.decoder_batch_min
            b_opt = trt_config.decoder_batch_opt
            b_max = trt_config.decoder_batch_max
            f = decoder_input_frames
            trt_opts["trt_profile_min_shapes"] = f"z:{b_min}x{inter_channels}x{f},g:{b_min}x{gin_channels}x1"
            trt_opts["trt_profile_opt_shapes"] = f"z:{b_opt}x{inter_channels}x{f},g:{b_opt}x{gin_channels}x1"
            trt_opts["trt_profile_max_shapes"] = f"z:{b_max}x{inter_channels}x{f},g:{b_max}x{gin_channels}x1"

        return [("TensorrtExecutionProvider", trt_opts)]

    @staticmethod
    def _create_onnx_session(
        ort: Any,
        onnx_path: str,
        sess_options: Any,
        providers: list[str | tuple[str, dict[str, Any]]],
        name: str,
    ) -> Any:
        """Create an ONNX Runtime InferenceSession and fail on provider errors."""
        logger.info("Loading %s ONNX session from %s", name, onnx_path)
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        expected_provider = providers[0] if isinstance(providers[0], str) else providers[0][0]
        actual_providers = session.get_providers()
        if not actual_providers or actual_providers[0] != expected_provider:
            raise RuntimeError(
                f"{name} ONNX session provider mismatch: expected {expected_provider}, got {actual_providers}"
            )
        return session

    def _preload_bert_sessions(self) -> None:
        """Preload batch BERT sessions (triggers TRT engine build at startup)."""
        assert self._text_config is not None
        trt_config = self._text_config.trt_config

        from charavoice_engine.nlp.fast_bert import preload_batch_bert_session

        for lang in Languages:
            preload_batch_bert_session(lang, self._text_config.onnx_providers, trt_config)

    def _warmup_bert_sessions(self) -> None:
        """Warmup batch BERT TRT sessions with representative inputs."""
        assert self._text_config is not None
        trt_config = self._text_config.trt_config
        if trt_config is None:
            return

        from charavoice_engine.nlp.fast_bert import warmup_batch_bert_sessions

        warmup_batch_bert_sessions(trt_config)

    @staticmethod
    def _get_vram_free() -> int:
        """Return free VRAM in bytes (CUDA driver level, includes non-PyTorch allocations)."""
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        return free

    def _warmup_decoder(self) -> None:
        """Warmup decoder session (B=1 + B=opt)."""
        assert self._hps is not None
        ic = self._hps.model.inter_channels
        gc = self._hps.model.gin_channels
        oc = self._overlap_config
        trt = self.config.trt

        logger.info("Warming up decoder (B=1)...")
        g_np_1 = np.zeros((1, gc, 1), dtype=np.float32)
        z_dec_1 = np.zeros((1, ic, oc.input_frames), dtype=np.float32)
        self._decoder_session.run(None, {"z": z_dec_1, "g": g_np_1})

        B_dec = trt.decoder_batch_opt
        logger.info("Warming up decoder (B=%d)...", B_dec)
        z_dec_b = np.zeros((B_dec, ic, oc.input_frames), dtype=np.float32)
        g_np_b = np.zeros((B_dec, gc, 1), dtype=np.float32)
        self._decoder_session.run(None, {"z": z_dec_b, "g": g_np_b})

    def _warmup_enc_dp(self) -> None:
        """Warmup enc_p+dp session (B=1 + B=opt)."""
        assert self._hps is not None
        assert self._enc_dp_graph is not None
        gc = self._hps.model.gin_channels
        trt = self.config.trt

        T = trt.enc_dp_t_phone_opt

        logger.info("Warming up enc_p+dp (B=1)...")
        self._enc_dp_session.run(None, self._enc_dp_graph.warmup_feed(batch=1, seq_len=T, gin_channels=gc))

        B_enc = trt.enc_dp_batch_opt
        logger.info("Warming up enc_p+dp (B=%d)...", B_enc)
        self._enc_dp_session.run(None, self._enc_dp_graph.warmup_feed(batch=B_enc, seq_len=T, gin_channels=gc))

    def _warmup_flow(self) -> None:
        """Warmup flow session (B=1 + B=opt)."""
        assert self._hps is not None
        ic = self._hps.model.inter_channels
        gc = self._hps.model.gin_channels
        trt = self.config.trt

        T_mel = trt.flow_t_mel_opt
        g_np_1 = np.zeros((1, gc, 1), dtype=np.float32)

        logger.info("Warming up flow (B=1)...")
        self._flow_session.run(
            None,
            {
                "z_p": np.zeros((1, ic, T_mel), dtype=np.float32),
                "y_mask": np.ones((1, 1, T_mel), dtype=np.float32),
                "g": g_np_1,
            },
        )

        B_flow = trt.flow_batch_opt
        logger.info("Warming up flow (B=%d)...", B_flow)
        self._flow_session.run(
            None,
            {
                "z_p": np.zeros((B_flow, ic, T_mel), dtype=np.float32),
                "y_mask": np.ones((B_flow, 1, T_mel), dtype=np.float32),
                "g": np.zeros((B_flow, gc, 1), dtype=np.float32),
            },
        )

    def _warmup_sessions(self) -> None:
        """Warmup all ONNX sessions to trigger initial TRT engine builds."""
        self._warmup_decoder()
        self._warmup_enc_dp()
        self._warmup_flow()
        logger.info("All ONNX session warmups complete")

    # -----------------------------------------------------------------------
    # Batched inference
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def _run_batch_stages_0_4(
        self,
        requests: list[InferenceRequest],
    ) -> tuple[list[torch.Tensor], NDArray[Any], tuple[float, float, float, float, float]]:
        """Run Stages 0-4 for a batch of requests.

        Returns:
            (z_list, g_np, (g2p_ms, bert_ms, enc_dp_ms, lenreg_ms, flow_ms))
            z_list: Per-request z_all tensor [1, 192, T_mel].
            g_np: [B, 512, 1] speaker embeddings.
        """
        assert self._net_g is not None
        assert self._hps is not None
        assert self._text_config is not None

        net_g = self._net_g
        device = self._device
        B = len(requests)

        do_profile = logger.isEnabledFor(logging.DEBUG)
        t0 = time.perf_counter() if do_profile else 0.0

        # Stage 0a: G2P for text-only requests (skip if pre-computed by app layer).
        # Mixed chunks are fully preprocessed in Stage 0b because their BERT
        # features must be concatenated after per-chunk boundary trimming.
        text_requests: list[InferenceRequest] = []
        text_indices: list[int] = []
        g2p_results = []
        preprocessed_slots: list[dict[str, Any] | None] = [None] * B
        for index, r in enumerate(requests):
            if r.chunks is not None:
                continue
            text_requests.append(r)
            text_indices.append(index)
            if r.g2p_result is not None:
                g2p_results.append(r.g2p_result.as_tuple())
            else:
                g2p_results.append(run_g2p(r.text, r.language, self._text_config))

        t_g2p = time.perf_counter() if do_profile else 0.0

        # Stage 0b: BERT extraction (batched per language for text-only requests).
        if text_requests:
            text_preprocessed = g2p_results_to_preprocessed(
                text_requests,
                g2p_results,
                self._bert_cache,
                self._text_config.onnx_providers,
                self._text_config.trt_config,
            )
            for index, item in zip(text_indices, text_preprocessed, strict=True):
                preprocessed_slots[index] = item

        for index, r in enumerate(requests):
            if r.chunks is not None:
                preprocessed_slots[index] = prepare_chunks_to_preprocessed(r, self._text_config, self._bert_cache)

        preprocessed_list = [item for item in preprocessed_slots if item is not None]

        t_bert = time.perf_counter() if do_profile else 0.0

        # Stage 1a: Speaker embedding (batched PyTorch lookup)
        sids = torch.LongTensor([p["sid"] for p in preprocessed_list]).to(device)
        g_all = net_g.emb_g(sids).unsqueeze(-1)  # [B, gin_channels, 1]
        g_np = g_all.cpu().numpy().astype(np.float32)

        # Stage 1b + 2: Batched enc_p + dp via ONNX
        enc_dp_results = self._run_enc_dp_batched(preprocessed_list, g_np)

        t_enc_dp = time.perf_counter() if do_profile else 0.0

        # Stage 3: Length regulator (batched CPU, no GPU transfer)
        flow_items = self._run_length_regulator_batched(
            enc_dp_results,
            [p["phones"] for p in preprocessed_list],
            [r.noise_scale for r in requests],
            [r.length_scale for r in requests],
        )

        t_lenreg = time.perf_counter() if do_profile else 0.0

        # Stage 4: Batched flow via ONNX
        z_list = self._run_flow_batched(flow_items, g_np, B)

        t_flow = time.perf_counter() if do_profile else 0.0

        if do_profile:
            times = (
                (t_g2p - t0) * 1000,
                (t_bert - t_g2p) * 1000,
                (t_enc_dp - t_bert) * 1000,
                (t_lenreg - t_enc_dp) * 1000,
                (t_flow - t_lenreg) * 1000,
            )
        else:
            times = (0.0, 0.0, 0.0, 0.0, 0.0)

        return z_list, g_np, times

    # -----------------------------------------------------------------------
    # Batched stage implementations
    # -----------------------------------------------------------------------

    def _run_enc_dp_batched(
        self,
        preprocessed_list: list[dict[str, Any]],
        g_np: NDArray[Any],
    ) -> list[tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]]:
        """Run enc_p+dp in a single batched ONNX call.

        Args:
            preprocessed_list: Per-request preprocessed dicts.
            g_np: [B, 512, 1] speaker embeddings.

        Returns:
            Per-request (m_p, logs_p, x_mask, logw) numpy arrays,
            each with shape [1, C, T_phone_i] (sliced to actual length).
        """
        assert self._enc_dp_graph is not None
        feed, t_phones = self._enc_dp_graph.runtime_feed(preprocessed_list, g_np)
        outputs = self._enc_dp_session.run(None, feed)
        m_p_all, logs_p_all, x_mask_all, logw_all = outputs

        # Slice per request to actual T_phone while preserving the stage ABI.
        results: list[tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]] = []
        for i in range(len(t_phones)):
            T = t_phones[i]
            results.append(
                (
                    m_p_all[i : i + 1, :, :T],
                    logs_p_all[i : i + 1, :, :T],
                    x_mask_all[i : i + 1, :, :T],
                    logw_all[i : i + 1, :, :T],
                )
            )
        return results

    @torch.inference_mode()
    def _run_length_regulator_batched(
        self,
        enc_dp_results: list[tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]],
        phones_list: list[NDArray[Any]],
        noise_scales: list[float],
        length_scales: list[float],
    ) -> list[tuple[int, int, NDArray[Any], NDArray[Any]]]:
        """Run length regulator for all requests as a GPU batch.

        Phase 1 (exp/ceil/split) runs on CPU (lightweight, data-dependent branching).
        Phase 2 (generate_path/matmul/randn) runs as a single GPU batch to keep
        the GPU busy on the inference thread.

        Args:
            enc_dp_results: Per-request (m_p, logs_p, x_mask, logw) numpy arrays [1,C,T].
            phones_list: Per-request phone ID arrays [T_phone].
            noise_scales: Per-request noise scale.
            length_scales: Per-request length scale.

        Returns:
            flow_items: (request_idx, segment_idx, z_p_np [1,192,T_mel], y_mask_np [1,1,T_mel]).
        """
        device = self._device
        chunk_size = min(32, self._overlap_config.chunk_size)
        max_t_mel = self.config.trt.flow_t_mel_max

        # Phase 1: Per-request exp/ceil/split on CPU (lightweight, needs branching)
        all_segments: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]] = []

        for i, (m_p_np, logs_p_np, x_mask_np, logw_np) in enumerate(enc_dp_results):
            x_mask = torch.from_numpy(x_mask_np)
            logw = torch.from_numpy(logw_np)

            w = torch.exp(logw) * x_mask * length_scales[i]
            w_ceil = torch.ceil(w)

            phones_t = torch.from_numpy(phones_list[i][np.newaxis, :])
            segments = split_by_duration_if_needed(w_ceil, phones_t, max_t_mel)

            if len(segments) > 1:
                seg_mels = [int(w_ceil[0, 0, s:e].sum().item()) for s, e in segments]
                logger.info("Long text split into %d segments (T_mel per seg: %s)", len(segments), seg_mels)

            m_p = torch.from_numpy(m_p_np)
            logs_p = torch.from_numpy(logs_p_np)

            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                w_ceil_seg = adjust_duration_for_chunk_boundary(w_ceil[:, :, seg_start:seg_end], chunk_size)
                all_segments.append(
                    (
                        i,
                        seg_idx,
                        w_ceil_seg,
                        m_p[:, :, seg_start:seg_end],
                        logs_p[:, :, seg_start:seg_end],
                        x_mask[:, :, seg_start:seg_end],
                        noise_scales[i],
                    )
                )

        if not all_segments:
            return []

        N = len(all_segments)

        # Phase 2: Pack into numpy padded batches (CPU), transfer once to GPU
        t_phones_seg = [seg[2].shape[2] for seg in all_segments]
        y_lengths_list = [max(1, int(seg[2].sum().item())) for seg in all_segments]
        T_phone_max = max(t_phones_seg)
        T_mel_max = max(y_lengths_list)

        # Build padded numpy arrays on CPU (zero-copy from Phase 1 tensors)
        w_ceil_np = np.zeros((N, 1, T_phone_max), dtype=np.float32)
        m_p_np = np.zeros((N, 192, T_phone_max), dtype=np.float32)
        logs_p_np = np.zeros((N, 192, T_phone_max), dtype=np.float32)
        x_mask_np = np.zeros((N, 1, T_phone_max), dtype=np.float32)

        for j, (_, _, w_ceil_seg, m_p_seg, logs_p_seg, x_mask_seg, _) in enumerate(all_segments):
            T = t_phones_seg[j]
            w_ceil_np[j, :, :T] = w_ceil_seg[0].numpy()
            m_p_np[j, :, :T] = m_p_seg[0].numpy()
            logs_p_np[j, :, :T] = logs_p_seg[0].numpy()
            x_mask_np[j, :, :T] = x_mask_seg[0].numpy()

        # Single CPU→GPU transfer per tensor (4 transfers total, regardless of N)
        w_ceil_batch = torch.from_numpy(w_ceil_np).to(device)
        m_p_batch = torch.from_numpy(m_p_np).to(device)
        logs_p_batch = torch.from_numpy(logs_p_np).to(device)
        x_mask_batch = torch.from_numpy(x_mask_np).to(device)
        noise_batch = torch.tensor([seg[6] for seg in all_segments], device=device).view(N, 1, 1)

        y_lengths = torch.tensor(y_lengths_list, dtype=torch.long, device=device)
        y_mask_batch = torch.unsqueeze(commons.sequence_mask(y_lengths, T_mel_max), 1).to(x_mask_batch.dtype)

        attn_mask = torch.unsqueeze(x_mask_batch, 2) * torch.unsqueeze(y_mask_batch, -1)
        attn = commons.generate_path(w_ceil_batch, attn_mask)

        m_p_exp = torch.matmul(attn.squeeze(1), m_p_batch.transpose(1, 2)).transpose(1, 2)
        logs_p_exp = torch.matmul(attn.squeeze(1), logs_p_batch.transpose(1, 2)).transpose(1, 2)

        z_p_batch = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_batch

        # Phase 3: Slice per segment → flow_items (GPU→CPU transfer, one batch)
        keep_on_cuda = self.config.throughput_optimized and device.startswith("cuda")
        z_p_cpu = z_p_batch if keep_on_cuda else z_p_batch.cpu().numpy()
        y_mask_cpu = y_mask_batch if keep_on_cuda else y_mask_batch.cpu().numpy()

        flow_items: list[tuple[int, int, Any, Any]] = []
        for j, (req_idx, seg_idx, _, _, _, _, _) in enumerate(all_segments):
            T_mel = y_lengths_list[j]
            latent = z_p_cpu[j : j + 1, :, :T_mel]
            mask = y_mask_cpu[j : j + 1, :, :T_mel]
            if isinstance(latent, np.ndarray):
                latent = latent.astype(np.float32)
            if isinstance(mask, np.ndarray):
                mask = mask.astype(np.float32)
            flow_items.append(
                (
                    req_idx,
                    seg_idx,
                    latent,
                    mask,
                )
            )

        return flow_items

    def _run_flow_batched(
        self,
        flow_items: list[tuple[int, int, Any, Any]],
        g_np: NDArray[Any],
        n_requests: int,
    ) -> list[torch.Tensor]:
        """Run flow inverse in batched ONNX calls.

        Args:
            flow_items: (request_idx, segment_idx, z_p_np [1,192,T], y_mask_np [1,1,T]).
            g_np: [B_total, 512, 1] speaker embeddings for all requests.
            n_requests: Total number of requests.

        Returns:
            Per-request z_all tensor [1, 192, T_mel] (segments concatenated).
        """
        if not flow_items:
            return []

        if self.config.throughput_optimized and self._device.startswith("cuda"):
            return self._run_flow_cuda(flow_items, g_np, n_requests)

        device = self._device
        batch_max = self.config.trt.flow_batch_max
        N = len(flow_items)
        t_mels = [item[2].shape[2] for item in flow_items]

        # Collect results indexed by position in flow_items
        z_results_flat: list[torch.Tensor] = [torch.empty(0)] * N

        # Process in mini-batches of flow_batch_max
        for batch_start in range(0, N, batch_max):
            batch_end = min(batch_start + batch_max, N)
            B_cur = batch_end - batch_start
            batch_t_mels = t_mels[batch_start:batch_end]
            T_max = max(batch_t_mels)

            z_p_batch = np.zeros((B_cur, 192, T_max), dtype=np.float32)
            y_mask_batch = np.zeros((B_cur, 1, T_max), dtype=np.float32)
            g_batch = np.zeros((B_cur, g_np.shape[1], 1), dtype=np.float32)

            for j in range(B_cur):
                idx = batch_start + j
                req_idx, _seg_idx, z_p_np, y_mask_np = flow_items[idx]
                T = batch_t_mels[j]
                z_p_batch[j, :, :T] = z_p_np[0]
                y_mask_batch[j, :, :T] = y_mask_np[0]
                g_batch[j] = g_np[req_idx]

            (z_batch_np,) = self._flow_session.run(
                None,
                {"z_p": z_p_batch, "y_mask": y_mask_batch, "g": g_batch},
            )

            # Slice results and apply mask
            for j in range(B_cur):
                idx = batch_start + j
                T = batch_t_mels[j]
                y_mask_t = torch.from_numpy(y_mask_batch[j : j + 1, :, :T]).to(device)
                z_t = torch.from_numpy(z_batch_np[j : j + 1, :, :T]).to(device) * y_mask_t
                z_results_flat[idx] = z_t

        # Reassemble per-request: group by request_idx, sort by segment_idx, cat
        per_request_segs: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(n_requests)]
        for idx, (req_idx, seg_idx, _, _) in enumerate(flow_items):
            per_request_segs[req_idx].append((seg_idx, z_results_flat[idx]))

        z_list: list[torch.Tensor] = []
        for segs in per_request_segs:
            segs.sort(key=lambda x: x[0])
            tensors = [s[1] for s in segs]
            z_list.append(torch.cat(tensors, dim=2) if len(tensors) > 1 else tensors[0])

        return z_list

    # -----------------------------------------------------------------------
    # Public API (pipeline-separated batcher uses these)
    # -----------------------------------------------------------------------

    @property
    def overlap_config(self) -> OverlapSaveConfig:
        """Overlap-save config (immutable, safe to read from any thread)."""
        return self._overlap_config

    def _run_flow_cuda(
        self, flow_items: list[tuple[int, int, Any, Any]], g_np: NDArray[Any], n_requests: int
    ) -> list[torch.Tensor]:
        from charavoice_engine.inference.onnx_session import run_cuda_bound

        results: list[torch.Tensor] = [torch.empty(0)] * len(flow_items)
        for start in range(0, len(flow_items), self.config.trt.flow_batch_max):
            group = flow_items[start : start + self.config.trt.flow_batch_max]
            width = max(item[2].shape[-1] for item in group)
            channels = group[0][2].shape[1]
            latent = torch.zeros((len(group), channels, width), device=self._device)
            mask = torch.zeros((len(group), 1, width), device=self._device)
            for index, (_, _, value, valid) in enumerate(group):
                count = value.shape[-1]
                latent[index, :, :count] = value[0]
                mask[index, :, :count] = valid[0]
            speakers = np.stack([g_np[item[0]] for item in group])
            output = (
                run_cuda_bound(
                    self._flow_session,
                    {"z_p": latent, "y_mask": mask, "g": speakers},
                    [tuple(latent.shape)],
                    self._device,
                )[0]
                * mask
            )
            for index, item in enumerate(group):
                results[start + index] = output[index : index + 1, :, : item[2].shape[-1]]
        per_request: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(n_requests)]
        for index, (request, segment, _, _) in enumerate(flow_items):
            per_request[request].append((segment, results[index]))
        return [
            torch.cat([value for _, value in sorted(group, key=lambda item: item[0])], dim=2) for group in per_request
        ]

    @torch.inference_mode()
    def run_stages_0_4(
        self,
        requests: list[InferenceRequest],
    ) -> tuple[list[torch.Tensor], NDArray[Any], tuple[float, float, float, float, float]]:
        """Run Stages 0-4 for a batch (public wrapper for _run_batch_stages_0_4).

        Returns:
            (z_list, g_np, (g2p_ms, bert_ms, enc_dp_ms, lenreg_ms, flow_ms))
        """
        return self._run_batch_stages_0_4(requests)

    @torch.inference_mode()
    def decode_chunk_batch(
        self,
        z_batch: NDArray[Any],
        g_batch: NDArray[Any],
    ) -> NDArray[Any]:
        """Decode a batch of chunks via the decoder ONNX session.

        Args:
            z_batch: [B, 192, input_frames] latent chunks.
            g_batch: [B, 512, 1] speaker embeddings.

        Returns:
            wav_batch: [B, 1, output_samples] raw decoder output.
        """
        (wav_batch,) = self._decoder_session.run(None, {"z": z_batch, "g": g_batch})
        return wav_batch

    # -----------------------------------------------------------------------
    # Public API (model lifecycle)
    # -----------------------------------------------------------------------

    def unload(self) -> None:
        self._net_g = None
        self._decoder_session = None
        self._enc_dp_session = None
        self._flow_session = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        if self._hps is not None:
            return self._hps.data.sampling_rate
        return 44100

    @property
    def text_config(self) -> TextProcessingConfig | None:
        return self._text_config

    @property
    def vram_breakdown(self) -> VramBreakdown | None:
        return self._vram_breakdown

    def check_and_split_text(
        self,
        text: str,
        language: Languages,
    ) -> list[str]:
        if self._text_config is None:
            return [text]
        from charavoice_engine.inference.text_processing import check_and_split_text as _check_and_split_text

        return _check_and_split_text(text, language, self._text_config)

    def check_and_split_text_with_len(
        self,
        text: str,
        language: Languages,
    ) -> list[tuple[str, int]]:
        if self._text_config is None:
            return [(text, 0)]
        from charavoice_engine.inference.text_processing import (
            check_and_split_text_with_len as _check_and_split_text_with_len,
        )

        return _check_and_split_text_with_len(text, language, self._text_config)
