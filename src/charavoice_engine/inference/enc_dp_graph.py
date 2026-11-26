"""enc_p+dp ONNX graph adapter for bilingual JP-Extra models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from charavoice_engine.config import TensorRtConfig


class _BilingualJPExtraEncPDPWrapper(nn.Module):
    """Traceable enc_p + dp wrapper for the bilingual JP-Extra model."""

    def __init__(self, enc_p: nn.Module, dp: nn.Module) -> None:
        super().__init__()
        self.enc_p = enc_p
        self.dp = dp

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        ja_bert: torch.Tensor,
        en_bert: torch.Tensor,
        style_vec: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_out, m_p, logs_p, x_mask = self.enc_p(
            x,
            x_lengths,
            tone,
            language,
            ja_bert,
            en_bert,
            style_vec,
            g=g,
        )
        logw_dp = self.dp(x_out, x_mask, g=g)
        return m_p, logs_p, x_mask, logw_dp


class BilingualJPExtraEncDpGraphAdapter:
    """Adapter for the single supported enc_p+dp ONNX subgraph."""

    input_names = ("x", "x_lengths", "tone", "language", "ja_bert", "en_bert", "style_vec", "g")

    def build_wrapper(self, enc_p: nn.Module, dp: nn.Module) -> nn.Module:
        """Return a traceable enc_p+dp wrapper."""
        return _BilingualJPExtraEncPDPWrapper(enc_p, dp)

    def export_args(self, batch: int, seq_len: int, gin_channels: int, device: str) -> tuple[torch.Tensor, ...]:
        """Return representative export tensors matching ``input_names``."""
        return (
            torch.randint(0, 100, (batch, seq_len), device=device),
            torch.full((batch,), seq_len, dtype=torch.long, device=device),
            torch.randint(0, 12, (batch, seq_len), device=device),
            torch.randint(0, 3, (batch, seq_len), device=device),
            torch.randn(batch, 1024, seq_len, device=device),
            torch.randn(batch, 1024, seq_len, device=device),
            torch.randn(batch, 256, device=device),
            torch.randn(batch, gin_channels, 1, device=device),
        )

    def warmup_feed(self, batch: int, seq_len: int, gin_channels: int) -> dict[str, NDArray[Any]]:
        """Return zero-filled ORT feed for session warmup."""
        return {
            "x": np.zeros((batch, seq_len), dtype=np.int64),
            "x_lengths": np.full(batch, seq_len, dtype=np.int64),
            "tone": np.zeros((batch, seq_len), dtype=np.int64),
            "language": np.zeros((batch, seq_len), dtype=np.int64),
            "ja_bert": np.zeros((batch, 1024, seq_len), dtype=np.float32),
            "en_bert": np.zeros((batch, 1024, seq_len), dtype=np.float32),
            "style_vec": np.zeros((batch, 256), dtype=np.float32),
            "g": np.zeros((batch, gin_channels, 1), dtype=np.float32),
        }

    def runtime_feed(
        self,
        preprocessed_list: list[dict[str, Any]],
        g_np: NDArray[Any],
    ) -> tuple[dict[str, NDArray[Any]], list[int]]:
        """Return padded ORT feed and per-request phone lengths."""
        B = len(preprocessed_list)
        t_phones = [p["phones"].shape[0] for p in preprocessed_list]
        T_max = max(t_phones)

        feed: dict[str, NDArray[Any]] = {
            "x": np.zeros((B, T_max), dtype=np.int64),
            "x_lengths": np.array(t_phones, dtype=np.int64),
            "tone": np.zeros((B, T_max), dtype=np.int64),
            "language": np.zeros((B, T_max), dtype=np.int64),
            "ja_bert": np.zeros((B, 1024, T_max), dtype=np.float32),
            "en_bert": np.zeros((B, 1024, T_max), dtype=np.float32),
            "style_vec": np.zeros((B, 256), dtype=np.float32),
            "g": g_np,
        }
        for i, p in enumerate(preprocessed_list):
            T = t_phones[i]
            feed["x"][i, :T] = p["phones"]
            feed["tone"][i, :T] = p["tones"]
            feed["language"][i, :T] = p["lang_ids"]
            feed["ja_bert"][i, :, :T] = p["ja_bert"]
            feed["en_bert"][i, :, :T] = p["en_bert"]
            feed["style_vec"][i] = p["style_vec"]
        return feed, t_phones

    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Dynamic axes for the ONNX export."""
        return {
            "x": {0: "batch", 1: "T_phone"},
            "x_lengths": {0: "batch"},
            "tone": {0: "batch", 1: "T_phone"},
            "language": {0: "batch", 1: "T_phone"},
            "ja_bert": {0: "batch", 2: "T_phone"},
            "en_bert": {0: "batch", 2: "T_phone"},
            "style_vec": {0: "batch"},
            "g": {0: "batch"},
            "m_p": {0: "batch", 2: "T_phone"},
            "logs_p": {0: "batch", 2: "T_phone"},
            "x_mask": {0: "batch", 2: "T_phone"},
            "logw_dp": {0: "batch", 2: "T_phone"},
        }

    def trt_profile_shapes(self, trt_config: TensorRtConfig, gin_channels: int, batch: int, seq_len: int) -> str:
        """Build a TensorRT profile shape string for this graph."""
        shapes = {
            "x": f"{batch}x{seq_len}",
            "x_lengths": f"{batch}",
            "tone": f"{batch}x{seq_len}",
            "language": f"{batch}x{seq_len}",
            "ja_bert": f"{batch}x1024x{seq_len}",
            "en_bert": f"{batch}x1024x{seq_len}",
            "style_vec": f"{batch}x256",
            "g": f"{batch}x{gin_channels}x1",
        }
        return ",".join(f"{name}:{shapes[name]}" for name in self.input_names)


def get_enc_dp_graph_adapter() -> BilingualJPExtraEncDpGraphAdapter:
    """Return the single supported enc_p+dp graph adapter."""
    return BilingualJPExtraEncDpGraphAdapter()


def export_enc_dp_graph_onnx(
    adapter: BilingualJPExtraEncDpGraphAdapter,
    net_g: nn.Module,
    output_path: str,
    device: str,
    opset_version: int,
    sample_len: int,
    gin_channels: int,
    simplify: Any,
) -> Path:
    """Export an enc_p+dp graph through the bilingual JP-Extra adapter."""
    wrapper = adapter.build_wrapper(net_g.enc_p, net_g.dp)  # type: ignore[attr-defined]
    wrapper.eval()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(output.stem + "_temp.onnx")

    args = adapter.export_args(batch=2, seq_len=sample_len, gin_channels=gin_channels, device=device)
    torch.onnx.export(
        model=wrapper,
        args=args,
        f=str(temp_path),
        verbose=False,
        dynamo=False,
        opset_version=opset_version,
        input_names=list(adapter.input_names),
        output_names=["m_p", "logs_p", "x_mask", "logw_dp"],
        dynamic_axes=adapter.dynamic_axes(),
    )
    simplify(temp_path, output)
    return output
