"""Export enc_p+dp and flow components to ONNX for TRT acceleration.

enc_p+dp are combined into a single ONNX model to eliminate intermediate
tensor transfers and allow TRT to optimize across component boundaries.

flow is exported separately with reverse=True baked in (trace-based export
captures only the reverse branch).

Usage:
    from charavoice_engine.convert.export_components import export_enc_p_dp_onnx, export_flow_onnx
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from charavoice_engine.inference.enc_dp_graph import export_enc_dp_graph_onnx, get_enc_dp_graph_adapter

logger = logging.getLogger(__name__)


class _FlowInverseWrapper(nn.Module):
    """Wraps flow.forward(reverse=True) so the trace only captures the reverse path."""

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, z_p: torch.Tensor, y_mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.flow(z_p, y_mask, g=g, reverse=True)


def _simplify_onnx(temp_path: Path, output: Path) -> None:
    """Run onnxsim simplification and fail if simplification is unavailable."""
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify
    except ImportError as exc:
        raise RuntimeError("onnxsim and onnx are required for production ONNX export") from exc

    logger.info("Running onnxsim simplification...")
    onnx_model = onnx.load(str(temp_path))
    simplified, check = onnx_simplify(onnx_model)
    if not check:
        raise RuntimeError(f"onnxsim validation failed for {temp_path}")

    onnx.save(simplified, str(output))
    temp_path.unlink()
    orig_size = onnx_model.ByteSize() / 1e6
    new_size = simplified.ByteSize() / 1e6
    logger.info("Simplified: %.1fMB -> %.1fMB", orig_size, new_size)


def export_enc_p_dp_onnx(
    net_g: nn.Module,
    output_path: str,
    device: str = "cpu",
    opset_version: int = 17,
    sample_len: int = 30,
    gin_channels: int = 512,
) -> Path:
    """Export enc_p + dp as a single ONNX model.

    Args:
        net_g: Loaded SynthesizerTrn in eval mode.
        output_path: Path to save ONNX model.
        device: Device for export tracing.
        opset_version: ONNX opset version.
        sample_len: Sample phone sequence length for tracing.
        gin_channels: Speaker conditioning channel count.

    Returns:
        Path to the saved ONNX model.
    """
    adapter = get_enc_dp_graph_adapter()
    logger.info(
        "Exporting bilingual JP-Extra enc_p+dp to ONNX (sample_len=%d)...",
        sample_len,
    )
    output = export_enc_dp_graph_onnx(
        adapter=adapter,
        net_g=net_g,
        output_path=output_path,
        device=device,
        opset_version=opset_version,
        sample_len=sample_len,
        gin_channels=gin_channels,
        simplify=_simplify_onnx,
    )
    logger.info("Saved enc_p+dp ONNX to %s (%.1f MB)", output, output.stat().st_size / 1e6)
    return output


def export_flow_onnx(
    net_g: nn.Module,
    output_path: str,
    device: str = "cpu",
    opset_version: int = 17,
    inter_channels: int = 192,
    gin_channels: int = 512,
    sample_mel_len: int = 100,
) -> Path:
    """Export flow (reverse=True) as ONNX.

    Args:
        net_g: Loaded bilingual JP-Extra SynthesizerTrn in eval mode.
        output_path: Path to save ONNX model.
        device: Device for export tracing.
        opset_version: ONNX opset version.
        inter_channels: Latent channels (192).
        gin_channels: Speaker embedding channels (512).
        sample_mel_len: Sample mel length for tracing.

    Returns:
        Path to the saved ONNX model.
    """
    wrapper = _FlowInverseWrapper(net_g.flow)  # type: ignore[reportArgumentType]
    wrapper.eval()

    T_mel = sample_mel_len
    z_p = torch.randn(2, inter_channels, T_mel, device=device)
    y_mask = torch.ones(2, 1, T_mel, device=device)
    g = torch.randn(2, gin_channels, 1, device=device)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(output.stem + "_temp.onnx")

    logger.info("Exporting flow (reverse) to ONNX (sample_mel_len=%d)...", T_mel)
    torch.onnx.export(
        model=wrapper,
        args=(z_p, y_mask, g),
        f=str(temp_path),
        verbose=False,
        dynamo=False,
        opset_version=opset_version,
        input_names=["z_p", "y_mask", "g"],
        output_names=["z"],
        dynamic_axes={
            "z_p": {0: "batch", 2: "T_mel"},
            "y_mask": {0: "batch", 2: "T_mel"},
            "g": {0: "batch"},
            "z": {0: "batch", 2: "T_mel"},
        },
    )
    logger.info("ONNX export complete: %s", temp_path)

    _simplify_onnx(temp_path, output)
    logger.info("Saved flow ONNX to %s (%.1f MB)", output, output.stat().st_size / 1e6)
    return output
