"""Export Generator (decoder) to ONNX with fixed input shape for TRT.

Usage:
    uv run python -m charavoice_engine.convert.export_decoder \
        --model Data/mix/models/G_272000.pth --config model_assets/mix/config.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_decoder_onnx(
    model_path: str,
    config_path: str,
    output_path: str,
    device: str = "cpu",
    opset_version: int = 17,
    simplify: bool = True,
    input_frames: int = 58,
) -> Path:
    """Export net_g.dec as fixed-shape ONNX for overlap-save decoding.

    Fixed input: z=[1, inter_channels, input_frames], g=[1, gin_channels, 1]
    No dynamic_axes — TRT can build a single optimized engine.

    Args:
        model_path: Path to .pth/.safetensors model file.
        config_path: Path to config.json.
        output_path: Path to save ONNX model.
        device: Device for export tracing.
        opset_version: ONNX opset version.
        simplify: Whether to run onnxsim.
        input_frames: Total mel frames per chunk (L + cs + R).

    Returns:
        Path to the saved ONNX model.
    """
    from charavoice_engine.config import HyperParameters
    from charavoice_engine.models.infer import get_net_g

    hps = HyperParameters.load_from_json(config_path)
    logger.info("Loading model from %s", model_path)
    net_g = get_net_g(model_path, hps.version, device, hps)

    decoder = net_g.dec
    decoder.remove_weight_norm()

    inter_channels = hps.model.inter_channels  # 192
    gin_channels = hps.model.gin_channels  # 512

    z_example = torch.randn(2, inter_channels, input_frames, device=device)
    g_example = torch.randn(2, gin_channels, 1, device=device)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(output.stem + "_temp.onnx")

    # Wrap forward to pass g as a positional argument accepted by torch.onnx.export.
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, dec: torch.nn.Module) -> None:
            super().__init__()
            self.dec = dec

        def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            return self.dec(z, g=g)

    wrapper = DecoderWrapper(decoder)
    wrapper.eval()

    logger.info("Exporting decoder to ONNX (input_frames=%d)...", input_frames)
    torch.onnx.export(
        model=wrapper,
        args=(z_example, g_example),
        f=str(temp_path),
        verbose=False,
        dynamo=False,
        opset_version=opset_version,
        input_names=["z", "g"],
        output_names=["audio"],
        dynamic_axes={"z": {0: "batch"}, "g": {0: "batch"}, "audio": {0: "batch"}},
    )
    logger.info("ONNX export complete: %s", temp_path)

    if simplify:
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
    else:
        temp_path.rename(output)

    logger.info("Saved decoder ONNX to %s (%.1f MB)", output, output.stat().st_size / 1e6)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Style-Bert-VITS2 decoder to ONNX")
    parser.add_argument("--model", "-m", required=True, help="Path to .pth/.safetensors model")
    parser.add_argument("--config", "-c", required=True, help="Path to config.json")
    parser.add_argument("--output", "-o", help="Output ONNX path (default: <model_dir>/decoder.onnx)")
    parser.add_argument("--device", default="cpu", help="Device for export")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--input-frames", type=int, default=58, help="Total mel frames per chunk (L+cs+R)")
    parser.add_argument("--no-simplify", action="store_true", help="Skip onnxsim")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.output is None:
        args.output = str(Path(args.config).parent / "decoder.onnx")

    export_decoder_onnx(
        args.model,
        args.config,
        args.output,
        args.device,
        args.opset,
        not args.no_simplify,
        args.input_frames,
    )


if __name__ == "__main__":
    main()
