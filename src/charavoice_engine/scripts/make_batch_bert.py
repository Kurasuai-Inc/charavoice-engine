"""Convert single-output BERT ONNX model to batch-output version.

The HuggingFace ONNX BERT models end with a Gather node that extracts
only the [CLS] token, producing output shape [batch, 1024].

The batch version removes only the Gather node and reconnects the Cast
node to the LayerNorm output, producing [batch, seq_len, 1024].

Usage:
    uv run python -m charavoice_engine.scripts.make_batch_bert bert/deberta-v2-large-japanese-char-wwm-onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import onnx

logger = logging.getLogger(__name__)


def convert_to_batch(input_path: Path, output_path: Path) -> None:
    """Convert a single-output BERT ONNX model to batch output.

    Original graph tail: LayerNorm → Gather([CLS]) → Cast(fp32→fp16) → "output"
    Batch graph tail:    LayerNorm → Cast(fp32→fp16) → "output"

    Only removes Gather; keeps Cast to preserve fp16 type consistency.
    """
    logger.info("Loading %s", input_path)
    model = onnx.load(str(input_path))

    nodes = list(model.graph.node)
    gather_idx = None
    cast_idx = None
    layernorm_output = None

    # Walk backwards to find Cast → Gather pattern
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if node.op_type == "Cast" and node.output[0] == "output":
            cast_idx = i
        elif node.op_type == "Gather" and any(n.op_type == "Cast" and node.output[0] in n.input for n in nodes):
            gather_idx = i
            layernorm_output = node.input[0]
            break

    if gather_idx is None or cast_idx is None or layernorm_output is None:
        logger.error("Could not find Gather->Cast pattern at end of graph")
        sys.exit(1)

    # Reconnect Cast input: Gather output → LayerNorm output
    cast_node = model.graph.node[cast_idx]
    gather_output = model.graph.node[gather_idx].output[0]
    for j, inp in enumerate(cast_node.input):
        if inp == gather_output:
            cast_node.input[j] = layernorm_output

    # Remove only Gather
    del model.graph.node[gather_idx]

    # Update output shape to [batch, seq, 1024] (keep original elem_type)
    out_info = model.graph.output[0]
    out_info.type.tensor_type.shape.ClearField("dim")
    d0 = out_info.type.tensor_type.shape.dim.add()
    d0.dim_param = "batch_size"
    d1 = out_info.type.tensor_type.shape.dim.add()
    d1.dim_param = "sequence_length"
    d2 = out_info.type.tensor_type.shape.dim.add()
    d2.dim_value = 1024

    logger.info("Saving batch model to %s", output_path)
    onnx.save(model, str(output_path))
    logger.info("Done (%.1f MB)", output_path.stat().st_size / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BERT ONNX model to batch version")
    parser.add_argument("bert_dir", type=Path, help="BERT model directory (contains model_fp16.onnx)")
    parser.add_argument("--input", default="model_fp16.onnx", help="Input model filename")
    parser.add_argument("--output", default="model_fp16_batch.onnx", help="Output model filename")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_path = args.bert_dir / args.input
    output_path = args.bert_dir / args.output

    if not input_path.exists():
        logger.error("Input model not found: %s", input_path)
        sys.exit(1)

    if output_path.exists():
        logger.info("Batch model already exists: %s (skipping)", output_path)
        return

    convert_to_batch(input_path, output_path)


if __name__ == "__main__":
    main()
