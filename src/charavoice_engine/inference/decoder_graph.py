"""Derive larger decoder shapes from the served ONNX, preserving its weights."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def resize_decoder_graph(source: Path, input_frames: int, hop_length: int, cache_root: Path) -> Path:
    """Change only a fully convolutional decoder's time axis in a content-keyed cache.

    The checkpoint in the model directory may differ from the served ONNX. Never
    re-export its decoder to implement a different chunk size.
    """
    import onnx

    model = onnx.load(source)
    latent = next(value for value in model.graph.input if value.name == "z")
    if latent.type.tensor_type.shape.dim[2].dim_value == input_frames:
        return source
    allowed = {"Conv", "ConvTranspose", "Add", "LeakyRelu", "Div", "Tanh"}
    unsupported = {node.op_type for node in model.graph.node} - allowed
    if unsupported:
        raise ValueError(f"Cannot resize decoder with unsupported time-dependent operators: {sorted(unsupported)}")
    if input_frames < 1 or hop_length < 1:
        raise ValueError("input_frames and hop_length must be positive")
    source_hash = hashlib.sha256(model.SerializeToString()).hexdigest()
    folder = cache_root / "derived_onnx" / source_hash
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"decoder_f{input_frames}_h{hop_length}.onnx"
    metadata = target.with_suffix(".json")
    if target.exists() and metadata.exists():
        try:
            saved = json.loads(metadata.read_text())
            if saved["output_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest():
                return target
        except (ValueError, KeyError):
            pass
    initializers = {value.name: value.SerializeToString() for value in model.graph.initializer}
    latent.type.tensor_type.shape.dim[2].dim_value = input_frames
    model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = input_frames * hop_length
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    if {value.name: value.SerializeToString() for value in model.graph.initializer} != initializers:
        raise RuntimeError("Decoder shape inference changed served ONNX weights")
    handle, temporary_name = tempfile.mkstemp(dir=folder, suffix=".onnx")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        onnx.save(model, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    metadata.write_text(
        json.dumps(
            {
                "source_sha256": source_hash,
                "input_frames": input_frames,
                "unchanged_initializers": len(initializers),
                "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    return target
