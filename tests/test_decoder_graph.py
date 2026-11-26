from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from charavoice_engine.inference.decoder_graph import resize_decoder_graph


def _graph(path, weight=1.0, operation="Conv"):
    node = helper.make_node(operation, ["z", "weight"], ["audio"])
    model = helper.make_model(
        helper.make_graph(
            [node],
            "decoder",
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 1, 58])],
            [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, 58])],
            [numpy_helper.from_array(np.array([[[weight]]], dtype=np.float32), "weight")],
        )
    )
    onnx.save(model, path)


def test_derivation_preserves_served_weights_and_source(tmp_path):
    source = tmp_path / "decoder.onnx"
    _graph(source)
    original = source.read_bytes()
    target = resize_decoder_graph(source, 154, 1, tmp_path / "cache")
    model = onnx.load(target)
    assert model.graph.input[0].type.tensor_type.shape.dim[2].dim_value == 154
    assert model.graph.output[0].type.tensor_type.shape.dim[2].dim_value == 154
    assert model.graph.initializer == onnx.load(source).graph.initializer
    assert source.read_bytes() == original
    assert resize_decoder_graph(source, 154, 1, tmp_path / "cache") == target
    metadata = json.loads(target.with_suffix(".json").read_text())
    assert metadata["unchanged_initializers"] == 1
    target.write_bytes(b"corrupt cache")
    assert resize_decoder_graph(source, 154, 1, tmp_path / "cache").read_bytes() != b"corrupt cache"
    _graph(source, weight=2.0)
    assert resize_decoder_graph(source, 154, 1, tmp_path / "cache") != target


def test_unknown_graph_cannot_silently_change_time_semantics(tmp_path):
    source = tmp_path / "decoder.onnx"
    _graph(source, operation="Reshape")
    with pytest.raises(ValueError, match="unsupported"):
        resize_decoder_graph(source, 154, 1, tmp_path / "cache")
