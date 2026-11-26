from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import onnxruntime


def torch_device_to_onnx_providers(
    device: str,
) -> Sequence[str | tuple[str, dict[str, Any]]]:
    """
    PyTorch のデバイス種別を ONNX の ExecutionProvider に変換する

    Args:
        device (str): PyTorch のデバイス種別

    Returns:
        Sequence[str | tuple[str, dict[str, Any]]]: ExecutionProvider のリスト
    """

    if device.startswith("cuda"):
        return [
            # cudnn_conv_algo_search を DEFAULT にすると推論速度が大幅に向上する
            ## ref: https://medium.com/neuml/debug-onnx-gpu-performance-c9290fe07459
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "DEFAULT"}),
            # CUDA が利用できない場合、可能であれば DirectML を利用する (明示的な device_id 指定が必要)
            ## device_id: 0 は、システムにインストールされているプライマリディスプレイ用 GPU に対応する
            ## プライマリディスプレイ用 GPU (GPU 0) よりも性能の高い GPU が接続されている環境では、 適宜 device_id を変更する必要がある
            ## ref: https://github.com/w-okada/voice-changer/issues/410#issuecomment-1627994911
            ("DmlExecutionProvider", {"device_id": 0}),
            # arena_extend_strategy を kSameAsRequested にすると、推論セッションによって作成される
            # メモリアリーナが、実際に推論に必要な容量以上にメモリを確保する問題を防ぐことができる
            ## ref: https://github.com/microsoft/onnxruntime/issues/11627#issuecomment-1137668551
            ## ref: https://skottmckay.github.io/onnxruntime/docs/reference/api/c-api.html
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ]  # fmt: skip
    else:
        return [
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ]


def get_onnx_device_options(
    onnx_session: onnxruntime.InferenceSession,
) -> tuple[str, int, onnxruntime.RunOptions]:
    """
    ONNX 推論時のデバイス関連のオプションを取得する

    Args:
        onnx_session (onnxruntime.InferenceSession): 初期化済みの ONNX セッション

    Returns:
        tuple[str, int, onnxruntime.RunOptions]: 入力テンソルの転送に使用するデバイス種別, デバイス ID, 実行オプション
    """

    # ONNX セッションに対応する SessionOptions を取得
    sess_options = onnx_session.get_session_options()

    # 実際に推論に用いられる ExecutionProvider を取得
    first_provider = onnx_session.get_providers()[0]

    # 入力テンソルを転送する推論デバイスを取得
    ## GPU への I/O Binding は CUDA と DirectML 以外はサポートされていないので、
    ## それ以外の ExecutionProvider が指定された場合は CPU に入力テンソルを転送する
    if first_provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        device_type = "cuda"
    elif first_provider == "DmlExecutionProvider":
        device_type = "dml"
    else:
        device_type = "cpu"

    # 入力テンソルを転送する GPU デバイスの ID をセッション自身から取得
    device_id = 0
    if device_type != "cpu":
        provider_options = onnx_session.get_provider_options()
        first_provider_opts = provider_options.get(first_provider, {})
        if "device_id" in first_provider_opts:
            device_id = int(first_provider_opts["device_id"])

    # 推論後にメモリアリーナを縮小し、確保されていたメモリを解放する
    run_options = onnxruntime.RunOptions()
    if sess_options.enable_cpu_mem_arena is True:
        if first_provider == "CPUExecutionProvider":
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")  # fmt: skip
        elif first_provider == "DmlExecutionProvider":
            pass
        elif first_provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"cpu:0;gpu:{device_id}")  # fmt: skip

    return device_type, device_id, run_options
