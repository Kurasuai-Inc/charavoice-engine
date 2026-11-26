"""Constants and language enum for CHARAVOICE Engine.

Ported from style_bert_vits2.constants to remove the vendored dependency.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

# ベースディレクトリ
# 環境変数 CHARAVOICE_BASE_DIR でオーバーライド可能
# デフォルト: src/ の親 = リポジトリルート
BASE_DIR = (
    Path(os.environ["CHARAVOICE_BASE_DIR"])
    if "CHARAVOICE_BASE_DIR" in os.environ
    else Path(__file__).parent.parent.parent
)


class Languages(StrEnum):
    """Supported synthesis languages for the bilingual JP-Extra model."""

    JP = "JP"
    EN = "EN"


# 言語ごとのデフォルト BERT モデル (ONNX 版) パス
DEFAULT_ONNX_BERT_MODEL_PATHS: dict[Languages, Path] = {
    Languages.JP: BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm-onnx",
    Languages.EN: BASE_DIR / "bert" / "deberta-v3-large-onnx",
}

# デフォルトの推論パラメータ
DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 1.0
DEFAULT_NOISE = 0.6
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 1.0
