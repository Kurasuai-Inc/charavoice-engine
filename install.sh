#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CHARAVOICE Engine install script
# セットアップを一括で行う: 前提チェック → uv → 依存 → BERT/SLM モデル
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- デフォルト設定 ---
SKIP_BERT=false
SKIP_SLM=false
INSTALL_TRAIN=false
JP_ONLY=false

# --- 色付き出力 ---
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }

usage() {
    cat <<'EOF'
Usage: bash install.sh [OPTIONS]

CHARAVOICE Engine の環境セットアップを一括で行います。

Options:
  --jp-only      JP BERT のみをダウンロード (デフォルト: JP/EN)
  --train        学習用の追加依存もインストール
  --skip-bert    BERT モデルのダウンロードをスキップ
  --skip-slm     学習用 WavLM SLM のダウンロードをスキップ
  -h, --help     このヘルプを表示

処理内容:
  1. 前提条件チェック (NVIDIA ドライバ, CUDA 12.x)
  2. uv インストール (未導入の場合)
  3. Python 依存パッケージのインストール (uv sync)
  4. BERT ONNX モデルのダウンロード (HuggingFace)
  5. --train 指定時に WavLM SLM を slm/wavlm-base-plus へダウンロード

Example:
  bash install.sh              # JP/EN BERT
  bash install.sh --train      # JP/EN BERT + 学習依存 + WavLM SLM
  bash install.sh --jp-only    # JP BERT のみ
EOF
    exit 0
}

# --- 引数パース ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jp-only) JP_ONLY=true; shift ;;
        --train) INSTALL_TRAIN=true; shift ;;
        --skip-bert) SKIP_BERT=true; shift ;;
        --skip-slm) SKIP_SLM=true; shift ;;
        -h|--help)   usage ;;
        *) error "Unknown option: $1"; usage ;;
    esac
done

# ============================================================
# 1. 前提条件チェック
# ============================================================
info "前提条件をチェックしています..."

# git
if ! command -v git &>/dev/null; then
    error "git が見つかりません。インストールしてください。"
    exit 1
fi
ok "git: $(git --version)"

# nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi が見つかりません。NVIDIA ドライバをインストールしてください。"
    exit 1
fi

# CUDA ドライババージョンチェック
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
NVIDIA_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
if [[ -z "$NVIDIA_CUDA" ]]; then
    warn "CUDA バージョンを検出できませんでした。nvidia-smi の出力を確認してください。"
elif [[ "$NVIDIA_CUDA" -lt 12 ]]; then
    error "CUDA 12.x 以上が必要です (検出: CUDA $NVIDIA_CUDA)。ドライバを更新してください。"
    exit 1
else
    ok "NVIDIA ドライバ: $CUDA_VERSION (CUDA $NVIDIA_CUDA)"
fi

# GPU 情報表示
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
info "GPU: $GPU_NAME ($GPU_VRAM)"

# ============================================================
# 2. uv インストール
# ============================================================
if command -v uv &>/dev/null; then
    ok "uv: $(uv --version)"
else
    info "uv をインストールしています..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # パスを通す (現在のシェルセッション用)
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv: $(uv --version)"
    else
        error "uv のインストールに失敗しました。"
        exit 1
    fi
fi

# ============================================================
# 3. Python 依存パッケージ
# ============================================================
info "Python 依存パッケージをインストールしています..."
if [[ "$INSTALL_TRAIN" == true ]]; then
    uv sync --extra server --extra tensorrt --extra train
else
    uv sync --extra server --extra tensorrt
fi
ok "依存パッケージのインストール完了"

# ============================================================
# 4. BERT ONNX モデルのダウンロード
# ============================================================
download_bert() {
    local repo="$1"
    local dest="$2"

    if [[ -f "$dest/model_fp16.onnx" ]]; then
        ok "BERT: $dest (既にダウンロード済み、スキップ)"
        return
    fi

    info "BERT モデルをダウンロード: $repo → $dest"
    mkdir -p "$dest"
    uv run huggingface-cli download "$repo" --local-dir "$dest"
    ok "BERT: $dest"
}

if [[ "$SKIP_BERT" == true ]]; then
    info "BERT モデルのダウンロードをスキップ (--skip-bert)"
else
    info "BERT ONNX モデルをダウンロードしています..."

    # JP (必須)
    download_bert \
        "tsukumijima/deberta-v2-large-japanese-char-wwm-onnx" \
        "bert/deberta-v2-large-japanese-char-wwm-onnx"

    if [[ "$JP_ONLY" != true ]]; then
        download_bert \
            "tsukumijima/deberta-v3-large-onnx" \
            "bert/deberta-v3-large-onnx"
    fi

    # Generate batch models
    info "バッチ BERT モデルを生成しています..."
    for bert_dir in bert/*-onnx; do
        if [[ -f "$bert_dir/model_fp16.onnx" ]]; then
            uv run python -m charavoice_engine.scripts.make_batch_bert "$bert_dir"
        fi
    done

    ok "BERT モデルのダウンロード完了"
fi

# ============================================================
# 5. 学習用 WavLM SLM モデルのダウンロード
# ============================================================
download_slm() {
    local repo="$1"
    local dest="$2"

    if [[ -f "$dest/pytorch_model.bin" ]]; then
        ok "SLM: $dest (既にダウンロード済み、スキップ)"
        return
    fi

    info "WavLM SLM をダウンロード: $repo → $dest"
    mkdir -p "$dest"
    uv run huggingface-cli download "$repo" --local-dir "$dest"
    ok "SLM: $dest"
}

if [[ "$INSTALL_TRAIN" == true ]]; then
    if [[ "$SKIP_SLM" == true ]]; then
        info "WavLM SLM のダウンロードをスキップ (--skip-slm)"
    else
        download_slm "microsoft/wavlm-base-plus" "slm/wavlm-base-plus"
    fi
else
    info "WavLM SLM のダウンロードをスキップ (--train 未指定)"
fi

# ============================================================
# 完了
# ============================================================
echo ""
info "=========================================="
info "  CHARAVOICE Engine セットアップ完了!"
info "=========================================="
echo ""
info "次のステップ:"
echo "  1. model_assets/ にモデルを配置"
echo "     model_assets/<モデル名>/"
echo "       ├── <モデル名>.safetensors"
echo "       ├── config.json"
echo "       ├── decoder.onnx / enc_p_dp.onnx / flow.onnx"
echo "       └── style_vectors.npy"
echo ""
echo "  2. サーバー起動"
echo "     uv run python -m charavoice_engine.scripts.run_server --dir model_assets"
echo ""
echo "  3. 学習を行う場合"
echo "     bash install.sh --train"
echo "     uv run python -m charavoice_engine.training.recipe --dataset Data/bilingual_v1 --model-name bilingual_v1 --assets-dir model_assets/bilingual_v1 ..."
echo "     ※ WavLM SLM は slm/wavlm-base-plus に配置されます"
echo ""
echo "  ※ 初回起動時は TensorRT エンジンのビルドに ~10分/モデルかかります。"
echo "  ※ VRAM が 12GB 以下の場合は --trt-workspace 2 を推奨:"
echo "     uv run python -m charavoice_engine.scripts.run_server --dir model_assets --trt-workspace 2"
