#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CHARAVOICE Engine Docker entrypoint
# BERT ONNX モデルの自動ダウンロード + batch 変換 → サーバー起動
# ============================================================

# --- BERT モデル定義 ---
declare -A BERT_REPOS=(
    [JP]="tsukumijima/deberta-v2-large-japanese-char-wwm-onnx"
    [EN]="tsukumijima/deberta-v3-large-onnx"
)
declare -A BERT_DIRS=(
    [JP]="/app/bert/deberta-v2-large-japanese-char-wwm-onnx"
    [EN]="/app/bert/deberta-v3-large-onnx"
)

# CHARAVOICE_LANGS: ダウンロードする言語 (カンマ区切り, デフォルト: JP,EN)
LANGS="${CHARAVOICE_LANGS:-JP,EN}"

info()  { printf '\033[1;34m[entrypoint]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[entrypoint]\033[0m %s\n' "$*"; }

download_bert() {
    local lang="$1"
    local repo="${BERT_REPOS[$lang]}"
    local dest="${BERT_DIRS[$lang]}"

    if [[ -f "$dest/model_fp16.onnx" ]]; then
        ok "BERT $lang: already downloaded"
    else
        info "Downloading BERT $lang: $repo"
        mkdir -p "$dest"
        uv run --no-sync huggingface-cli download "$repo" --local-dir "$dest"
        ok "BERT $lang: download complete"
    fi

    # Generate batch model if missing
    if [[ ! -f "$dest/model_fp16_batch.onnx" ]]; then
        info "Generating batch BERT model for $lang..."
        uv run --no-sync python -m charavoice_engine.scripts.make_batch_bert "$dest"
        ok "BERT $lang: batch model generated"
    fi
}

# --- BERT ダウンロード + batch 変換 ---
IFS=',' read -ra LANG_ARRAY <<< "$LANGS"
for lang in "${LANG_ARRAY[@]}"; do
    lang=$(echo "$lang" | tr -d ' ' | tr '[:lower:]' '[:upper:]')
    if [[ -n "${BERT_REPOS[$lang]+x}" ]]; then
        download_bert "$lang"
    else
        info "Unknown language: $lang (supported: JP, EN)"
    fi
done

# --- サーバー起動 ---
info "Starting CHARAVOICE Engine server..."
exec uv run --no-sync python -m charavoice_engine.scripts.run_server "$@"
