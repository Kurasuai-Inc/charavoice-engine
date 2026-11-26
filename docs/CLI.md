# CLI

## 0. インストール

```bash
git clone https://github.com/Kurasuai-Inc/charavoice-engine.git
cd charavoice-engine
bash install.sh
```

オプション:

- `--train`: 学習用依存もインストールします。
- `--jp-only`: 日本語 BERT だけをダウンロードします。
- `--skip-bert`: BERT download と batch ONNX 生成をスキップします。

## 1. データセット

学習データは `Data/<model_name>/raw/<style>/<speaker>/<language>/` に配置します。

```text
Data/bilingual_v1/
  raw/
    Neutral/
      speaker_a/
        JP/a_0001.wav
        JP/a_0001.txt
        EN/a_0002.wav
        EN/a_0002.txt
```

対応する language directory は `JP` と `EN` です。話者ごとに片方の言語だけでも構いません。同名 `.txt` がない場合は `Data/<model_name>/esd.list` を用意します。

`esd.list` 形式:

```text
/abs/path/to/audio.wav|speaker_name|JP|本文
/abs/path/to/audio.wav|speaker_name|EN|Text
```

## 2. 学習

```bash
uv run python -m charavoice_engine.training.recipe \
  --dataset Data/bilingual_v1 \
  --model-name bilingual_v1 \
  --assets-dir model_assets/bilingual_v1 \
  --epochs 1000 \
  --batch-size 6 \
  --source-jp-extra-g /path/to/G_0.safetensors \
  --initial-d /path/to/D_0.safetensors \
  --initial-wd /path/to/WD_0.safetensors
```

主なオプション:

- `--batch-size`: 学習 batch size。WavLM discriminator 有効時の RTX 4090 class GPU では、まず `6` 前後から開始します。
- `--epochs`: 総 epoch 数。
- `--save-every-epoch`: checkpoint 保存間隔。
- `--source-jp-extra-g`: bilingual JP-Extra 初期重みに使う JP-Extra generator checkpoint。
- `--initial-d`: discriminator checkpoint。
- `--initial-wd`: WavLM discriminator checkpoint。
- `--no-wavlm-discriminator`: WavLM discriminator を無効化します。
- `--skip-train`: dataset preparation と export validation だけを実行し、学習は行いません。

## 3. サーバー

```bash
uv run python -m charavoice_engine.scripts.run_server \
  --dir model_assets \
  --host 0.0.0.0 \
  --port 5000
```

主なオプション:

- `--max-batch-size`: HTTP batcher の最大 batch size。
- `--inference-profile throughput128`: 採用済み128フレーム構成。[動作・検証範囲](throughput128.md)。省略時は従来の `default`。
- `--batch-timeout-min` / `--batch-timeout-max`: adaptive batching window。単位は millisecond。
- `--trt-cache`: TensorRT engine cache directory。
- `--trt-workspace`: TensorRT workspace size。単位は GiB。
- `--stream-buffer-chunks`: PCM streaming でHTTPへまとめて書き出す decoder chunk 数。標準値は `10`。TTFBを優先する場合は `1` を指定します。
- `--watermark`: `none`、`audioseal`、`audioseal_trt`。
- `--watermark-core16`: default 16-bit watermark word。
- `--watermark-extension`: default extension payload。hex 文字列で指定します。
- `--g2p-workers`: 日本語 G2P process pool size。

## 4. ベンチマーク

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language JP \
  --input-mode text \
  -c 8 -n 1000
```

Streaming PCM:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language EN \
  --input-mode text \
  --streaming --format pcm \
  -c 8 -n 1000
```

主なベンチマーク用オプション:

- `--api-key` / `--api-key-env` / `--api-key-header`: public LB など API key が必要な経路で認証 header を付与します。
- `--header "Name: value"`: 任意の HTTP header を追加します。複数回指定できます。
- `--insecure`: 固定 IP の HTTPS smoke/bench などで TLS 証明書検証を無効化します。
- `--status-url`: ベンチ後に表示する stats endpoint を指定します。複数 worker の aggregate endpoint では `/cluster-status` を指定すると全 worker の healthy 状態と request 数を確認できます。
- `--csv`: request ごとの latency、音声長、status、失敗 body を CSV に保存します。

local aggregate の例:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://127.0.0.1:5000 \
  --status-url http://127.0.0.1:5000/cluster-status \
  --language JP \
  --input-mode text \
  --format wav \
  -c 8 -n 200 -w 24 \
  --seed 20260622
```

## 5. Watermark ツール

AudioSeal watermark 検出:

```bash
uv run python tools/detect_audioseal_watermark.py output.wav \
  --core16 0x1234 \
  --extension beef
```

頑健性評価:

```bash
uv run python tools/eval_watermark_robustness.py \
  --input output.wav \
  --core16 0x1234
```
