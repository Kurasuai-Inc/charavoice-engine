# CHARAVOICE Engine

CHARAVOICE Engine は、Style-Bert-VITS2 JP-Extra を母体にした高速 TTS エンジンです。最優先の目的は、低レイテンシと高スループットを同時に伸ばすことです。その上で、ストリーミング配信、AudioSeal ウォーターマーク、外部からの `phones + tones + word2ph` 完全指定、日英統合モデル学習を 1 つの運用系として扱えるようにしています。

このリポジトリにはデフォルト音声モデルは含まれていません。利用前に [開発陣からのお願いとモデル利用上の注意](docs/TERMS_OF_USE.md) と、使用する各モデル・データセットの利用規約を確認してください。

## 性能優先

CHARAVOICE Engine は、GPU 上で TensorRT FP16 の分割推論を行い、HTTP batcher と streaming decoder で実運用時の待ち時間を抑えます。RTX 4090 では、通常テキストの WAV 生成で 448倍実時間、ストリーミング PCM でも 441倍実時間を確認しています。低負荷のストリーミングでは、TTFB 平均 33ms / P99 39ms で最初の音声チャンクを返せます。

| シナリオ | 条件 | スループット | 音声生成速度 | 平均レイテンシ | P99 レイテンシ | TTFB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WAV / TensorRT | JP テキスト、C=8、200リクエスト | 48.2 RPS | 448倍実時間 | 165.4ms | 239.1ms | - |
| ストリーミング PCM / TensorRT | JP テキスト、C=8、200リクエスト | 47.4 RPS | 441倍実時間 | 165.5ms | 249.8ms | 平均129.7ms |
| 低 TTFB ストリーミング | JP テキスト、C=1、`--stream-buffer-chunks 1` | 24.2 RPS | 218倍実時間 | 41.3ms | 58.1ms | 平均33.1ms / P99 38.7ms |
| WAV + AudioSeal TRT | JP テキスト、C=8、core `0x1234` | 40.7 RPS | 378倍実時間 | 194.6ms | 321.0ms | - |

TTFB はストリーミング時にクライアントが最初の音声バイトを受け取るまでの時間です。AudioSeal TRT はウォーターマークを実際に付与した条件で測り、検出側で `0x1234` をスコア `1.0` で復元できることも確認しています。計測条件、EN 入力、完全指定入力、ストリーミング書き出し単位比較、再現コマンドは [docs/BENCHMARKS.md](docs/BENCHMARKS.md) を参照してください。

## 主な機能

- TensorRT FP16 の `enc_p+dp` / `flow` / ストリーミング decoder 推論
- HTTP batcher とストリーミング PCM/OGG による低レイテンシ配信
- AudioSeal ウォーターマークと TensorRT ウォーターマーク backend
- `text` だけの通常入力と、`chunks[].surface_text + phones + tones + word2ph` の完全指定入力
- JP-Extra ベースの日英 bilingual モデル学習
- 日本語のみ、英語のみ、日英両方の話者データを同一モデルで扱えるデータセット recipe
- Prometheus metrics と Docker 運用

## インストール

```bash
git clone https://github.com/Kurasuai-Inc/charavoice-engine.git
cd charavoice-engine
bash install.sh
```

学習も行う環境では training 依存を追加します。

```bash
bash install.sh --train
```

JP-only 運用で EN BERT を入れたくない場合だけ `--jp-only` を指定します。

```bash
bash install.sh --jp-only
```

詳細な CLI は [docs/CLI.md](docs/CLI.md) を参照してください。

## データセット

データは 1 モデル 1 ディレクトリで、`raw/<style>/<speaker>/<language>/` に置きます。話者ごとに日本語だけ、英語だけ、または両方があっても構いません。

```text
Data/bilingual_v1/
  raw/
    Neutral/
      speaker_a/
        JP/a_jp_0001.wav
        JP/a_jp_0001.txt
        EN/a_en_0001.wav
        EN/a_en_0001.txt
    happy/
      speaker_a/
        JP/a_jp_happy_0001.wav
      speaker_b/
        EN/b_en_happy_0001.wav
```

同名 `.txt` がない場合は `esd.list` を用意します。

```text
/abs/path/Data/bilingual_v1/raw/Neutral/speaker_a/JP/a_jp_0001.wav|speaker_a|JP|今日はいい天気ですね。
/abs/path/Data/bilingual_v1/raw/Neutral/speaker_a/EN/a_en_0001.wav|speaker_a|EN|The weather is nice today.
```

`JP` と `EN` は学習時の G2P/BERT 経路を決めます。日本語文中の英語フレーズを英語として読ませたい場合は、推論時に `chunks` で分けて `language: "EN"` を指定します。

## 学習

1 コマンド recipe で前処理、BERT feature 生成、style vector 生成、学習、ONNX export を実行します。

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

JP-Extra generator checkpoint から開始する場合は `--source-jp-extra-g` を指定します。recipe が bilingual JP-Extra 用の初期重みを作成します。WavLM discriminator はデフォルトで有効です。無効化する場合だけ `--no-wavlm-discriminator` を指定します。

主な成果物:

```text
Data/bilingual_v1/
  esd.list
  esd.list.cleaned
  train.list
  val.list
  config.json
  dataset_report.json
  models/

model_assets/bilingual_v1/
  config.json
  style_vectors.npy
  bilingual_v1_e*_s*.safetensors
  decoder.onnx
  enc_p_dp.onnx
  flow.onnx
```

## サーバー

起動には、ライセンスの表示に使う 2 つが要ります（無ければ起動しません）: 今の版の commit（`CHARAVOICE_SOURCE_COMMIT`）と NOTICE の場所（`CHARAVOICE_NOTICE_PATH`、image の中では既定の `/app/NOTICE`）。公開の source の URL（`CHARAVOICE_SOURCE_URL`）は、Docker image と docker compose では既定で https://github.com/Kurasuai-Inc/charavoice-engine です（fork した時は自分の repo に向けてください）。無い時、`GET /source` は `source_available: false` を返します。

```bash
export CHARAVOICE_SOURCE_COMMIT=$(git rev-parse HEAD) CHARAVOICE_NOTICE_PATH=$PWD/NOTICE
uv run python -m charavoice_engine.scripts.run_server --dir model_assets
```

`GET /licenses` はクレジットと NOTICE を、`GET /source` は source の URL と commit を返します（アプリの「ライセンス」の画面が使います）。

VRAM を抑える場合:

```bash
uv run python -m charavoice_engine.scripts.run_server --dir model_assets --trt-workspace 2
```

初回起動時は TensorRT engine のビルドに時間がかかります。生成物は `trt_cache/` に保存されます。

## API

通常テキスト入力:

```bash
curl -X POST http://localhost:5000/voice \
  -H 'Content-Type: application/json' \
  --output jp.wav \
  -d '{
    "text": "こんにちは。今日はいい天気ですね。",
    "language": "JP",
    "model_id": 0,
    "speaker_id": 0,
    "style": "Neutral",
    "format": "wav"
  }'
```

日英混在入力:

```bash
curl -X POST http://localhost:5000/voice \
  -H 'Content-Type: application/json' \
  --output mixed.wav \
  -d '{
    "chunks": [
      {"language": "JP", "text": "Facebookは2021年10月28日にMetaという社名となったが、"},
      {"language": "EN", "text": "Extended Reality"},
      {"language": "JP", "text": "を前面に押し出している。"}
    ],
    "model_id": 0,
    "speaker_id": 0,
    "style": "Neutral",
    "format": "wav"
  }'
```

完全指定入力:

```bash
curl -X POST http://localhost:5000/voice \
  -H 'Content-Type: application/json' \
  --output control.wav \
  -d '{
    "chunks": [
      {
        "language": "JP",
        "surface_text": "今日",
        "phones": ["ky", "o:"],
        "tones": [1, 0],
        "word2ph": [1, 1]
      }
    ],
    "model_id": 0,
    "speaker_id": 0,
    "format": "wav"
  }'
```

外部 API では先頭末尾の `_` と blank intersperse 後の phone は渡しません。サーバー内部で付与します。

## Docker

```bash
CHARAVOICE_SOURCE_COMMIT=$(git rev-parse HEAD) docker compose up -d --build
docker compose logs -f
```

`CHARAVOICE_SOURCE_COMMIT` は build の時だけ要ります（`docker compose logs` などは無くても動きます）。無いまま build すると、Dockerfile がそこで止めます。

Docker entrypoint はデフォルトで JP/EN BERT をダウンロードし、batch BERT ONNX を生成します。

### 別の手順で build・起動する時

image を別の script や CI で作る時、また image の外の source を mount して `run_server` を動かす時は、次の 2 つが要ります。どちらも、無ければ止まります。

1. build: `docker build --build-arg CHARAVOICE_SOURCE_COMMIT=$(git -C <engine の source> rev-parse HEAD) …`。無いと、build が止まります。source を tar などで送って、送り先で build する時は、送る前に手元で commit を取って渡してください（`.git` は送られないため）。
2. 起動: `CHARAVOICE_SOURCE_COMMIT` と、読める・空でない NOTICE（`CHARAVOICE_NOTICE_PATH`、image では `/app/NOTICE`）。どちらかが欠けると、`run_server` は model を読む前に exit 2 で止まります。
   - image の外の source を mount して動かす時は、その source の commit と、その source の `NOTICE` を指してください（image の中の `/app/NOTICE` は、image を作った時の版です）。

```bash
CHARAVOICE_LANGS=JP docker compose up -d
```

## ベンチマーク

実測結果の詳細は [docs/BENCHMARKS.md](docs/BENCHMARKS.md) を参照してください。ローカル環境で測る場合は、サーバー起動後に以下を実行します。

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language JP \
  --input-mode text \
  --format wav \
  -c 8 -n 200 -w 16 \
  --seed 20260616
```

ストリーミング PCM:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language JP \
  --input-mode text \
  --streaming --format pcm \
  -c 8 -n 200 -w 16 \
  --seed 20260616
```

## 検証

```bash
uv run --extra dev --extra server pytest
uv run --extra dev ruff check src tests tools
uv run --extra dev --extra server pyright
```

## クレジット

CHARAVOICE Engine には [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) 由来のコードが含まれます。Style-Bert-VITS2 は [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) 由来です。また、日本語モーラ表には [VOICEVOX ENGINE](https://github.com/VOICEVOX/voicevox_engine) 由来のコード（その中に [Open JTalk](http://open-jtalk.sourceforge.net/) のモーラ表）が、英語の発音辞書には [The CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) が含まれます。

実行時に使うモデル（このリポジトリには含みません）:

- 日本語 BERT: [ku-nlp/deberta-v2-large-japanese-char-wwm](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm)（[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)）。ONNX 版は [tsukumijima/deberta-v2-large-japanese-char-wwm-onnx](https://huggingface.co/tsukumijima/deberta-v2-large-japanese-char-wwm-onnx)（CC BY-SA 4.0）で、エンジンはさらに FP16・バッチ版に変換して使います。
- 英語 BERT: [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)（MIT）
- 音声透かし: [AudioSeal](https://github.com/facebookresearch/audioseal)（MIT）

詳細は [NOTICE](NOTICE) と [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照してください。

## ライセンス

このリポジトリは GNU Affero General Public License v3.0 only でライセンスされています。詳細は [LICENSE](LICENSE) を参照してください。

このリポジトリには、学習済みのモデルの重み・声のデータ・話者のデータセットは含まれません。
