# ベンチマーク

CHARAVOICE Engine は、Style-Bert-VITS2 系の音声合成を低レイテンシ・高スループットで動かすことを最優先にした推論エンジンです。ストリーミング配信、AudioSeal ウォーターマーク、`phones + tones + word2ph` の完全指定入力、JP/EN 統合モデルを同じ推論経路で扱います。

このページでは README に載せる代表値よりも詳しく、計測条件、指標の意味、補助計測、再現コマンドをまとめます。数値は実測値であり、モデルサイズ、文章長、GPU clock、driver/runtime のバージョン、TensorRT キャッシュ、batcher の状態、ウォーターマーク設定で変わります。

## 指標の見方

| 指標 | 意味 |
| --- | --- |
| スループット | 1秒あたりに完了したリクエスト数。HTTP、G2P/BERT、batcher、decoder、ウォーターマーク処理を含みます。 |
| 音声生成速度 | 1秒の処理時間で何秒分の音声を生成できたか。`500倍実時間` は 1秒で約500秒分の音声を生成する意味です。 |
| 平均レイテンシ | リクエスト開始から全音声受信完了までの平均時間です。 |
| P99 レイテンシ | 遅い側 1% のリクエストを見る指標です。 |
| TTFB | ストリーミング時に最初の音声バイトを受け取るまでの時間です。 |

## 計測環境

| 項目 | 値 |
| --- | --- |
| 計測日 | 2026-06-16 |
| GPU | NVIDIA GeForce RTX 4090 / 24GB |
| ドライバー | 591.74 |
| Python | 3.13.12 |
| PyTorch | 2.10.0+cu126 |
| CUDA runtime | 12.6 |
| ONNX Runtime | 1.24.2 |
| TensorRT | 10.15.1.29 |
| 実行方式 | ONNX Runtime TensorRT Execution Provider / TensorRT FP16 |
| モデル | 2話者、7 styles、JP/EN bilingual JP-Extra モデル |
| TensorRT キャッシュ | warmup 済み |
| サーバ | `max_batch_size=8`、adaptive batcher、G2P workers 4 |
| PCM ストリーミング標準書き出し単位 | decoder chunk 10個ごと |

入力文は、性能計測用に seed 固定のランダムテキストを使っています。音質評価用の文章ではありません。

## 代表結果

JP 通常テキスト入力、同時接続数8、200リクエストの代表値です。ストリーミング PCM は標準設定の `--stream-buffer-chunks 10` です。

| シナリオ | ウォーターマーク | 平均テキスト長 | 平均音声長 | スループット | 音声生成速度 | 平均レイテンシ | P99 レイテンシ | 平均 TTFB | P99 TTFB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WAV / TensorRT | なし | 53文字 | 9.30秒 | 48.2 RPS | 448倍実時間 | 165.4ms | 239.1ms | - | - |
| ストリーミング PCM / TensorRT | なし | 53文字 | 9.30秒 | 47.4 RPS | 441倍実時間 | 165.5ms | 249.8ms | 129.7ms | 196.0ms |
| WAV / TensorRT + AudioSeal TRT | core `0x1234` | 53文字 | 9.29秒 | 40.7 RPS | 378倍実時間 | 194.6ms | 321.0ms | - | - |
| ストリーミング PCM / TensorRT + AudioSeal TRT | core `0x1234` | 53文字 | 9.29秒 | 44.1 RPS | 410倍実時間 | 180.0ms | 287.8ms | 151.5ms | 225.1ms |

ウォーターマークなしの WAV 計測では、4.15秒の処理時間で 1859.1秒分の音声を生成しました。ストリーミング PCM は同じ入力列で 4.22秒の処理時間で 1859.1秒分の音声を生成し、WAV とほぼ同じ total latency 帯に収まっています。

AudioSeal TRT ウォーターマークありの計測では、watermark core word `0x1234` を実際に付与しています。短い検証音声に対する検出では `core16_hex: 0x1234`、`score: 1.0`、`detect_prob_mean: 1.0` を確認しました。

## 3-worker aggregate example

複数 worker を束ねる HTTP aggregate endpoint を用意している場合は、`--status-url` に `/cluster-status` を指定すると、ベンチ後に各 worker の healthy 状態と request 数も確認できます。以下は 24話者・7 styles の bilingual JP-Extra モデルを RTX 3060 x3 worker 構成で計測した参考値です。

| 項目 | 値 |
| --- | --- |
| モデル | 24 speakers / 7 styles bilingual JP-Extra model |
| worker | RTX 3060 x 3 |
| endpoint | local aggregate `http://127.0.0.1:5000` |
| ベンチツール | `charavoice_engine.scripts.bench_server` |

| シナリオ | リクエスト数 | 同時接続数 | warmup | 平均テキスト長 | 平均音声長 | スループット | 音声生成速度 | 平均レイテンシ | P95 | P99 | 平均 TTFB | P99 TTFB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JP WAV | 200 | 8 | 16 | 53文字 | 7.72秒 | 44.0 RPS | 340倍実時間 | 178.1ms | 257.1ms | 275.7ms | - | - |
| JP streaming PCM | 200 | 8 | 24 | 54文字 | 7.76秒 | 45.5 RPS | 353倍実時間 | 174.2ms | 282.3ms | 317.8ms | 150.1ms | 246.1ms |
| EN WAV | 100 | 8 | 24 | 54文字 | 3.93秒 | 61.4 RPS | 242倍実時間 | 127.6ms | 205.7ms | 233.6ms | - | - |

EN は最初に warmup 8 で測った際、2.4秒級の外れ値が2件出ました。warmup 24 で再計測すると P99 233.6ms に収まったため、aggregate 経由のベンチでは各 worker の lazy init を吸収するために `-w 24` 以上を推奨します。

低同時接続の計測結果です。全て `-n 100 -w 24 --seed 20260622` で、失敗は0件でした。

| シナリオ | 同時接続数 | スループット | 音声生成速度 | 平均レイテンシ | P95 | P99 | 平均 TTFB | P99 TTFB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JP WAV | 1 | 10.8 RPS | 90倍実時間 | 92.9ms | 138.3ms | 149.7ms | - | - |
| JP WAV | 2 | 21.5 RPS | 180倍実時間 | 92.2ms | 138.6ms | 157.2ms | - | - |
| JP WAV | 3 | 29.5 RPS | 248倍実時間 | 99.8ms | 153.7ms | 208.2ms | - | - |
| JP streaming PCM | 1 | 10.5 RPS | 88倍実時間 | 95.6ms | 140.1ms | 152.8ms | 76.5ms | 107.3ms |
| JP streaming PCM | 2 | 20.9 RPS | 175倍実時間 | 94.9ms | 144.6ms | 152.8ms | 75.8ms | 101.4ms |
| JP streaming PCM | 3 | 29.3 RPS | 246倍実時間 | 100.5ms | 151.0ms | 173.2ms | 80.8ms | 149.2ms |
| EN WAV | 1 | 17.1 RPS | 67倍実時間 | 58.5ms | 80.5ms | 84.8ms | - | - |
| EN WAV | 2 | 34.4 RPS | 135倍実時間 | 58.0ms | 80.1ms | 84.8ms | - | - |
| EN WAV | 3 | 50.2 RPS | 197倍実時間 | 59.1ms | 82.6ms | 104.8ms | - | - |

再計測する場合は `--csv` で request ごとの latency、音声長、status を保存できます。

## 言語別

同じ 2話者 bilingual JP-Extra モデルで、JP と EN の通常テキスト経路を測りました。EN は同じ文字数でも生成音声が短くなったため RPS は高く見えますが、音声生成速度は JP と同じく 400倍実時間台です。

| 言語 | モード | リクエスト数 | 同時接続数 | 平均テキスト長 | 平均音声長 | スループット | 音声生成速度 | 平均レイテンシ | P99 レイテンシ | 平均 TTFB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JP | WAV | 200 | 8 | 53文字 | 9.30秒 | 48.2 RPS | 448倍実時間 | 165.4ms | 239.1ms | - |
| JP | ストリーミング PCM | 200 | 8 | 53文字 | 9.30秒 | 47.4 RPS | 441倍実時間 | 165.5ms | 249.8ms | 129.7ms |
| EN | WAV | 100 | 8 | 58文字 | 4.39秒 | 100.7 RPS | 442倍実時間 | 74.5ms | 124.1ms | - |
| EN | ストリーミング PCM | 100 | 8 | 58文字 | 4.39秒 | 95.5 RPS | 419倍実時間 | 78.8ms | 125.1ms | 72.5ms |

## 完全指定入力

`chunks[].surface_text + phones + tones + word2ph` を完全指定した入力も同じ API 経路で処理できます。以下は固定短文 `今日` のため、通常の 3-100文字ランダムテキストとは直接比較しません。短いリクエストを大量に処理した場合の API オーバーヘッド確認用です。

| 入力 | モード | リクエスト数 | 同時接続数 | 平均音声長 | スループット | 音声生成速度 | 平均レイテンシ | P99 レイテンシ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 完全指定 JP 固定短文 | WAV | 200 | 8 | 0.74秒 | 372.4 RPS | 277倍実時間 | 21.4ms | 59.9ms |

## ストリーミング TTFB

同時接続数8では、TTFB は decoder の初回出力だけでなく HTTP batcher の batch window に強く影響されます。標準設定では、同時接続数8・200リクエストの平均 TTFB は 129.7ms、P99 TTFB は 196.0ms でした。

低負荷で最初の音声チャンクをできるだけ早く返したい場合は、`--stream-buffer-chunks 1` を指定できます。同時接続数1・100リクエストの比較では、標準の `10` は HTTP chunk 数を抑え、`1` は TTFB をさらに詰めます。

| 設定 | 同時接続数 | 平均 TTFB | P95 TTFB | P99 TTFB | 平均レイテンシ | P99 レイテンシ | 平均 HTTP chunk 数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `--stream-buffer-chunks 10` | 1 | 35.4ms | 44.6ms | 53.9ms | 41.3ms | 57.5ms | 7.5/リクエスト |
| `--stream-buffer-chunks 1` | 1 | 33.1ms | 38.2ms | 38.7ms | 41.3ms | 58.1ms | 24.0/リクエスト |

標準値は転送効率を優先して `10` です。TTFB 最優先の用途では `1` を指定します。

## AudioSeal TRT ウォーターマーク

AudioSeal ウォーターマークは TensorRT backend で実行できます。サーバー起動時に `--watermark audioseal_trt` だけを指定した場合、backend は有効になりますが、既定の core word がないためリクエストに `watermark_core16` を指定しない限りウォーターマークは付与されません。全リクエストに付与して測る場合は `--watermark-core16` を併用します。

今回のウォーターマーク計測では `--watermark-core16 4660` を指定し、core `0x1234` を全リクエストに付与しました。

| モード | リクエスト数 | 同時接続数 | スループット | 音声生成速度 | 平均レイテンシ | P99 レイテンシ | 平均 TTFB | P99 TTFB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WAV + AudioSeal TRT | 200 | 8 | 40.7 RPS | 378倍実時間 | 194.6ms | 321.0ms | - | - |
| ストリーミング PCM + AudioSeal TRT | 200 | 8 | 44.1 RPS | 410倍実時間 | 180.0ms | 287.8ms | 151.5ms | 225.1ms |

ウォーターマーク処理の統計:

| モード | ウォーターマーク chunk 数 | ウォーターマーク平均 microbatch | ウォーターマーク平均処理時間 |
| --- | ---: | ---: | ---: |
| WAV + AudioSeal TRT | 5352 chunks | 15.29 | 8.49ms/microbatch |
| ストリーミング PCM + AudioSeal TRT | 10704 chunks | 15.47 | 8.85ms/microbatch |

検出確認:

| 項目 | 値 |
| --- | --- |
| 検出対象 | `/tmp/charavoice_bench_results/watermark_check.wav` |
| core16 | `4660` / `0x1234` |
| mode | `core_only` |
| score | `1.0` |
| detect_prob_mean | `1.0` |
| crc_ok | `true` |

## VRAM

サーバー `/status` の内部計測では、2話者 bilingual JP-Extra モデルの TensorRT runtime は以下の VRAM 使用量でした。

| 条件 | PyTorch model | Decoder TRT | Enc+DP TRT | Flow TRT | 合計 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ウォーターマークなし | 470MB | 192MB | 564MB | 3536MB | 4762MB |
| AudioSeal TRT backend あり | 470MB | 194MB | 596MB | 3508MB | 4768MB |

AudioSeal TRT backend 自体の engine キャッシュは `audioseal_chunk_b16_opt16` を使用しました。

## 再現コマンド

標準サーバー:

```bash
uv run --extra server --extra tensorrt --extra watermark \
  python -m charavoice_engine.scripts.run_server \
  --dir model_assets \
  --host 0.0.0.0 \
  --port 5000 \
  --trt-cache trt_cache \
  --stream-buffer-chunks 10 \
  --watermark none
```

JP WAV:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language JP \
  --input-mode text \
  --format wav \
  -c 8 -n 200 -w 16 \
  --seed 20260616
```

local aggregate で worker 全体を見る場合:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://127.0.0.1:5000 \
  --status-url http://127.0.0.1:5000/cluster-status \
  --language JP \
  --input-mode text \
  --format wav \
  -c 8 -n 200 -w 24 \
  --seed 20260622 \
  --csv /tmp/charavoice_bench_results/aggregate_jp_wav_c8_n200.csv
```

API key が必要な HTTPS 経路を測る場合:

```bash
CHARAVOICE_API_KEY=... uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url https://tts.example.com \
  --api-key-env CHARAVOICE_API_KEY \
  --language JP \
  --input-mode text \
  --format wav \
  -c 4 -n 50 -w 8
```

テスト用の固定 IP や自己署名証明書など、証明書検証を通せない経路では `--insecure` を追加できます。

JP ストリーミング PCM:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language JP \
  --input-mode text \
  --streaming --format pcm \
  -c 8 -n 200 -w 16 \
  --seed 20260616
```

低 TTFB ストリーミング:

```bash
uv run --extra server --extra tensorrt --extra watermark \
  python -m charavoice_engine.scripts.run_server \
  --dir model_assets \
  --host 0.0.0.0 \
  --port 5000 \
  --trt-cache trt_cache \
  --stream-buffer-chunks 1 \
  --watermark none
```

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language JP \
  --input-mode text \
  --streaming --format pcm \
  -c 1 -n 100 -w 8 \
  --seed 20260616
```

EN テキスト:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language EN \
  --input-mode text \
  --format wav \
  -c 8 -n 100 -w 8 \
  --seed 20260616
```

完全指定入力:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language JP \
  --input-mode full-control \
  --format wav \
  -c 8 -n 200 -w 16 \
  --seed 20260616
```

AudioSeal TRT ウォーターマークサーバー:

```bash
uv run --extra server --extra tensorrt --extra watermark \
  python -m charavoice_engine.scripts.run_server \
  --dir model_assets \
  --host 0.0.0.0 \
  --port 5000 \
  --trt-cache trt_cache \
  --stream-buffer-chunks 10 \
  --watermark audioseal_trt \
  --watermark-core16 4660
```

ウォーターマークあり WAV:

```bash
uv run --extra server python -m charavoice_engine.scripts.bench_server \
  --url http://localhost:5000 \
  --language JP \
  --input-mode text \
  --format wav \
  -c 8 -n 200 -w 16 \
  --seed 20260616
```

AudioSeal ウォーターマーク検出:

```bash
uv run --extra watermark \
  python tools/detect_audioseal_watermark.py output.wav \
  --json --scan-offsets auto
```
