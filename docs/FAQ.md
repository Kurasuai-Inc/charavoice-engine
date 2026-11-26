# よくある質問

## 日本語だけ、英語だけの話者を同じモデルに入れられますか？

入れられます。`raw/<style>/<speaker>/JP` または `EN` の片方だけでも dataset recipe は扱えます。ただし、その話者が持っていない言語の発音品質はデータ量と他話者からの転移に依存します。

## 日英混在文はどう送りますか？

`chunks` で言語ごとに分けます。日本語として読ませたい `Facebook` や `Meta` は `language: "JP"` の chunk に残し、英語として読ませたいフレーズだけ `language: "EN"` にします。

## `surface_text` は何に使われますか？

`surface_text` は BERT alignment 用の表記です。完全指定 chunk では `surface_text + phones + tones + word2ph` を渡すことで G2P をスキップし、BERT feature と phone 列の対応を明示できます。

## `phones` に先頭末尾の `_` は必要ですか？

不要です。外部 API では `_` を渡しません。CHARAVOICE Engine が内部で先頭末尾の silence と blank intersperse を付与します。

## `word2ph` は何ですか？

`surface_text` の各文字に何個の phone が対応するかを示す配列です。BERT は文字単位、VITS は phone 単位で扱うため、BERT feature を phone 長に展開するために使います。

## WavLM discriminator は必要ですか？

本番品質を優先する場合は有効のままにしてください。学習速度と VRAM 使用量は増えますが、音質・自然性の安定に効きます。

## TensorRT cache は消してもいいですか？

消しても再生成されます。ただし初回起動時に engine build が再実行されるため、起動に時間がかかります。

## Watermark は OGG streaming でも検出できますか？

生成自体はできますが、Opus 圧縮後の検出安定性は WAV/PCM より落ちます。確実に検出したい用途では WAV または PCM を使ってください。

## デフォルトモデルはありますか？

このリポジトリには同梱していません。利用するモデルごとに、音声権利者・データセット・配布元の利用条件を確認してください。
