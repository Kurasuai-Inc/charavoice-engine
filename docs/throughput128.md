# 採用済み128フレーム構成

2026-10-25のRTX 3060での試聴・計測に基づく任意の推論プロファイルです。

```bash
uv run python -m charavoice_engine.scripts.run_server \
  --dir model_assets --trt-cache ./trt_cache \
  --inference-profile throughput128 --watermark audioseal_trt
```

要求側では `format=ogg` と `watermark_core16` を指定します。`streaming` はtrue/falseの両方に対応します。
`--watermark-core16` を起動時に指定すれば、要求で省略した場合の既定値になります。
ウォーターマークのbackendを有効にするだけでは、識別子を省略した要求には埋め込みません。
ベンチマーク用の識別子4660を運用設定へコピーせず、運用で定めた識別子を指定してください。

このプロファイルは次をまとめて選びます。

- Decoderの有効区間を32から128フレームへ拡張。左右13フレームを含む入力は154フレーム。
- 提供中の `decoder.onnx` の形状だけを変更し、全initializerの一致を確認してキャッシュへ保存。
  モデルと同居するcheckpointからの再exportは行いません。元のONNXは書き換えません。
- 音声長の丸め、末尾の有効区間、AudioSealの識別子計画、配信単位は元の32フレーム・16,384サンプルを維持。
- Flowの入出力をGPUに保持し、PyTorchとONNX Runtime間を明示的に同期。
- adaptive batchingを停止して待ち時間を0msに固定。前の処理を待っている間に到着した要求も回収。
- AudioSeal処理後の配信をまとめ、ストリーミングのOggページを200msに設定。
- BERTのTensorRT初期化を終えてから他のエンジン・G2P workerを初期化。

`--inference-profile default` が従来の32フレーム構成です。省略時も同じです。
128フレーム版のTensorRTキャッシュは `decoder_f154`、派生ONNXは内容hash別の `derived_onnx/` に保存します。
初回はエンジンの構築時間が必要です。モデルファイルは読み取り専用で使用できます。

HTTP切断時には、ラッパーから合成元のiteratorまでcloseを伝播し、満杯の配信キューを解除します。
この修正は両プロファイルに適用します。

## 計測上の意味

同重みでも、チャンク境界とTensorRTの計算手順が変わるため、波形は従来版と数値一致しません。
既存モデルの346ケースで有限値と音声長を確認し、ユーザーが比較音声を試聴して採用を承認しました。
圧縮後のAudioSeal判定は344/346ケースで、対照から新たな未通過はありません。完全な検出保証ではありません。
別モデルでは音質・識別子・音声長を再確認してください。

AudioSealとmono Ogg/Opus 48kbpsを固定した各6,000要求の比較では、採用版の最大処理量は通常約121.2音声秒/秒、
streaming約116.3音声秒/秒でした。これはRTX 3060 1枚に複数要求を並列実行した合計です。
単発の発話速度、全GPUの容量、NVIDIA unlock単独の効果を表す値ではありません。
最新ドライバーとunlockだけのHTTP処理量変化は約±1%以内でした。

