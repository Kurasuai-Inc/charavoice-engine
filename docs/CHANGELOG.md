# 変更履歴

## 1.0.0 (2026-11-26)

- CHARAVOICE Engine の初回リリース。
- Python package は `charavoice_engine`、distribution name は `charavoice-engine`。
- HTTP API は text-only 入力と、言語指定つき `chunks` 入力に対応。
- bilingual JP-Extra 学習 recipe と runtime export validation を追加。
- `enc_p+dp`、`flow`、streaming decoder の TensorRT component runtime を追加。
- AudioSeal watermark と AudioSeal TensorRT watermark 経路を追加。
- RTX 4090 でのベンチマーク結果と再現コマンドを追加。
- Docker、Prometheus metrics、benchmark utility、公開ドキュメントを追加。
