from pathlib import Path

import numpy as np

from charavoice_engine.config import HyperParameters
from charavoice_engine.constants import Languages
from charavoice_engine.scripts.run_server import (
    ModelScanError,
    _find_pytorch_model,
    parse_bert_languages,
    parse_model_keys,
    scan_models,
)


def test_find_pytorch_model_ignores_discriminator_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "D_100.safetensors").write_bytes(b"d")
    (tmp_path / "WD_100.safetensors").write_bytes(b"wd")
    generator = tmp_path / "voice_e1_s100.safetensors"
    generator.write_bytes(b"g")

    assert _find_pytorch_model(tmp_path) == generator


def test_scan_models_rejects_incomplete_model_directory(tmp_path: Path) -> None:
    (tmp_path / "broken").mkdir()

    try:
        scan_models(tmp_path)
    except ModelScanError as exc:
        assert "No config.json" in str(exc)
    else:
        raise AssertionError("incomplete model directory was skipped instead of rejected")


def test_scan_models_loads_complete_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "voice"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        HyperParameters(model_name="voice").model_dump_json(indent=2), encoding="utf-8"
    )
    np.save(model_dir / "style_vectors.npy", np.zeros((1, 256), dtype=np.float32))
    model_path = model_dir / "voice_e1_s1.safetensors"
    model_path.write_bytes(b"g")

    models = scan_models(tmp_path)

    assert len(models) == 1
    assert models[0]["name"] == "voice"
    assert models[0]["model_path"] == str(model_path)


def test_scan_models_loads_only_requested_models(tmp_path: Path) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "config.json").write_text(HyperParameters(model_name="good").model_dump_json(), encoding="utf-8")
    np.save(good / "style_vectors.npy", np.zeros((1, 256), dtype=np.float32))
    (good / "good.safetensors").write_bytes(b"g")
    (tmp_path / "broken").mkdir()

    models = scan_models(tmp_path, requested_models=["good"])

    assert [model["name"] for model in models] == ["good"]


def test_scan_models_rejects_missing_requested_model(tmp_path: Path) -> None:
    try:
        scan_models(tmp_path, requested_models=["missing"])
    except ModelScanError as exc:
        assert "Requested model(s) missing: missing" in str(exc)
    else:
        raise AssertionError("missing requested model was accepted")


def test_parse_model_keys_rejects_duplicates() -> None:
    try:
        parse_model_keys("a,b,a")
    except ValueError as exc:
        assert "Duplicate model keys" in str(exc)
    else:
        raise AssertionError("duplicate model keys were accepted")


def test_parse_bert_languages_defaults_to_jp_en() -> None:
    assert parse_bert_languages(None) == [Languages.JP, Languages.EN]


def test_parse_bert_languages_rejects_invalid_or_duplicate_values() -> None:
    for value, expected in (("JP,JP", "Duplicate BERT language"), ("ZH", "Unsupported BERT language")):
        try:
            parse_bert_languages(value)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"invalid BERT language value was accepted: {value}")
