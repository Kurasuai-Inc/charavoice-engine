import numpy as np
from fastapi.responses import Response
from fastapi.testclient import TestClient

import charavoice_engine.nlp as nlp
from charavoice_engine.server import app as app_module
from charavoice_engine.server.app import LoadedModel


class _FakeBatcher:
    max_queue_size = 32

    def stats(self) -> dict[str, object]:
        return {
            "total_requests": 3,
            "total_batches": 2,
            "queue_size": 1,
            "total_text_length": 42,
            "total_dropped_stale": 0,
        }


class _FakeEngine:
    vram_breakdown = None


def _fake_model(name: str = "voice") -> LoadedModel:
    return LoadedModel(
        name=name,
        engine=_FakeEngine(),  # type: ignore[arg-type]
        batcher=_FakeBatcher(),  # type: ignore[arg-type]
        style_vectors=np.zeros((1, 256), dtype=np.float32),
        spk2id={"speaker_a": 0},
        id2spk={0: "speaker_a"},
        style2id={"Neutral": 0},
        config_path="/tmp/config.json",
        model_path="/tmp/model.safetensors",
        device="cuda",
    )


def test_voice_accepts_query_param_post(monkeypatch) -> None:
    captured = {}

    async def _fake_handle(req):
        captured["request"] = req
        return Response(content=b"ok", media_type="audio/ogg")

    monkeypatch.setattr(app_module, "_handle_tts_request", _fake_handle)
    client = TestClient(app_module.app)

    resp = client.post(
        "/voice",
        params={
            "text": "hello",
            "model_id": "0",
            "speaker_id": "0",
            "streaming": "true",
            "format": "ogg",
            "language": "EN",
        },
    )

    assert resp.status_code == 200
    req = captured["request"]
    assert req.text == "hello"
    assert req.streaming is True
    assert req.format == "ogg"
    assert req.language.value == "EN"


def test_voice_preserves_json_text_request(monkeypatch) -> None:
    captured = {}

    async def _fake_handle(req):
        captured["request"] = req
        return Response(content=b"ok", media_type="audio/wav")

    monkeypatch.setattr(app_module, "_handle_tts_request", _fake_handle)
    client = TestClient(app_module.app)

    resp = client.post(
        "/voice",
        json={
            "text": "こんにちは",
            "model_id": 0,
            "speaker_id": 0,
            "language": "JP",
            "streaming": False,
            "format": "wav",
        },
    )

    assert resp.status_code == 200
    req = captured["request"]
    assert req.text == "こんにちは"
    assert req.chunks is None
    assert req.streaming is False
    assert req.format == "wav"
    assert req.language.value == "JP"


def test_voice_preserves_json_mixed_chunks_request(monkeypatch) -> None:
    captured = {}

    async def _fake_handle(req):
        captured["request"] = req
        return Response(content=b"ok", media_type="application/octet-stream")

    monkeypatch.setattr(app_module, "_handle_tts_request", _fake_handle)
    client = TestClient(app_module.app)

    resp = client.post(
        "/voice",
        json={
            "chunks": [
                {"language": "JP", "text": "これは"},
                {"language": "EN", "text": "mixed language"},
                {"language": "JP", "text": "のテストです"},
            ],
            "model_id": 0,
            "speaker_id": 0,
            "streaming": True,
            "format": "pcm",
        },
    )

    assert resp.status_code == 200
    req = captured["request"]
    assert req.text is None
    assert req.chunks is not None
    assert [chunk.language.value for chunk in req.chunks] == ["JP", "EN", "JP"]
    assert req.streaming is True
    assert req.format == "pcm"


def test_full_control_convert_converts_jp_text_chunks_and_keeps_en_text(monkeypatch) -> None:
    def _fake_clean_text(text, language, raise_yomi_error=False):
        assert raise_yomi_error is False
        if language.value == "JP":
            return text, ["_", "k", "o", "_"], [0, 1, 0, 0], [1, 2, 1]
        return text, ["_", "m", "_"], [0, 3, 0], [1, 1, 1]

    monkeypatch.setattr(nlp, "clean_text", _fake_clean_text)
    client = TestClient(app_module.app)

    resp = client.post(
        "/full-control/convert",
        json={
            "chunks": [
                {"language": "JP", "text": "こ"},
                {"language": "EN", "text": "mixed language"},
            ]
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["warnings"] == []
    assert payload["chunks"] == [
        {
            "language": "JP",
            "text": None,
            "surface_text": "こ",
            "phones": ["k", "o"],
            "tones": [1, 0],
            "word2ph": [2],
        },
        {
            "language": "EN",
            "text": "mixed language",
            "surface_text": None,
            "phones": None,
            "tones": None,
            "word2ph": None,
        },
    ]


def test_full_control_convert_allows_zero_word2ph(monkeypatch) -> None:
    def _fake_clean_text(text, language, raise_yomi_error=False):
        assert raise_yomi_error is False
        assert language.value == "JP"
        return text, ["_", "ky", "o:", "_"], [0, 1, 0, 0], [1, 2, 0, 1]

    monkeypatch.setattr(nlp, "clean_text", _fake_clean_text)
    client = TestClient(app_module.app)

    resp = client.post(
        "/full-control/convert",
        json={
            "chunks": [
                {"language": "JP", "text": "今日"},
            ]
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["warnings"] == []
    assert payload["chunks"][0]["word2ph"] == [2, 0]


def test_status_includes_worker_metadata() -> None:
    model = _fake_model("voice")
    app_module.set_models({model.name: model}, [model])
    app_module.set_worker_metadata(
        {
            "worker_id": "worker-1",
            "worker_site": "local",
            "capacity_audio_sec_per_sec": 91,
            "requested_model_keys": [model.name],
            "bert_languages": ["JP", "EN"],
        }
    )

    status = app_module.get_status()

    assert status["worker"]["worker_id"] == "worker-1"
    assert status["worker"]["loaded_model_keys"] == [model.name]
    assert status["worker"]["bert_languages"] == ["JP", "EN"]
    assert status["loaded_model_keys"] == [model.name]
    assert status["model_metadata"][model.name]["speaker_count"] == 1
    assert status["queue"]["per_model"][model.name]["max_queue_size"] == 32


def test_cluster_status_includes_single_worker_metadata(monkeypatch) -> None:
    model = _fake_model("voice")
    app_module.set_models({model.name: model}, [model])
    app_module.set_worker_metadata(
        {
            "worker_id": "worker-1",
            "worker_site": "local",
            "capacity_audio_sec_per_sec": 91,
            "requested_model_keys": [model.name],
            "bert_languages": ["JP", "EN"],
        }
    )
    monkeypatch.delenv("CHARAVOICE_CLUSTER_BACKENDS", raising=False)
    client = TestClient(app_module.app)

    resp = client.get("/cluster-status")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["backends"] == 1
    assert payload["backends_healthy"] == 1
    assert payload["worker"]["worker_id"] == "worker-1"
    assert payload["loaded_model_keys"] == [model.name]
    assert payload["model_metadata"][model.name]["style_count"] == 1
