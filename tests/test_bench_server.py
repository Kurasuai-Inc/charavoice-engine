import pytest

from charavoice_engine.scripts import bench_server
from charavoice_engine.scripts.bench_server import BenchResult, ModelSpec


def test_bench_server_builds_headers_from_explicit_api_key() -> None:
    headers = bench_server._build_headers(
        api_key="secret",
        api_key_env="",
        api_key_header="X-API-Key",
        extra_headers=[("X-Trace", "bench")],
    )

    assert headers == {"X-Trace": "bench", "X-API-Key": "secret"}


def test_bench_server_builds_headers_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_API_KEY", "from-env")

    headers = bench_server._build_headers(
        api_key="",
        api_key_env="MISSING,TTS_API_KEY",
        api_key_header="X-API-Key",
        extra_headers=[],
    )

    assert headers == {"X-API-Key": "from-env"}


def test_bench_server_print_stats_handles_all_failures(capsys: pytest.CaptureFixture[str]) -> None:
    bench_server.print_stats(
        [
            BenchResult(
                latency_ms=10.0,
                status=401,
                audio_bytes=0,
                text_len=5,
                model_id=0,
                speaker_id=1,
                style="Neutral",
                error="unauthorized",
            )
        ],
        "failure case",
        wall_time_s=0.1,
    )

    out = capsys.readouterr().out
    assert "0 OK, 1 failed" in out
    assert "401: 1" in out
    assert "unauthorized" in out


def test_bench_server_builds_model_specs_from_models_info() -> None:
    specs = bench_server._build_model_specs(
        {
            "0": {
                "id2spk": {"15": "speaker_b", "1": "speaker_a"},
                "style2id": {"Neutral": 0, "happy": 4},
            }
        }
    )

    assert specs == [ModelSpec(model_id=0, speakers=[1, 15], styles=["Neutral", "happy"])]


def test_bench_server_prints_cluster_stats(capsys: pytest.CaptureFixture[str]) -> None:
    bench_server._print_cluster_stats(
        {
            "backends": 3,
            "backends_healthy": 3,
            "total_in_flight": 0,
            "total_requests": 12,
            "loaded_model_keys": ["voice"],
            "gpu_names": ["NVIDIA GeForce RTX 3060"],
            "per_backend": [
                {
                    "host": "127.0.0.1:5001",
                    "healthy": True,
                    "in_flight": 0,
                    "total_requests": 4,
                    "worker": {"worker_id": "worker-a"},
                }
            ],
        }
    )

    out = capsys.readouterr().out
    assert "Backends:       3" in out
    assert "Healthy:        3" in out
    assert "worker-a" in out


def test_bench_server_builds_text_json_payload() -> None:
    bench_server._language = "JP"
    bench_server._input_mode = "text"

    text, payload = bench_server._make_request_payload(
        [ModelSpec(model_id=0, speakers=[1], styles=["Neutral"])],
        streaming=False,
        audio_format="ogg",
    )

    assert text
    assert payload["text"] == text
    assert "chunks" not in payload
    assert payload["language"] == "JP"
    assert payload["streaming"] is False
    assert payload["format"] == "ogg"


def test_bench_server_builds_full_control_json_payload() -> None:
    bench_server._language = "EN"
    bench_server._input_mode = "full-control"

    text, payload = bench_server._make_request_payload(
        [ModelSpec(model_id=0, speakers=[1], styles=["Neutral"])],
        streaming=True,
        audio_format="pcm",
    )

    assert text == "SQL"
    assert "text" not in payload
    assert payload["language"] == "EN"
    assert payload["streaming"] is True
    assert payload["format"] == "pcm"
    assert payload["chunks"] == [
        {
            "language": "EN",
            "surface_text": "SQL",
            "phones": ["s", "iy", "k", "w", "ah", "l"],
            "tones": [0, 1, 0, 0, 1, 0],
            "word2ph": [2, 2, 2],
        }
    ]


@pytest.mark.asyncio
async def test_streaming_request_passes_audio_format() -> None:
    bench_server._language = "JP"
    bench_server._input_mode = "text"
    bench_server._text_min_len = 3
    bench_server._text_max_len = 3

    class _FakeContent:
        async def iter_any(self):
            yield b"pcm"

    class _FakeResponse:
        status = 200
        content = _FakeContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeSession:
        payload = None

        def post(self, _url, json):
            self.payload = json
            return _FakeResponse()

    session = _FakeSession()
    result = await bench_server.send_streaming_request(
        session,  # type: ignore[arg-type]
        "http://example.invalid",
        [ModelSpec(model_id=0, speakers=[1], styles=["Neutral"])],
        bench_server.asyncio.Semaphore(1),
        audio_format="pcm",
    )

    assert result.status == 200
    assert session.payload is not None
    assert session.payload["streaming"] is True
    assert session.payload["format"] == "pcm"


def _ogg_page(granule: int, body: bytes) -> bytes:
    header = bytearray(27)
    header[:4] = b"OggS"
    header[6:14] = granule.to_bytes(8, "little")
    header[26] = 1
    return bytes(header) + bytes([len(body)]) + body


@pytest.mark.asyncio
async def test_streaming_ogg_duration_uses_opus_granules() -> None:
    opus_head = b"OpusHead" + bytes([1, 2]) + (312).to_bytes(2, "little") + bytes(7)
    ogg = _ogg_page(0, opus_head) + _ogg_page(48_312, b"x")

    class _FakeContent:
        async def iter_any(self):
            yield ogg[:40]
            yield ogg[40:]

    class _FakeResponse:
        status = 200
        content = _FakeContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeSession:
        def post(self, _url, json):
            return _FakeResponse()

    result = await bench_server.send_streaming_request(
        _FakeSession(), "http://example.invalid",
        [ModelSpec(model_id=0, speakers=[1], styles=["Neutral"])],
        bench_server.asyncio.Semaphore(1), audio_format="ogg",
    )
    assert result.audio_duration == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Truncated"):
        bench_server.ogg_opus_duration(ogg[:-1])
