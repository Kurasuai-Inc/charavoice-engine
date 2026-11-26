import pytest

from charavoice_engine.nlp.japanese.pyopenjtalk_worker import worker_client, worker_server
from charavoice_engine.nlp.japanese.pyopenjtalk_worker.worker_client import WorkerClient
from charavoice_engine.nlp.japanese.pyopenjtalk_worker.worker_server import WorkerServer


def test_worker_client_defaults_to_loopback(monkeypatch) -> None:
    addresses = []

    class FakeSocket:
        def settimeout(self, timeout):
            pass

        def connect(self, address):
            addresses.append(address)
            raise OSError("stop before real network access")

    monkeypatch.setattr(worker_client.socket, "socket", lambda *_args, **_kwargs: FakeSocket())

    with pytest.raises(OSError):
        WorkerClient(12345)

    assert addresses == [("127.0.0.1", 12345)]


def test_worker_server_defaults_to_loopback_bind(monkeypatch) -> None:
    addresses = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def setsockopt(self, *_args):
            pass

        def bind(self, address):
            addresses.append(address)

        def listen(self):
            raise RuntimeError("stop before real network access")

    monkeypatch.setattr(worker_server.socket, "socket", lambda *_args, **_kwargs: FakeSocket())

    with pytest.raises(RuntimeError, match="stop before real network access"):
        WorkerServer().start_server(port=12345)

    assert addresses == [("127.0.0.1", 12345)]
