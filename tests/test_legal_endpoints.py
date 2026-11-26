"""GET /licenses and GET /source: values the apps show on their license screen (AGPL-3.0 §13).

The build commit and NOTICE are required: missing ones stop the server at startup instead of
being filled in or returned as null. Only the public source URL may be unset for now.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from charavoice_engine.server import app as app_module
from charavoice_engine.server import legal

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv(legal.NOTICE_PATH_ENV, str(REPO / "NOTICE"))
    monkeypatch.setenv(legal.SOURCE_COMMIT_ENV, "96e6418f")
    monkeypatch.delenv(legal.SOURCE_URL_ENV, raising=False)
    legal._read_notice.cache_clear()
    yield
    legal._read_notice.cache_clear()


def test_source_url_is_null_until_decided_but_commit_is_always_there():
    body = TestClient(app_module.app).get("/source").json()
    assert body == {
        "name": "CHARAVOICE Engine",
        "license": "AGPL-3.0-only",
        "source_url": None,
        "commit": "96e6418f",
        "source_available": False,
    }


def test_source_reports_configured_url(monkeypatch):
    monkeypatch.setenv(legal.SOURCE_URL_ENV, "https://example.invalid/charavoice-engine")
    body = TestClient(app_module.app).get("/source").json()
    assert body["source_url"] == "https://example.invalid/charavoice-engine"
    assert body["source_available"] is True


def test_licenses_credit_the_japanese_bert_and_include_notice(monkeypatch):
    monkeypatch.setenv(legal.SOURCE_URL_ENV, "https://example.invalid/charavoice-engine/")
    body = TestClient(app_module.app).get("/licenses").json()
    assert body["engine"]["license"] == "AGPL-3.0-only"
    bert = next(c for c in body["credits"] if c.get("id") == "ku-nlp/deberta-v2-large-japanese-char-wwm")
    assert bert["license"] == "CC-BY-SA-4.0"
    assert bert["license_url"] == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert "ONNX" in bert["changes"] and bert["included_in_source"] is False
    assert body["notice"] == (REPO / "NOTICE").read_text(encoding="utf-8")
    assert body["third_party_licenses_url"] == (
        "https://example.invalid/charavoice-engine/blob/96e6418f/THIRD_PARTY_LICENSES.md"
    )


def test_notice_path_defaults_to_the_image_location(monkeypatch):
    monkeypatch.delenv(legal.NOTICE_PATH_ENV)
    assert legal.notice_path() == Path("/app/NOTICE")


@pytest.mark.parametrize("missing", ["commit", "notice_file", "empty_notice"])
def test_missing_required_values_stop_startup(monkeypatch, tmp_path, missing):
    if missing == "commit":
        monkeypatch.delenv(legal.SOURCE_COMMIT_ENV)
    elif missing == "notice_file":
        monkeypatch.setenv(legal.NOTICE_PATH_ENV, str(tmp_path / "absent-NOTICE"))
    else:
        empty = tmp_path / "NOTICE"
        empty.write_text("  \n", encoding="utf-8")
        monkeypatch.setenv(legal.NOTICE_PATH_ENV, str(empty))
    with pytest.raises(legal.LegalConfigError):
        legal.validate_startup()
    # The app lifespan runs the same check before serving anything.
    with pytest.raises(legal.LegalConfigError), TestClient(app_module.app):
        pass


def test_valid_configuration_starts():
    legal.validate_startup()
    with TestClient(app_module.app) as client:
        assert client.get("/source").status_code == 200


def test_every_credit_is_also_in_notice_and_third_party_licenses():
    # The endpoint must not drift from the files that are published with the source.
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    third_party = (REPO / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    for credit in legal.CREDITS:
        key = str(credit["id"])
        assert key.lower() in notice.lower(), key
        assert key.lower() in third_party.lower(), key
