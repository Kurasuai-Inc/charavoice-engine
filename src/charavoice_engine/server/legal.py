"""License credits and the AGPL-3.0 §13 source offer, served by GET /licenses and GET /source.

Apps show these values on their license screen instead of keeping their own copy, so the text
follows the deployed engine version.

Required at startup (the server refuses to start without them; nothing is filled in silently):
- CHARAVOICE_SOURCE_COMMIT: the commit the image was built from (Dockerfile build arg).
- NOTICE at CHARAVOICE_NOTICE_PATH (default /app/NOTICE, where the image puts it).
Optional: CHARAVOICE_SOURCE_URL, the public source location (the Docker image defaults it to the upstream
repository). When it is unset, GET /source reports source_available: false.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

SOURCE_URL_ENV = "CHARAVOICE_SOURCE_URL"
SOURCE_COMMIT_ENV = "CHARAVOICE_SOURCE_COMMIT"
NOTICE_PATH_ENV = "CHARAVOICE_NOTICE_PATH"
DEFAULT_NOTICE_PATH = "/app/NOTICE"


class LegalConfigError(RuntimeError):
    """The license/source information the engine must serve is missing."""


ENGINE = {
    "name": "CHARAVOICE Engine",
    "license": "AGPL-3.0-only",
    "license_url": "https://www.gnu.org/licenses/agpl-3.0.html",
    "copyright": "Copyright 2024-2026 Kurasuai, Inc. and CHARAVOICE Engine contributors",
}

# Keep in step with NOTICE and THIRD_PARTY_LICENSES.md.
CREDITS: list[dict[str, object]] = [
    {
        "name": "DeBERTa V2 large Japanese char WWM",
        "id": "ku-nlp/deberta-v2-large-japanese-char-wwm",
        "used_for": "Japanese BERT features",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "url": "https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm",
        "changes": "Converted to ONNX (tsukumijima/deberta-v2-large-japanese-char-wwm-onnx, CC-BY-SA-4.0); "
        "the engine may further convert it to FP16 and batched ONNX.",
        "included_in_source": False,
    },
    {
        "name": "DeBERTa V3 large",
        "id": "microsoft/deberta-v3-large",
        "used_for": "English BERT features",
        "license": "MIT",
        "url": "https://huggingface.co/microsoft/deberta-v3-large",
        "changes": "Converted to ONNX (tsukumijima/deberta-v3-large-onnx, MIT).",
        "included_in_source": False,
    },
    {
        "name": "AudioSeal",
        "id": "facebook/audioseal",
        "used_for": "Audio watermark",
        "license": "MIT",
        "url": "https://github.com/facebookresearch/audioseal",
        "copyright": "Copyright (c) Meta Platforms, Inc. and affiliates.",
        "included_in_source": False,
    },
    {
        "name": "Style-Bert-VITS2",
        "id": "Style-Bert-VITS2",
        "used_for": "Engine code (derived)",
        "license": "AGPL-3.0",
        "url": "https://github.com/litagin02/Style-Bert-VITS2",
        "copyright": "Copyright (c) 2023 litagin02",
        "included_in_source": True,
    },
    {
        "name": "Bert-VITS2",
        "id": "Bert-VITS2",
        "used_for": "Engine code (via Style-Bert-VITS2)",
        "license": "AGPL-3.0",
        "url": "https://github.com/fishaudio/Bert-VITS2",
        "copyright": "Copyright (c) 2023 Fish Audio",
        "included_in_source": True,
    },
    {
        "name": "VOICEVOX ENGINE",
        "id": "VOICEVOX",
        "used_for": "Japanese mora table code",
        "license": "LGPL-3.0",
        "url": "https://github.com/VOICEVOX/voicevox_engine",
        "included_in_source": True,
    },
    {
        "name": "Open JTalk",
        "id": "Open JTalk",
        "used_for": "Japanese mora table (via VOICEVOX ENGINE)",
        "license": "BSD-3-Clause",
        "url": "http://open-jtalk.sourceforge.net/",
        "copyright": "Copyright (c) 2008-2014 Nagoya Institute of Technology, Department of Computer Science",
        "included_in_source": True,
    },
    {
        "name": "The Carnegie Mellon Pronouncing Dictionary (cmudict 0.6)",
        "id": "cmudict",
        "used_for": "English pronunciation dictionary",
        "license": "Unrestricted use; acknowledgement requested",
        "url": "http://www.speech.cs.cmu.edu/cgi-bin/cmudict",
        "copyright": "Copyright 1998 by Carnegie Mellon University",
        "included_in_source": True,
    },
]


def notice_path() -> Path:
    return Path(os.environ.get(NOTICE_PATH_ENV, "").strip() or DEFAULT_NOTICE_PATH)


@lru_cache(maxsize=4)
def _read_notice(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise LegalConfigError(f"NOTICE not readable at {path} ({NOTICE_PATH_ENV}): {exc}") from exc
    if not text.strip():
        raise LegalConfigError(f"NOTICE at {path} is empty")
    return text


def notice_text() -> str:
    return _read_notice(str(notice_path()))


def source_commit() -> str:
    commit = os.environ.get(SOURCE_COMMIT_ENV, "").strip()
    if not commit:
        raise LegalConfigError(
            f"{SOURCE_COMMIT_ENV} is not set; build the image with "
            f"--build-arg {SOURCE_COMMIT_ENV}=$(git rev-parse HEAD), or export it for a local run"
        )
    return commit


def validate_startup() -> None:
    """Fail fast before serving: AGPL-3.0 §13 needs the build commit and the NOTICE text."""
    source_commit()
    notice_text()


def source_info() -> dict[str, object]:
    url = os.environ.get(SOURCE_URL_ENV, "").strip() or None
    commit = source_commit()
    return {
        "name": ENGINE["name"],
        "license": ENGINE["license"],
        "source_url": url,
        "commit": commit,
        # Null when no public source location is configured; apps can show "準備中" then.
        "source_available": url is not None,
    }


def licenses_info() -> dict[str, object]:
    source = source_info()
    third_party = None
    if source["source_url"] and source["commit"]:
        third_party = f"{str(source['source_url']).rstrip('/')}/blob/{source['commit']}/THIRD_PARTY_LICENSES.md"
    return {
        "engine": ENGINE,
        "credits": CREDITS,
        "notice": notice_text(),
        "third_party_licenses_url": third_party,
        "source": source,
    }
