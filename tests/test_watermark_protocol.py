from __future__ import annotations

import pytest

from charavoice_engine.watermark.protocol import (
    CONTROL_PREFIX,
    build_control_word,
    build_watermark_plan,
    extension_hex_to_bytes,
    normalize_extension_hex,
    parse_control_word,
    recover_watermark,
)


def test_normalize_extension_hex_accepts_prefix_and_case() -> None:
    assert normalize_extension_hex("  0xDeAdBEEF  ") == "deadbeef"
    assert extension_hex_to_bytes("0x00ff") == bytes.fromhex("00ff")


def test_build_plan_core_only_for_short_audio() -> None:
    plan = build_watermark_plan(core16=0x1234, extension_bytes=bytes.fromhex("beef"), chunk_count=2)
    assert plan.mode == "core_only"
    assert plan.frame_words == (0x1234,)
    assert plan.embedded_extension_bytes == b""
    assert plan.truncated is True


def test_build_plan_truncates_extension_to_fit_chunk_budget() -> None:
    plan = build_watermark_plan(core16=0x1234, extension_bytes=bytes.fromhex("001122334455"), chunk_count=4)
    assert plan.mode == "extension"
    assert plan.embedded_extension_bytes == bytes.fromhex("0011")
    assert len(plan.frame_words) == 4


def test_control_word_round_trip() -> None:
    control = build_control_word(5)
    assert (control & 0xFF00) == CONTROL_PREFIX
    assert parse_control_word(control) == (1, 5)


def test_recover_extension_frame_from_repeated_words() -> None:
    plan = build_watermark_plan(core16=0x1234, extension_bytes=bytes.fromhex("beef01"), chunk_count=10)
    detected_words = [plan.word_for_chunk(i) for i in range(10)]
    recovery = recover_watermark(detected_words)
    assert recovery.mode == "extension"
    assert recovery.core16 == 0x1234
    assert recovery.extension_bytes == bytes.fromhex("beef01")
    assert recovery.crc_ok is True


def test_recover_extension_frame_with_single_word_errors() -> None:
    plan = build_watermark_plan(core16=0x1234, extension_bytes=bytes.fromhex("beef01"), chunk_count=10)
    detected_words = [plan.word_for_chunk(i) for i in range(10)]
    detected_words[0] = 0x9999
    detected_words[1 + plan.frame_len] = 0x7777
    recovery = recover_watermark(detected_words)
    assert recovery.mode == "extension"
    assert recovery.core16 == 0x1234
    assert recovery.extension_bytes == bytes.fromhex("beef01")
    assert recovery.crc_ok is True


def test_recover_core_only_when_no_valid_control_word() -> None:
    recovery = recover_watermark([0x1234, 0x1234, 0x1234])
    assert recovery.mode == "core_only"
    assert recovery.core16 == 0x1234
    assert recovery.extension_bytes == b""
    assert recovery.crc_ok is True


def test_invalid_extension_hex_rejected() -> None:
    with pytest.raises(ValueError, match="valid hex string"):
        normalize_extension_hex("zz")
