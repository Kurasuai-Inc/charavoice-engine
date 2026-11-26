from __future__ import annotations

from charavoice_engine.nlp.symbols import (
    EN_SYMBOLS,
    JP_SYMBOLS,
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    NUM_LANGUAGES,
    NUM_RESERVED_TONES,
    NUM_TONES,
    RESERVED_SYMBOLS,
    SYMBOLS,
)


class TestSymbolsLength:
    def test_symbols_not_empty(self) -> None:
        assert len(SYMBOLS) > 0

    def test_symbols_starts_with_pad(self) -> None:
        assert SYMBOLS[0] == "_"

    def test_symbols_contains_unk(self) -> None:
        assert "UNK" in SYMBOLS

    def test_symbols_contains_sp(self) -> None:
        assert "SP" in SYMBOLS


class TestPhonemeExistence:
    def test_jp_n(self) -> None:
        assert "N" in JP_SYMBOLS

    def test_jp_q(self) -> None:
        assert "q" in JP_SYMBOLS

    def test_en_aa(self) -> None:
        assert "aa" in EN_SYMBOLS

    def test_reserved_symbols_preserve_checkpoint_abi(self) -> None:
        assert "zh" in RESERVED_SYMBOLS

    def test_all_jp_in_symbols(self) -> None:
        for s in JP_SYMBOLS:
            assert s in SYMBOLS, f"JP symbol {s!r} not in SYMBOLS"

    def test_all_en_in_symbols(self) -> None:
        for s in EN_SYMBOLS:
            assert s in SYMBOLS, f"EN symbol {s!r} not in SYMBOLS"

    def test_all_reserved_in_symbols(self) -> None:
        for s in RESERVED_SYMBOLS:
            assert s in SYMBOLS, f"reserved symbol {s!r} not in SYMBOLS"


class TestLanguageIdMap:
    def test_reserved_language_row_is_kept(self) -> None:
        assert NUM_LANGUAGES == 3

    def test_jp_id(self) -> None:
        assert LANGUAGE_ID_MAP["JP"] == 1

    def test_en_id(self) -> None:
        assert LANGUAGE_ID_MAP["EN"] == 2


class TestToneStartMap:
    def test_jp_after_reserved_tones(self) -> None:
        assert LANGUAGE_TONE_START_MAP["JP"] == NUM_RESERVED_TONES

    def test_en_after_jp(self) -> None:
        assert LANGUAGE_TONE_START_MAP["EN"] > LANGUAGE_TONE_START_MAP["JP"]

    def test_num_tones_positive(self) -> None:
        assert NUM_TONES == 12
