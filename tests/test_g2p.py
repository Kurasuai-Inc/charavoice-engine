from __future__ import annotations

import pytest

from charavoice_engine.nlp.japanese.g2p import g2p


class TestAlignTonesRobustness:
    """g2p の基本動作とエッジケースで正常動作することを確認。"""

    @pytest.mark.parametrize(
        "text",
        [
            "こんにちは",
            "私はそう思う",
            "東京タワーに行きたい",
            "今日はいい天気ですね",
            "ありがとうございます",
        ],
    )
    def test_normal_text_returns_valid_output(self, text: str) -> None:
        phones, tones, word2ph, _ = g2p(text)
        assert len(phones) == len(tones)
        assert len(phones) == sum(word2ph)
        assert phones[0] == "_"
        assert phones[-1] == "_"

    @pytest.mark.parametrize(
        "text",
        [
            "アイウエオカキクケコ",
            "ガギグゲゴザジズゼゾ",
            "パピプペポ",
            "ニャニュニョ",
            "ッテッタッチ",
        ],
    )
    def test_katakana_sequences_no_crash(self, text: str) -> None:
        """カタカナ列で ValueError が出ないことを確認。"""
        phones, tones, word2ph, _ = g2p(text)
        assert len(phones) == len(tones)
        assert len(phones) == sum(word2ph)

    @pytest.mark.parametrize(
        "text",
        [
            "えーっと、あのー",
            "わーい!すごーい!",
            "うーん、そうだなー。",
            "はぁ?なんだって?",
        ],
    )
    def test_edge_case_text_no_crash(self, text: str) -> None:
        """長音・感嘆符混じり等のエッジケースで ValueError が出ないことを確認。"""
        phones, tones, word2ph, _ = g2p(text)
        assert len(phones) == len(tones)
        assert len(phones) == sum(word2ph)

    def test_mismatched_phone_does_not_raise(self) -> None:
        """g2p が ValueError を投げずにフォールバックすることを確認。"""
        phones, tones, word2ph, _ = g2p("テスト")
        assert len(phones) == len(tones)
        assert len(phones) == sum(word2ph)
