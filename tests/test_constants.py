"""Tests for charavoice_engine.constants module."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from charavoice_engine.constants import (
    BASE_DIR,
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_ONNX_BERT_MODEL_PATHS,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)


class TestLanguages:
    def test_is_strenum(self):
        assert issubclass(Languages, StrEnum)

    def test_members(self):
        assert Languages.JP == "JP"
        assert Languages.EN == "EN"

    def test_member_count(self):
        assert len(Languages) == 2

    def test_string_coercion(self):
        assert str(Languages.JP) == "JP"
        assert f"{Languages.EN}" == "EN"


class TestBaseDir:
    def test_is_path(self):
        assert isinstance(BASE_DIR, Path)

    def test_exists(self):
        assert BASE_DIR.is_dir()


class TestDefaultInferenceParams:
    def test_style(self):
        assert DEFAULT_STYLE == "Neutral"

    def test_style_weight(self):
        assert DEFAULT_STYLE_WEIGHT == 1.0

    def test_noise(self):
        assert DEFAULT_NOISE == 0.6

    def test_length(self):
        assert DEFAULT_LENGTH == 1.0

    def test_line_split(self):
        assert DEFAULT_LINE_SPLIT is True

    def test_split_interval(self):
        assert DEFAULT_SPLIT_INTERVAL == 0.5

    def test_assist_text_weight(self):
        assert DEFAULT_ASSIST_TEXT_WEIGHT == 1.0


class TestBertModelPaths:
    def test_all_languages_present(self):
        for lang in Languages:
            assert lang in DEFAULT_ONNX_BERT_MODEL_PATHS

    def test_paths_are_path_objects(self):
        for path in DEFAULT_ONNX_BERT_MODEL_PATHS.values():
            assert isinstance(path, Path)

    def test_paths_under_base_dir(self):
        for path in DEFAULT_ONNX_BERT_MODEL_PATHS.values():
            assert path.is_relative_to(BASE_DIR / "bert")
