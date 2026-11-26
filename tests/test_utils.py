"""Tests for charavoice_engine.models.commons.intersperse."""

from __future__ import annotations

from charavoice_engine.models.commons import intersperse


class TestIntersperse:
    def test_basic(self):
        assert intersperse([1, 2, 3], 0) == [0, 1, 0, 2, 0, 3, 0]

    def test_single_element(self):
        assert intersperse([1], 0) == [0, 1, 0]

    def test_empty_list(self):
        assert intersperse([], 0) == [0]

    def test_string_items(self):
        assert intersperse(["a", "b"], "-") == ["-", "a", "-", "b", "-"]

    def test_result_length(self):
        lst = [1, 2, 3, 4, 5]
        result = intersperse(lst, 0)
        assert len(result) == len(lst) * 2 + 1

    def test_original_elements_at_odd_indices(self):
        lst = [10, 20, 30]
        result = intersperse(lst, 0)
        assert result[1::2] == lst
