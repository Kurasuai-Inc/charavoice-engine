"""cmudict is built from cmudict.rep at first use; no pickle is shipped or loaded (room #4228)."""

from pathlib import Path

from charavoice_engine.nlp.english import cmudict


def test_dictionary_is_built_from_the_text_file_once():
    cmudict.get_dict.cache_clear()
    first = cmudict.get_dict()
    assert cmudict.get_dict() is first  # one parse per process
    assert len(first) == 129464
    assert first["HERRING"] == [["HH", "EH1"], ["R", "IH0", "NG"]]


def test_no_pickle_is_shipped_or_used():
    package = Path(cmudict.__file__).parent
    assert not list(package.glob("*.pickle"))
    source = Path(cmudict.__file__).read_text(encoding="utf-8")
    assert "import pickle" not in source and "pickle.load" not in source
