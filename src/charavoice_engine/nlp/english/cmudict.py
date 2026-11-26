from __future__ import annotations

from functools import lru_cache
from pathlib import Path

CMU_DICT_PATH = Path(__file__).parent / "cmudict.rep"


@lru_cache(maxsize=1)
def get_dict() -> dict[str, list[list[str]]]:
    """Build the dictionary from cmudict.rep once per process (about 0.5 s).

    The source tree no longer ships a pickled cache: loading a pickle runs code from the file,
    which a published repository should not do at import time.
    """
    return read_dict()


def read_dict() -> dict[str, list[list[str]]]:
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict
