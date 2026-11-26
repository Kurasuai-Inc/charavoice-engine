import numpy as np

from charavoice_engine.nlp import _align_hidden_to_chars


class _NoOffsetTokenizer:
    def __call__(self, *_args, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise NotImplementedError("return_offset_mapping is not available when using Python tokenizers.")
        return {}


def test_align_hidden_to_chars_falls_back_when_tokenizer_has_no_offset_mapping() -> None:
    hidden = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)

    aligned = _align_hidden_to_chars(
        hidden=hidden,
        tokenizer=_NoOffsetTokenizer(),
        text="うーん……そうですね。",
        n_word2ph=10,
    )

    assert aligned.shape == (10, 3)
    np.testing.assert_array_equal(aligned[0], hidden[0])
    np.testing.assert_array_equal(aligned[-1], hidden[-1])
