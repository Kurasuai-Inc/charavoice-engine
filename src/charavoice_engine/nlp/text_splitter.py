"""Split long text at word2ph boundaries to fit TRT profile max_seq_len.

Pure functions — no model or engine dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

# Preferred split points (Japanese + ASCII punctuation)
_PUNCT = set("、。！？!?,.…")


@dataclass
class TextChunk:
    """A segment of norm_text after splitting."""

    text: str


def estimate_seq_len(word2ph: list[int], add_blank: bool) -> int:
    """Estimate post-intersperse sequence length from word2ph.

    word2ph includes start/end "_" entries (len = len(norm_text) + 2).
    After intersperse: each phone doubles, plus word2ph[0] gets +1.
    """
    n_phones = sum(word2ph)
    if add_blank:
        return 2 * n_phones + 1
    return n_phones


def split_text_by_seq_len(
    norm_text: str,
    word2ph: list[int],
    max_seq_len: int,
    add_blank: bool,
) -> list[TextChunk]:
    """Split norm_text at word2ph character boundaries so each chunk fits max_seq_len.

    Args:
        norm_text: Normalized text from G2P (no start/end markers).
        word2ph: Phone counts per character, including start/end "_"
                 (len = len(norm_text) + 2). word2ph[0] and word2ph[-1]
                 are for the implicit start/end tokens.
        max_seq_len: Maximum post-intersperse sequence length (e.g. 512).
        add_blank: Whether intersperse (add_blank) is enabled.

    Returns:
        List of TextChunk. If text fits, returns single chunk with full text.
    """
    assert len(word2ph) == len(norm_text) + 2, (
        f"word2ph length {len(word2ph)} != len(norm_text)+2 ({len(norm_text) + 2})"
    )

    # Check if the full text fits
    full_seq_len = estimate_seq_len(word2ph, add_blank)
    if full_seq_len <= max_seq_len:
        return [TextChunk(text=norm_text)]

    # Per-character phone counts (excluding start/end "_")
    char_phones = word2ph[1:-1]  # len = len(norm_text)
    assert len(char_phones) == len(norm_text)

    # Start/end "_" overhead: 2 phones (1 for start, 1 for end)
    # After add_blank: overhead = 2*2 + 1 = 5 (2 phones * 2 for intersperse + 1 for leading blank)
    # More precisely: each chunk gets its own start/end, so overhead per chunk:
    #   start_phones = word2ph[0] typically = 1
    #   end_phones = word2ph[-1] typically = 1
    #   overhead_phones = 2
    overhead_phones = 2  # start "_" + end "_"

    chunks: list[TextChunk] = []
    start = 0
    n_chars = len(norm_text)

    while start < n_chars:
        # Greedily find the largest end position that fits
        cum_phones = 0
        end = start
        last_punct = -1

        while end < n_chars:
            chunk_phones = cum_phones + char_phones[end] + overhead_phones
            chunk_seq_len = _phones_to_seq_len(chunk_phones, add_blank)
            if chunk_seq_len > max_seq_len:
                break
            cum_phones += char_phones[end]
            # Track punctuation for preferred split
            if norm_text[end] in _PUNCT:
                last_punct = end
            end += 1

        if end == start:
            # Single character exceeds limit — shouldn't happen in practice,
            # but force-include it to avoid infinite loop
            end = start + 1

        # Prefer splitting at punctuation — use the last punctuation in the
        # chunk as long as it's past the halfway point (avoid tiny chunks)
        halfway = start + (end - start) // 2
        if end < n_chars and last_punct >= halfway:
            end = last_punct + 1  # include the punctuation in this chunk

        chunks.append(TextChunk(text=norm_text[start:end]))
        start = end

    return chunks


def _phones_to_seq_len(n_phones: int, add_blank: bool) -> int:
    """Convert phone count to post-intersperse sequence length."""
    if add_blank:
        return 2 * n_phones + 1
    return n_phones
