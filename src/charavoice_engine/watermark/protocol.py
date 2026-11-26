from __future__ import annotations

import binascii
from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from charavoice_engine.watermark.common import pack_watermark_words, validate_watermark_word

CONTROL_MAGIC = 0xA
CONTROL_VERSION = 0x1
CONTROL_MAGIC_MASK = 0xF000
CONTROL_VERSION_MASK = 0x0F00
CONTROL_LENGTH_MASK = 0x00FF
CONTROL_PREFIX = (CONTROL_MAGIC << 12) | (CONTROL_VERSION << 8)
MIN_EXTENSION_FRAME_WORDS = 4


@dataclass
class WatermarkPlan:
    """Per-request chunk watermark plan for the fixed 16-bit AudioSeal primitive."""

    core16: int
    requested_extension_bytes: bytes
    embedded_extension_bytes: bytes
    frame_words: tuple[int, ...]
    frame_bits: NDArray[np.int64]
    mode: Literal["core_only", "extension"]
    control_word: int | None = None
    crc16: int | None = None

    @property
    def frame_len(self) -> int:
        return len(self.frame_words)

    @property
    def truncated(self) -> bool:
        return len(self.embedded_extension_bytes) < len(self.requested_extension_bytes)

    @property
    def embedded_extension_hex(self) -> str:
        return self.embedded_extension_bytes.hex()

    def word_for_chunk(self, chunk_idx: int) -> int:
        return self.frame_words[chunk_idx % self.frame_len]

    def bits_for_chunk(self, chunk_idx: int) -> NDArray[np.int64]:
        return self.frame_bits[chunk_idx % self.frame_len]


@dataclass
class WatermarkRecovery:
    """Recovered watermark payload from detected 16-bit chunk words."""

    mode: Literal["core_only", "extension"]
    core16: int
    extension_bytes: bytes
    frame_words: tuple[int, ...]
    crc16: int | None
    crc_ok: bool
    start_chunk: int
    frame_repetitions: int
    score: float

    @property
    def extension_hex(self) -> str:
        return self.extension_bytes.hex()

    @property
    def control_word(self) -> int | None:
        return self.frame_words[1] if self.mode == "extension" and len(self.frame_words) >= 2 else None


def normalize_extension_hex(extension: str | None) -> str | None:
    """Normalize an optional extension hex string.

    Accepts an optional ``0x`` prefix and ignores surrounding whitespace.
    """
    if extension is None:
        return None
    normalized = extension.strip().lower()
    if not normalized:
        return None
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    if not normalized:
        return None
    if len(normalized) % 2 != 0:
        raise ValueError("watermark_extension must have an even number of hex characters")
    try:
        bytes.fromhex(normalized)
    except ValueError as e:
        raise ValueError("watermark_extension must be a valid hex string") from e
    return normalized


def extension_hex_to_bytes(extension: str | None) -> bytes:
    normalized = normalize_extension_hex(extension)
    if normalized is None:
        return b""
    return bytes.fromhex(normalized)


def bytes_to_words(data: bytes) -> tuple[int, ...]:
    """Convert bytes into big-endian 16-bit words, padding the last byte with zero."""
    if not data:
        return ()
    padded = data if len(data) % 2 == 0 else data + b"\x00"
    return tuple(int.from_bytes(padded[i : i + 2], byteorder="big") for i in range(0, len(padded), 2))


def words_to_bytes(words: tuple[int, ...] | list[int], byte_length: int | None = None) -> bytes:
    """Convert 16-bit words back to bytes and optionally trim to the original byte length."""
    out = bytearray()
    for word in words:
        out.extend(validate_watermark_word(int(word)).to_bytes(2, byteorder="big"))
    data = bytes(out)
    if byte_length is not None:
        data = data[:byte_length]
    return data


def build_control_word(extension_nbytes: int) -> int:
    if not 0 < extension_nbytes <= CONTROL_LENGTH_MASK:
        raise ValueError(f"extension byte length must be in 1..{CONTROL_LENGTH_MASK}")
    return CONTROL_PREFIX | extension_nbytes


def parse_control_word(word: int) -> tuple[int, int] | None:
    word = validate_watermark_word(int(word), field_name="control_word")
    if (word & (CONTROL_MAGIC_MASK | CONTROL_VERSION_MASK)) != CONTROL_PREFIX:
        return None
    extension_nbytes = word & CONTROL_LENGTH_MASK
    if extension_nbytes <= 0:
        return None
    return CONTROL_VERSION, extension_nbytes


def compute_frame_crc16(core16: int, control_word: int, extension_bytes: bytes) -> int:
    payload = (
        validate_watermark_word(core16, field_name="watermark_core16").to_bytes(2, byteorder="big")
        + validate_watermark_word(control_word, field_name="control_word").to_bytes(2, byteorder="big")
        + extension_bytes
    )
    return binascii.crc_hqx(payload, 0xFFFF)


def build_watermark_plan(core16: int, extension_bytes: bytes, chunk_count: int) -> WatermarkPlan:
    """Plan a per-chunk frame using ``core`` + optional ``extension`` + frame repeat."""
    if chunk_count <= 0:
        raise ValueError("chunk_count must be >= 1")

    core16 = validate_watermark_word(core16, field_name="watermark_core16")
    requested_extension = bytes(extension_bytes)

    if chunk_count < MIN_EXTENSION_FRAME_WORDS or not requested_extension:
        frame_words = (core16,)
        return WatermarkPlan(
            core16=core16,
            requested_extension_bytes=requested_extension,
            embedded_extension_bytes=b"",
            frame_words=frame_words,
            frame_bits=pack_watermark_words(frame_words),
            mode="core_only",
        )

    max_extension_bytes = min(CONTROL_LENGTH_MASK, 2 * (chunk_count - 3))
    embedded_extension = requested_extension[:max_extension_bytes]
    if not embedded_extension:
        frame_words = (core16,)
        return WatermarkPlan(
            core16=core16,
            requested_extension_bytes=requested_extension,
            embedded_extension_bytes=b"",
            frame_words=frame_words,
            frame_bits=pack_watermark_words(frame_words),
            mode="core_only",
        )

    control_word = build_control_word(len(embedded_extension))
    extension_words = bytes_to_words(embedded_extension)
    crc16 = compute_frame_crc16(core16, control_word, embedded_extension)
    frame_words = (core16, control_word, *extension_words, crc16)
    if len(frame_words) > chunk_count:
        raise RuntimeError(f"planned watermark frame ({len(frame_words)} words) exceeds chunk_count={chunk_count}")

    return WatermarkPlan(
        core16=core16,
        requested_extension_bytes=requested_extension,
        embedded_extension_bytes=embedded_extension,
        frame_words=frame_words,
        frame_bits=pack_watermark_words(frame_words),
        mode="extension",
        control_word=control_word,
        crc16=crc16,
    )


def _ranked_votes(words: list[int], weights: list[float]) -> list[tuple[int, float]]:
    totals: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    total_weight = 0.0
    for idx, (word, weight) in enumerate(zip(words, weights, strict=True)):
        validated_word = validate_watermark_word(word)
        if validated_word not in first_seen:
            first_seen[validated_word] = idx
        totals[validated_word] = totals.get(validated_word, 0.0) + float(weight)
        total_weight += float(weight)
    ranked = sorted(totals.items(), key=lambda item: (item[1], -first_seen[item[0]]), reverse=True)
    if total_weight <= 0:
        return [(word, 0.0) for word, _weight in ranked]
    return [(word, weight / total_weight) for word, weight in ranked]


def _weighted_vote(words: list[int], weights: list[float]) -> tuple[int, float]:
    ranked = _ranked_votes(words, weights)
    if not ranked:
        raise ValueError("cannot vote on an empty word list")
    return ranked[0]


def recover_watermark(
    detected_words: list[int],
    detect_probs: list[float] | None = None,
) -> WatermarkRecovery:
    """Recover a frame-repeat watermark from per-chunk 16-bit detections.

    The embedding always starts at chunk 0. Recovery scans for a valid control
    word and uses repeated-frame voting when available. If no extension frame
    can be validated, it falls back to a core-only majority vote.
    """
    if not detected_words:
        raise ValueError("detected_words must not be empty")

    weights = detect_probs if detect_probs is not None else [1.0] * len(detected_words)
    if len(weights) != len(detected_words):
        raise ValueError("detect_probs length must match detected_words length")

    candidates: list[WatermarkRecovery] = []

    for control_idx in range(1, len(detected_words) - 1):
        parsed = parse_control_word(detected_words[control_idx])
        if parsed is None:
            continue

        _version, extension_nbytes = parsed
        frame_len = 3 + ceil(extension_nbytes / 2)
        start = control_idx - 1
        if len(detected_words) - start < frame_len:
            continue

        voted_words: list[int] = []
        voted_options: list[list[tuple[int, float]]] = []
        support_total = 0.0
        repetitions = 0
        for pos in range(frame_len):
            idxs = list(range(start + pos, len(detected_words), frame_len))
            if not idxs:
                break
            repetitions = max(repetitions, len(idxs))
            pos_words = [detected_words[i] for i in idxs]
            pos_weights = [weights[i] for i in idxs]
            ranked = _ranked_votes(pos_words, pos_weights)
            voted_word, support = ranked[0]
            voted_words.append(voted_word)
            voted_options.append(ranked)
            support_total += support

        if len(voted_words) != frame_len:
            continue

        candidate_words = voted_words[:]
        control_word = candidate_words[1]
        parsed_control = parse_control_word(control_word)
        if parsed_control is None:
            continue
        _version, voted_extension_nbytes = parsed_control
        if voted_extension_nbytes != extension_nbytes:
            continue

        extension_words = tuple(candidate_words[2:-1])
        extension_bytes = words_to_bytes(extension_words, voted_extension_nbytes)
        crc_expected = compute_frame_crc16(candidate_words[0], control_word, extension_bytes)
        crc_ok = crc_expected == candidate_words[-1]

        if not crc_ok:
            for pos, ranked in enumerate(voted_options):
                for alt_word, _support in ranked[1:]:
                    trial_words = candidate_words[:]
                    trial_words[pos] = alt_word
                    trial_control = parse_control_word(trial_words[1])
                    if trial_control is None:
                        continue
                    _version, trial_extension_nbytes = trial_control
                    if trial_extension_nbytes != extension_nbytes:
                        continue
                    trial_extension_bytes = words_to_bytes(tuple(trial_words[2:-1]), trial_extension_nbytes)
                    trial_crc = compute_frame_crc16(trial_words[0], trial_words[1], trial_extension_bytes)
                    if trial_crc == trial_words[-1]:
                        candidate_words = trial_words
                        control_word = trial_words[1]
                        extension_bytes = trial_extension_bytes
                        crc_ok = True
                        break
                if crc_ok:
                    break

        candidates.append(
            WatermarkRecovery(
                mode="extension",
                core16=candidate_words[0],
                extension_bytes=extension_bytes,
                frame_words=tuple(candidate_words),
                crc16=candidate_words[-1],
                crc_ok=crc_ok,
                start_chunk=start,
                frame_repetitions=repetitions,
                score=support_total / frame_len,
            )
        )

    valid_candidates = [candidate for candidate in candidates if candidate.crc_ok]
    if valid_candidates:
        return max(valid_candidates, key=lambda candidate: (candidate.score, candidate.frame_repetitions))
    if candidates:
        return max(candidates, key=lambda candidate: (candidate.score, candidate.frame_repetitions))

    core16, support = _weighted_vote(detected_words, weights)
    return WatermarkRecovery(
        mode="core_only",
        core16=core16,
        extension_bytes=b"",
        frame_words=(core16,),
        crc16=None,
        crc_ok=True,
        start_chunk=0,
        frame_repetitions=len(detected_words),
        score=support,
    )
