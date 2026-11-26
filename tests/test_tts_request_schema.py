from pydantic import ValidationError

from charavoice_engine.constants import Languages
from charavoice_engine.server.schemas import FullControlConvertRequest, TTSRequest


def _jp_full_control_chunk() -> dict[str, object]:
    return {
        "language": "JP",
        "surface_text": "今日",
        "phones": ["ky", "o:"],
        "tones": [1, 0],
        "word2ph": [1, 1],
    }


def test_tts_request_accepts_text_only_input() -> None:
    request = TTSRequest(text="こんにちは", speaker_name="speaker_a", streaming=False, format="wav")

    assert request.text == "こんにちは"
    assert request.chunks is None
    assert request.language == Languages.JP
    assert request.format == "wav"


def test_tts_request_accepts_full_control_chunks() -> None:
    request = TTSRequest(chunks=[_jp_full_control_chunk()], speaker_name="speaker_a", streaming=True, format="pcm")

    assert request.text is None
    assert request.chunks is not None
    assert request.chunks[0].language == Languages.JP
    assert request.chunks[0].surface_text == "今日"
    assert request.chunks[0].phones == ["ky", "o:"]
    assert request.streaming is True
    assert request.format == "pcm"


def test_tts_request_accepts_zero_word2ph_full_control_chunks() -> None:
    request = TTSRequest(
        chunks=[
            {
                "language": "JP",
                "surface_text": "今日",
                "phones": ["ky", "o:"],
                "tones": [1, 0],
                "word2ph": [2, 0],
            }
        ],
        speaker_name="speaker_a",
    )

    assert request.chunks is not None
    assert request.chunks[0].word2ph == [2, 0]


def test_tts_request_accepts_mixed_text_chunks() -> None:
    request = TTSRequest(
        chunks=[
            {"language": "JP", "text": "今日は"},
            {"language": "EN", "text": "Extended Reality"},
            {"language": "JP", "text": "の話です。"},
        ],
        speaker_name="speaker_a",
    )

    assert request.chunks is not None
    assert [chunk.language for chunk in request.chunks] == [Languages.JP, Languages.EN, Languages.JP]
    assert request.chunks[1].text == "Extended Reality"


def test_tts_request_accepts_mixed_full_control_and_text_chunks() -> None:
    request = TTSRequest(
        chunks=[
            _jp_full_control_chunk(),
            {"language": "EN", "text": "mixed language"},
        ],
        speaker_name="speaker_a",
    )

    assert request.chunks is not None
    assert request.chunks[0].surface_text == "今日"
    assert request.chunks[1].text == "mixed language"


def test_full_control_convert_request_defaults_to_jp_conversion() -> None:
    request = FullControlConvertRequest(
        chunks=[
            {"language": "JP", "text": "今日は"},
            {"language": "EN", "text": "mixed language"},
        ]
    )

    assert request.convert_languages == [Languages.JP]


def test_tts_request_rejects_unknown_chunk_fields() -> None:
    display_key = "display_text"
    pronunciation_key = "pronunciation_hint"
    try:
        TTSRequest(
            chunks=[
                {
                    "language": "JP",
                    display_key: "こんにちは",
                    pronunciation_key: "コンニチハ",
                }
            ],
        )
    except ValidationError as error:
        error_text = str(error)
        assert display_key in error_text
        assert pronunciation_key in error_text
        assert "extra_forbidden" in error_text
    else:
        raise AssertionError("TTSRequest accepted unknown chunk field names")


def test_tts_request_rejects_unknown_top_level_key() -> None:
    try:
        TTSRequest(utterances=[_jp_full_control_chunk()])
    except ValidationError as error:
        assert "extra_forbidden" in str(error)
    else:
        raise AssertionError("TTSRequest accepted an unknown top-level key")


def test_tts_request_requires_exactly_one_input_mode() -> None:
    for payload in ({}, {"text": "こんにちは", "chunks": [_jp_full_control_chunk()]}):
        try:
            TTSRequest(**payload)
        except ValidationError as error:
            assert "Exactly one of 'text' or 'chunks' must be provided" in str(error)
        else:
            raise AssertionError(f"TTSRequest accepted invalid input mode: {payload}")


def test_tts_request_rejects_bad_full_control_alignment() -> None:
    try:
        TTSRequest(
            chunks=[
                {
                    "language": "JP",
                    "surface_text": "今日",
                    "phones": ["ky", "o:"],
                    "tones": [1],
                    "word2ph": [1, 1],
                }
            ],
        )
    except ValidationError as error:
        assert "len(phones) must equal len(tones)" in str(error)
    else:
        raise AssertionError("TTSRequest accepted mismatched phones/tones")


def test_tts_request_rejects_negative_word2ph_full_control_chunks() -> None:
    try:
        TTSRequest(
            chunks=[
                {
                    "language": "JP",
                    "surface_text": "今日",
                    "phones": ["ky"],
                    "tones": [1],
                    "word2ph": [1, -1],
                }
            ],
        )
    except ValidationError as error:
        assert "word2ph values must be non-negative" in str(error)
    else:
        raise AssertionError("TTSRequest accepted negative word2ph")


def test_tts_request_rejects_unsupported_language() -> None:
    try:
        TTSRequest(text="你好", language="Z" + "H")
    except ValidationError as error:
        assert "Input should be 'JP' or 'EN'" in str(error)
    else:
        raise AssertionError("TTSRequest accepted unsupported language")


def test_tts_request_rejects_unknown_audio_format() -> None:
    try:
        TTSRequest(text="こんにちは", format="mp3")
    except ValidationError as error:
        assert "format" in str(error)
    else:
        raise AssertionError("TTSRequest accepted unsupported audio format")


def test_tts_request_rejects_incoherent_streaming_format() -> None:
    invalid_payloads = (
        {"text": "こんにちは", "streaming": True, "format": "wav"},
        {"text": "こんにちは", "streaming": False, "format": "pcm"},
    )
    for payload in invalid_payloads:
        try:
            TTSRequest(**payload)
        except ValidationError as error:
            assert "format" in str(error) or "streaming" in str(error)
        else:
            raise AssertionError(f"TTSRequest accepted incoherent format: {payload}")
