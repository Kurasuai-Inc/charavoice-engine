from __future__ import annotations

import re

import inflect

__INFLECT = inflect.engine()
__COMMA_NUMBER_PATTERN = re.compile(r"([0-9][0-9\,]+[0-9])")
__DECIMAL_NUMBER_PATTERN = re.compile(r"([0-9]+\.[0-9]+)")
__POUNDS_PATTERN = re.compile(r"£([0-9\,]*[0-9]+)")
__DOLLARS_PATTERN = re.compile(r"\$([0-9\.\,]*[0-9]+)")
__ORDINAL_PATTERN = re.compile(r"[0-9]+(st|nd|rd|th)")
__NUMBER_PATTERN = re.compile(r"[0-9]+")


def normalize_text(text: str) -> str:
    text = __normalize_numbers(text)
    text = replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text


def replace_punctuation(text: str) -> str:
    REPLACE_MAP = {
        "\uff1a": ",",  # fullwidth colon
        "\uff1b": ",",  # fullwidth semicolon
        "\uff0c": ",",  # fullwidth comma
        "\u3002": ".",  # ideographic full stop
        "\uff01": "!",  # fullwidth exclamation mark
        "\uff1f": "?",  # fullwidth question mark
        "\n": ".",
        "\uff0e": ".",  # fullwidth full stop
        "\u2026": "...",  # horizontal ellipsis
        "\u00b7\u00b7\u00b7": "...",  # middle dot x3
        "\u30fb\u30fb\u30fb": "...",  # katakana middle dot x3
        "\u00b7": ",",  # middle dot
        "\u30fb": ",",  # katakana middle dot
        "\u3001": ",",  # ideographic comma
        "$": ".",
        "\u201c": "'",  # left double quotation mark
        "\u201d": "'",  # right double quotation mark
        '"': "'",
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\uff08": "'",  # fullwidth left parenthesis
        "\uff09": "'",  # fullwidth right parenthesis
        "(": "'",
        ")": "'",
        "\u300a": "'",  # left double angle bracket
        "\u300b": "'",  # right double angle bracket
        "\u3010": "'",  # left black lenticular bracket
        "\u3011": "'",  # right black lenticular bracket
        "[": "'",
        "]": "'",
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\uff5e": "-",  # fullwidth tilde
        "~": "-",
        "\u300c": "'",  # left corner bracket
        "\u300d": "'",  # right corner bracket
    }
    pattern = re.compile("|".join(re.escape(p) for p in REPLACE_MAP))
    replaced_text = pattern.sub(lambda x: REPLACE_MAP[x.group()], text)
    return replaced_text


def __normalize_numbers(text: str) -> str:
    text = re.sub(__COMMA_NUMBER_PATTERN, __remove_commas, text)
    text = re.sub(__POUNDS_PATTERN, r"\1 pounds", text)
    text = re.sub(__DOLLARS_PATTERN, __expand_dollars, text)
    text = re.sub(__DECIMAL_NUMBER_PATTERN, __expand_decimal_point, text)
    text = re.sub(__ORDINAL_PATTERN, __expand_ordinal, text)
    text = re.sub(__NUMBER_PATTERN, __expand_number, text)
    return text


def __expand_dollars(m: re.Match[str]) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{dollars} {dollar_unit}, {cents} {cent_unit}"
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return f"{dollars} {dollar_unit}"
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{cents} {cent_unit}"
    else:
        return "zero dollars"


def __remove_commas(m: re.Match[str]) -> str:
    return m.group(1).replace(",", "")


def __expand_ordinal(m: re.Match[str]) -> str:
    return __INFLECT.number_to_words(m.group(0))  # type: ignore


def __expand_number(m: re.Match[str]) -> str:
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + __INFLECT.number_to_words(num % 100)  # type: ignore
        elif num % 100 == 0:
            return __INFLECT.number_to_words(num // 100) + " hundred"  # type: ignore
        else:
            result: str = __INFLECT.number_to_words(num, andword="", zero="oh", group=2)  # type: ignore[reportArgumentType]
            return result.replace(", ", " ")
    else:
        return __INFLECT.number_to_words(num, andword="")  # type: ignore


def __expand_decimal_point(m: re.Match[str]) -> str:
    return m.group(1).replace(".", " point ")
