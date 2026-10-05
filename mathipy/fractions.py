"""Fraction values in an item, in digit and word form.

Denominator magnitude, unit fractions, unlike denominators and unreduced
forms are properties of the quantity rather than of the notation, and each is
a known source of difficulty in fraction items.
"""

import re
from math import gcd

_fraction = re.compile(r"(?<![\w.])(\d+)\s*[/⁄]\s*(\d+)(?!\w|\.\d)")

_vulgar = {
    "½": (1, 2), "⅓": (1, 3), "⅔": (2, 3), "¼": (1, 4), "¾": (3, 4),
    "⅕": (1, 5), "⅖": (2, 5), "⅗": (3, 5), "⅘": (4, 5), "⅙": (1, 6),
    "⅚": (5, 6), "⅐": (1, 7), "⅛": (1, 8), "⅜": (3, 8), "⅝": (5, 8),
    "⅞": (7, 8), "⅑": (1, 9), "⅒": (1, 10),
}

_word_numerators = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_word_denominators = {
    "half": 2, "halves": 2, "third": 3, "fourth": 4, "quarter": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
    "tenth": 10, "twelfth": 12, "sixteenth": 16,
}
_word_fraction = re.compile(
    r"\b(" + "|".join(_word_numerators) + r")[\s-]+("
    + "|".join(_word_denominators) + r")s?\b", re.IGNORECASE)
_article_fraction = re.compile(
    r"\ban?[\s-]+(" + "|".join(_word_denominators) + r")\b", re.IGNORECASE)
_article_context = re.compile(
    r"\s+of\b|\s+(?:and|or)\s+an?[\s-]+(?:" + "|".join(_word_denominators) + r")\b",
    re.IGNORECASE)
_mixed_number = re.compile(r"\band\s+$", re.IGNORECASE)

_empty = {
    "fraction_count": 0,
    "fraction_max_denominator": 0,
    "fraction_mean_denominator": 0.0,
    "fraction_unit_count": 0,
    "fraction_distinct_denominators": 0,
    "fraction_unreduced_count": 0,
}


def _word_pairs(text: str) -> list[tuple[int, int]]:
    """Fractions written in words, as (numerator, denominator)."""
    pairs = []
    for num, den in _word_fraction.findall(text):
        n = _word_numerators[num.lower()]
        d = _word_denominators[den.lower()]
        if n == 1 and den.lower().endswith("s") and den.lower() != "halves":
            continue
        pairs.append((n, d))
    pairs += [(1, _word_denominators[m.group(1).lower()]) for m in _article_fraction.finditer(text)
              if _article_context.match(text, m.end()) or _mixed_number.search(text[:m.start()])]
    return pairs


def fraction_features(text: str) -> dict[str, float]:
    """Structural features of every fraction found in text, in digit or word form."""
    text = text or ""
    matches = [(int(n), int(d)) for n, d in _fraction.findall(text) if int(d) != 0]
    matches += _word_pairs(text)
    matches += [_vulgar[c] for c in text if c in _vulgar]
    if not matches:
        return dict(_empty)
    denominators = [d for _, d in matches]
    return {
        "fraction_count": len(matches),
        "fraction_max_denominator": max(denominators),
        "fraction_mean_denominator": sum(denominators) / len(denominators),
        "fraction_unit_count": sum(1 for n, _ in matches if n == 1),
        "fraction_distinct_denominators": len(set(denominators)),
        "fraction_unreduced_count": sum(1 for n, d in matches if gcd(n, d) > 1),
    }
