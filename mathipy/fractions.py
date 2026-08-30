"""Fraction values in an item, in digit and word form.

Denominator magnitude, unit fractions, unlike denominators and unreduced
forms are properties of the quantity rather than of the notation, and each is
a known source of difficulty in fraction items.
"""

import re
from math import gcd

_fraction = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")

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
    return pairs


def fraction_features(text: str) -> dict[str, float]:
    """Structural features of every fraction found in text, in digit or word form."""
    text = text or ""
    matches = [(int(n), int(d)) for n, d in _fraction.findall(text) if int(d) != 0]
    matches += _word_pairs(text)
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
