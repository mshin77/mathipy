"""Relations carried by notation rather than by words."""

import re

from mathipy.utils import strip_ranges

_number = r"(?<![\d.])\d+(?:\.\d+)?"

_operand = rf"(?:{_number}|[a-z]\b)"

_patterns = {
    "sym_addition": re.compile(rf"{_number}\s*\+\s*(?={_number})"),
    "sym_subtraction": re.compile(rf"{_number}\s*[-−]\s*(?={_number})"),
    "sym_multiplicative": re.compile(
        rf"{_number}\s*[*×·]\s*(?={_number})"),
    "sym_division": re.compile(rf"{_number}\s*[/÷]\s*(?={_number})"),
    "sym_comparison": re.compile(
        rf"(?:{_number}|[a-z])\s*(?:[<>]=?|[≤≥])\s*(?={_operand})", re.I),
    "sym_equality": re.compile(r"(?<![=<>!])=(?!=)"),
    "sym_exponent": re.compile(rf"(?:{_number}|[a-z])\s*\^\s*(?:{_number}|[a-z])", re.I),
}

channel_pairs = {
    "sym_multiplicative": "rel_multiplicative",
    "sym_division": "rel_division",
    "sym_comparison": "rel_comparison",
}


def symbolic_features(text: str) -> dict[str, int]:
    """Counts of relations carried by mathematical notation."""
    text = strip_ranges(text)
    counts = {name: len(pattern.findall(text)) for name, pattern in _patterns.items()}
    counts["sym_total"] = sum(counts.values())
    return counts
