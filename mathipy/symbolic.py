"""Relations carried by notation rather than by words.

O'Halloran (2015) treats language and symbolism as separate resources.
``register.relational_features`` measures the wording; these count the same
relations in notation, kept apart so the channels stay comparable.

``sym_division`` counts a solidus between numbers, which is also how a
fraction is written; ``fractions`` measures that separately.
"""

import re

_number = r"\d+(?:\.\d+)?"

_patterns = {
    "sym_addition": re.compile(rf"{_number}\s*\+\s*{_number}"),
    "sym_subtraction": re.compile(rf"{_number}\s*[-−]\s*{_number}"),
    "sym_multiplicative": re.compile(
        rf"{_number}\s*[*×·]\s*{_number}"),
    "sym_division": re.compile(rf"{_number}\s*[/÷]\s*{_number}"),
    "sym_comparison": re.compile(
        rf"{_number}\s*[<>≤≥]\s*{_number}"
        rf"|[a-z]\s*[<>≤≥]\s*{_number}", re.I),
    "sym_equality": re.compile(r"[^=<>!]=[^=]"),
    "sym_exponent": re.compile(rf"(?:{_number}|[a-z])\s*\^\s*(?:{_number}|[a-z])", re.I),
}

channel_pairs = {
    "sym_multiplicative": "rel_multiplicative",
    "sym_division": "rel_division",
    "sym_comparison": "rel_comparison",
}


def symbolic_features(text: str) -> dict[str, int]:
    """Counts of relations carried by mathematical notation."""
    text = text or ""
    counts = {name: len(pattern.findall(text)) for name, pattern in _patterns.items()}
    counts["sym_total"] = sum(counts.values())
    return counts
