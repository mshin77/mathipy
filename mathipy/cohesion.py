"""Cohesion and discourse measures for extended writing and dialogue."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

connectives = {
    "additive": ["in addition", "as well as", "also", "moreover", "furthermore",
                 "besides", "similarly", "likewise"],
    "causal": ["as a result", "because of", "because", "therefore", "thus", "hence",
               "consequently", "since", "so that"],
    "temporal": ["at the same time", "meanwhile", "afterward", "finally", "then",
                 "next", "before", "after", "first", "second"],
    "adversative": ["on the other hand", "even though", "however", "although",
                    "nevertheless", "whereas", "instead", "yet", "but"],
    "clarifying": ["for example", "for instance", "in other words", "that is",
                   "specifically", "namely"],
}

pronouns = {
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "this", "that", "these", "those", "who", "which",
}

_function_words = pronouns | set(
    "a an the of to in for is are was were be been am do does did have has had "
    "will would can could should may might must and or if not no yes".split()
)


_abbreviations = {"mr", "mrs", "ms", "dr", "st", "vs", "fig", "no", "approx",
                  "e.g", "i.e", "etc", "in", "ft", "cm", "mm", "km", "lb", "oz"}
_answer_label = re.compile(r"^[A-E]$")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation.

    Splitting on a bare period fragments an assessment item in two ways that
    matter: answer-choice labels ("A. 225 B. 233") become separate sentences,
    and titles ("Mr. Chapman") break mid-name. Both inflate the sentence count,
    and because answer lists occur mostly in multiple-choice items the
    inflation tracks item format rather than prose structure.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged:
            tail = merged[-1].rstrip(".").rsplit(None, 1)
            last = tail[-1].lower() if tail else ""
            if last in _abbreviations or _answer_label.match(last.upper()):
                merged[-1] = f"{merged[-1]} {part}"
                continue
        merged.append(part)
    return merged


_word_re = re.compile(r"[a-z]+'?[a-z]*|\d+(?:[./]\d+)*")


def _words(text: str) -> list[str]:
    return _word_re.findall(text.lower())


def _content(text: str) -> set[str]:
    return {w for w in _words(text)
            if (w[0].isdigit() or (len(w) > 2 and w not in _function_words))}


def connective_density(text: str) -> dict[str, Any]:
    """Count connectives by category, normalized per 100 words."""
    lowered = " " + re.sub(r"[^a-z\s]", " ", text.lower()) + " "
    total_words = len(_words(text)) or 1
    counts = {}
    for category, terms in connectives.items():
        hits = 0
        remaining = lowered
        for term in sorted(terms, key=len, reverse=True):
            pattern = r"\s" + re.escape(term) + r"\s"
            hits += len(re.findall(pattern, remaining))
            remaining = re.sub(pattern, " ", remaining)
        counts[category] = hits
    total = sum(counts.values())
    return {
        **{f"connective_{k}": v for k, v in counts.items()},
        "connective_total": total,
        "connective_per_100w": round(100 * total / total_words, 3),
    }


def lexical_overlap(units: Sequence[str]) -> dict[str, float]:
    """Mean content-word overlap between adjacent units."""
    scores = []
    for first, second in zip(units, units[1:]):
        base = _content(first)
        if base:
            scores.append(len(_content(second) & base) / len(base))
    mean = sum(scores) / len(scores) if scores else 0.0
    return {"overlap_adjacent_mean": round(mean, 4),
            "overlap_pairs": len(scores)}


def lexical_diversity(text: str, window: int = 50) -> dict[str, float]:
    """Moving-average type-token ratio, falling back to plain ratio when short.

    Numerals are excluded here and only here. A type-token ratio measures how
    varied the vocabulary is, and "3, 7, 12, 45" contributes four types with no
    lexical variety at all, so counting numerals would make a numerically dense
    item look lexically rich. Length denominators elsewhere in this module do
    count them, because there the question is how long the text is.
    """
    tokens = [w for w in _words(text) if not w[0].isdigit()]
    if not tokens:
        return {"lexical_diversity": 0.0, "token_count": 0}
    if len(tokens) < window:
        return {"lexical_diversity": round(len(set(tokens)) / len(tokens), 4),
                "token_count": len(tokens)}
    ratios = [len(set(tokens[i:i + window])) / window
              for i in range(len(tokens) - window + 1)]
    return {"lexical_diversity": round(sum(ratios) / len(ratios), 4),
            "token_count": len(tokens)}


def pronoun_density(text: str) -> dict[str, float]:
    """Pronouns per 100 words."""
    tokens = _words(text)
    if not tokens:
        return {"pronoun_per_100w": 0.0, "pronoun_count": 0}
    hits = sum(1 for w in tokens if w in pronouns)
    return {"pronoun_per_100w": round(100 * hits / len(tokens), 3),
            "pronoun_count": hits}


def cohesion_features(source: str | Sequence[str]) -> dict[str, Any]:
    """Return all cohesion measures for a text or a sequence of units."""
    if isinstance(source, str):
        units = split_sentences(source)
        text = source
    else:
        units = list(source)
        text = " ".join(units)

    return {
        **connective_density(text),
        **lexical_overlap(units),
        **lexical_diversity(text),
        **pronoun_density(text),
        "unit_count": len(units),
    }
