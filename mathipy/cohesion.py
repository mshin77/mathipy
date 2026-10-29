"""Cohesion and discourse measures for extended writing and dialogue."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from mathipy.utils import option_label_spans

connectives = {
    "additive": ["in addition", "as well as", "also", "moreover", "furthermore",
                 "besides", "similarly", "likewise"],
    "causal": ["as a result", "because of", "because", "therefore", "thus", "hence",
               "consequently", "since", "so that"],
    "temporal": ["at the same time", "meanwhile", "afterward", "finally",
                 "next", "before", "after", "first", "second"],
    "conditional": ["if and only if", "provided that", "given that", "in that case",
                    "whenever", "otherwise", "unless", "if", "then"],
    "adversative": ["on the other hand", "even though", "however", "although",
                    "nevertheless", "whereas", "instead", "yet", "but"],
    "clarifying": ["for example", "for instance", "in other words", "that is",
                   "specifically", "namely"],
}

negations = {
    "not", "no", "never", "none", "neither", "nor", "without", "cannot",
    "can't", "don't", "doesn't", "isn't", "aren't", "won't", "didn't",
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


# Units and times end a sentence when a capital follows ("4 cm. What..."), but not
# before a lowercase word ("6 yd. long").
_units = {"in", "no", "ft", "cm", "mm", "km", "lb", "oz", "yd", "mi", "min", "hr", "hrs",
          "sec", "kg", "mg", "ml", "pt", "qt", "gal", "tsp", "tbsp", "a.m", "p.m"}
_abbreviations = {"mr", "mrs", "ms", "dr", "st", "vs", "fig", "approx",
                  "e.g", "i.e", "etc"} | _units
_answer_label = re.compile(r"^[A-E]$")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation."""
    text = (text or "").strip()
    for _, end in option_label_spans(text):
        text = text[:end - 1] + ("\x00" if text[end - 1] == "." else text[end - 1]) + text[end:]
    parts = [p.strip().replace("\x00", ".")
             for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    merged: list[str] = []
    for previous, part in zip([""] + parts, parts):
        tail = previous.rstrip(".").rsplit(None, 1)
        last = tail[-1] if tail else ""
        continues = not part[0].isupper()
        label = bool(_answer_label.match(last)) and (previous == f"{last}." or continues)
        abbreviation = last.lower() in _abbreviations and (last.lower() not in _units or continues)
        if merged and (abbreviation or label):
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


def lexical_diversity(text: str, window: int = 10) -> dict[str, float]:
    """Moving-average type-token ratio (MATTR; Covington and McFall, 2010)."""
    tokens = [w for w in _words(text) if not w[0].isdigit()]
    if not tokens:
        return {"lexical_diversity": 0.0, "token_count": 0}
    size = min(window, len(tokens))
    ratios = [len(set(tokens[i:i + size])) / size
              for i in range(len(tokens) - size + 1)]
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


def negation_density(text: str) -> dict[str, float]:
    """Negations per 100 words."""
    tokens = _words(text)
    if not tokens:
        return {"negation_per_100w": 0.0, "negation_count": 0}
    hits = sum(1 for w in tokens if w in negations)
    return {"negation_per_100w": round(100 * hits / len(tokens), 3),
            "negation_count": hits}


def cohesion_features(source: str | Sequence[str]) -> dict[str, Any]:
    """Return all cohesion measures for a text or a sequence of units."""
    source = "" if source is None else source
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
        **negation_density(text),
        "unit_count": len(units),
    }
