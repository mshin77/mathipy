"""Shared utility functions for text pattern extraction."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_number_pattern = re.compile(
    r"(?<![\w.])\d+\s*/\s*\d+"
    r"|(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d|,\d)"
    r"|(?:(?<![\w\s])-|(?<=^)-|(?<=[(\s])-(?=\d))?(?<![\w.])\d+\.?\d*")
_letter_edge = r"A-Za-z'’_"
_lower_variable = re.compile(
    rf"(?<![{_letter_edge}])(?!a(?![{_letter_edge}]))(?!(?<=\d)s\b)[a-z](?![{_letter_edge}])")
_upper_candidate = re.compile(rf"(?<![{_letter_edge}])(?![IO])[A-Z](?![{_letter_edge}])")
_option_label = re.compile(r"[A-E][.)]")
_math_operators = "=+−×÷*/<>≤≥^"
_math_after = re.compile(rf"\s*(?:[{_math_operators}]|-(?![A-Za-z]))")
_answer_label_line = re.compile(r"(?m)^\s*[A-E][.)](?:\s+|$)")
_option_value_run = re.compile(r"(?<![A-Za-z])A\s+-?[\d$.]\S*(?:\s+[B-E]\s+-?[\d$.]\S*)+")
_article = re.compile(r"A\s+[a-z]")
_label_candidate = re.compile(r"(?<![\w.])([A-E])[.)](?=\s|$)")

_latex_patterns = [
    re.compile(r"\$\$[\s\S]*?\$\$"),
    re.compile(r"\$(?![\s\d])[^$]+?(?<!\s)\$"),
    re.compile(r"\\\([\s\S]*?\\\)"),
    re.compile(r"\\\[[\s\S]*?\\\]"),
    re.compile(r"\\frac\{[\s\S]*?\}\{[\s\S]*?\}"),
    re.compile(r"\\sqrt\{[\s\S]*?\}"),
    re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL),
]

_latex_command_pattern = re.compile(
    r"\\int|\\sum|\\lim|\\log|\\ln|\\sin|\\cos|\\tan"
)


def _previous_char(text: str, index: int, skip: str = " \t") -> str:
    while index and text[index - 1] in skip:
        index -= 1
    return text[index - 1] if index else ""


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values (integers and decimals, including negatives) from text."""
    numbers = []
    for m in _number_pattern.findall((text or "").replace("−", "-")):
        try:
            if "/" in m:
                top, bottom = m.split("/")
                numbers.append(float(top) / float(bottom))
            else:
                numbers.append(float(m.replace(",", "")))
        except (ValueError, ZeroDivisionError):
            continue
    return numbers


def extract_variables(text: str) -> list[str]:
    """Extract single-letter variable names (e.g., x, y, n) from text."""
    text = _option_value_run.sub(" ", (text or "").replace("−", "-"))
    upper = [m.group() for m in _upper_candidate.finditer(text)
             if not _option_label.match(text, m.start())
             and not _article.match(text, m.start())
             and (_math_after.match(text, m.end())
                  or text[max(0, m.start() - 1):m.start()].isdigit()
                  or _previous_char(text, m.start(), " \t\r\n") in set(_math_operators + "-"))]
    return sorted(set(_lower_variable.findall(text)) | set(upper))


def extract_math_expressions(text: str) -> list[str]:
    """Extract LaTeX expressions and equations from text."""
    expressions = []
    for pattern in _latex_patterns:
        expressions.extend(pattern.findall(text))
    equation_pattern = r"[^=\s]+\s*=\s*[^=\s]+"
    expressions.extend(re.findall(equation_pattern, text))
    return sorted(set(expressions))


def consume_phrases(text: str, phrases) -> dict[str, int]:
    """Count whole-word phrase hits, longest first, removing each hit so it counts once."""
    remaining = text.lower()
    counts = {}
    for phrase in sorted(set(phrases), key=lambda p: (-len(p), p)):
        pattern = re.compile(r"\b" + re.escape(phrase.lower()) + r"\b")
        remaining, hits = pattern.subn(" | ", remaining)
        counts.update({phrase: hits} if hits else {})
    return counts


def safe_get(d, *keys, default=None):
    """Retrieve a nested value from a dict, returning *default* on any miss."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def option_label_spans(text: str) -> list[tuple[int, int]]:
    """Spans of ordered answer labels: three or more anywhere, two only after a list opening."""
    spans, chain, opened = [], [], False
    candidates = list(_label_candidate.finditer(text))
    for m, following in zip(candidates + [None], candidates[1:] + [None, None]):
        expected = chr(ord("A") + len(chain))
        if m and chain and m.group(1) == expected:
            chain.append(m.span())
            continue
        if m and len(chain) > 1 and following and following.group(1) == expected:
            continue
        spans.extend(chain if len(chain) > 2 or (opened and len(chain) == 2) else [])
        chain = [m.span()] if m and m.group(1) == "A" else []
        opened = bool(chain) and _previous_char(text, m.start()) in {"", "?", ":", "!", ".", ")", "\n"}
    return spans


def strip_option_labels(text: str) -> str:
    """Remove ordered answer-label runs and line-initial labels, leaving the option content."""
    for start, end in reversed(option_label_spans(text)):
        text = f"{text[:start]} {text[end:]}"
    return _answer_label_line.sub(" ", text)


def normalize_math_text(text: str) -> str:
    """Replace math notation and symbols with placeholders for readability analysis."""
    normalized = text or ""
    for pattern in _latex_patterns:
        normalized = pattern.sub(" MATH ", normalized)
    normalized = _latex_command_pattern.sub(" MATH ", normalized)
    normalized = strip_option_labels(normalized)
    normalized = re.sub(r"[+\-−×*·÷/=<>≤≥≈^]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def compute_interrater_reliability(
    coder1: Sequence, coder2: Sequence
) -> dict[str, Any]:
    """Compute agreement and Cohen's kappa between two raters.

    Manual implementation — no sklearn dependency required.

    Args:
        coder1: Sequence of ratings from coder 1.
        coder2: Sequence of ratings from coder 2 (same length).

    Returns:
        ``{"agreement": float, "kappa": float, "n": int}``
    """
    c1, c2 = list(coder1), list(coder2)
    n = len(c1)
    if n != len(c2):
        raise ValueError("coder1 and coder2 must have the same length")
    if n == 0:
        return {"agreement": 0.0, "kappa": 0.0, "n": 0}

    agreement = sum(a == b for a, b in zip(c1, c2)) / n

    labels = sorted(set(c1) | set(c2), key=repr)
    p_e = sum(
        (c1.count(k) / n) * (c2.count(k) / n) for k in labels
    )
    kappa = (agreement - p_e) / (1 - p_e) if p_e < 1 else 0.0

    return {"agreement": round(agreement, 4), "kappa": round(kappa, 4), "n": n}
