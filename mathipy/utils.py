"""Shared utility functions for text pattern extraction."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_number_pattern = re.compile(r"-?\d+\.?\d*")
_variable_pattern = re.compile(r"\b[a-zA-Z]\b(?!\w)")

_latex_patterns = [
    re.compile(r"\$\$[\s\S]*?\$\$"),
    re.compile(r"\$[\s\S]*?\$"),
    re.compile(r"\\\([\s\S]*?\\\)"),
    re.compile(r"\\\[[\s\S]*?\\\]"),
    re.compile(r"\\frac\{[\s\S]*?\}\{[\s\S]*?\}"),
    re.compile(r"\\sqrt\{[\s\S]*?\}"),
    re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL),
]

_latex_command_pattern = re.compile(
    r"\\int|\\sum|\\lim|\\log|\\ln|\\sin|\\cos|\\tan"
)


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values (integers and decimals, including negatives) from text."""
    matches = _number_pattern.findall(text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def extract_variables(text: str) -> list[str]:
    """Extract single-letter variable names (e.g., x, y, n) from text."""
    return list(set(_variable_pattern.findall(text)))


def extract_math_expressions(text: str) -> list[str]:
    """Extract LaTeX expressions and equations from text."""
    expressions = []
    for pattern in _latex_patterns:
        expressions.extend(pattern.findall(text))
    equation_pattern = r"[^=\s]+\s*=\s*[^=\s]+"
    expressions.extend(re.findall(equation_pattern, text))
    return list(set(expressions))


def safe_get(d, *keys, default=None):
    """Retrieve a nested value from a dict, returning *default* on any miss."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def normalize_math_text(text: str) -> str:
    """Replace math notation and symbols with placeholders for readability analysis."""
    normalized = text
    for pattern in _latex_patterns:
        normalized = pattern.sub(" MATH ", normalized)
    normalized = _latex_command_pattern.sub(" MATH ", normalized)
    normalized = re.sub(r"[+\-−×*·÷/=<>≤≥≈^]", " ", normalized)
    normalized = re.sub(r"\b[A-D][\.)]\s*", " ", normalized)
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

    # Cohen's kappa
    labels = sorted(set(c1) | set(c2))
    p_e = sum(
        (c1.count(k) / n) * (c2.count(k) / n) for k in labels
    )
    kappa = (agreement - p_e) / (1 - p_e) if p_e < 1 else 0.0

    return {"agreement": round(agreement, 4), "kappa": round(kappa, 4), "n": n}
