"""Shared utility functions for text pattern extraction."""

import re
from typing import List

_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")
_VARIABLE_PATTERN = re.compile(r"\b[a-zA-Z]\b(?!\w)")

_LATEX_PATTERNS = [
    re.compile(r"\$\$[\s\S]*?\$\$"),
    re.compile(r"\$[\s\S]*?\$"),
    re.compile(r"\\\([\s\S]*?\\\)"),
    re.compile(r"\\\[[\s\S]*?\\\]"),
    re.compile(r"\\frac\{[\s\S]*?\}\{[\s\S]*?\}"),
    re.compile(r"\\sqrt\{[\s\S]*?\}"),
    re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL),
]

_LATEX_COMMAND_PATTERN = re.compile(
    r"\\int|\\sum|\\lim|\\log|\\ln|\\sin|\\cos|\\tan"
)


def extract_numbers(text: str) -> List[float]:
    """Extract all numeric values (integers and decimals, including negatives) from text."""
    matches = _NUMBER_PATTERN.findall(text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def extract_variables(text: str) -> List[str]:
    """Extract single-letter variable names (e.g., x, y, n) from text."""
    return list(set(_VARIABLE_PATTERN.findall(text)))


def extract_math_expressions(text: str) -> List[str]:
    """Extract LaTeX expressions and equations from text."""
    expressions = []
    for pattern in _LATEX_PATTERNS:
        expressions.extend(pattern.findall(text))
    equation_pattern = r"[^=\s]+\s*=\s*[^=\s]+"
    expressions.extend(re.findall(equation_pattern, text))
    return list(set(expressions))


def normalize_math_text(text: str) -> str:
    """Replace math notation and symbols with placeholders for readability analysis."""
    normalized = text
    for pattern in _LATEX_PATTERNS:
        normalized = pattern.sub(" MATH ", normalized)
    normalized = _LATEX_COMMAND_PATTERN.sub(" MATH ", normalized)
    normalized = re.sub(r"[+\-−×*·÷/=<>≤≥≈^]", " ", normalized)
    normalized = re.sub(r"\b[A-D][\.)]\s*", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
