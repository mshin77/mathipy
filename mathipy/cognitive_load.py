"""Cognitive load estimation for mathematical assessment items."""

from __future__ import annotations

import logging
import re
from typing import Any

from mathipy.utils import extract_numbers, extract_variables

logger = logging.getLogger(__name__)


class CognitiveLoadEstimator:
    """Estimate cognitive load components for math assessment items.

    Computes intrinsic (item complexity), extraneous (linguistic demand),
    and germane (schema-building) load from text features.
    """

    # CCSSM-aligned math vocabulary for germane load indicator counting.
    math_keywords = {
        "add", "subtract", "multiply", "divide", "sum", "difference",
        "product", "quotient", "fraction", "decimal", "percent",
        "equation", "variable", "solve", "function", "graph",
        "area", "perimeter", "volume", "angle", "triangle", "circle",
        "mean", "median", "mode", "probability", "ratio", "proportion",
    }

    def estimate(
        self,
        text: str,
        readability_grade: float | None = None,
        math_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extract raw cognitive load indicators from text.

        Returns raw counts and ratios only — no composite scores or
        scaling constants. Weights should be determined empirically.

        Args:
            text: Input text to analyze.
            readability_grade: Optional Flesch-Kincaid grade level.
            math_terms: Optional list of math terms found in the text.

        Returns:
            Dictionary with raw element counts and ratios.
        """
        if not text or not text.strip():
            return self._empty_estimate()

        numbers = extract_numbers(text)
        variables = extract_variables(text)
        operations = sum(1 for c in text if c in "+-*/^=<>")
        words = text.split()
        word_count = len(words)

        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        math_term_count = len(math_terms) if math_terms else self._count_math_keywords(text)

        return {
            "numeric_elements": len(numbers),
            "variable_count": len(variables),
            "operation_count": operations,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "math_term_count": math_term_count,
            "element_density": (len(numbers) + len(variables)) / word_count if word_count else 0,
            "avg_sentence_length": word_count / sentence_count,
        }

    def _count_math_keywords(self, text: str) -> int:
        text_lower = text.lower()
        return sum(
            1 for term in self.math_keywords
            if re.search(r"\b" + re.escape(term) + r"\b", text_lower)
        )

    def _empty_estimate(self) -> dict[str, Any]:
        return {
            "numeric_elements": 0,
            "variable_count": 0,
            "operation_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "math_term_count": 0,
            "element_density": 0.0,
            "avg_sentence_length": 0.0,
        }
