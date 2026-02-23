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

    def estimate(
        self,
        text: str,
        readability_grade: float | None = None,
        math_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate cognitive load for the given text.

        Args:
            text: Input text to analyze.
            readability_grade: Optional Flesch-Kincaid grade level. Estimated from text if not provided.
            math_terms: Optional list of math terms found in the text. Estimated from keywords if not provided.

        Returns:
            Dictionary with ``intrinsic_cognitive_load``, ``extraneous_cognitive_load``,
            ``germane_cognitive_load``, ``total_cognitive_load``, and element counts.
        """
        if not text or not text.strip():
            return self._empty_estimate()

        numbers = extract_numbers(text)
        variables = extract_variables(text)
        operations = sum(1 for c in text if c in "+-*/^=<>")
        word_count = len(text.split())

        intrinsic = (len(numbers) + len(variables)) / word_count if word_count else 0
        intrinsic = min(1.0, intrinsic * 2)

        if readability_grade is not None:
            extraneous = min(1.0, readability_grade / 12)
        else:
            extraneous = self._estimate_extraneous(text)

        math_term_count = len(math_terms) if math_terms else 0
        if math_term_count:
            germane = min(1.0, math_term_count / 10)
        else:
            germane = self._estimate_germane(text)

        total = intrinsic * 0.4 + extraneous * 0.3 + germane * 0.3

        return {
            "intrinsic_cognitive_load": round(intrinsic, 3),
            "extraneous_cognitive_load": round(extraneous, 3),
            "germane_cognitive_load": round(germane, 3),
            "total_cognitive_load": round(total, 3),
            "numeric_elements": len(numbers),
            "variable_count": len(variables),
            "operation_count": operations,
        }

    def _estimate_extraneous(self, text: str) -> float:
        words = text.split()
        word_count = len(words)
        if not word_count:
            return 0.0

        avg_word_length = sum(len(w) for w in words) / word_count
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        avg_sentence_length = word_count / max(len(sentences), 1)

        estimated_grade = (avg_word_length * 1.5) + (avg_sentence_length * 0.3) - 3
        estimated_grade = max(1, min(16, estimated_grade))
        return min(1.0, estimated_grade / 12)

    def _estimate_germane(self, text: str) -> float:
        math_keywords = {
            "add", "subtract", "multiply", "divide", "sum", "difference",
            "product", "quotient", "fraction", "decimal", "percent",
            "equation", "variable", "solve", "function", "graph",
            "area", "perimeter", "volume", "angle", "triangle", "circle",
            "mean", "median", "mode", "probability", "ratio", "proportion",
        }
        text_lower = text.lower()
        found = sum(1 for term in math_keywords if term in text_lower)
        return min(1.0, found / 10) if found else 0.3

    def _empty_estimate(self) -> dict[str, Any]:
        return {
            "intrinsic_cognitive_load": 0.0,
            "extraneous_cognitive_load": 0.0,
            "germane_cognitive_load": 0.0,
            "total_cognitive_load": 0.0,
            "numeric_elements": 0,
            "variable_count": 0,
            "operation_count": 0,
        }
