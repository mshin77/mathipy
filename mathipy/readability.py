"""Readability analysis for mathematical text."""

from __future__ import annotations

import logging
import re
from typing import Any

from mathipy.utils import normalize_math_text

logger = logging.getLogger(__name__)

try:
    from textstat import (
        automated_readability_index,
        coleman_liau_index,
        dale_chall_readability_score,
        flesch_kincaid_grade,
        flesch_reading_ease,
        gunning_fog,
        linsear_write_formula,
        smog_index,
    )
    textstat_available = True
except ImportError:
    textstat_available = False
    logger.warning("textstat not available - using estimation methods")


class ReadabilityAnalyzer:
    """Analyze text readability with math-aware normalization.

    Computes Flesch-Kincaid, Gunning Fog, SMOG, and other readability metrics.
    LaTeX and math symbols are replaced with placeholders before analysis
    so they don't inflate complexity scores.

    Requires ``pip install mathipy[nlp]`` for full metrics.
    Falls back to estimation methods when textstat is not installed.
    """

    def __init__(self):
        self._cmu_dict: dict | None = None
        self._load_cmu_dict()

    def _load_cmu_dict(self) -> None:
        try:
            import nltk
            from nltk.corpus import cmudict
            try:
                nltk.data.find('corpora/cmudict')
            except LookupError:
                nltk.download('cmudict', quiet=True)
            self._cmu_dict = cmudict.dict()
        except Exception as e:
            logger.debug(f"CMU dictionary not available: {e}")

    def analyze(self, text: str) -> dict[str, Any]:
        """Compute readability metrics for the given text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with readability scores including ``flesch_kincaid_grade``,
            ``flesch_reading_ease``, ``average_grade_level``, and others.
            Includes ``low_confidence=True`` when text is shorter than 20 words.
        """
        if not text or not text.strip():
            return self._empty_metrics()

        normalized = self._normalize_for_readability(text)
        word_count = len(normalized.split())
        low_confidence = word_count < 20

        if textstat_available:
            return self._compute_metrics(normalized, low_confidence)
        else:
            return self._estimate_metrics(normalized, low_confidence)

    def _normalize_for_readability(self, text: str) -> str:
        return normalize_math_text(text)

    def _compute_metrics(self, text: str, low_confidence: bool) -> dict[str, Any]:
        try:
            fk = flesch_kincaid_grade(text)
            fog = gunning_fog(text)
            smog = smog_index(text)

            return {
                "flesch_reading_ease": flesch_reading_ease(text),
                "flesch_kincaid_grade": fk,
                "gunning_fog": fog,
                "smog_index": smog,
                "automated_readability_index": automated_readability_index(text),
                "coleman_liau_index": coleman_liau_index(text),
                "linsear_write_formula": linsear_write_formula(text),
                "dale_chall_readability": dale_chall_readability_score(text),
                "average_grade_level": (fk + fog + smog) / 3,
                "low_confidence": low_confidence,
            }
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return self._estimate_metrics(text, True)

    def _estimate_metrics(self, text: str, low_confidence: bool) -> dict[str, Any]:
        words = text.split()
        word_count = len(words)

        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        total_syllables = sum(self._count_syllables(w) for w in words)
        avg_syllables = total_syllables / word_count if word_count else 0

        avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0
        avg_sentence_length = word_count / sentence_count

        estimated_grade = (avg_word_length * 1.5) + (avg_sentence_length * 0.3) - 3
        estimated_grade = max(1, min(16, estimated_grade))

        flesch = max(0, min(100, 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables))

        return {
            "flesch_reading_ease": flesch,
            "flesch_kincaid_grade": estimated_grade,
            "gunning_fog": estimated_grade * 1.1,
            "smog_index": estimated_grade * 0.9,
            "automated_readability_index": estimated_grade,
            "coleman_liau_index": estimated_grade,
            "linsear_write_formula": estimated_grade,
            "dale_chall_readability": estimated_grade,
            "average_grade_level": estimated_grade,
            "low_confidence": low_confidence,
            "estimated": True,
        }

    def _count_syllables(self, word: str) -> int:
        """Uses CMU dictionary if available, falls back to vowel-counting heuristic."""
        word = word.lower().strip()
        if not word:
            return 0

        if self._cmu_dict and word in self._cmu_dict:
            pronunciation = self._cmu_dict[word][0]
            return len([p for p in pronunciation if p[-1].isdigit()])

        vowels = "aeiou"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        if word.endswith("e") and count > 1:
            count -= 1

        return max(1, count)

    def _empty_metrics(self) -> dict[str, Any]:
        return {
            "flesch_reading_ease": 50.0,
            "flesch_kincaid_grade": 8.0,
            "gunning_fog": 8.0,
            "smog_index": 8.0,
            "automated_readability_index": 8.0,
            "coleman_liau_index": 8.0,
            "linsear_write_formula": 8.0,
            "dale_chall_readability": 8.0,
            "average_grade_level": 8.0,
            "low_confidence": True,
        }
