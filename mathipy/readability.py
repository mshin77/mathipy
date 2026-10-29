"""Readability analysis for mathematical text."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from mathipy._api import _optional_import
from mathipy.utils import normalize_math_text

logger = logging.getLogger(__name__)

_textstat, textstat_available = _optional_import("textstat", "textstat")
if textstat_available:
    automated_readability_index = _textstat.automated_readability_index
    coleman_liau_index = _textstat.coleman_liau_index
    dale_chall_readability_score = _textstat.dale_chall_readability_score
    flesch_kincaid_grade = _textstat.flesch_kincaid_grade
    flesch_reading_ease = _textstat.flesch_reading_ease
    gunning_fog = _textstat.gunning_fog
    linsear_write_formula = _textstat.linsear_write_formula
    smog_index = _textstat.smog_index

_metric_keys = (
    "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "smog_index",
    "automated_readability_index", "coleman_liau_index", "linsear_write_formula",
    "dale_chall_readability", "average_grade_level",
)
_dale_chall_skip = re.compile(r"\b\d+(?:[.,/]\d+)*\b|\bMATH\b")


def _dale_chall_text(text: str) -> str:
    return re.sub(r"\s+", " ", _dale_chall_skip.sub(" ", text)).strip()


@lru_cache(maxsize=1)
def _warn_missing_cmudict() -> None:
    logger.warning("CMU dictionary not installed; syllables use a vowel heuristic. "
                   "Install with: python -m nltk.downloader cmudict")


class ReadabilityAnalyzer:
    """Analyze text readability with math-aware normalization."""

    def __init__(self):
        self._cmu_dict: dict | None = None
        self._cmu_missing = False
        self._load_cmu_dict()

    def _load_cmu_dict(self) -> None:
        try:
            import nltk
            from nltk.corpus import cmudict
            nltk.data.find('corpora/cmudict')
            self._cmu_dict = cmudict.dict()
        except LookupError:
            self._cmu_missing = True
            _warn_missing_cmudict()
        except Exception as e:
            logger.debug(f"CMU dictionary not available: {e}")

    def analyze(self, text: str) -> dict[str, Any]:
        """Compute readability metrics for the given text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with readability scores including ``flesch_kincaid_grade``,
            ``flesch_reading_ease``, ``average_grade_level``, and others.
            Includes ``low_confidence=True`` when text is shorter than 20 words."""
        if not text or not text.strip():
            return self._empty_metrics()

        normalized = self._normalize_for_readability(text)
        if not any(re.search(r"[^\W\d_]", w) and w != "MATH" for w in normalized.split()):
            return self._empty_metrics()  # "3 + 4 = 7" has no words to read
        word_count = len(normalized.split())
        low_confidence = word_count < 20

        if textstat_available and not self._cmu_missing:
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
                "dale_chall_readability": dale_chall_readability_score(_dale_chall_text(text)),
                "average_grade_level": (fk + fog + smog) / 3,
                "low_confidence": low_confidence,
                "estimated": False,
            }
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return self._estimate_metrics(text, True)

    def _estimate_metrics(self, text: str, low_confidence: bool) -> dict[str, Any]:
        words = [w for w in text.split() if re.search(r"[A-Za-z0-9]", w)]
        word_count = len(words)
        if not word_count:
            return self._empty_metrics()

        sentences = re.split(r"[.!?]+(?=\s|$)", text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        total_syllables = sum(self._count_syllables(w) for w in words)
        avg_syllables = total_syllables / word_count
        avg_sentence_length = word_count / sentence_count

        fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59
        estimated_grade = float(max(1, min(16, fk_grade)))

        flesch = float(max(0, min(100, 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables)))

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
        token = word.lower().strip()
        if not token:
            return 0
        word = re.sub(r"[^a-z']", "", token).strip("'")
        if not word:
            return 1

        if self._cmu_dict and word in self._cmu_dict:
            pronunciation = self._cmu_dict[word][0]
            return len([p for p in pronunciation if p[-1].isdigit()])

        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for i, char in enumerate(word):
            is_vowel = char in vowels and not (char == "y" and i == 0)
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        count += len(re.findall(r"(?<![cstg])ia", word))
        if count > 1:
            if word.endswith("e") and not re.search(r"[^aeiouy]le$", word):
                count -= 1
            elif re.search(r"[^aeiouystdxzcg](?:es|ed)$", word):
                count -= 1

        return max(1, count)

    def _empty_metrics(self) -> dict[str, Any]:
        return {
            **dict.fromkeys(_metric_keys, float("nan")),
            "low_confidence": True,
            "estimated": False,
        }
