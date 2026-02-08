"""Tests for mathipy.readability module."""

import re
from unittest.mock import patch

import pytest

from mathipy.readability import ReadabilityAnalyzer


EXPECTED_KEYS = {
    "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
    "smog_index", "automated_readability_index", "coleman_liau_index",
    "linsear_write_formula", "dale_chall_readability", "average_grade_level",
    "low_confidence",
}

LONG_TEXT = (
    "A farmer has a rectangular field that is 120 meters long and 80 meters wide. "
    "He wants to divide the field into smaller plots, each measuring 20 meters by 20 meters. "
    "How many smaller plots can the farmer create from the rectangular field? "
    "Show your work and explain your reasoning step by step."
)

SHORT_TEXT = "What is 3 + 4?"


class TestAnalyzerInit:
    def test_constructor(self):
        analyzer = ReadabilityAnalyzer()
        assert isinstance(analyzer, ReadabilityAnalyzer)

    def test_cmu_dict_loaded_or_none(self):
        analyzer = ReadabilityAnalyzer()
        assert analyzer._cmu_dict is None or isinstance(analyzer._cmu_dict, dict)


class TestEmptyInput:
    def test_empty_string(self):
        analyzer = ReadabilityAnalyzer()
        result = analyzer.analyze("")
        assert set(result.keys()) >= EXPECTED_KEYS
        assert result["low_confidence"] is True
        assert result["flesch_kincaid_grade"] == 8.0

    def test_none_like_whitespace(self):
        analyzer = ReadabilityAnalyzer()
        result = analyzer.analyze("   \n\t  ")
        assert result["low_confidence"] is True


class TestNormalization:
    def test_latex_inline_replaced(self):
        analyzer = ReadabilityAnalyzer()
        normalized = analyzer._normalize_for_readability("Solve $x^2 + 1 = 0$ for x.")
        assert "$" not in normalized
        assert "MATH" in normalized

    def test_latex_display_replaced(self):
        analyzer = ReadabilityAnalyzer()
        normalized = analyzer._normalize_for_readability("$$\\frac{a}{b}$$")
        assert "\\frac" not in normalized

    def test_math_operators_removed(self):
        analyzer = ReadabilityAnalyzer()
        normalized = analyzer._normalize_for_readability("3 + 4 = 7")
        assert "+" not in normalized
        assert "=" not in normalized

    def test_choice_markers_removed(self):
        analyzer = ReadabilityAnalyzer()
        normalized = analyzer._normalize_for_readability("A) ten B) twenty")
        assert not re.search(r"\bA\)", normalized)

    def test_whitespace_collapsed(self):
        analyzer = ReadabilityAnalyzer()
        normalized = analyzer._normalize_for_readability("word   $x$    word")
        assert "  " not in normalized


class TestSyllableCounting:
    def test_known_word(self):
        analyzer = ReadabilityAnalyzer()
        count = analyzer._count_syllables("rectangle")
        assert count >= 3

    def test_single_syllable(self):
        analyzer = ReadabilityAnalyzer()
        assert analyzer._count_syllables("add") >= 1

    def test_empty_word(self):
        analyzer = ReadabilityAnalyzer()
        assert analyzer._count_syllables("") == 0

    def test_minimum_one(self):
        analyzer = ReadabilityAnalyzer()
        assert analyzer._count_syllables("xyz") >= 1


class TestMetrics:
    def test_long_text_all_keys_present(self):
        result = ReadabilityAnalyzer().analyze(LONG_TEXT)
        assert set(result.keys()) >= EXPECTED_KEYS

    def test_long_text_not_low_confidence(self):
        result = ReadabilityAnalyzer().analyze(LONG_TEXT)
        assert result["low_confidence"] is False

    def test_short_text_low_confidence(self):
        result = ReadabilityAnalyzer().analyze(SHORT_TEXT)
        assert result["low_confidence"] is True

    def test_flesch_reading_ease_range(self):
        result = ReadabilityAnalyzer().analyze(LONG_TEXT)
        assert -100 <= result["flesch_reading_ease"] <= 200

    def test_grade_level_positive(self):
        result = ReadabilityAnalyzer().analyze(LONG_TEXT)
        assert result["flesch_kincaid_grade"] > 0

    def test_average_grade_is_mean(self):
        result = ReadabilityAnalyzer().analyze(LONG_TEXT)
        fk = result["flesch_kincaid_grade"]
        fog = result["gunning_fog"]
        smog = result["smog_index"]
        expected_avg = (fk + fog + smog) / 3
        assert abs(result["average_grade_level"] - expected_avg) < 0.01

    def test_math_text_not_inflated(self):
        plain = "The farmer divides the field into smaller equal plots."
        math_heavy = "Solve $\\frac{x^2 + 3x}{2} = \\sqrt{y}$ for $x$ given $y = 16$."
        plain_result = ReadabilityAnalyzer().analyze(plain)
        math_result = ReadabilityAnalyzer().analyze(math_heavy)
        assert math_result["flesch_kincaid_grade"] < plain_result["flesch_kincaid_grade"] + 5


class TestEstimationFallback:
    def test_estimation_mode(self):
        analyzer = ReadabilityAnalyzer()
        result = analyzer._estimate_metrics(LONG_TEXT, low_confidence=False)
        assert "estimated" in result
        assert result["estimated"] is True
        assert result["low_confidence"] is True
        assert result["flesch_kincaid_grade"] >= 1
        assert result["flesch_kincaid_grade"] <= 16


class TestClassBasedAPI:
    def test_analyzer_analyze_method(self):
        result = ReadabilityAnalyzer().analyze(LONG_TEXT)
        assert isinstance(result, dict)
        assert "flesch_kincaid_grade" in result
