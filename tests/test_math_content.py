"""Tests for mathipy.math_content module."""

import pytest

from mathipy.math_content import MathContentAnalyzer


ALGEBRA_TEXT = "Solve for x: 2x + 5 = 15. What is the value of x?"
GEOMETRY_TEXT = "Find the area of a triangle with a base of 10 cm and a height of 6 cm."
STATS_TEXT = "Calculate the mean and median of the data set: 3, 7, 8, 12, 15."
ARITHMETIC_TEXT = "What is 24 × 6? Find the product of 24 and 6."
FRACTIONS_TEXT = "Simplify the fraction 3/4 + 1/2 to find the sum."
CALCULUS_TEXT = "Find d/dx of f(x) = x^3 + 2x. What is the derivative?"


class TestInit:
    def test_constructor(self):
        analyzer = MathContentAnalyzer()
        assert len(analyzer.patterns) > 0
        assert len(analyzer.domains) == 6
        assert len(analyzer.all_terms) > 0


class TestEmptyInput:
    def test_empty_string(self):
        analyzer = MathContentAnalyzer()
        result = analyzer.analyze("")
        assert result["pattern_matches"] == {}
        assert result["total_math_symbols"] == 0
        assert result["numbers"]["count"] == 0
        assert result["domain_classification"]["primary"] == "unknown"
        assert result["math_density"] == 0

    def test_whitespace_only(self):
        analyzer = MathContentAnalyzer()
        result = analyzer.analyze("   ")
        assert result["domain_classification"]["primary"] == "unknown"


class TestPatternDetection:
    def test_addition(self):
        result = MathContentAnalyzer().analyze("3 + 5")
        assert "addition" in result["pattern_matches"]

    def test_subtraction(self):
        result = MathContentAnalyzer().analyze("10 - 3")
        assert "subtraction" in result["pattern_matches"]

    def test_multiplication(self):
        result = MathContentAnalyzer().analyze("4 × 5")
        assert "multiplication" in result["pattern_matches"]

    def test_division(self):
        result = MathContentAnalyzer().analyze("12 ÷ 3")
        assert "division" in result["pattern_matches"]

    def test_equation(self):
        result = MathContentAnalyzer().analyze("2x + 5 = 15")
        assert "equation" in result["pattern_matches"]

    def test_fraction(self):
        result = MathContentAnalyzer().analyze("The value 3/4 is a fraction.")
        assert "fraction" in result["pattern_matches"]

    def test_decimal(self):
        result = MathContentAnalyzer().analyze("The answer is 3.14.")
        assert "decimal" in result["pattern_matches"]

    def test_percentage(self):
        result = MathContentAnalyzer().analyze("About 25% of students passed.")
        assert "percentage" in result["pattern_matches"]

    def test_ratio(self):
        result = MathContentAnalyzer().analyze("The ratio is 3:4.")
        assert "ratio" in result["pattern_matches"]

    def test_derivative(self):
        result = MathContentAnalyzer().analyze("Find d/dx of x^2.")
        assert "derivative" in result["pattern_matches"]

    def test_variable(self):
        result = MathContentAnalyzer().analyze("Solve for x in the equation.")
        assert "variable" in result["pattern_matches"]


class TestSymbolCounting:
    def test_counts_operators(self):
        result = MathContentAnalyzer().analyze("3 + 5 - 2 = 6")
        assert result["total_math_symbols"] > 0
        assert "addition" in result["symbol_counts"]
        assert "subtraction" in result["symbol_counts"]
        assert "equals" in result["symbol_counts"]

    def test_unique_types(self):
        result = MathContentAnalyzer().analyze("3 + 5 = 8")
        assert result["unique_symbol_types"] >= 2


class TestNumberExtraction:
    def test_integers(self):
        result = MathContentAnalyzer().analyze("Add 10 and 20.")
        assert result["numbers"]["count"] >= 2
        assert 10.0 in result["numbers"]["values"]
        assert 20.0 in result["numbers"]["values"]

    def test_negative_numbers(self):
        result = MathContentAnalyzer().analyze("The temperature was -5 degrees.")
        assert result["numbers"]["has_negative"] is True

    def test_decimal_numbers(self):
        result = MathContentAnalyzer().analyze("Pi is approximately 3.14.")
        assert result["numbers"]["has_decimal"] is True

    def test_range_computation(self):
        result = MathContentAnalyzer().analyze("Values: 2 and 10")
        assert result["numbers"]["range"] == 8.0


class TestVocabulary:
    def test_geometry_terms(self):
        result = MathContentAnalyzer().analyze(GEOMETRY_TEXT)
        terms = result["vocabulary"]["math_terms"]
        assert any(t in terms for t in ["area", "triangle"])

    def test_statistics_terms(self):
        result = MathContentAnalyzer().analyze(STATS_TEXT)
        terms = result["vocabulary"]["math_terms"]
        assert "mean" in terms
        assert "median" in terms

    def test_term_count_positive(self):
        result = MathContentAnalyzer().analyze(ALGEBRA_TEXT)
        assert result["vocabulary"]["term_count"] > 0

    def test_no_terms_in_plain_text(self):
        result = MathContentAnalyzer().analyze("The dog ran across the park.")
        assert result["vocabulary"]["unique_terms"] == 0


class TestDomainClassification:
    def test_algebra_domain(self):
        result = MathContentAnalyzer().analyze(ALGEBRA_TEXT)
        assert result["domain_classification"]["primary"] == "algebra"

    def test_geometry_domain(self):
        result = MathContentAnalyzer().analyze(GEOMETRY_TEXT)
        assert result["domain_classification"]["primary"] == "geometry"

    def test_statistics_domain(self):
        result = MathContentAnalyzer().analyze(STATS_TEXT)
        assert result["domain_classification"]["primary"] == "statistics"

    def test_arithmetic_domain(self):
        result = MathContentAnalyzer().analyze(ARITHMETIC_TEXT)
        assert result["domain_classification"]["primary"] == "arithmetic"

    def test_calculus_boost(self):
        result = MathContentAnalyzer().analyze(CALCULUS_TEXT)
        scores = result["domain_classification"]["scores"]
        assert scores.get("calculus", 0) > 0

    def test_confidence_between_0_and_1(self):
        result = MathContentAnalyzer().analyze(ALGEBRA_TEXT)
        assert 0 <= result["domain_classification"]["confidence"] <= 1

    def test_secondary_domains_present(self):
        result = MathContentAnalyzer().analyze(ALGEBRA_TEXT)
        assert isinstance(result["domain_classification"]["secondary"], list)


class TestMathDensity:
    def test_density_positive_for_math_text(self):
        result = MathContentAnalyzer().analyze("3 + 5 = 8 and 10 - 2 = 8")
        assert result["math_density"] > 0

    def test_density_zero_for_plain_text(self):
        result = MathContentAnalyzer().analyze("The dog ran across the park quickly.")
        assert result["math_density"] == 0


class TestClassBasedAPI:
    def test_analyzer_analyze_method(self):
        result = MathContentAnalyzer().analyze(GEOMETRY_TEXT)
        assert isinstance(result, dict)
        assert "domain_classification" in result
        assert "pattern_matches" in result
