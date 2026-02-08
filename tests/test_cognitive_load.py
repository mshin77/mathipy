"""Tests for mathipy.cognitive_load module."""

import pytest

from mathipy.cognitive_load import CognitiveLoadEstimator


EXPECTED_KEYS = {
    "intrinsic_cognitive_load", "extraneous_cognitive_load",
    "germane_cognitive_load", "total_cognitive_load",
    "numeric_elements", "variable_count", "operation_count",
}

SIMPLE_TEXT = "What is 3 + 4?"
COMPLEX_TEXT = (
    "Given the quadratic equation 2x^2 + 5x - 3 = 0, use the quadratic formula "
    "x = (-b +/- sqrt(b^2 - 4ac)) / 2a to find both solutions for x. "
    "Show all work and simplify your final answers."
)
FRACTION_TEXT = "What is 3/4 + 1/2? Simplify the fraction to find the sum."


class TestInit:
    def test_constructor(self):
        estimator = CognitiveLoadEstimator()
        assert isinstance(estimator, CognitiveLoadEstimator)


class TestEmptyInput:
    def test_empty_string(self):
        estimator = CognitiveLoadEstimator()
        result = estimator.estimate("")
        assert set(result.keys()) == EXPECTED_KEYS
        assert result["total_cognitive_load"] == 0.0
        assert result["numeric_elements"] == 0
        assert result["variable_count"] == 0

    def test_whitespace_only(self):
        result = CognitiveLoadEstimator().estimate("   ")
        assert result["total_cognitive_load"] == 0.0


class TestIntrinsicLoad:
    def test_more_numbers_higher_intrinsic(self):
        few_nums = CognitiveLoadEstimator().estimate("Add 3 and 5.")
        many_nums = CognitiveLoadEstimator().estimate("Compute 3 + 5 + 7 + 9 + 11 + 13 + 15.")
        assert many_nums["intrinsic_cognitive_load"] >= few_nums["intrinsic_cognitive_load"]

    def test_variables_increase_intrinsic(self):
        no_vars = CognitiveLoadEstimator().estimate("Find the answer.")
        with_vars = CognitiveLoadEstimator().estimate("Find x and y.")
        assert with_vars["intrinsic_cognitive_load"] > no_vars["intrinsic_cognitive_load"]

    def test_intrinsic_capped_at_one(self):
        result = CognitiveLoadEstimator().estimate("x y z 1 2 3 4 5 6 7 8 9")
        assert result["intrinsic_cognitive_load"] <= 1.0


class TestExtraneousLoad:
    def test_with_readability_grade(self):
        result = CognitiveLoadEstimator().estimate("test text", readability_grade=6.0)
        assert result["extraneous_cognitive_load"] == round(6.0 / 12, 3)

    def test_high_readability_higher_extraneous(self):
        low = CognitiveLoadEstimator().estimate("test text here", readability_grade=3.0)
        high = CognitiveLoadEstimator().estimate("test text here", readability_grade=10.0)
        assert high["extraneous_cognitive_load"] > low["extraneous_cognitive_load"]

    def test_extraneous_capped_at_one(self):
        result = CognitiveLoadEstimator().estimate("test", readability_grade=20.0)
        assert result["extraneous_cognitive_load"] <= 1.0

    def test_estimated_when_no_grade(self):
        result = CognitiveLoadEstimator().estimate(COMPLEX_TEXT)
        assert result["extraneous_cognitive_load"] > 0


class TestGermaneLoad:
    def test_with_math_terms(self):
        terms = ["add", "subtract", "multiply", "sum", "product"]
        result = CognitiveLoadEstimator().estimate("test", math_terms=terms)
        assert result["germane_cognitive_load"] == round(5 / 10, 3)

    def test_more_terms_higher_germane(self):
        few = CognitiveLoadEstimator().estimate("test", math_terms=["add"])
        many = CognitiveLoadEstimator().estimate("test", math_terms=["add", "subtract", "multiply", "divide", "sum"])
        assert many["germane_cognitive_load"] > few["germane_cognitive_load"]

    def test_germane_capped_at_one(self):
        terms = [f"term{i}" for i in range(20)]
        result = CognitiveLoadEstimator().estimate("test", math_terms=terms)
        assert result["germane_cognitive_load"] <= 1.0

    def test_estimated_with_keywords(self):
        result = CognitiveLoadEstimator().estimate("Find the area of the triangle and the perimeter of the circle.")
        assert result["germane_cognitive_load"] > 0

    def test_estimated_no_keywords_default(self):
        result = CognitiveLoadEstimator().estimate("The dog ran quickly across the park.")
        assert result["germane_cognitive_load"] == 0.3


class TestTotalLoad:
    def test_weighted_formula(self):
        result = CognitiveLoadEstimator().estimate(SIMPLE_TEXT, readability_grade=4.0, math_terms=["add"])
        intrinsic = result["intrinsic_cognitive_load"]
        extraneous = result["extraneous_cognitive_load"]
        germane = result["germane_cognitive_load"]
        expected = round(intrinsic * 0.4 + extraneous * 0.3 + germane * 0.3, 3)
        assert result["total_cognitive_load"] == expected

    def test_total_between_0_and_1(self):
        result = CognitiveLoadEstimator().estimate(COMPLEX_TEXT)
        assert 0 <= result["total_cognitive_load"] <= 1.0

    def test_complex_higher_than_simple(self):
        simple = CognitiveLoadEstimator().estimate(SIMPLE_TEXT)
        complex_result = CognitiveLoadEstimator().estimate(COMPLEX_TEXT)
        assert complex_result["total_cognitive_load"] >= simple["total_cognitive_load"]


class TestElementCounts:
    def test_numeric_elements(self):
        result = CognitiveLoadEstimator().estimate("Add 3, 5, and 10.")
        assert result["numeric_elements"] == 3

    def test_variable_count(self):
        result = CognitiveLoadEstimator().estimate("Solve x + y = z.")
        assert result["variable_count"] == 3

    def test_operation_count(self):
        result = CognitiveLoadEstimator().estimate("3 + 5 - 2 = 6")
        assert result["operation_count"] >= 3


class TestClassBasedAPI:
    def test_estimator_estimate_method(self):
        result = CognitiveLoadEstimator().estimate(FRACTION_TEXT)
        assert isinstance(result, dict)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_all_optional_params(self):
        result = CognitiveLoadEstimator().estimate(
            SIMPLE_TEXT,
            readability_grade=3.5,
            math_terms=["add", "sum"],
        )
        assert result["total_cognitive_load"] > 0
