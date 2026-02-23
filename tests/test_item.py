"""Tests for mathipy.item module."""

import os
from pathlib import Path

import pytest

from mathipy.item import ItemFeatureExtractor


@pytest.fixture
def extractor():
    return ItemFeatureExtractor()


sample_text = "What is 3 + 5? Choose the correct answer. A) 7 B) 8 C) 9 D) 10"

data_dir = Path(os.environ.get("MATHIPY_TEST_DATA", Path(__file__).resolve().parent / "data"))
images_dir = data_dir / "images"


class TestTextOnly:
    def test_returns_readability(self, extractor):
        r = extractor.extract(sample_text)
        assert "readability_flesch_kincaid_grade" in r
        assert r["readability_flesch_kincaid_grade"] is not None

    def test_returns_math(self, extractor):
        r = extractor.extract(sample_text)
        assert "math_density" in r

    def test_returns_cognitive(self, extractor):
        r = extractor.extract(sample_text)
        assert "cognitive_total" in r
        assert r["cognitive_total"] is not None

    def test_visual_none_without_images(self, extractor):
        r = extractor.extract(sample_text)
        assert r["visual_complexity_overall"] is None
        assert r["visual_complexity_level"] is None

    def test_math_pattern_keys(self, extractor):
        r = extractor.extract(sample_text)
        for pat in ["equation", "fraction", "variable", "percentage", "ratio", "exponent"]:
            assert f"math_pat_{pat}" in r


class TestWithImages:
    def test_visual_populated_with_real_image(self, extractor):
        images = sorted(images_dir.glob("*.png"))
        if not images:
            pytest.skip("No PNG images in data/images")
        r = extractor.extract(sample_text, image_paths=[images[0]])
        assert r["visual_complexity_overall"] is not None
        assert r["visual_width"] is not None

    def test_nonexistent_images_give_none(self, extractor):
        r = extractor.extract(sample_text, image_paths=["nonexistent.png"])
        assert r["visual_complexity_overall"] is None

    def test_multiple_images_aggregate(self, extractor):
        images = sorted(images_dir.glob("*.png"))
        if len(images) < 2:
            pytest.skip("Need at least 2 images")
        r = extractor.extract(sample_text, image_paths=images[:2])
        assert r["visual_complexity_overall"] is not None


class TestEmptyText:
    def test_empty_string_returns_defaults(self, extractor):
        r = extractor.extract("")
        assert "readability_flesch_kincaid_grade" in r
        assert "cognitive_total" in r
