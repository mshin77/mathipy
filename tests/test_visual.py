"""Tests for mathipy.visual module."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mathipy.visual import VisualFeatureExtractor


DATA_DIR = Path(os.environ.get("MATHIPY_TEST_DATA", Path(__file__).resolve().parent / "data"))
IMAGES_DIR = DATA_DIR / "images"


@pytest.fixture
def sample_image_path():
    images = sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        pytest.skip("No PNG images found in data/images")
    return images[0]


@pytest.fixture
def white_array():
    return np.ones((100, 150, 3), dtype=np.uint8) * 255


@pytest.fixture
def gradient_array():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    for i in range(200):
        img[:, i, :] = int(255 * i / 199)
    return img


@pytest.fixture
def shapes_array():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    try:
        import cv2
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
        cv2.circle(img, (225, 150), 50, (0, 0, 0), 2)
        pts = np.array([[150, 250], [200, 200], [250, 250]], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)
    except ImportError:
        pytest.skip("OpenCV required for shapes_array fixture")
    return img


class TestInit:
    def test_constructor(self):
        extractor = VisualFeatureExtractor()
        assert isinstance(extractor, VisualFeatureExtractor)


class TestLoadImage:
    def test_load_numpy_array(self, white_array):
        extractor = VisualFeatureExtractor()
        loaded = extractor._load_image(white_array)
        assert loaded is white_array

    def test_load_from_path(self, sample_image_path):
        extractor = VisualFeatureExtractor()
        loaded = extractor._load_image(sample_image_path)
        assert loaded is not None
        assert isinstance(loaded, np.ndarray)
        assert len(loaded.shape) >= 2

    def test_load_nonexistent_returns_none(self):
        extractor = VisualFeatureExtractor()
        result = extractor._load_image("nonexistent_image.png")
        assert result is None

    def test_nonexistent_returns_empty_features(self):
        extractor = VisualFeatureExtractor()
        result = extractor.extract("nonexistent_image.png")
        assert result["complexity_score"]["level"] == "unknown"
        assert result["dimensions"] == {}


class TestDimensions:
    def test_dimensions_from_array(self, white_array):
        extractor = VisualFeatureExtractor()
        dims = extractor._extract_dimensions(white_array)
        assert dims["width"] == 150
        assert dims["height"] == 100
        assert dims["channels"] == 3
        assert dims["total_pixels"] == 15000
        assert abs(dims["aspect_ratio"] - 1.5) < 0.01

    def test_dimensions_from_file(self, sample_image_path):
        result = VisualFeatureExtractor().extract(sample_image_path)
        assert result["dimensions"]["width"] > 0
        assert result["dimensions"]["height"] > 0
        assert result["dimensions"]["channels"] >= 1

    def test_grayscale_channels(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        extractor = VisualFeatureExtractor()
        dims = extractor._extract_dimensions(gray)
        assert dims["channels"] == 1


class TestPixelStatistics:
    def test_white_image_stats(self, white_array):
        extractor = VisualFeatureExtractor()
        stats = extractor._extract_pixel_stats(white_array)
        assert stats["mean"] == 255.0
        assert stats["std"] == 0.0
        assert stats["contrast"] == 0.0

    def test_gradient_image_stats(self, gradient_array):
        extractor = VisualFeatureExtractor()
        stats = extractor._extract_pixel_stats(gradient_array)
        assert 100 < stats["mean"] < 140
        assert stats["std"] > 0
        assert stats["contrast"] > 200

    def test_stats_keys(self, white_array):
        extractor = VisualFeatureExtractor()
        stats = extractor._extract_pixel_stats(white_array)
        expected = {"mean", "std", "min", "max", "median", "contrast"}
        assert set(stats.keys()) == expected


class TestEdgeMetrics:
    def test_requires_opencv(self, white_array):
        extractor = VisualFeatureExtractor()
        try:
            import cv2
            gray = cv2.cvtColor(white_array, cv2.COLOR_BGR2GRAY)
            metrics = extractor._extract_edge_metrics(gray)
            assert "edge_ratio" in metrics
            assert metrics["edge_ratio"] >= 0
        except ImportError:
            metrics = extractor._extract_edge_metrics(white_array[:, :, 0])
            assert metrics == {}

    def test_uniform_image_low_edges(self):
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        extractor = VisualFeatureExtractor()
        uniform = np.ones((100, 100), dtype=np.uint8) * 128
        metrics = extractor._extract_edge_metrics(uniform)
        assert metrics["edge_ratio"] < 0.01

    def test_real_image_has_edges(self, sample_image_path):
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        result = VisualFeatureExtractor().extract(sample_image_path)
        assert result["edge_metrics"]["edge_ratio"] > 0


class TestStructuralElements:
    def test_shapes_detected(self, shapes_array):
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        extractor = VisualFeatureExtractor()
        result = extractor.extract(shapes_array)
        elements = result["structural_elements"]
        assert elements["total_shapes"] >= 0
        assert "shapes" in elements
        assert set(elements["shapes"].keys()) == {"triangles", "rectangles", "circles", "polygons"}

    def test_real_image_structure(self, sample_image_path):
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        result = VisualFeatureExtractor().extract(sample_image_path)
        assert "structural_elements" in result
        assert "contour_count" in result["structural_elements"]


class TestFrequencyDomain:
    def test_frequency_ratios_sum_to_one(self, sample_image_path):
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        result = VisualFeatureExtractor().extract(sample_image_path)
        freq = result["frequency_domain"]
        total = freq["low_freq_ratio"] + freq["mid_freq_ratio"] + freq["high_freq_ratio"]
        assert abs(total - 1.0) < 0.01

    def test_uniform_mostly_low_freq(self):
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        uniform = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = VisualFeatureExtractor().extract(uniform)
        assert result["frequency_domain"]["low_freq_ratio"] > 0.5


class TestComplexity:
    def test_complexity_keys(self, sample_image_path):
        result = VisualFeatureExtractor().extract(sample_image_path)
        score = result["complexity_score"]
        assert "overall" in score
        assert "level" in score
        assert score["level"] in ("low", "medium", "high")

    def test_uniform_low_complexity(self):
        uniform = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = VisualFeatureExtractor().extract(uniform)
        assert result["complexity_score"]["level"] in ("low", "medium")

    def test_complexity_between_0_and_1(self, sample_image_path):
        result = VisualFeatureExtractor().extract(sample_image_path)
        assert 0 <= result["complexity_score"]["overall"] <= 1.0

    def test_no_opencv_fallback(self, white_array):
        extractor = VisualFeatureExtractor()
        features = {
            "dimensions": extractor._extract_dimensions(white_array),
            "pixel_statistics": extractor._extract_pixel_stats(white_array),
        }
        score = extractor._calculate_complexity(features)
        assert score["overall"] == 0.5
        assert score["level"] == "medium"


class TestClassBasedAPI:
    def test_extractor_extract_method(self, sample_image_path):
        result = VisualFeatureExtractor().extract(sample_image_path)
        assert isinstance(result, dict)
        assert "dimensions" in result
        assert "pixel_statistics" in result
        assert "complexity_score" in result

    def test_accepts_numpy_array(self, white_array):
        result = VisualFeatureExtractor().extract(white_array)
        assert result["dimensions"]["width"] == 150
