"""Tests for mathipy.data module."""

from pathlib import Path

import pytest

from mathipy.data import get_sample_csv, get_sample_image, list_sample_images


class TestGetSampleCsv:
    def test_returns_path(self):
        assert isinstance(get_sample_csv(), Path)

    def test_file_exists(self):
        assert get_sample_csv().exists()

    def test_is_csv(self):
        assert get_sample_csv().suffix == ".csv"


class TestGetSampleImage:
    def test_returns_path(self):
        assert isinstance(get_sample_image("2024-4M10 #2"), Path)

    def test_valid_id_resolves(self):
        path = get_sample_image("2024-4M10 #2")
        assert path.name == "2024-4M10 #2.png"

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError, match="Invalid item_id"):
            get_sample_image("../../etc/passwd")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid item_id"):
            get_sample_image("")


class TestListSampleImages:
    def test_returns_list_of_strings(self):
        result = list_sample_images()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)

    def test_all_png(self):
        assert all(name.endswith(".png") for name in list_sample_images())

    def test_sorted(self):
        result = list_sample_images()
        assert result == sorted(result)

    def test_known_image_present(self):
        assert "2024-4M10 #2.png" in list_sample_images()
