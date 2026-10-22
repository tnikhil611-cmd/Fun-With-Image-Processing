"""Tests for analysis module."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.analysis import get_image_info, compute_histogram, compute_histogram_comparison


class TestGetImageInfo:
    def test_bgr_image_info(self, rgb_image):
        info = get_image_info(rgb_image)
        assert info["width"] == 100
        assert info["height"] == 100
        assert info["channels"] == 3
        assert info["color_mode"] == "RGB"
        assert info["dtype"] == "uint8"
        assert info["total_pixels"] == 10000

    def test_grayscale_image_info(self, grayscale_image):
        info = get_image_info(grayscale_image, file_size_bytes=2048)
        assert info["channels"] == 1
        assert info["color_mode"] == "Grayscale"
        assert info["file_size_kb"] == 2.0

    def test_rgba_image_info(self, rgba_image):
        info = get_image_info(rgba_image)
        assert info["channels"] == 4
        assert info["color_mode"] == "RGBA"

    def test_invalid_image_none(self):
        with pytest.raises(ValueError):
            get_image_info(None)

    def test_invalid_image_empty(self):
        with pytest.raises(ValueError):
            get_image_info(np.array([]))


class TestComputeHistogram:
    def test_grayscale_histogram(self, grayscale_image):
        hists = compute_histogram(grayscale_image)
        assert "gray" in hists and len(hists["gray"]) == 256

    def test_bgr_histogram(self, rgb_image):
        hists = compute_histogram(rgb_image)
        assert set(hists.keys()) == {"blue", "green", "red"}
        for ch in ("blue", "green", "red"):
            assert len(hists[ch]) == 256

    def test_bgra_histogram_ignores_alpha(self, rgba_image):
        hists = compute_histogram(rgba_image)
        assert set(hists.keys()) == {"blue", "green", "red"}
        assert "alpha" not in hists

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            compute_histogram(None)
        with pytest.raises(ValueError):
            compute_histogram(np.array([]))


class TestComputeHistogramComparison:
    def test_identical_bgr_images(self, rgb_image):
        result = compute_histogram_comparison(rgb_image, rgb_image)
        assert "correlation" in result
        assert pytest.approx(result["correlation"], 0.01) == 1.0

    def test_identical_grayscale_images(self, grayscale_image):
        result = compute_histogram_comparison(grayscale_image, grayscale_image)
        assert pytest.approx(result["correlation"], 0.01) == 1.0

    def test_different_images(self, rgb_image, black_image):
        result = compute_histogram_comparison(rgb_image, black_image)
        assert result["correlation"] < 1.0

    def test_cross_color_mode_comparison(self, rgb_image, grayscale_image):
        result = compute_histogram_comparison(rgb_image, grayscale_image)
        assert "correlation" in result
        assert isinstance(result["correlation"], float)

    def test_invalid_inputs(self, rgb_image):
        with pytest.raises(ValueError):
            compute_histogram_comparison(None, rgb_image)
        with pytest.raises(ValueError):
            compute_histogram_comparison(rgb_image, None)