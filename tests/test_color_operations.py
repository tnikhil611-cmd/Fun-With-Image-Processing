"""Tests for color operations."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.color_operations import to_grayscale, grayscale_to_rgb, to_hsv, to_lab


class TestToGrayscale:
    def test_bgr_to_gray(self, rgb_image):
        result = to_grayscale(rgb_image)
        assert len(result.shape) == 2
        assert result.shape[:2] == rgb_image.shape[:2]
        assert result.dtype == np.uint8

    def test_bgra_to_gray(self, rgba_image):
        result = to_grayscale(rgba_image)
        assert len(result.shape) == 2
        assert result.shape[:2] == rgba_image.shape[:2]

    def test_gray_to_gray(self, grayscale_image):
        result = to_grayscale(grayscale_image)
        assert len(result.shape) == 2
        np.testing.assert_array_equal(result, grayscale_image)

    def test_output_dtype(self, rgb_image):
        result = to_grayscale(rgb_image)
        assert result.dtype == np.uint8


class TestGrayscaleToRgb:
    def test_gray_to_bgr(self, grayscale_image):
        result = grayscale_to_rgb(grayscale_image)
        assert len(result.shape) == 3
        assert result.shape[2] == 3
        assert result.shape[:2] == grayscale_image.shape[:2]

    def test_bgr_passthrough(self, rgb_image):
        result = grayscale_to_rgb(rgb_image)
        assert result.shape == rgb_image.shape
        assert result.shape[2] == 3

    def test_output_dtype(self, grayscale_image):
        result = grayscale_to_rgb(grayscale_image)
        assert result.dtype == np.uint8


class TestToHsv:
    def test_bgr_to_hsv(self, rgb_image):
        result = to_hsv(rgb_image)
        assert result.shape == rgb_image.shape
        assert result.dtype == np.uint8

    def test_gray_to_hsv(self, grayscale_image):
        result = to_hsv(grayscale_image)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_bgra_to_hsv(self, rgba_image):
        result = to_hsv(rgba_image)
        assert len(result.shape) == 3
        assert result.shape[2] == 3


class TestToLab:
    def test_bgr_to_lab(self, rgb_image):
        result = to_lab(rgb_image)
        assert result.shape == rgb_image.shape
        assert result.dtype == np.uint8

    def test_gray_to_lab(self, grayscale_image):
        result = to_lab(grayscale_image)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_bgra_to_lab(self, rgba_image):
        result = to_lab(rgba_image)
        assert len(result.shape) == 3
        assert result.shape[2] == 3