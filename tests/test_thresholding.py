"""Tests for thresholding operations."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.thresholding import (
    binary_threshold, binary_inv_threshold,
    adaptive_threshold, otsu_threshold
)


class TestBinaryThreshold:
    def test_grayscale(self, grayscale_image):
        res = binary_threshold(grayscale_image, thresh_value=127)
        assert res.shape == grayscale_image.shape
        assert set(np.unique(res)).issubset({0, 255})

    def test_bgr_converts_to_gray(self, rgb_image):
        res = binary_threshold(rgb_image, 127)
        assert res.ndim == 2 and res.shape == rgb_image.shape[:2]

    def test_bgra_converts_to_gray(self, rgba_image):
        res = binary_threshold(rgba_image, 127)
        assert res.ndim == 2 and res.shape == rgba_image.shape[:2]

    def test_threshold_values(self):
        img = np.array([[50, 150], [200, 100]], dtype=np.uint8)
        res = binary_threshold(img, thresh_value=127)
        expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(res, expected)


class TestBinaryInvThreshold:
    def test_grayscale(self, grayscale_image):
        res = binary_inv_threshold(grayscale_image, 127)
        assert set(np.unique(res)).issubset({0, 255})

    def test_inverse_of_binary(self, grayscale_image):
        b = binary_threshold(grayscale_image, 127)
        bi = binary_inv_threshold(grayscale_image, 127)
        np.testing.assert_array_equal(b + bi, np.full_like(b, 255))


class TestAdaptiveThreshold:
    def test_grayscale(self, grayscale_image):
        res = adaptive_threshold(grayscale_image, block_size=11, c=2)
        assert res.shape == grayscale_image.shape

    def test_bgr(self, rgb_image):
        res = adaptive_threshold(rgb_image, block_size=11, c=2)
        assert res.ndim == 2

    def test_gaussian_method(self, grayscale_image):
        res = adaptive_threshold(grayscale_image, method='gaussian')
        assert res.shape == grayscale_image.shape

    def test_mean_method(self, grayscale_image):
        res = adaptive_threshold(grayscale_image, method='mean')
        assert res.shape == grayscale_image.shape

    def test_even_block_size_corrected(self, grayscale_image):
        res = adaptive_threshold(grayscale_image, block_size=10)
        assert res.shape == grayscale_image.shape


class TestOtsuThreshold:
    def test_grayscale(self, grayscale_image):
        res = otsu_threshold(grayscale_image)
        assert res.shape == grayscale_image.shape
        assert set(np.unique(res)).issubset({0, 255})

    def test_bgr(self, rgb_image):
        res = otsu_threshold(rgb_image)
        assert res.ndim == 2

    def test_bimodal_image(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 50
        img[50:, :] = 200
        res = otsu_threshold(img)
        assert res[0, 0] == 0 and res[99, 0] == 255