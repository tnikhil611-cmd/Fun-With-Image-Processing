"""Tests for image filters."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.filters import (
    gaussian_blur, median_blur, bilateral_filter,
    average_blur, sharpen, denoise
)


class TestGaussianBlur:
    def test_bgr(self, rgb_image):
        res = gaussian_blur(rgb_image, kernel_size=5)
        assert res.shape == rgb_image.shape and res.dtype == np.uint8

    def test_grayscale(self, grayscale_image):
        res = gaussian_blur(grayscale_image, kernel_size=5)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = gaussian_blur(rgba_image, kernel_size=5)
        assert res.shape == rgba_image.shape
        np.testing.assert_array_equal(res[:, :, 3], rgba_image[:, :, 3])

    def test_even_kernel_corrected(self, rgb_image):
        res = gaussian_blur(rgb_image, kernel_size=4)
        assert res.shape == rgb_image.shape

    def test_result_is_blurred(self, rgb_image):
        res = gaussian_blur(rgb_image, kernel_size=15)
        assert np.std(res.astype(float)) <= np.std(rgb_image.astype(float))


class TestMedianBlur:
    def test_bgr(self, rgb_image):
        res = median_blur(rgb_image, kernel_size=5)
        assert res.shape == rgb_image.shape

    def test_grayscale(self, grayscale_image):
        res = median_blur(grayscale_image, kernel_size=5)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = median_blur(rgba_image, kernel_size=5)
        assert res.shape == rgba_image.shape


class TestBilateralFilter:
    def test_bgr(self, rgb_image):
        res = bilateral_filter(rgb_image, d=9)
        assert res.shape == rgb_image.shape

    def test_grayscale(self, grayscale_image):
        res = bilateral_filter(grayscale_image, d=9)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = bilateral_filter(rgba_image, d=9)
        assert res.shape == rgba_image.shape


class TestAverageBlur:
    def test_bgr(self, rgb_image):
        res = average_blur(rgb_image, kernel_size=5)
        assert res.shape == rgb_image.shape

    def test_grayscale(self, grayscale_image):
        res = average_blur(grayscale_image, kernel_size=5)
        assert res.shape == grayscale_image.shape


class TestSharpen:
    def test_bgr(self, rgb_image):
        res = sharpen(rgb_image, strength=1.0)
        assert res.shape == rgb_image.shape

    def test_grayscale(self, grayscale_image):
        res = sharpen(grayscale_image, strength=1.0)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = sharpen(rgba_image, strength=1.0)
        assert res.shape == rgba_image.shape

    def test_zero_strength(self, rgb_image):
        res = sharpen(rgb_image, strength=0.0)
        np.testing.assert_array_equal(res, rgb_image)


class TestDenoise:
    def test_bgr(self, noisy_image):
        res = denoise(noisy_image, strength=10.0)
        assert res.shape == noisy_image.shape

    def test_grayscale(self, grayscale_image):
        res = denoise(grayscale_image, strength=10.0)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = denoise(rgba_image, strength=10.0)
        assert res.shape == rgba_image.shape