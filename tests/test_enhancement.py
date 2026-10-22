"""Tests for image enhancement algorithms."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.enhancement import (
    adjust_brightness, adjust_contrast,
    histogram_equalization, apply_clahe, gamma_correction
)


class TestAdjustBrightness:
    def test_increase_brightness(self, rgb_image):
        res = adjust_brightness(rgb_image, value=50)
        assert np.mean(res) >= np.mean(rgb_image)

    def test_decrease_brightness(self, rgb_image):
        res = adjust_brightness(rgb_image, value=-50)
        assert np.mean(res) <= np.mean(rgb_image)

    def test_zero_brightness(self, rgb_image):
        res = adjust_brightness(rgb_image, value=0)
        np.testing.assert_array_equal(res, rgb_image)

    def test_grayscale(self, grayscale_image):
        res = adjust_brightness(grayscale_image, value=30)
        assert res.shape == grayscale_image.shape

    def test_no_overflow(self, white_image):
        res = adjust_brightness(white_image, value=50)
        assert np.max(res) <= 255

    def test_no_underflow(self, black_image):
        res = adjust_brightness(black_image, value=-50)
        assert np.min(res) >= 0

    def test_bgra(self, rgba_image):
        res = adjust_brightness(rgba_image, value=30)
        np.testing.assert_array_equal(res[:, :, 3], rgba_image[:, :, 3])


class TestAdjustContrast:
    def test_increase_contrast(self, rgb_image):
        res = adjust_contrast(rgb_image, factor=1.5)
        assert res.shape == rgb_image.shape

    def test_decrease_contrast(self, rgb_image):
        res = adjust_contrast(rgb_image, factor=0.5)
        assert np.std(res.astype(float)) <= np.std(rgb_image.astype(float))

    def test_identity_contrast(self, rgb_image):
        res = adjust_contrast(rgb_image, factor=1.0)
        np.testing.assert_array_equal(res, rgb_image)

    def test_grayscale(self, grayscale_image):
        res = adjust_contrast(grayscale_image, factor=1.2)
        assert res.shape == grayscale_image.shape

    def test_no_overflow(self, rgb_image):
        res = adjust_contrast(rgb_image, factor=3.0)
        assert np.max(res) <= 255 and np.min(res) >= 0


class TestHistogramEqualization:
    def test_grayscale(self, grayscale_image):
        res = histogram_equalization(grayscale_image)
        assert res.shape == grayscale_image.shape

    def test_bgr(self, rgb_image):
        res = histogram_equalization(rgb_image)
        assert res.shape == rgb_image.shape

    def test_bgra(self, rgba_image):
        res = histogram_equalization(rgba_image)
        assert res.shape == rgba_image.shape
        np.testing.assert_array_equal(res[:, :, 3], rgba_image[:, :, 3])

    def test_improves_contrast(self):
        low_con = np.full((100, 100), 120, dtype=np.uint8)
        low_con[40:60, 40:60] = 135
        res = histogram_equalization(low_con)
        assert np.std(res.astype(float)) > np.std(low_con.astype(float))


class TestApplyClahe:
    def test_grayscale(self, grayscale_image):
        res = apply_clahe(grayscale_image, clip_limit=2.0)
        assert res.shape == grayscale_image.shape

    def test_bgr(self, rgb_image):
        res = apply_clahe(rgb_image, clip_limit=2.0)
        assert res.shape == rgb_image.shape

    def test_bgra(self, rgba_image):
        res = apply_clahe(rgba_image, clip_limit=2.0)
        assert res.shape == rgba_image.shape
        np.testing.assert_array_equal(res[:, :, 3], rgba_image[:, :, 3])

    def test_different_clip_limits(self, rgb_image):
        res1 = apply_clahe(rgb_image, clip_limit=1.0)
        res2 = apply_clahe(rgb_image, clip_limit=5.0)
        assert not np.array_equal(res1, res2)


class TestGammaCorrection:
    def test_brighten(self, rgb_image):
        res = gamma_correction(rgb_image, gamma=0.5)
        assert np.mean(res) >= np.mean(rgb_image)

    def test_darken(self, rgb_image):
        res = gamma_correction(rgb_image, gamma=2.0)
        assert np.mean(res) <= np.mean(rgb_image)

    def test_identity(self, rgb_image):
        res = gamma_correction(rgb_image, gamma=1.0)
        np.testing.assert_array_equal(res, rgb_image)

    def test_grayscale(self, grayscale_image):
        res = gamma_correction(grayscale_image, gamma=0.8)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = gamma_correction(rgba_image, gamma=1.5)
        assert res.shape == rgba_image.shape
        np.testing.assert_array_equal(res[:, :, 3], rgba_image[:, :, 3])

    def test_no_overflow(self, rgb_image):
        res = gamma_correction(rgb_image, gamma=0.1)
        assert np.max(res) <= 255 and np.min(res) >= 0