"""Tests for morphological operations."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.morphology import erode, dilate, morph_open, morph_close


class TestErode:
    def test_bgr(self, rgb_image):
        result = erode(rgb_image, kernel_size=3)
        assert result.shape == rgb_image.shape and result.dtype == np.uint8

    def test_grayscale(self, grayscale_image):
        result = erode(grayscale_image, kernel_size=3)
        assert result.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        result = erode(rgba_image, kernel_size=3)
        assert result.shape == rgba_image.shape

    def test_shrinks_bright(self, binary_image):
        result = erode(binary_image, kernel_size=3, iterations=1)
        assert np.sum(result == 255) < np.sum(binary_image == 255)

    def test_multiple_iterations(self, binary_image):
        r1 = erode(binary_image, kernel_size=3, iterations=1)
        r2 = erode(binary_image, kernel_size=3, iterations=3)
        assert np.sum(r2 == 255) <= np.sum(r1 == 255)

    def test_kernel_shapes(self, binary_image):
        r_rect = erode(binary_image, kernel_size=5, kernel_shape='rect')
        r_cross = erode(binary_image, kernel_size=5, kernel_shape='cross')
        r_ellipse = erode(binary_image, kernel_size=5, kernel_shape='ellipse')
        assert r_rect.shape == binary_image.shape
        assert r_cross.shape == binary_image.shape
        assert r_ellipse.shape == binary_image.shape


class TestDilate:
    def test_bgr(self, rgb_image):
        result = dilate(rgb_image, kernel_size=3)
        assert result.shape == rgb_image.shape

    def test_grayscale(self, grayscale_image):
        result = dilate(grayscale_image, kernel_size=3)
        assert result.shape == grayscale_image.shape

    def test_expands_bright(self, binary_image):
        result = dilate(binary_image, kernel_size=3, iterations=1)
        assert np.sum(result == 255) > np.sum(binary_image == 255)

    def test_even_kernel_corrected(self, binary_image):
        result = dilate(binary_image, kernel_size=4)
        assert result.shape == binary_image.shape


class TestMorphOpen:
    def test_bgr(self, rgb_image):
        result = morph_open(rgb_image, kernel_size=3)
        assert result.shape == rgb_image.shape

    def test_grayscale(self, grayscale_image):
        result = morph_open(grayscale_image, kernel_size=3)
        assert result.shape == grayscale_image.shape

    def test_removes_small_bright_spots(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[50, 50] = 255
        img[10:40, 10:40] = 255
        result = morph_open(img, kernel_size=5)
        assert result[50, 50] == 0


class TestMorphClose:
    def test_bgr(self, rgb_image):
        result = morph_close(rgb_image, kernel_size=3)
        assert result.shape == rgb_image.shape

    def test_grayscale(self, grayscale_image):
        result = morph_close(grayscale_image, kernel_size=3)
        assert result.shape == grayscale_image.shape

    def test_fills_small_dark_holes(self):
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[50, 50] = 0
        result = morph_close(img, kernel_size=5)
        assert result[50, 50] == 255