"""Tests for edge detection algorithms."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.edge_detection import canny_edge, sobel_edge, laplacian_edge


class TestCannyEdge:
    def test_bgr(self, rgb_image):
        res = canny_edge(rgb_image, 50, 150)
        assert res.ndim == 2 and res.shape == rgb_image.shape[:2]

    def test_grayscale(self, grayscale_image):
        res = canny_edge(grayscale_image, 50, 150)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = canny_edge(rgba_image, 50, 150)
        assert res.shape == rgba_image.shape[:2]

    def test_binary_output(self, rgb_image):
        res = canny_edge(rgb_image, 50, 150)
        unique = set(np.unique(res))
        assert unique.issubset({0, 255})

    def test_detects_edges(self, binary_image):
        res = canny_edge(binary_image, 50, 150)
        assert np.sum(res == 255) > 0


class TestSobelEdge:
    def test_bgr_both(self, rgb_image):
        res = sobel_edge(rgb_image, ksize=3, direction='both')
        assert res.ndim == 2 and res.shape == rgb_image.shape[:2]

    def test_direction_x(self, rgb_image):
        res = sobel_edge(rgb_image, ksize=3, direction='x')
        assert res.ndim == 2

    def test_direction_y(self, rgb_image):
        res = sobel_edge(rgb_image, ksize=3, direction='y')
        assert res.ndim == 2

    def test_grayscale(self, grayscale_image):
        res = sobel_edge(grayscale_image)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = sobel_edge(rgba_image)
        assert res.shape == rgba_image.shape[:2]


class TestLaplacianEdge:
    def test_bgr(self, rgb_image):
        res = laplacian_edge(rgb_image, ksize=3)
        assert res.ndim == 2 and res.shape == rgb_image.shape[:2]

    def test_grayscale(self, grayscale_image):
        res = laplacian_edge(grayscale_image, ksize=3)
        assert res.shape == grayscale_image.shape

    def test_bgra(self, rgba_image):
        res = laplacian_edge(rgba_image, ksize=3)
        assert res.shape == rgba_image.shape[:2]

    def test_detects_edges(self, binary_image):
        res = laplacian_edge(binary_image, ksize=3)
        assert np.sum(res > 0) > 0