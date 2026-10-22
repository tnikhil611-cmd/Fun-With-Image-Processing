"""Tests for image transformations."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.transformations import (
    rotate_image, flip_horizontal, flip_vertical,
    resize_image, crop_image
)


class TestRotateImage:
    def test_90_degrees(self, rgb_image):
        result = rotate_image(rgb_image, 90)
        assert result.dtype == np.uint8
        assert result.shape[2] == 3

    def test_180_degrees(self, rgb_image):
        result = rotate_image(rgb_image, 180)
        assert result.dtype == np.uint8

    def test_0_degrees(self, rgb_image):
        result = rotate_image(rgb_image, 0)
        assert result.shape == rgb_image.shape

    def test_grayscale(self, grayscale_image):
        result = rotate_image(grayscale_image, 45)
        assert result.dtype == np.uint8

    def test_bgra(self, rgba_image):
        result = rotate_image(rgba_image, 30)
        assert result.shape[2] == 4

    def test_negative_angle(self, rgb_image):
        result = rotate_image(rgb_image, -45)
        assert result.dtype == np.uint8


class TestFlipHorizontal:
    def test_bgr(self, rgb_image):
        result = flip_horizontal(rgb_image)
        assert result.shape == rgb_image.shape
        np.testing.assert_array_equal(result[:, 0, :], rgb_image[:, -1, :])

    def test_grayscale(self, grayscale_image):
        result = flip_horizontal(grayscale_image)
        assert result.shape == grayscale_image.shape
        np.testing.assert_array_equal(result[:, 0], grayscale_image[:, -1])

    def test_double_flip_identity(self, rgb_image):
        result = flip_horizontal(flip_horizontal(rgb_image))
        np.testing.assert_array_equal(result, rgb_image)


class TestFlipVertical:
    def test_bgr(self, rgb_image):
        result = flip_vertical(rgb_image)
        assert result.shape == rgb_image.shape
        np.testing.assert_array_equal(result[0, :, :], rgb_image[-1, :, :])

    def test_grayscale(self, grayscale_image):
        result = flip_vertical(grayscale_image)
        assert result.shape == grayscale_image.shape

    def test_double_flip_identity(self, rgb_image):
        result = flip_vertical(flip_vertical(rgb_image))
        np.testing.assert_array_equal(result, rgb_image)


class TestResizeImage:
    def test_downscale(self, rgb_image):
        result = resize_image(rgb_image, 50, 50, keep_aspect_ratio=False)
        assert result.shape == (50, 50, 3)

    def test_upscale(self, small_rgb_image):
        result = resize_image(small_rgb_image, 100, 100, keep_aspect_ratio=False)
        assert result.shape == (100, 100, 3)

    def test_keep_aspect_ratio(self):
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        result = resize_image(img, 50, 50, keep_aspect_ratio=True)
        assert result.shape[1] <= 50 and result.shape[0] <= 50

    def test_grayscale(self, grayscale_image):
        result = resize_image(grayscale_image, 50, 50, keep_aspect_ratio=False)
        assert result.shape == (50, 50)

    def test_bgra(self, rgba_image):
        result = resize_image(rgba_image, 50, 50, keep_aspect_ratio=False)
        assert result.shape == (50, 50, 4)

    def test_minimum_size(self, rgb_image):
        result = resize_image(rgb_image, 1, 1, keep_aspect_ratio=False)
        assert result.shape[0] >= 1 and result.shape[1] >= 1


class TestCropImage:
    def test_basic_crop(self, rgb_image):
        result = crop_image(rgb_image, 10, 10, 50, 50)
        assert result.shape == (50, 50, 3)

    def test_crop_grayscale(self, grayscale_image):
        result = crop_image(grayscale_image, 10, 10, 50, 50)
        assert result.shape == (50, 50)

    def test_crop_bgra(self, rgba_image):
        result = crop_image(rgba_image, 10, 10, 50, 50)
        assert result.shape == (50, 50, 4)

    def test_crop_clamped_to_bounds(self, rgb_image):
        result = crop_image(rgb_image, 80, 80, 50, 50)
        assert 0 < result.shape[0] <= 50 and 0 < result.shape[1] <= 50

    def test_full_image_crop(self, rgb_image):
        h, w = rgb_image.shape[:2]
        result = crop_image(rgb_image, 0, 0, w, h)
        np.testing.assert_array_equal(result, rgb_image)