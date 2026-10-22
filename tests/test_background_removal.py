"""Tests for background removal module."""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.background_removal import (
    is_rembg_available,
    remove_background,
    replace_background_color,
    replace_background_image,
)


class TestIsRembgAvailable:
    def test_returns_bool(self):
        res = is_rembg_available()
        assert isinstance(res, bool)


class TestRemoveBackground:
    def test_invalid_input_none(self):
        with pytest.raises(ValueError):
            remove_background(None)

    def test_invalid_input_empty(self):
        with pytest.raises(ValueError):
            remove_background(np.array([]))

    def test_rembg_call_or_importerror(self, rgb_image):
        if not is_rembg_available():
            with pytest.raises(ImportError):
                remove_background(rgb_image)
        else:
            from unittest.mock import patch
            from PIL import Image
            h, w = rgb_image.shape[:2]
            dummy_rgba = Image.new("RGBA", (w, h), (100, 150, 200, 255))
            with patch("rembg.remove", return_value=dummy_rgba) as mock_remove:
                result = remove_background(rgb_image)
                assert mock_remove.called
                assert result.ndim == 3 and result.shape[2] == 4
                assert result.dtype == np.uint8
                assert result.shape[:2] == rgb_image.shape[:2]


class TestReplaceBackgroundColor:
    def test_replace_transparent_quarter(self, rgba_image):
        res = replace_background_color(rgba_image, color=(0, 255, 0))
        assert res.ndim == 3 and res.shape[2] == 3
        assert res.shape[:2] == rgba_image.shape[:2]
        np.testing.assert_array_equal(res[25, 25], [0, 255, 0])

    def test_no_alpha_returns_copy(self, rgb_image):
        res = replace_background_color(rgb_image, color=(255, 255, 255))
        assert res.shape == rgb_image.shape
        np.testing.assert_array_equal(res, rgb_image)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            replace_background_color(None)
        with pytest.raises(ValueError):
            replace_background_color(np.array([]))


class TestReplaceBackgroundImage:
    def test_composite_with_bgr_background(self, rgba_image, white_image):
        res = replace_background_image(rgba_image, white_image)
        assert res.ndim == 3 and res.shape[2] == 3
        np.testing.assert_array_equal(res[25, 25], [255, 255, 255])

    def test_background_auto_resize(self, rgba_image):
        bg = np.zeros((50, 50, 3), dtype=np.uint8)
        bg[:, :] = [10, 20, 30]
        res = replace_background_image(rgba_image, bg)
        assert res.shape[:2] == rgba_image.shape[:2]
        np.testing.assert_array_equal(res[25, 25], [10, 20, 30])

    def test_no_alpha_returns_copy(self, rgb_image, white_image):
        res = replace_background_image(rgb_image, white_image)
        np.testing.assert_array_equal(res, rgb_image)

    def test_invalid_input(self, rgba_image):
        with pytest.raises(ValueError):
            replace_background_image(None, rgba_image)
        with pytest.raises(ValueError):
            replace_background_image(rgba_image, None)