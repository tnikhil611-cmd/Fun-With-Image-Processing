"""Tests for image I/O operations."""
import numpy as np
import pytest
import cv2
import io
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.image_io import (
    validate_uploaded_file,
    load_image,
    prepare_download,
    create_checkerboard,
    overlay_on_checkerboard,
    get_display_image,
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""

    def __init__(self, name: str, data: bytes, size: int = None):
        self.name = name
        self._data = data
        self.size = size if size is not None else len(data)
        self._pos = 0

    def read(self):
        return self._data

    def seek(self, pos):
        self._pos = pos

    def getvalue(self):
        return self._data


def _create_test_image_bytes(fmt='PNG', mode='RGB', size=(50, 50)):
    """Create valid image bytes in the specified format."""
    img = Image.new(mode, size, color=(128, 64, 32) if mode in ('RGB', 'RGBA') else 128)
    buf = io.BytesIO()
    save_fmt = fmt.upper()
    if save_fmt == 'JPG':
        save_fmt = 'JPEG'
    if save_fmt == 'JPEG' and mode == 'RGBA':
        img = img.convert('RGB')
    img.save(buf, format=save_fmt)
    return buf.getvalue()


class TestValidation:
    def test_valid_png(self):
        data = _create_test_image_bytes('PNG')
        f = MockUploadedFile('test.png', data)
        valid, msg = validate_uploaded_file(f)
        assert valid is True
        assert msg == ''

    def test_valid_jpg(self):
        data = _create_test_image_bytes('JPEG')
        f = MockUploadedFile('photo.jpg', data)
        valid, msg = validate_uploaded_file(f)
        assert valid is True

    def test_valid_jpeg(self):
        data = _create_test_image_bytes('JPEG')
        f = MockUploadedFile('photo.jpeg', data)
        valid, msg = validate_uploaded_file(f)
        assert valid is True

    def test_valid_webp(self):
        data = _create_test_image_bytes('WEBP')
        f = MockUploadedFile('photo.webp', data)
        valid, msg = validate_uploaded_file(f)
        assert valid is True

    def test_invalid_extension(self):
        data = _create_test_image_bytes('PNG')
        f = MockUploadedFile('test.bmp', data)
        valid, msg = validate_uploaded_file(f)
        assert valid is False
        assert 'format' in msg.lower() or 'extension' in msg.lower() or 'allowed' in msg.lower()

    def test_file_too_large(self):
        data = _create_test_image_bytes('PNG')
        f = MockUploadedFile('test.png', data, size=25 * 1024 * 1024)
        valid, msg = validate_uploaded_file(f)
        assert valid is False
        assert 'size' in msg.lower() or 'large' in msg.lower()

    def test_corrupted_file(self):
        f = MockUploadedFile('test.png', b'not an image at all')
        valid, msg = validate_uploaded_file(f)
        assert valid is False

    def test_empty_file(self):
        f = MockUploadedFile('test.png', b'')
        valid, msg = validate_uploaded_file(f)
        assert valid is False


class TestLoadImage:
    def test_load_rgb_png(self):
        data = _create_test_image_bytes('PNG', 'RGB', (50, 50))
        f = MockUploadedFile('test.png', data)
        img, err = load_image(f)
        assert img is not None
        assert err == ''
        assert img.shape == (50, 50, 3)
        assert img.dtype == np.uint8

    def test_load_rgba_png(self):
        data = _create_test_image_bytes('PNG', 'RGBA', (50, 50))
        f = MockUploadedFile('test.png', data)
        img, err = load_image(f)
        assert img is not None
        assert img.shape[2] == 4  # BGRA

    def test_load_grayscale(self):
        data = _create_test_image_bytes('PNG', 'L', (50, 50))
        f = MockUploadedFile('test.png', data)
        img, err = load_image(f)
        assert img is not None
        assert len(img.shape) == 2  # 2D grayscale

    def test_load_jpeg(self):
        data = _create_test_image_bytes('JPEG', 'RGB', (50, 50))
        f = MockUploadedFile('test.jpg', data)
        img, err = load_image(f)
        assert img is not None
        assert img.shape == (50, 50, 3)

    def test_load_corrupted(self):
        f = MockUploadedFile('test.png', b'corrupted data here')
        img, err = load_image(f)
        assert img is None
        assert err != ''


class TestPrepareDownload:
    def test_download_png_rgb(self, rgb_image):
        data = prepare_download(rgb_image, 'PNG')
        assert isinstance(data, bytes)
        assert len(data) > 0
        img = Image.open(io.BytesIO(data))
        assert img.format == 'PNG'

    def test_download_png_rgba(self, rgba_image):
        data = prepare_download(rgba_image, 'PNG')
        assert isinstance(data, bytes)
        img = Image.open(io.BytesIO(data))
        assert img.mode == 'RGBA'

    def test_download_jpeg(self, rgb_image):
        data = prepare_download(rgb_image, 'JPEG')
        assert isinstance(data, bytes)
        img = Image.open(io.BytesIO(data))
        assert img.format == 'JPEG'


class TestCheckerboard:
    def test_create_checkerboard(self):
        cb = create_checkerboard(100, 100, 10)
        assert cb.shape == (100, 100, 3)
        assert cb.dtype == np.uint8

    def test_checkerboard_has_two_colors(self):
        cb = create_checkerboard(100, 100, 10)
        unique_pixels = len(np.unique(cb.reshape(-1, 3), axis=0))
        assert unique_pixels == 2

    def test_overlay_on_checkerboard_bgra(self, rgba_image):
        result = overlay_on_checkerboard(rgba_image)
        assert result.shape[:2] == rgba_image.shape[:2]
        assert len(result.shape) == 3 and result.shape[2] == 3

    def test_overlay_on_checkerboard_bgr(self, rgb_image):
        result = overlay_on_checkerboard(rgb_image)
        assert result.shape == rgb_image.shape


class TestDisplayImage:
    def test_bgr_to_rgb(self, rgb_image):
        display = get_display_image(rgb_image)
        assert display.shape == rgb_image.shape
        np.testing.assert_array_equal(display[:, :, 0], rgb_image[:, :, 2])
        np.testing.assert_array_equal(display[:, :, 2], rgb_image[:, :, 0])

    def test_gray_passthrough(self, grayscale_image):
        display = get_display_image(grayscale_image)
        assert display.shape == grayscale_image.shape

    def test_bgra_to_rgba(self, rgba_image):
        display = get_display_image(rgba_image)
        assert display.shape == rgba_image.shape
        assert display.shape[2] == 4