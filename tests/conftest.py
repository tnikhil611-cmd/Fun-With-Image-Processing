"""Shared test fixtures for Fun With Image Processing tests."""
import numpy as np
import pytest
import cv2


@pytest.fixture
def rgb_image():
    """Create a 100x100 BGR test image with a gradient pattern."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = [i * 2 % 256, j * 2 % 256, (i + j) % 256]
    return img


@pytest.fixture
def grayscale_image():
    """Create a 100x100 grayscale test image with a gradient."""
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = (i * 2 + j) % 256
    return img


@pytest.fixture
def rgba_image():
    """Create a 100x100 BGRA test image with semi-transparent regions."""
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = [i * 2 % 256, j * 2 % 256, (i + j) % 256, 200]
    img[0:50, 0:50, 3] = 0  # Quarter transparent
    return img


@pytest.fixture
def white_image():
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


@pytest.fixture
def black_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def binary_image():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255
    return img


@pytest.fixture
def noisy_image():
    img = np.random.randint(100, 156, (100, 100, 3), dtype=np.uint8)
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)