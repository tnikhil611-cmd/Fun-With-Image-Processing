"""Mathematical morphology operations."""
import cv2
import numpy as np


def _get_kernel(size: int, shape: str = 'rect') -> np.ndarray:
    k = max(1, int(size))
    if k % 2 == 0:
        k += 1
    s = cv2.MORPH_RECT
    if shape == 'ellipse':
        s = cv2.MORPH_ELLIPSE
    elif shape == 'cross':
        s = cv2.MORPH_CROSS
    return cv2.getStructuringElement(s, (k, k))


def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1, kernel_shape: str = 'rect') -> np.ndarray:
    kernel = _get_kernel(kernel_size, kernel_shape)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.erode(image[:, :, :3], kernel, iterations=int(iterations))
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.erode(image, kernel, iterations=int(iterations))


def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1, kernel_shape: str = 'rect') -> np.ndarray:
    kernel = _get_kernel(kernel_size, kernel_shape)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.dilate(image[:, :, :3], kernel, iterations=int(iterations))
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.dilate(image, kernel, iterations=int(iterations))


def morph_open(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    kernel = _get_kernel(kernel_size, kernel_shape)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.morphologyEx(image[:, :, :3], cv2.MORPH_OPEN, kernel)
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def morph_close(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    kernel = _get_kernel(kernel_size, kernel_shape)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.morphologyEx(image[:, :, :3], cv2.MORPH_CLOSE, kernel)
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)