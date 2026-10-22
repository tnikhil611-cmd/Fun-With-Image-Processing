"""Binarization and thresholding techniques."""
import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binary_threshold(image: np.ndarray, thresh_value: int = 127) -> np.ndarray:
    gray = _to_gray(image)
    _, res = cv2.threshold(gray, int(thresh_value), 255, cv2.THRESH_BINARY)
    return res


def binary_inv_threshold(image: np.ndarray, thresh_value: int = 127) -> np.ndarray:
    gray = _to_gray(image)
    _, res = cv2.threshold(gray, int(thresh_value), 255, cv2.THRESH_BINARY_INV)
    return res


def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2, method: str = 'gaussian') -> np.ndarray:
    gray = _to_gray(image)
    bs = max(3, int(block_size))
    if bs % 2 == 0:
        bs += 1
    m = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == 'gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(gray, 255, m, cv2.THRESH_BINARY, bs, int(c))


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    _, res = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res