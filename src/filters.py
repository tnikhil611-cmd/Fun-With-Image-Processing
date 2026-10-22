"""Spatial filtering and smoothing algorithms."""
import cv2
import numpy as np


def _ensure_odd(val: int) -> int:
    val = max(1, int(val))
    return val if val % 2 == 1 else val + 1


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0.0) -> np.ndarray:
    k = _ensure_odd(kernel_size)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.GaussianBlur(image[:, :, :3], (k, k), sigma)
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.GaussianBlur(image, (k, k), sigma)


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = _ensure_odd(kernel_size)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.medianBlur(image[:, :, :3], k)
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.medianBlur(image, k)


def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    d = max(1, int(d))
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.bilateralFilter(image[:, :, :3], d, sigma_color, sigma_space)
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def average_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = _ensure_odd(kernel_size)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.blur(image[:, :, :3], (k, k))
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.blur(image, (k, k))


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    strength = max(0.0, float(strength))
    if strength == 0.0:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3.0)
        sharp = cv2.addWeighted(bgr, 1.0 + strength, blurred, -strength, 0)
        return np.dstack([sharp, image[:, :, 3]])
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def denoise(image: np.ndarray, strength: float = 10.0) -> np.ndarray:
    strength = max(1.0, float(strength))
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.fastNlMeansDenoisingColored(image[:, :, :3], None, strength, strength, 7, 21)
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)