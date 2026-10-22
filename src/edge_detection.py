"""Edge detection and gradient operators."""
import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny_edge(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    gray = _to_gray(image)
    return cv2.Canny(gray, int(low_threshold), int(high_threshold))


def sobel_edge(image: np.ndarray, ksize: int = 3, direction: str = 'both') -> np.ndarray:
    gray = _to_gray(image)
    k = ksize if ksize in [1, 3, 5, 7] else 3
    if direction == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        return cv2.convertScaleAbs(grad)
    elif direction == 'y':
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        return cv2.convertScaleAbs(grad)
    else:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.magnitude(gx, gy)
        return cv2.convertScaleAbs(mag)


def laplacian_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = _to_gray(image)
    k = ksize if ksize in [1, 3, 5, 7] else 3
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    return cv2.convertScaleAbs(lap)