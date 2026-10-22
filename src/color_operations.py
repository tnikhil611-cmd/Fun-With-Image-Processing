"""Color space conversion operations."""
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA image to 2D grayscale."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale to 3-channel BGR."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")
    if image.ndim == 3 and image.shape[2] >= 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert image to HSV color space."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def to_lab(image: np.ndarray) -> np.ndarray:
    """Convert image to CIE LAB color space."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)