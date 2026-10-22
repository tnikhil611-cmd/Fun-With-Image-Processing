"""Contrast, brightness, CLAHE, and gamma enhancements."""
import cv2
import numpy as np


def adjust_brightness(image: np.ndarray, value: int = 0) -> np.ndarray:
    if value == 0:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.convertScaleAbs(image[:, :, :3], alpha=1.0, beta=value)
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.convertScaleAbs(image, alpha=1.0, beta=value)


def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    factor = max(0.0, float(factor))
    if factor == 1.0:
        return image.copy()
    f = image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = np.clip(factor * (f[:, :, :3] - 128.0) + 128.0, 0, 255).astype(np.uint8)
        return np.dstack([bgr, image[:, :, 3]])
    return np.clip(factor * (f - 128.0) + 128.0, 0, 255).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        eq_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return np.dstack([eq_bgr, image[:, :, 3]])
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    grid = max(1, int(tile_grid_size))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid))
    if image.ndim == 2:
        return clahe.apply(image)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return np.dstack([clahe_bgr, image[:, :, 3]])
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    gamma = max(0.01, float(gamma))
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.LUT(image[:, :, :3], lut)
        return np.dstack([bgr, image[:, :, 3]])
    return cv2.LUT(image, lut)