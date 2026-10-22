"""Spatial transformations and affine warping."""
import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def flip_vertical(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 0)


def resize_image(image: np.ndarray, width: int, height: int, keep_aspect_ratio: bool = True) -> np.ndarray:
    h, w = image.shape[:2]
    target_w, target_h = max(1, int(width)), max(1, int(height))
    if keep_aspect_ratio:
        scale = min(target_w / w, target_h / h)
        final_w, final_h = max(1, int(w * scale)), max(1, int(h * scale))
    else:
        final_w, final_h = target_w, target_h
    inter = cv2.INTER_AREA if (final_w < w or final_h < h) else cv2.INTER_CUBIC
    return cv2.resize(image, (final_w, final_h), interpolation=inter)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, min(int(x), w - 1))
    y1 = max(0, min(int(y), h - 1))
    x2 = max(x1 + 1, min(x1 + int(width), w))
    y2 = max(y1 + 1, min(y1 + int(height), h))
    return image[y1:y2, x1:x2].copy()