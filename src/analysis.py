"""Spectral and metadata analytics."""
from typing import Dict, Any, Optional
import cv2
import numpy as np


def get_image_info(image: np.ndarray, file_size_bytes: Optional[int] = None) -> Dict[str, Any]:
    if image is None or image.size == 0:
        return {}
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    mode = "Grayscale" if channels == 1 else ("RGBA" if channels == 4 else "RGB")
    res = {
        "width": w,
        "height": h,
        "channels": channels,
        "color_mode": mode,
        "dtype": str(image.dtype),
        "total_pixels": h * w,
    }
    if file_size_bytes is not None:
        res["file_size_kb"] = round(file_size_bytes / 1024.0, 2)
    return res


def compute_histogram(image: np.ndarray) -> Dict[str, np.ndarray]:
    if image is None or image.size == 0:
        return {}
    if image.ndim == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return {"gray": hist.flatten()}
    # 3 or 4 channel BGR
    b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
    r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
    return {"blue": b, "green": g, "red": r}


def compute_histogram_comparison(original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
    h1 = compute_histogram(original)
    h2 = compute_histogram(processed)
    k1 = "gray" if "gray" in h1 else "green"
    k2 = "gray" if "gray" in h2 else "green"
    v1 = h1[k1].astype(np.float32)
    v2 = h2[k2].astype(np.float32)
    score = float(cv2.compareHist(v1, v2, cv2.HISTCMP_CORREL))
    return {"correlation": round(score, 4), "original_hist": h1, "processed_hist": h2}