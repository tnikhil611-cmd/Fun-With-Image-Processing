"""Background segmentation and compositing."""
from typing import Tuple
import cv2
import numpy as np
from PIL import Image

try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def is_rembg_available() -> bool:
    return REMBG_AVAILABLE


def remove_background(image: np.ndarray) -> np.ndarray:
    if not is_rembg_available():
        raise ImportError("rembg is not installed.")
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_in = Image.fromarray(rgb)
    rembg_out = rembg.remove(pil_in)
    rgba = np.array(rembg_out.convert("RGBA"))
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def replace_background_color(image: np.ndarray, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")
    if image.ndim != 3 or image.shape[2] != 4:
        return image.copy()
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    fg = image[:, :, :3].astype(np.float32)
    bg = np.full_like(fg, color, dtype=np.float32)
    blended = fg * alpha + bg * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def replace_background_image(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    if foreground is None or background is None or foreground.size == 0 or background.size == 0:
        raise ValueError("Invalid image input.")
    if foreground.ndim != 3 or foreground.shape[2] != 4:
        return foreground.copy()
    h, w = foreground.shape[:2]
    bg_resized = cv2.resize(background[:, :, :3], (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    fg = foreground[:, :, :3].astype(np.float32)
    bg = bg_resized.astype(np.float32)
    blended = fg * alpha + bg * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)