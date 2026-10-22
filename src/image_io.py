"""Image I/O, download preparation, EXIF fix, and sample generators."""
import io
from typing import Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def validate_uploaded_file(uploaded_file: Any) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "No file provided."
    if hasattr(uploaded_file, "size") and uploaded_file.size is not None:
        if uploaded_file.size == 0:
            return False, "File is empty."
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File exceeds {MAX_FILE_SIZE_MB}MB limit."
    if hasattr(uploaded_file, "name") and uploaded_file.name:
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported extension .{ext}."
    return True, ""


def fix_exif_orientation(image: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


def load_image(uploaded_file: Any) -> Tuple[Optional[np.ndarray], str]:
    try:
        if hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
        else:
            data = bytes(uploaded_file)
        pil_img = Image.open(io.BytesIO(data))
        pil_img = fix_exif_orientation(pil_img)
        mode = pil_img.mode
        if mode == "RGBA":
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA), ""
        elif mode == "L":
            return np.array(pil_img), ""
        else:
            rgb = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR), ""
    except Exception as e:
        return None, str(e)


def prepare_download(image: np.ndarray, format: str = "PNG") -> bytes:
    fmt = format.upper()
    has_alpha = image.ndim == 3 and image.shape[2] == 4
    if fmt == "PNG":
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
    else:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        if has_alpha:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            image = np.clip(image[:, :, :3] * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    success, buffer = cv2.imencode(ext, image, params)
    return buffer.tobytes() if success else b""


def create_checkerboard(height: int, width: int, square_size: int = 10) -> np.ndarray:
    y, x = np.indices((height, width))
    checker = ((x // square_size) + (y // square_size)) % 2 == 0
    val = np.where(checker, np.uint8(255), np.uint8(200))
    return np.repeat(val[:, :, np.newaxis], 3, axis=2)


def overlay_on_checkerboard(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return image.copy()
    h, w = image.shape[:2]
    checker = create_checkerboard(h, w, 10)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    bgr = image[:, :, :3].astype(np.float32)
    blended = bgr * alpha + checker.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def get_display_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


def get_sample_image(sample_type: str = "Sunset Landscape") -> np.ndarray:
    h, w = 480, 640
    if sample_type == "Geometric Shapes":
        img = np.full((h, w, 3), (35, 30, 25), dtype=np.uint8)
        cv2.circle(img, (140, 140), 90, (230, 100, 40), -1)
        cv2.circle(img, (140, 140), 60, (40, 200, 240), -1)
        cv2.circle(img, (140, 140), 30, (255, 255, 255), -1)
        rect_pts = np.array([[380, 70], [520, 120], [470, 240], [330, 190]], np.int32)
        cv2.fillPoly(img, [rect_pts], (60, 220, 110))
        tri_pts = np.array([[160, 420], [60, 280], [260, 280]], np.int32)
        cv2.fillPoly(img, [tri_pts], (210, 50, 180))
        star_pts = np.array([[460, 280], [500, 370], [600, 370], [520, 420], [550, 470],
                             [460, 430], [370, 470], [400, 420], [320, 370], [420, 370]], np.int32)
        cv2.fillPoly(img, [star_pts], (240, 200, 50))
        return img
    elif sample_type == "Portrait Silhouette":
        img = np.zeros((h, w, 3), dtype=np.uint8)
        top_color, bot_color = [120, 50, 20], [220, 140, 50]
        for c in range(3):
            img[:, :, c] = np.tile(np.linspace(top_color[c], bot_color[c], h)[:, None].astype(np.uint8), (1, w))
        cv2.ellipse(img, (320, 190), (75, 100), 0, 0, 360, (25, 25, 25), -1)
        cv2.ellipse(img, (320, 160), (90, 80), 0, 0, 360, (15, 15, 15), -1)
        cv2.rectangle(img, (295, 270), (345, 340), (25, 25, 25), -1)
        shoulder_pts = np.array([[120, 480], [260, 340], [380, 340], [520, 480]], np.int32)
        cv2.fillPoly(img, [shoulder_pts], (30, 30, 30))
        cv2.ellipse(img, (320, 190), (75, 100), 0, 190, 350, (230, 200, 160), 3)
        return img
    elif sample_type == "Noisy Texture":
        y, x = np.mgrid[0:h, 0:w]
        r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
        pattern = (np.sin(r / 12.0) * 100 + 128).astype(np.float32)
        noise = np.random.normal(0, 25, (h, w)).astype(np.float32)
        noisy = np.clip(pattern + noise, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
        num_sp = int(0.02 * h * w)
        img[np.random.randint(0, h, num_sp), np.random.randint(0, w, num_sp)] = 255
        img[np.random.randint(0, h, num_sp), np.random.randint(0, w, num_sp)] = 0
        return img
    else:  # Sunset Landscape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        sky_h = int(h * 0.7)
        sky_top, sky_bot = [80, 20, 90], [40, 130, 245]
        for c in range(3):
            img[:sky_h, :, c] = np.tile(np.linspace(sky_top[c], sky_bot[c], sky_h)[:, None].astype(np.uint8), (1, w))
        cv2.circle(img, (440, 180), 45, (100, 220, 255), -1)
        cv2.circle(img, (440, 180), 65, (60, 180, 250), 3)
        pts_m1 = np.array([[0, 340], [130, 220], [280, 310], [420, 240], [540, 300], [640, 250], [640, 340]], np.int32)
        cv2.fillPoly(img, [pts_m1], (70, 35, 55))
        pts_m2 = np.array([[0, 340], [80, 280], [200, 340], [330, 270], [490, 340], [640, 310], [640, 340]], np.int32)
        cv2.fillPoly(img, [pts_m2], (45, 20, 35))
        water_h = h - sky_h
        water_top, water_bot = [35, 70, 130], [25, 30, 50]
        for c in range(3):
            img[sky_h:, :, c] = np.tile(np.linspace(water_top[c], water_bot[c], water_h)[:, None].astype(np.uint8), (1, w))
        for y_offset in range(sky_h + 5, h - 20, 12):
            length = int(np.random.uniform(20, 90))
            cv2.line(img, (440 - length // 2, y_offset), (440 + length // 2, y_offset), (60, 150, 240), 2)
        return img