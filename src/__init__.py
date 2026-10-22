# Fun With Image Processing - Source Package
"""
Core image processing modules for the Fun With Image Processing application.
"""

from .image_io import load_image, prepare_download, get_display_image
from .color_operations import to_grayscale, grayscale_to_rgb, to_hsv, to_lab
from .filters import gaussian_blur, median_blur, bilateral_filter, average_blur, sharpen, denoise
from .edge_detection import canny_edge, sobel_edge, laplacian_edge
from .thresholding import binary_threshold, binary_inv_threshold, adaptive_threshold, otsu_threshold
from .transformations import rotate_image, flip_horizontal, flip_vertical, resize_image, crop_image
from .enhancement import adjust_brightness, adjust_contrast, histogram_equalization, apply_clahe, gamma_correction
from .morphology import erode, dilate, morph_open, morph_close
from .analysis import get_image_info, compute_histogram
