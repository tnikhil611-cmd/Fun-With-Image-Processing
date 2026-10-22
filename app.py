import streamlit as st
import numpy as np
import cv2
import io
import time
from PIL import Image
# Import core processing modules
from src.image_io import (
    load_image,
    prepare_download,
    overlay_on_checkerboard,
    get_display_image,
    validate_uploaded_file,
    get_sample_image,
)
from src.color_operations import to_grayscale, grayscale_to_rgb, to_hsv, to_lab
from src.filters import (
    gaussian_blur,
    median_blur,
    bilateral_filter,
    average_blur,
    sharpen,
    denoise,
)
from src.edge_detection import canny_edge, sobel_edge, laplacian_edge
from src.thresholding import (
    binary_threshold,
    binary_inv_threshold,
    adaptive_threshold,
    otsu_threshold,
)
from src.transformations import (
    rotate_image,
    flip_horizontal,
    flip_vertical,
    resize_image,
    crop_image,
)
from src.enhancement import (
    adjust_brightness,
    adjust_contrast,
    histogram_equalization,
    apply_clahe,
    gamma_correction,
)
from src.morphology import erode, dilate, morph_open, morph_close
from src.analysis import get_image_info, compute_histogram, compute_histogram_comparison
# Try importing background removal (optional)
try:
    from src.background_removal import (
        remove_background,
        replace_background_color,
        replace_background_image,
        is_rembg_available,
    )
    BG_REMOVAL_AVAILABLE = is_rembg_available()
except ImportError:
    BG_REMOVAL_AVAILABLE = False
# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Studio | Fun With Image Processing",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ──────────────────────────────────────────────
# Modern Dark Theme Glassmorphic CSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Global layout */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1350px;
    }
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(22, 27, 46, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 18px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }
    .hero-banner h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(120deg, #FFFFFF 20%, #A5B4FC 60%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #94A3B8;
        font-size: 1.15rem;
        max-width: 760px;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        border-radius: 9999px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-primary {
        background: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    .badge-pink {
        background: rgba(236, 72, 153, 0.2);
        color: #F472B6;
        border: 1px solid rgba(236, 72, 153, 0.4);
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.2);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    /* Stat bar */
    .stat-pill {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .stat-title {
        color: #94A3B8;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    .stat-value {
        color: #F8FAFC;
        font-weight: 700;
        font-size: 1rem;
    }
    /* Operation Info card */
    .algo-card {
        background: rgba(30, 41, 59, 0.65);
        border-left: 4px solid #6366F1;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.8rem 0;
        color: #CBD5E1;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)
# ──────────────────────────────────────────────
# State Initialization
# ──────────────────────────────────────────────
if "current_image" not in st.session_state:
    st.session_state.current_image = get_sample_image("Sunset Landscape")
    st.session_state.image_name = "sample_sunset.png"
if "original_image" not in st.session_state:
    st.session_state.original_image = st.session_state.current_image.copy()
if "navigation_tab" not in st.session_state:
    st.session_state.navigation_tab = "🏠 Studio Home"
if "bg_removed_cache" not in st.session_state:
    st.session_state.bg_removed_cache = None
# ──────────────────────────────────────────────
# Algorithm Dictionary
# ──────────────────────────────────────────────
OPERATION_INFO = {
    "RGB to Grayscale": "Converts a color image to grayscale using human perceptual weights: `Y = 0.299R + 0.587G + 0.114B`. Green is weighted most heavily due to human optical sensitivity.",
    "Grayscale to RGB": "Promotes a single-channel intensity array into 3-channel BGR by replicating gray levels across all planes.",
    "RGB to HSV": "Decouples chromaticity (Hue) and vibrancy (Saturation) from illumination (Value). Standard for color segmentation.",
    "RGB to LAB": "Perceptually uniform space: Euclidean distance corresponds to human-perceived color difference.",
    "Gaussian Blur": "Convolves image with a 2D bell-curve Gaussian kernel. Attenuates high-frequency noise while preserving boundaries.",
    "Median Blur": "Non-linear filter replacing each pixel with the neighborhood median. Highly effective on salt-and-pepper noise.",
    "Bilateral Filter": "Edge-preserving smoothing. Weights neighbor pixels by geometric spatial closeness and radiometric color similarity.",
    "Average Blur": "Uniform box filter averaging all pixels within a square neighborhood.",
    "Sharpening": "Unsharp masking: adds high-frequency residual back to the original image: `Output = Image + Strength × (Image − Blurred)`.",
    "Noise Reduction": "Non-Local Means Denoising. Compares self-similar patches across the canvas and computes non-local weighted averages.",
    "Canny Edge": "Multi-stage detector: (1) Gaussian smoothing, (2) Sobel gradient computation, (3) Non-maximum suppression, and (4) Hysteresis tracking.",
    "Sobel Edge": "Computes spatial derivatives using horizontal and vertical 3×3 approximation kernels.",
    "Laplacian Edge": "Second-order isotropic derivative operator detecting rapid intensity transitions in all directions.",
    "Binary Threshold": "Global binarization: intensities above a threshold map to pure white (255), others map to black (0).",
    "Binary Inverse": "Inverted binarization: pixels above threshold become black (0), below become white (255).",
    "Adaptive Threshold": "Calculates local thresholds for individual pixels based on Gaussian or mean statistics within a moving neighborhood.",
    "Otsu Threshold": "Minimizes intra-class intensity variance to automatically discover the optimal bimodal separation threshold.",
    "Rotation": "Rotates by an arbitrary angle about center. Dynamically expands bounding box with trigonometric bounds so no pixels are cut off.",
    "Horizontal Flip": "Mirrors the image left-to-right across the vertical axis.",
    "Vertical Flip": "Mirrors the image top-to-bottom across the horizontal axis.",
    "Resize": "Resamples image dimensions. Uses `INTER_AREA` for downscaling and `INTER_CUBIC` for bicubic upscaling.",
    "Crop": "Extracts an exact region of interest bounded safely within image coordinates.",
    "Brightness": "Direct additive intensity shift clipped safely to [0, 255].",
    "Contrast": "Multiplicative scaling relative to mid-gray: `Output = Factor × (Input − 128) + 128`.",
    "Histogram Equalization": "Flattens cumulative distribution function (CDF) to maximize dynamic contrast. Applied to Y channel in YCrCb for color images.",
    "CLAHE": "Contrast Limited Adaptive Histogram Equalization. Divides canvas into contextual tiles and limits amplification.",
    "Gamma Correction": "Non-linear power-law curve adjustment using precomputed 256-element lookup tables: `Output = 255 × (Input / 255)^γ`.",
    "Erosion": "Morphological minimum filter: shrinks white foreground regions and dissolves small speckles.",
    "Dilation": "Morphological maximum filter: expands white foreground regions and bridges small dark gaps.",
    "Opening": "Erosion followed by dilation: eliminates tiny bright noise while preserving macro object geometry.",
    "Closing": "Dilation followed by erosion: seals small dark holes and cracks inside solid foreground features.",
    "Background Removal": "Deep learning semantic segmentation utilizing neural matting models to extract foreground subjects.",
    "Background Replacement": "Composites the isolated foreground subject over a solid studio color or secondary image.",
    "Image Information": "Comprehensive structural audit displaying spatial resolution, channel depth, color mode, and memory footprint.",
    "Histogram": "256-bin intensity histogram revealing color balance, dynamic range, and tonal clipping.",
}
def show_info(operation_name: str):
    if operation_name in OPERATION_INFO:
        st.markdown(
            f'<div class="algo-card">💡 <strong>Computer Vision Concept — {operation_name}</strong><br>{OPERATION_INFO[operation_name]}</div>',
            unsafe_allow_html=True,
        )
def display_image_canvas(image: np.ndarray, caption: str = "", use_checkerboard: bool = True):
    if image is None:
        return
    has_alpha = len(image.shape) == 3 and image.shape[2] == 4
    if has_alpha and use_checkerboard:
        preview = overlay_on_checkerboard(image)
        display = get_display_image(preview)
    else:
        display = get_display_image(image)
    st.image(display, caption=caption, use_container_width=True)
def download_action_bar(image: np.ndarray, key_prefix: str = "dl"):
    has_alpha = len(image.shape) == 3 and image.shape[2] == 4
    c1, c2 = st.columns(2)
    with c1:
        png_bytes = prepare_download(image, "PNG")
        st.download_button(
            label="📥 Export PNG (Lossless" + (" + Alpha)" if has_alpha else ")"),
            data=png_bytes,
            file_name="vision_studio_export.png",
            mime="image/png",
            key=f"{key_prefix}_png",
            use_container_width=True,
        )
    with c2:
        if not has_alpha:
            jpg_bytes = prepare_download(image, "JPEG")
            st.download_button(
                label="📥 Export JPEG (Web Ready)",
                data=jpg_bytes,
                file_name="vision_studio_export.jpg",
                mime="image/jpeg",
                key=f"{key_prefix}_jpg",
                use_container_width=True,
            )
        else:
            st.caption("ℹ️ Transparency requires PNG export.")
# ──────────────────────────────────────────────
# Top Navigation Bar
# ──────────────────────────────────────────────
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
with nav_col1:
    if st.button("🏠 Studio Home", use_container_width=True, type="primary" if st.session_state.navigation_tab == "🏠 Studio Home" else "secondary"):
        st.session_state.navigation_tab = "🏠 Studio Home"
        st.rerun()
with nav_col2:
    if st.button("🖌️ Image Editor Studio", use_container_width=True, type="primary" if st.session_state.navigation_tab == "🖌️ Image Editor Studio" else "secondary"):
        st.session_state.navigation_tab = "🖌️ Image Editor Studio"
        st.rerun()
with nav_col3:
    if st.button("⚡ Filter Preset Lab", use_container_width=True, type="primary" if st.session_state.navigation_tab == "⚡ Filter Preset Lab" else "secondary"):
        st.session_state.navigation_tab = "⚡ Filter Preset Lab"
        st.rerun()
with nav_col4:
    if st.button("📊 Spectral & Histogram Lab", use_container_width=True, type="primary" if st.session_state.navigation_tab == "📊 Spectral & Histogram Lab" else "secondary"):
        st.session_state.navigation_tab = "📊 Spectral & Histogram Lab"
        st.rerun()
with nav_col5:
    if st.button("📚 Vision CV Guide", use_container_width=True, type="primary" if st.session_state.navigation_tab == "📚 Vision CV Guide" else "secondary"):
        st.session_state.navigation_tab = "📚 Vision CV Guide"
        st.rerun()
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
# ══════════════════════════════════════════════
# PAGE 1: 🏠 STUDIO HOME
# ══════════════════════════════════════════════
if st.session_state.navigation_tab == "🏠 Studio Home":
    st.markdown(
        """
    <div class="hero-banner">
        <span class="badge badge-primary">Computer Vision Suite 2.0</span>
        <span class="badge badge-emerald">OpenCV • NumPy • Pillow</span>
        <span class="badge badge-pink">Neural Segmentation</span>
        <h1>Fun With Image Processing</h1>
        <p>
            An interactive digital image processing studio and visual computing workbench.
            Seamlessly upload, transform, filter, detect edges, enhance contrast, and segment images
            with real-time feedback, parameter controls, and mathematical transparency.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🚀 Quick Start: Load an Image")
    q_col1, q_col2 = st.columns([1.2, 1.8])
    with q_col1:
        st.markdown(
            """
        <div class="glass-card">
            <h4>📁 Upload Your Own Image</h4>
            <p style="color: #94A3B8; font-size: 0.88rem;">
                Support for high-resolution PNG, JPG, JPEG, and WebP files up to 20 MB with EXIF orientation auto-correction.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "webp"],
            help="PNG, JPG, JPEG, WebP up to 20MB",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            is_valid, err_msg = validate_uploaded_file(uploaded_file)
            if is_valid:
                img_array, load_err = load_image(uploaded_file)
                if img_array is not None:
                    st.session_state.current_image = img_array
                    st.session_state.original_image = img_array.copy()
                    st.session_state.image_name = uploaded_file.name
                    st.session_state.bg_removed_cache = None
                    st.success(f"Loaded: {uploaded_file.name}")
                else:
                    st.error(f"Failed to load image: {load_err}")
            else:
                st.error(err_msg)
    with q_col2:
        st.markdown(
            """
        <div class="glass-card">
            <h4>🎨 Or Choose a Built-in Demo Canvas</h4>
            <p style="color: #94A3B8; font-size: 0.88rem;">
                Instant synthetic test images specifically designed to showcase filters, morphology, edge detectors, and color spaces.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            if st.button("🌄 Sunset", use_container_width=True):
                st.session_state.current_image = get_sample_image("Sunset Landscape")
                st.session_state.original_image = st.session_state.current_image.copy()
                st.session_state.image_name = "sample_sunset.png"
                st.session_state.bg_removed_cache = None
                st.success("Loaded Sunset Landscape!")
        with s_col2:
            if st.button("📐 Geometry", use_container_width=True):
                st.session_state.current_image = get_sample_image("Geometric Shapes")
                st.session_state.original_image = st.session_state.current_image.copy()
                st.session_state.image_name = "sample_geometry.png"
                st.session_state.bg_removed_cache = None
                st.success("Loaded Geometric Shapes!")
        with s_col3:
            if st.button("👤 Portrait", use_container_width=True):
                st.session_state.current_image = get_sample_image("Portrait Silhouette")
                st.session_state.original_image = st.session_state.current_image.copy()
                st.session_state.image_name = "sample_portrait.png"
                st.session_state.bg_removed_cache = None
                st.success("Loaded Portrait Silhouette!")
        with s_col4:
            if st.button("🔬 Noisy Text", use_container_width=True):
                st.session_state.current_image = get_sample_image("Noisy Texture")
                st.session_state.original_image = st.session_state.current_image.copy()
                st.session_state.image_name = "sample_noisy.png"
                st.session_state.bg_removed_cache = None
                st.success("Loaded Noisy Texture!")
        curr = st.session_state.current_image
        h, w = curr.shape[:2]
        ch = curr.shape[2] if curr.ndim == 3 else 1
        st.markdown(
            f"""
        <div style="display: flex; gap: 1rem; align-items: center; margin-top: 0.6rem; background: rgba(15, 23, 42, 0.6); padding: 0.6rem 1rem; border-radius: 8px;">
            <span style="color: #6EE7B7; font-weight: 600;">Active Image:</span>
            <span style="color: #F8FAFC;">{st.session_state.image_name}</span>
            <span style="color: #94A3B8;">({w} × {h} px • {ch} ch)</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Open in Studio Editor →", type="primary", use_container_width=True):
            st.session_state.navigation_tab = "🖌️ Image Editor Studio"
            st.rerun()
    st.markdown("---")
    st.markdown("### 🧰 Studio Feature Suite")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown(
            """
        <div class="glass-card">
            <h4>🎨 Color Science Lab</h4>
            <span class="badge badge-primary">Color Spaces</span>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.5rem;">
                Transform color coordinates between sRGB, Grayscale, HSV, and CIELAB perceptual space.
            </p>
        </div>
        <div class="glass-card">
            <h4>🔍 Spatial Filtering & Denoising</h4>
            <span class="badge badge-emerald">Convolution Kernels</span>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.5rem;">
                Gaussian bell-curve blur, non-linear median filtering, bilateral edge-preserving filtering, and Non-Local Means.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown(
            """
        <div class="glass-card">
            <h4>⚡ Edge & Boundary Detection</h4>
            <span class="badge badge-pink">Differential Operators</span>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.5rem;">
                Multi-stage Canny hysteresis detection, Sobel directional gradients, and isotropic Laplacian operators.
            </p>
        </div>
        <div class="glass-card">
            <h4>🌓 Segmentation & Thresholding</h4>
            <span class="badge badge-amber">Binarization</span>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.5rem;">
                Otsu variance minimization, adaptive local Gaussian/Mean thresholding, and global binary inversion.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with fc3:
        st.markdown(
            """
        <div class="glass-card">
            <h4>✨ Dynamic Range & Enhancement</h4>
            <span class="badge badge-primary">Tone Mapping</span>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.5rem;">
                Tile-based CLAHE, luminance histogram equalization, power-law Gamma correction, and linear contrast scaling.
            </p>
        </div>
        <div class="glass-card">
            <h4>🤖 AI Matting & Background Studio</h4>
            <span class="badge badge-emerald">Neural Segmentation</span>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.5rem;">
                U²-Net deep learning foreground matting with transparency export, studio color replacement, and compositing.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
# ══════════════════════════════════════════════
# PAGE 2: 🖌️ IMAGE EDITOR STUDIO
# ══════════════════════════════════════════════
elif st.session_state.navigation_tab == "🖌️ Image Editor Studio":
    info = get_image_info(st.session_state.current_image)
    top_c1, top_c2, top_c3, top_c4 = st.columns([1.5, 1, 1, 1.2])
    with top_c1:
        st.markdown(
            f"**File:** `{st.session_state.image_name}` &nbsp;|&nbsp; `{info.get('width', 0)} × {info.get('height', 0)} px` &nbsp;|&nbsp; `{info.get('color_mode', 'RGB')}`"
        )
    with top_c2:
        view_mode = st.radio(
            "View Mode",
            ["Side-by-Side", "Processed Only", "Checkerboard Alpha"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with top_c3:
        if st.button("🔄 Reset Original", use_container_width=True):
            st.session_state.current_image = st.session_state.original_image.copy()
            st.session_state.bg_removed_cache = None
            st.rerun()
    with top_c4:
        if "last_processed" in st.session_state and st.session_state.last_processed is not None:
            if st.button("⚡ Apply & Chain Next Edit", use_container_width=True):
                st.session_state.current_image = st.session_state.last_processed.copy()
                st.session_state.bg_removed_cache = None
                st.success("Committed! You can now apply another operation on top.")
                st.rerun()
    st.markdown("---")
    ctrl_col, canvas_col = st.columns([1.1, 1.9])
    with ctrl_col:
        st.markdown("#### ⚙️ Operation Suite")
        category = st.selectbox(
            "Select Processing Module",
            [
                "🎨 Color Space Conversion",
                "🔍 Image Filtering",
                "✏️ Edge Detection",
                "⬛ Thresholding",
                "🔄 Transformations",
                "✨ Image Enhancement",
                "🔲 Morphological Operations",
                "🖼️ Background Removal",
            ],
            index=0,
        )
        input_img = st.session_state.current_image
        processed = input_img.copy()
        op_name = ""
        if category == "🎨 Color Space Conversion":
            op_name = st.radio(
                "Color Transform",
                ["RGB to Grayscale", "Grayscale to RGB", "RGB to HSV", "RGB to LAB"],
            )
            show_info(op_name)
            if op_name == "RGB to Grayscale":
                processed = to_grayscale(input_img)
            elif op_name == "Grayscale to RGB":
                processed = grayscale_to_rgb(input_img)
            elif op_name == "RGB to HSV":
                processed = to_hsv(input_img)
            elif op_name == "RGB to LAB":
                processed = to_lab(input_img)
        elif category == "🔍 Image Filtering":
            op_name = st.selectbox(
                "Filter Technique",
                ["Gaussian Blur", "Median Blur", "Bilateral Filter", "Average Blur", "Sharpening", "Noise Reduction"],
            )
            show_info(op_name)
            if op_name == "Gaussian Blur":
                k = st.slider("Kernel Size (odd)", 3, 31, 5, step=2)
                sig = st.slider("Sigma (0 = auto)", 0.0, 10.0, 0.0, 0.5)
                processed = gaussian_blur(input_img, kernel_size=k, sigma=sig)
            elif op_name == "Median Blur":
                k = st.slider("Kernel Size (odd)", 3, 31, 5, step=2)
                processed = median_blur(input_img, kernel_size=k)
            elif op_name == "Bilateral Filter":
                d = st.slider("Diameter (d)", 3, 25, 9, step=2)
                sc = st.slider("Sigma Color", 10.0, 200.0, 75.0, 5.0)
                ss = st.slider("Sigma Space", 10.0, 200.0, 75.0, 5.0)
                processed = bilateral_filter(input_img, d=d, sigma_color=sc, sigma_space=ss)
            elif op_name == "Average Blur":
                k = st.slider("Kernel Size (odd)", 3, 31, 5, step=2)
                processed = average_blur(input_img, kernel_size=k)
            elif op_name == "Sharpening":
                strength = st.slider("Sharpness Strength", 0.0, 3.0, 1.0, 0.1)
                processed = sharpen(input_img, strength=strength)
            elif op_name == "Noise Reduction":
                h_val = st.slider("Denoise Filter Strength", 1.0, 30.0, 10.0, 1.0)
                with st.spinner("Processing Non-Local Means Denoising..."):
                    processed = denoise(input_img, strength=h_val)
        elif category == "✏️ Edge Detection":
            op_name = st.selectbox("Algorithm", ["Canny Edge", "Sobel Edge", "Laplacian Edge"])
            show_info(op_name)
            if op_name == "Canny Edge":
                low = st.slider("Low Threshold", 0, 255, 50)
                high = st.slider("High Threshold", 0, 255, 150)
                processed = canny_edge(input_img, low_threshold=low, high_threshold=high)
            elif op_name == "Sobel Edge":
                direction = st.selectbox("Direction", ["both", "x", "y"])
                k = st.select_slider("Kernel Size", options=[1, 3, 5, 7], value=3)
                processed = sobel_edge(input_img, ksize=k, direction=direction)
            elif op_name == "Laplacian Edge":
                k = st.select_slider("Kernel Size", options=[1, 3, 5, 7], value=3)
                processed = laplacian_edge(input_img, ksize=k)
        elif category == "⬛ Thresholding":
            op_name = st.selectbox(
                "Method",
                ["Binary Threshold", "Binary Inverse", "Adaptive Threshold", "Otsu Threshold"],
            )
            show_info(op_name)
            if op_name in ["Binary Threshold", "Binary Inverse"]:
                t_val = st.slider("Threshold Value", 0, 255, 127)
                if op_name == "Binary Threshold":
                    processed = binary_threshold(input_img, thresh_value=t_val)
                else:
                    processed = binary_inv_threshold(input_img, thresh_value=t_val)
            elif op_name == "Adaptive Threshold":
                m = st.selectbox("Method", ["gaussian", "mean"])
                bs = st.slider("Block Size (odd >= 3)", 3, 51, 11, step=2)
                c_val = st.slider("Constant C", 0, 20, 2)
                processed = adaptive_threshold(input_img, block_size=bs, c=c_val, method=m)
            elif op_name == "Otsu Threshold":
                processed = otsu_threshold(input_img)
        elif category == "🔄 Transformations":
            op_name = st.selectbox(
                "Transform",
                ["Rotation", "Horizontal Flip", "Vertical Flip", "Resize", "Crop"],
            )
            show_info(op_name)
            if op_name == "Rotation":
                angle = st.slider("Rotation Angle (degrees)", -180.0, 180.0, 0.0, 1.0)
                processed = rotate_image(input_img, angle=angle)
            elif op_name == "Horizontal Flip":
                processed = flip_horizontal(input_img)
            elif op_name == "Vertical Flip":
                processed = flip_vertical(input_img)
            elif op_name == "Resize":
                cur_h, cur_w = input_img.shape[:2]
                rw = st.number_input("Target Width", min_value=1, max_value=4000, value=int(cur_w * 0.8))
                rh = st.number_input("Target Height", min_value=1, max_value=4000, value=int(cur_h * 0.8))
                keep = st.checkbox("Preserve Aspect Ratio", value=True)
                processed = resize_image(input_img, width=rw, height=rh, keep_aspect_ratio=keep)
            elif op_name == "Crop":
                cur_h, cur_w = input_img.shape[:2]
                cx = st.slider("Left (X)", 0, max(0, cur_w - 10), 0)
                cy = st.slider("Top (Y)", 0, max(0, cur_h - 10), 0)
                cw = st.slider("Width", 10, cur_w - cx, max(10, cur_w - cx))
                ch = st.slider("Height", 10, cur_h - cy, max(10, cur_h - cy))
                processed = crop_image(input_img, x=cx, y=cy, width=cw, height=ch)
        elif category == "✨ Image Enhancement":
            op_name = st.selectbox(
                "Enhancement Technique",
                ["Brightness", "Contrast", "Histogram Equalization", "CLAHE", "Gamma Correction"],
            )
            show_info(op_name)
            if op_name == "Brightness":
                b_val = st.slider("Brightness Offset", -100, 100, 0)
                processed = adjust_brightness(input_img, value=b_val)
            elif op_name == "Contrast":
                c_val = st.slider("Contrast Multiplier", 0.1, 3.0, 1.0, 0.05)
                processed = adjust_contrast(input_img, factor=c_val)
            elif op_name == "Histogram Equalization":
                processed = histogram_equalization(input_img)
            elif op_name == "CLAHE":
                cl = st.slider("Clip Limit", 1.0, 10.0, 2.0, 0.5)
                gs = st.slider("Tile Grid Size", 2, 16, 8, step=1)
                processed = apply_clahe(input_img, clip_limit=cl, tile_grid_size=gs)
            elif op_name == "Gamma Correction":
                gam = st.slider("Gamma Value", 0.1, 3.0, 1.0, 0.05)
                processed = gamma_correction(input_img, gamma=gam)
        elif category == "🔲 Morphological Operations":
            op_name = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing"])
            show_info(op_name)
            shape = st.selectbox("Kernel Shape", ["rect", "ellipse", "cross"])
            k_size = st.slider("Kernel Size (odd)", 3, 25, 3, step=2)
            iters = st.slider("Iterations", 1, 5, 1)
            if op_name == "Erosion":
                processed = erode(input_img, kernel_size=k_size, iterations=iters, kernel_shape=shape)
            elif op_name == "Dilation":
                processed = dilate(input_img, kernel_size=k_size, iterations=iters, kernel_shape=shape)
            elif op_name == "Opening":
                processed = morph_open(input_img, kernel_size=k_size, kernel_shape=shape)
            elif op_name == "Closing":
                processed = morph_close(input_img, kernel_size=k_size, kernel_shape=shape)
        elif category == "🖼️ Background Removal":
            op_name = st.selectbox("Feature", ["Remove Background", "Replace Background Color", "Replace Background Image"])
            show_info(op_name)
            if not BG_REMOVAL_AVAILABLE:
                st.warning("⚠️ `rembg` library is not loaded or runtime is in fallback mode.")
            else:
                if op_name == "Remove Background":
                    if st.button("🚀 Extract Foreground", type="primary", use_container_width=True):
                        with st.spinner("AI model segmenting subject..."):
                            try:
                                processed = remove_background(input_img)
                                st.session_state.bg_removed_cache = processed
                                st.success("Background extracted successfully!")
                            except Exception as ex:
                                st.error(f"Background removal notice: {ex}")
                    elif st.session_state.bg_removed_cache is not None:
                        processed = st.session_state.bg_removed_cache
                elif op_name == "Replace Background Color":
                    if st.session_state.bg_removed_cache is None:
                        st.info("💡 Please run 'Remove Background' first to extract the transparent subject.")
                    else:
                        color_hex = st.color_picker("Pick Studio Backdrop Color", "#10B981")
                        r_c = int(color_hex[1:3], 16)
                        g_c = int(color_hex[3:5], 16)
                        b_c = int(color_hex[5:7], 16)
                        processed = replace_background_color(
                            st.session_state.bg_removed_cache, color=(b_c, g_c, r_c)
                        )
                elif op_name == "Replace Background Image":
                    if st.session_state.bg_removed_cache is None:
                        st.info("💡 Please run 'Remove Background' first to extract the transparent subject.")
                    else:
                        bg_choice = st.selectbox(
                            "Choose Background Canvas",
                            ["Sunset Landscape", "Geometric Shapes", "Custom Upload"],
                        )
                        if bg_choice == "Custom Upload":
                            bg_file = st.file_uploader("Upload Backdrop Image", type=["png", "jpg", "jpeg", "webp"])
                            if bg_file:
                                bg_arr, _ = load_image(bg_file)
                                if bg_arr is not None:
                                    processed = replace_background_image(st.session_state.bg_removed_cache, bg_arr)
                        else:
                            sample_bg = get_sample_image(bg_choice)
                            processed = replace_background_image(st.session_state.bg_removed_cache, sample_bg)
        st.session_state.last_processed = processed
        st.markdown("---")
        st.markdown("##### 💾 Export Result")
        download_action_bar(processed, key_prefix="studio_ctrl")
    with canvas_col:
        st.markdown(f"#### 🖼️ Canvas Viewport — `{op_name}`")
        use_check = (view_mode == "Checkerboard Alpha")
        if view_mode == "Side-by-Side":
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.markdown("**Original Reference**")
                display_image_canvas(input_img, caption="Original", use_checkerboard=False)
            with v_col2:
                st.markdown("**Processed Result**")
                display_image_canvas(processed, caption="Active Edit", use_checkerboard=True)
        else:
            display_image_canvas(
                processed,
                caption=f"Result: {op_name}",
                use_checkerboard=use_check or (processed.ndim == 3 and processed.shape[2] == 4),
            )
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            p_h, p_w = processed.shape[:2]
            st.metric("Canvas Size", f"{p_w} × {p_h} px")
        with m_c2:
            p_ch = processed.shape[2] if processed.ndim == 3 else 1
            st.metric("Channels", f"{p_ch} Channel{'s' if p_ch > 1 else ''}")
        with m_c3:
            st.metric("Data Type", str(processed.dtype))
# ══════════════════════════════════════════════
# PAGE 3: ⚡ FILTER PRESET LAB
# ══════════════════════════════════════════════
elif st.session_state.navigation_tab == "⚡ Filter Preset Lab":
    st.markdown("### ⚡ Professional One-Click Presets")
    base = st.session_state.current_image.copy()
    preset_choice = st.radio(
        "Choose Preset Look",
        [
            "🌆 Cyberpunk Neon (Sobel + High Saturation)",
            "📜 Vintage Warm Sepia (LAB Color Grading)",
            "✏️ Architectural Pencil Sketch (Inverted Gaussian Dodge)",
            "🌅 Vivid HDR Landscape (CLAHE + Contrast Pop)",
            "🎬 Dramatic Film Noir (High Contrast Monochromatic)",
            "🧼 Studio Soft Portrait (Bilateral Smoothing + Micro-Sharpening)",
        ],
        horizontal=True,
    )
    preset_out = base.copy()
    if "Cyberpunk Neon" in preset_choice:
        edges = sobel_edge(base, ksize=3, direction="both")
        neon_mask = np.zeros_like(base)
        neon_mask[:, :, 0] = edges
        neon_mask[:, :, 2] = edges
        boosted = adjust_contrast(base, factor=1.4)
        preset_out = cv2.addWeighted(boosted, 0.75, neon_mask, 0.7, 0)
    elif "Vintage Warm Sepia" in preset_choice:
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_bgr = np.clip(cv2.transform(base, sepia_filter), 0, 255).astype(np.uint8)
        preset_out = adjust_contrast(sepia_bgr, factor=1.1)
    elif "Pencil Sketch" in preset_choice:
        gray = to_grayscale(base)
        inv_gray = 255 - gray
        blurred = gaussian_blur(inv_gray, kernel_size=21, sigma=0.0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        preset_out = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif "Vivid HDR Landscape" in preset_choice:
        clahe_img = apply_clahe(base, clip_limit=3.0, tile_grid_size=8)
        contrast = adjust_contrast(clahe_img, factor=1.3)
        sharpened = sharpen(contrast, strength=1.2)
        preset_out = gamma_correction(sharpened, gamma=0.9)
    elif "Dramatic Film Noir" in preset_choice:
        gray = to_grayscale(base)
        high_con = adjust_contrast(gray, factor=1.8)
        preset_out = cv2.cvtColor(high_con, cv2.COLOR_GRAY2BGR)
    elif "Studio Soft Portrait" in preset_choice:
        soft = bilateral_filter(base, d=9, sigma_color=50, sigma_space=50)
        preset_out = sharpen(soft, strength=0.6)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original**")
        display_image_canvas(base, caption="Original Input", use_checkerboard=False)
    with c2:
        st.markdown(f"**Applied Preset: {preset_choice.split('(')[0]}**")
        display_image_canvas(preset_out, caption=preset_choice, use_checkerboard=True)
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("🚀 Send to Editor Studio as Working Canvas", type="primary", use_container_width=True):
            st.session_state.current_image = preset_out.copy()
            st.session_state.navigation_tab = "🖌️ Image Editor Studio"
            st.rerun()
    with b2:
        download_action_bar(preset_out, key_prefix="preset_dl")
# ══════════════════════════════════════════════
# PAGE 4: 📊 SPECTRAL & HISTOGRAM LAB
# ══════════════════════════════════════════════
elif st.session_state.navigation_tab == "📊 Spectral & Histogram Lab":
    st.markdown("### 📊 Spectral & Intensity Analytics")
    curr = st.session_state.current_image
    orig = st.session_state.original_image
    hist_curr = compute_histogram(curr)
    hist_orig = compute_histogram(orig)
    comparison = compute_histogram_comparison(orig, curr)
    import pandas as pd
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Width × Height", f"{curr.shape[1]} × {curr.shape[0]} px")
    with m2:
        st.metric("Total Pixels", f"{curr.shape[0] * curr.shape[1]:,}")
    with m3:
        corr_val = comparison.get("correlation", 1.0)
        st.metric("Histogram Correlation vs Original", f"{corr_val:.3f}")
    with m4:
        st.metric("Luminance Mean", f"{np.mean(curr):.1f} / 255")
    st.markdown("---")
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        st.markdown("#### 📈 Active Canvas Histogram")
        if "gray" in hist_curr:
            df = pd.DataFrame({"Luminance": hist_curr["gray"].flatten()}, index=range(256))
            st.line_chart(df, use_container_width=True)
        else:
            df = pd.DataFrame(
                {
                    "Blue": hist_curr["blue"].flatten(),
                    "Green": hist_curr["green"].flatten(),
                    "Red": hist_curr["red"].flatten(),
                },
                index=range(256),
            )
            st.line_chart(df, use_container_width=True)
    with ch_col2:
        st.markdown("#### 📉 Original Reference Histogram")
        if "gray" in hist_orig:
            df_orig = pd.DataFrame({"Luminance": hist_orig["gray"].flatten()}, index=range(256))
            st.line_chart(df_orig, use_container_width=True)
        else:
            df_orig = pd.DataFrame(
                {
                    "Blue": hist_orig["blue"].flatten(),
                    "Green": hist_orig["green"].flatten(),
                    "Red": hist_orig["red"].flatten(),
                },
                index=range(256),
            )
            st.line_chart(df_orig, use_container_width=True)
# ══════════════════════════════════════════════
# PAGE 5: 📚 VISION CV GUIDE
# ══════════════════════════════════════════════
elif st.session_state.navigation_tab == "📚 Vision CV Guide":
    st.markdown("### 📚 Computer Vision Knowledge Handbook")
    t1, t2, t3, t4 = st.tabs(["Spatial Kernels & Filtering", "Differential Edge Operators", "Thresholding Science", "Dynamic Range Enhancement"])
    with t1:
        st.markdown(
            r"""
        #### 1. Gaussian Convolution
        $$G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
        Gaussian smoothing suppresses high-frequency Gaussian noise. The parameter $\sigma$ dictates the spatial dispersion width.
        #### 2. Bilateral Filter
        $$I^{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \, f_r(\|I(p) - I(q)\|) \, g_s(\|p - q\|)$$
        Combines spatial Euclidean distance $g_s$ with intensity distance $f_r$. Edges are strictly preserved.
        """
        )
    with t2:
        st.markdown(
            r"""
        #### 1. Sobel Gradient Kernels
        $$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$
        $$G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$
        #### 2. Canny Edge Hysteresis
        Tracks weak edges only if topologically connected to strong edges, eliminating false boundary noise.
        """
        )
    with t3:
        st.markdown(
            r"""
        #### Otsu's Global Variance Minimization
        $$\sigma_b^2(t) = \omega_0(t) \, \omega_1(t) \left[\mu_0(t) - \mu_1(t)\right]^2$$
        Exhaustively minimizes intra-class intensity variance between foreground and background classes.
        """
        )
    with t4:
        st.markdown(
            r"""
        #### 1. CLAHE
        Contrast Limited Adaptive Histogram Equalization operates on local tiles (e.g. $8 \times 8$) and clips the histogram bins to prevent noise amplification in uniform regions.
        #### 2. Gamma Correction
        $$V_{\text{out}} = 255 \times \left(\frac{V_{\text{in}}}{255}\right)^\gamma$$
        - $\gamma < 1$: Illuminates dark underexposed regions.
        - $\gamma > 1$: Compresses shadows, enhancing high-key highlights.
        """
        )