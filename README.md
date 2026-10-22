# 🎨 Fun With Image Processing — Digital Vision Studio

An interactive, browser-based digital image processing studio and algorithmic workbench built from scratch with **Python**, **OpenCV**, **NumPy**, **Pillow**, and **Streamlit**.

---

## 📋 Features

### Core Image Processing
| Category | Operations |
|---|---|
| **🎨 Color Space Conversion** | RGB ↔ Grayscale, RGB → HSV, RGB → LAB |
| **🔍 Image Filtering** | Gaussian Blur, Median Blur, Bilateral Filter, Average Blur, Sharpening, Noise Reduction |
| **✏️ Edge Detection** | Canny Hysteresis, Sobel (X/Y/Both), Laplacian |
| **⬛ Thresholding** | Binary, Binary Inverse, Adaptive (Gaussian/Mean), Otsu |
| **🔄 Transformations** | Full-canvas Rotation (no clipping), Horizontal/Vertical Flip, Resizing (with aspect ratio), Crop |
| **✨ Image Enhancement** | Brightness, Contrast, Histogram Equalization, CLAHE, Gamma Correction |
| **🔲 Morphological Operations** | Erosion, Dilation, Opening, Closing (Rectangular, Elliptical, Cross) |

### Advanced Features & Presets
- **🖼️ AI Background Removal:** Foreground extraction using U²-Net neural segmentation with transparent PNG download.
- **🎨 Backdrop Replacement:** Solid studio color replacement or custom image compositing.
- **⚡ Filter Preset Lab:** 1-click creative looks (*Cyberpunk Neon*, *Vintage Sepia*, *Pencil Sketch*, *Vivid HDR*, *Film Noir*).
- **📊 Spectral Analytics:** 256-bin per-channel histograms and statistical correlation comparisons.
- **📚 CV Handbook:** Theory and LaTeX formulas for all implemented algorithms.

---

## 🛠️ Tech Stack
- **Python 3.10+** (tested on 3.13)
- **OpenCV** (`opencv-python-headless`)
- **NumPy & SciPy**
- **Pillow**
- **Streamlit**
- **rembg & ONNX Runtime** (CPU-compatible)
- **pytest** (177 automated tests)

---

## 💻 Installation & Local Run (Windows 11)

```powershell
# 1. Navigate to the project directory
cd "C:\Users\Nikhil nitro\.gemini\antigravity\scratch\Fun-With-Image-Processing"

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py