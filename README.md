## Document Scanner — Low-Contrast Optimized (v8)

A Python tool that detects and flattens documents from photos, with special handling for grey/white low-contrast backgrounds.
Built as **Day 3** of my Python learning journey.

---

### Project Overview

This project implements a classic “phone photo → flat scan” pipeline with extra care for hard cases (washed-out pages, faint edges, bright tables).

Key steps:
- Contrast analysis and adaptive enhancement
- Multi-method edge detection (Canny, Sobel, Laplacian)
- Robust contour search and rectangle validation
- Perspective transform to a top-down view
- Clean black/white output via improved adaptive thresholding

---

### Features

- **Adaptive Contrast**: CLAHE + dynamic scaling based on image stats
- **Multi-Scale Edges**: Canny (fine/medium), Sobel magnitude, Laplacian cues
- **Low-Contrast Heuristics**: Gentler denoise & morphology, relaxed rectangle checks
- **Corner Recovery**: Polygon simplification, convex hull fallback, min-area rectangle last resort
- **Smart Thresholding**: Majority vote across Gaussian/Mean adaptive + Otsu
- **Preview Mode**: Step-by-step windows to visualize the pipeline

---

### Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

Install via pip:

```bash
pip install opencv-python numpy
# (Optional for development)
# pip install pytest black ruff
```

---

### Usage

#### CLI

```bash
# Basic usage – saves <photo>_scanned_v8.png (PNG ensures crisp binary edges)
python scan_improved.py -i path/to/photo.jpg

# Specify output
python scan_improved.py -i path/to/photo.jpg -o scanned.png

# Also save the color-warp (perspective-corrected color image)
python scan_improved.py -i path/to/photo.jpg -o scanned.png --save-color warped_color.jpg

# Visualize processing steps (multiple windows will open)
python scan_improved.py -i path/to/photo.jpg --preview
```

Notes
- **B/W scans are always saved as PNG** for crisp edges (if you pass a non-PNG name, it will append .png).
- If corners aren’t detected, the script uses the full image and still outputs a thresholded scan.

#### Python API (import)

```python
from scan_improved import scan_image

color_warp, bw_scan = scan_image("path/to/photo.jpg", show_preview=False)
# color_warp: perspective-corrected color image (H×W×3, uint8)
# bw_scan:    binarized scan (H×W, uint8, {0, 255})
```

---

### How It Works

1. Resize & Grayscale → speed + stability
2. Contrast Analysis → mean/std/min/max, RMS & Michelson metrics
3. Adaptive Enhancement → CLAHE + scaling depending on brightness/contrast
4. Noise Reduction → bilateral filter (gentler for low contrast)
5. Multi-Method Edge Detection → Canny, Sobel, Laplacian combined
6. Morphology → closing gaps with adaptive kernels
7. Contour & Corners → approx polygons, convex hulls, fallback min-rect
8. Perspective Transform → warp to top-down scan
9. Adaptive Thresholding → combine Gaussian, Mean, Otsu by majority vote

---

### Example Repository Structure

```
.
├─ scan_improved.py
├─ README.md
├─ examples/            # (optional)
│  ├─ invoice_01.jpg
│  ├─ invoice_01_scanned_v8.png
│  └─ invoice_01_warped_color.jpg
└─ tests/               # (optional)
```

---

###  Results (What to Expect)

- B/W scan with sharp text and clean background
- Color warp (optional) for forms or images you don’t want binarized
- Better performance on grey desks, white tables, faint page edges

---

###  Troubleshooting

- “Could not read image” → check `-i` path and file permissions
- Wrong page detected → run with `--preview` to inspect detection steps
- All white / all black → try with better lighting or place on darker surface
- Corners not found → script falls back to full frame (still usable)

---

### Roadmap

- Batch mode (scan a folder)
- Auto-rotate (detect text orientation)
- Illumination correction for uneven lighting
- Multi-page PDF export

---

### Acknowledgments

Built while learning OpenCV/NumPy, inspired by common doc-scan recipes and optimized for low contrast scenarios.

---

### License

This project is open source under the MIT License.




