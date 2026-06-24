# Logic.md — Underwater Image Enhancement System

## 1. Overview

System enhances underwater images degraded by water absorption, scattering, and color distortion. Pipeline applies sequential DSP stages (contrast, color, gamma, denoising) then fuses two parallel enhancement paths via Laplacian pyramid blending. Outputs enhanced image + depth estimate + object detections + quality metrics.

**Core problem addressed:** Water absorbs red wavelengths first (at ~1–2m depth), then green, leaving images blue-green and low-contrast. Standard enhancement fails because the degradation is wavelength-dependent and spatially non-uniform.

---

## 2. Tech Stack

### Backend
| Library | Role |
|---------|------|
| **Flask** | Web framework, REST API |
| **OpenCV (`opencv-python`)** | All image processing (BGR ops, CLAHE, bilateral filter, pyramid ops) |
| **NumPy** | Array math, channel manipulation |
| **SciPy** | Scientific utilities |
| **scikit-image** | SSIM/PSNR metrics |
| **Pillow** | Image I/O format conversion |
| **Ultralytics** | YOLOv8 model loading and inference |
| **CLIP (Ultralytics fork)** | Vision-language backbone for YOLOv8-World open-vocabulary |
| **Gunicorn** | Production WSGI server |

### Frontend
| Library | Role |
|---------|------|
| **Alpine.js v3** | Reactive state management, UI logic |
| **Tailwind CSS** | Utility-first styling (CDN) |
| **Chart.js** | Histogram charts, composition pie chart |
| **img-comparison-slider v7** | Before/after image slider |

### Infrastructure
| Component | Details |
|-----------|---------|
| **Docker** | `python:3.9-slim` base, libgl1 + libglib2.0-0 |
| **Server** | Gunicorn, 1 worker, 120s timeout |
| **Port** | Dynamic via `PORT` env var (default 5000) |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│  Alpine.js + Tailwind CSS + Chart.js + img-comparison-slider │
│  - Drag/drop upload   - Pipeline toggle switches            │
│  - Before/after slider - Detection overlay                  │
│  - Depth map view     - RGB histograms + composition chart  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (multipart + JSON)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK APP (app.py)                        │
│  POST /upload        POST /reprocess     GET /download       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Enhancement Pipeline                    │   │
│  │  src/adaptive_enhancement.py (4 sequential stages)   │   │
│  │  src/clahe.py         src/histogram_linearization.py │   │
│  │  src/weight_maps.py   src/multiscale_fusion.py       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ src/         │  │ src/         │  │ src/             │  │
│  │ detection.py │  │ depth_       │  │ metrics.py       │  │
│  │ (YOLO-World) │  │ estimation.py│  │ (UCIQE/Entropy)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │   Model Weights     │
            │  yolov8s-world.pt   │
            │  CLIP (git install) │
            └─────────────────────┘
```

---

## 4. Enhancement Pipeline (Core DSP)

All stages operate on **uint8 BGR** images. Source: `src/adaptive_enhancement.py`, `src/clahe.py`, `src/histogram_linearization.py`, `src/weight_maps.py`, `src/multiscale_fusion.py`.

### Stage 1 — Adaptive Histogram Stretching
**File:** `src/adaptive_enhancement.py:8–39`  
**Purpose:** Expand the compressed dynamic range caused by water scattering.

```
For each channel (B, G, R):
  low  = percentile(channel, 1)
  high = percentile(channel, 99)
  stretched = (channel - low) * 255 / (high - low)
  clip to [0, 255]
```

Percentile clipping (1%/99%) prevents extreme pixels from compressing the stretch range. Applied independently per channel to compensate for differential wavelength absorption.

---

### Stage 2 — Adaptive Color Correction (White Balance)
**File:** `src/adaptive_enhancement.py:41–92`  
**Purpose:** Remove the blue-green cast from water absorption.

**Step A — LAB neutral shift:**
```
Convert BGR → LAB
A_channel += (128 - mean(A)) * 0.5   # push toward neutral green-red
B_channel += (128 - mean(B)) * 0.5   # push toward neutral blue-yellow
Convert LAB → BGR
```

**Step B — HSV saturation boost (conditional):**
```
Convert BGR → HSV
if mean(S_channel) < 80:   # image is dull/desaturated
    S_channel *= 1.2
    clip S to [0, 255]
Convert HSV → BGR
```

---

### Stage 3 — Adaptive Gamma Correction
**File:** `src/adaptive_enhancement.py:94–123`  
**Purpose:** Compensate for darkness/overexposure based on actual image luminance.

```
Convert BGR → LAB
L_normalized = L_channel / 255.0
mean_L = mean(L_normalized), clip to (0.01, 0.99)
gamma = log(0.5) / log(mean_L)    # target mean luminance = 0.5
gamma = clamp(gamma, 0.5, 2.5)

Build LUT: lut[i] = (i/255)^gamma * 255
Apply LUT to L channel
Convert LAB → BGR
```

Gamma < 1: brightens dark images. Gamma > 1: darkens overexposed. Auto-calculated, no manual tuning.

---

### Stage 4 — Edge-Preserving Denoising
**File:** `src/adaptive_enhancement.py:125–155`  
**Purpose:** Remove noise introduced by water particles without blurring edges.

```
cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
```

Bilateral filter weights pixels by both spatial proximity AND color similarity — edges (high color difference) are not smoothed, noise (low color difference) is removed.

---

### Stage 5A — CLAHE (Contrast Limited Adaptive Histogram Equalization)
**File:** `src/clahe.py:4–30`  
**Purpose:** Enhance local contrast without blowing out highlights.

```
Convert BGR → LAB
Apply CLAHE to L channel only:
  clipLimit = 2.0
  tileGridSize = (8, 8)
Convert LAB → BGR
```

Operates only on luminance (L channel) to avoid hue shifts. Divides image into 8×8 tiles, equalizes each locally, clips redistribution at 2.0 to prevent noise amplification.

---

### Stage 5B — Histogram Linearization (Global Equalization)
**File:** `src/histogram_linearization.py:4–33`  
**Purpose:** Provide a contrasting global enhancement for fusion.

```
Convert BGR → HSV
Apply cv2.equalizeHist() to V (Value) channel
Convert HSV → BGR
```

Standard global histogram equalization on brightness. Runs in parallel to CLAHE to provide a complementary enhancement perspective for the fusion stage.

---

### Stage 6 — Weight Map Generation
**File:** `src/weight_maps.py:4–67`  
**Purpose:** Determine which pixels each enhancement path handles better.

**Three sub-weights computed per image:**

**W_Laplacian (local contrast):**
```
gray = cv2.cvtColor(img, COLOR_BGR2GRAY)
W_L = |cv2.Laplacian(gray, CV_64F)|
```

**W_Saliency (perceptual importance):**
```
Convert to LAB
W_S = sqrt((L - mean_L)^2 + (a - mean_a)^2 + (b - mean_b)^2)
```
Frequency-tuned saliency — pixels far from global LAB mean are considered more salient.

**W_Saturation (color richness):**
```
R, G, B = split channels
mean_rgb = (R + G + B) / 3
W_Sat = sqrt(((R-mean)^2 + (G-mean)^2 + (B-mean)^2) / 3)
```

**Combined weight:**
```
W = W_Laplacian + W_Saliency + W_Saturation + 1e-12  (epsilon prevents /0)
```

Computed for both the CLAHE image and the HE image independently.

---

### Stage 7 — Multiscale Fusion (Laplacian Pyramid)
**File:** `src/multiscale_fusion.py:30–81`, `levels=5`  
**Purpose:** Blend CLAHE and HE outputs seamlessly using multi-resolution decomposition.

```
1. Normalize weights:
   W_clahe_norm = W_clahe / (W_clahe + W_he)
   W_he_norm    = W_he    / (W_clahe + W_he)

2. Build Gaussian pyramids (5 levels) from both weight maps

3. Build Laplacian pyramids (5 levels) from both enhanced images:
   For level k:
     Gauss_k → upsample → Gauss_{k-1} → Laplacian_k = Gauss_{k-1} - upsampled

4. Fuse at each pyramid level:
   Fused_k = W_clahe_gauss_k * Laplacian_clahe_k + W_he_gauss_k * Laplacian_he_k

5. Reconstruct by collapsing pyramid (upsample + add, bottom up)

6. Clip to [0, 255], convert to uint8
```

Multi-resolution fusion avoids visible seams that occur in direct pixel blending. Low frequencies blended coarsely; high frequencies (edges, texture) blended finely.

---

## 5. ML Models

### YOLOv8s-World
| Property | Value |
|----------|-------|
| **Type** | Open-vocabulary object detection |
| **Weights** | `yolov8s-world.pt` (auto-downloaded by Ultralytics) |
| **Backbone** | YOLOv8s + CLIP visual encoder |
| **Inference** | `model.predict(img, conf=0.10)` |
| **Input** | BGR image (any size, YOLO auto-resizes) |
| **Output** | Bounding boxes [x1,y1,x2,y2], class label, confidence |

**Detection classes defined in** `src/detection.py:14–32`:
```python
CLASSES = [
    "Fish", "Coral", "Diver", "Rock",
    "Sea Turtle", "Shark", "Starfish", "Jellyfish"
]
```

Each class has synonym lists to improve recall via YOLO-World's text-based matching (e.g., "Fish" also matches "fish", "tropical fish", "school of fish").

**Confidence threshold:** 0.10 (low, to catch partially-visible objects in murky water).

**Composition calculation:**
```
For each detected class:
  area = (x2-x1) * (y2-y1)
  composition[label] += area
Normalize: composition[label] / total_image_area * 100
```

### CLIP (Ultralytics fork)
- Installed from `git+https://github.com/ultralytics/CLIP.git`
- Powers YOLO-World's open-vocabulary capability — encodes class name text into the same embedding space as visual features
- Not called directly; loaded internally by Ultralytics when YOLOv8s-World initializes

---

## 6. API Endpoints

**Framework:** Flask  
**File:** `app.py`

### `POST /upload`
Upload and process a new image.

**Request:** `multipart/form-data`
```
file:   <image binary>   (PNG, JPG, JPEG, BMP, WEBP; max 16MB)
config: '{"stretch": true, "wb": true, "gamma": true, "sharp": true}'
```

**Response:** `application/json`
```json
{
  "success": true,
  "filename": "image.jpg",
  "images": {
    "original":  "static/images/uploads/image.jpg",
    "wb":        "static/images/results/image_1_adaptive_color.jpg",
    "gamma":     "static/images/results/image_2_adaptive_gamma.jpg",
    "sharp":     "static/images/results/image_3_edge_preserving.jpg",
    "clahe":     "static/images/results/image_4_clahe.jpg",
    "hist":      "static/images/results/image_5_hist_linear.jpg",
    "final":     "static/images/results/image_6_final.jpg",
    "depth_map": "static/images/results/image_depth_color.jpg",
    "depth_raw": "static/images/results/image_depth_raw.png",
    "final_b64": "data:image/jpeg;base64,<encoded>",
    "detections": [
      {"box": [x1, y1, x2, y2], "label": "Fish", "confidence": 0.85}
    ],
    "composition": {"Fish": 45.2, "Coral": 34.1},
    "histograms": {
      "before": {"r": [...256], "g": [...256], "b": [...256], "y": [...256]},
      "after":  {"r": [...256], "g": [...256], "b": [...256], "y": [...256]}
    }
  },
  "metrics": {
    "UCIQE": 5.234,
    "Entropy": 7.456,
    "UIQM (Est)": 0.0,
    "Depth": "5–15 meters",
    "DepthConf": 0.85
  }
}
```

### `POST /reprocess`
Re-run pipeline on already-uploaded image with new config.

**Request:** `application/json`
```json
{"filename": "image.jpg", "config": {"stretch": true, "wb": false, "gamma": true, "sharp": true}}
```
**Response:** Same schema as `/upload` (no `filename` key).

### `GET /download/<filename>`
Download result with optional format conversion.

**Query params:** `?format=png` or `?format=jpg`  
**Response:** Binary file download with correct MIME type.

### `GET /`
Serves `templates/index.html`.

---

## 7. Frontend

**File:** `templates/index.html`, `static/js/main.js`, `static/css/styles.css`

### Components

**Upload Panel**
- Drag-and-drop zone + file input (`accept="image/*"`)
- Camera capture via `navigator.mediaDevices.getUserMedia`
- Preview of selected image before processing

**Pipeline Configuration**
- 4 toggle switches: Stretch / WB / Gamma / Sharp
- Visual pipeline flow: `IN → [1] → [2] → [3] → [4] → F`
- Toggles map to `config` JSON sent with upload

**Results Panel**
- `img-comparison-slider`: side-by-side original vs. final
- Detection view: canvas overlay drawing bounding boxes from `detections[]`
- Depth map view: color-coded (Red=shallow, Blue=deep) with legend
- Intermediate steps strip: WB → Gamma → Sharp → CLAHE → HE

**Metrics Panel**
- UCIQE score, Entropy, depth range, detection count

**Charts**
- RGB + Luminance histograms (Chart.js bar): before vs. after
- Composition pie chart: % area per detected class

**Export**
- Download button → `GET /download/<filename>?format=jpg|png`
- "Process New Image" resets Alpine.js state

### State Management (Alpine.js)
```javascript
{
  pipelineConfig: {stretch, wb, gamma, sharp},
  results: {images, detections, composition, histograms},
  metrics: {UCIQE, Entropy, Depth, ...},
  loading: bool,
  showDetections: bool,
  showDepth: bool
}
```

---

## 8. Quality Metrics

**File:** `src/metrics.py`

### UCIQE (Underwater Color Image Quality Evaluation)
Standard reference-free metric for underwater images.

```
1. Convert image to LAB
2. σ_chroma  = std(chroma) where chroma = sqrt(a^2 + b^2)
3. contrast_L = (top10% of L) - (bottom10% of L)
4. μ_saturation = mean(saturation in HSV)

UCIQE = 0.4680 * σ_chroma + 0.2745 * contrast_L + 0.2576 * μ_saturation
```

Higher = better quality. Typical range: 0–10.

### Entropy (Shannon)
```
gray = cv2.cvtColor(img, COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0,256]) / total_pixels
entropy = -sum(p * log2(p) for p > 0)
```

Higher entropy = more information/detail in image. Range: 0–8.

### UIQM
Placeholder in current implementation — returns 0.0. Field reserved for future Underwater Image Quality Measure implementation.

---

## 9. Depth Estimation

**File:** `src/depth_estimation.py`

Uses red-channel attenuation as a depth proxy. Red light is absorbed fastest in water (~1–2m half-depth), so low red ratio indicates depth.

```
red_ratio = mean(R_channel) / (mean(R_channel) + mean(G_channel) + mean(B_channel))

if red_ratio > 0.25:   depth = "0–5 meters"
elif red_ratio > 0.15: depth = "5–15 meters"
elif red_ratio > 0.05: depth = "15–30 meters"
else:                  depth = ">30 meters"

confidence = based on how far ratio is from thresholds
```

**Depth Map Generation:**
```
depth_map_raw = 255 - R_channel        # invert red: deeper = brighter
Apply cv2.applyColorMap(depth_raw, COLORMAP_JET)
  Red   = shallow (high red, near surface)
  Blue  = deep    (low red, greater depth)
```

Saved as both `_depth_raw.png` (grayscale) and `_depth_color.jpg` (JET colormap).

---

## 10. Full Data Flow

```
USER UPLOAD (multipart/form-data)
         │
         ▼
┌─────────────────────────────────────┐
│  app.py: validate + resize          │
│  - Check file extension             │
│  - Decode with cv2.imdecode         │
│  - If max(H,W) > 1024: resize       │
│    (maintain aspect ratio)          │
│  - Convert to uint8                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 1: Histogram Stretching      │
│  src/adaptive_enhancement.py:8–39   │
│  Per-channel percentile stretch     │
│  (1%, 99%) → linear remap [0,255]   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 2: Color Correction          │
│  src/adaptive_enhancement.py:41–92  │
│  LAB A/B channel neutral shift +    │
│  HSV saturation boost (if S < 80)   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 3: Gamma Correction          │
│  src/adaptive_enhancement.py:94–123 │
│  gamma = log(0.5)/log(mean_L)       │
│  Applied via 256-entry LUT          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 4: Bilateral Filter          │
│  src/adaptive_enhancement.py:125–155│
│  d=5, sigmaColor=50, sigmaSpace=50  │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────────┐
│  Stage 5A    │ │  Stage 5B        │
│  CLAHE       │ │  Histogram       │
│  clahe.py    │ │  Linearization   │
│  L-channel   │ │  hist_linear.py  │
│  clip=2.0    │ │  V-channel       │
│  tiles=8x8   │ │  equalizeHist()  │
└──────┬───────┘ └────────┬─────────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────────┐
│  Stage 6: Weight Map Generation     │
│  src/weight_maps.py                 │
│  W = W_Laplacian + W_Saliency       │
│      + W_Saturation                 │
│  (computed for BOTH images)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 7: Multiscale Fusion         │
│  src/multiscale_fusion.py           │
│  5-level Laplacian pyramid          │
│  Weighted blend at each level       │
│  Collapse → uint8 BGR               │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────────┐
│YOLO-   │ │Depth   │ │Metrics     │
│World   │ │Estim.  │ │UCIQE       │
│detect  │ │Red     │ │Entropy     │
│.py     │ │ratio   │ │metrics.py  │
└────┬───┘ └───┬────┘ └─────┬──────┘
     │         │             │
     └────┬────┘             │
          ▼                  ▼
┌─────────────────────────────────────┐
│  app.py: Save intermediate images   │
│  Base64 encode final for inline     │
│  Compute histograms before/after    │
│  Build JSON response                │
└──────────────┬──────────────────────┘
               │
               ▼
         JSON RESPONSE
```

---

## 11. Configuration

### Flask App Config (`app.py:45–55`)
```python
SECRET_KEY          = 'dev-super-secret-key'   # change in production
UPLOAD_FOLDER       = 'static/images/uploads'
RESULT_FOLDER       = 'static/images/results'
MAX_CONTENT_LENGTH  = 16 * 1024 * 1024         # 16MB upload limit
MAX_IMAGE_DIMENSION = 1024                      # auto-resize threshold (px)
```

### Pipeline Config (per-request)
```json
{
  "stretch": true,   // Stage 1: histogram stretching
  "wb":      true,   // Stage 2: color correction
  "gamma":   true,   // Stage 3: gamma correction
  "sharp":   true    // Stage 4: bilateral filter
}
```
Stages 5–7 (CLAHE, HE, fusion) always run.

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server listen port |
| `FLASK_ENV` | `development` | Set to `production` to disable debug |

### Accepted Image Formats
PNG, JPG, JPEG, BMP, WEBP

### Output Files Per Upload
```
static/images/uploads/
  <filename>                         # original

static/images/results/
  <name>_1_adaptive_color.jpg        # after Stage 2
  <name>_2_adaptive_gamma.jpg        # after Stage 3
  <name>_3_edge_preserving.jpg       # after Stage 4
  <name>_4_clahe.jpg                 # Stage 5A output
  <name>_5_hist_linear.jpg           # Stage 5B output
  <name>_6_final.jpg                 # fusion output
  <name>_depth_color.jpg             # JET colormap depth map
  <name>_depth_raw.png               # raw grayscale depth map
```

---

## 12. Docker & Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \           # OpenCV requires OpenGL runtime
    libglib2.0-0 \     # GLib runtime for OpenCV
    git \              # needed to pip install CLIP from GitHub
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE $PORT

CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 app:app
```

### Key Deployment Decisions
| Setting | Value | Reason |
|---------|-------|--------|
| `--workers 1` | Single worker | YOLO model loaded once globally; multi-worker would duplicate 27GB weight |
| `--timeout 120` | 2 minutes | Large images + YOLO inference can exceed default 30s |
| `python:3.9-slim` | Minimal base | Reduces image size; only adds 3 required system libs |
| `git` in apt | Required | CLIP installed directly from GitHub repo |

### Logging
```
handlers: [StreamHandler (stdout), FileHandler ('app.log')]
level: INFO
format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

---

## 13. File Structure

```
Underwater-image-enhancement/
├── app.py                          # Flask app, orchestrates full pipeline
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container build
├── yolov8s-world.pt                # YOLO-World weights (auto-downloaded)
├── src/
│   ├── adaptive_enhancement.py     # Stages 1–4 (stretch/WB/gamma/bilateral)
│   ├── clahe.py                    # Stage 5A (CLAHE)
│   ├── histogram_linearization.py  # Stage 5B (histogram equalization)
│   ├── weight_maps.py              # Stage 6 (Laplacian+saliency+saturation)
│   ├── multiscale_fusion.py        # Stage 7 (Laplacian pyramid fusion)
│   ├── detection.py                # YOLOv8-World inference
│   ├── depth_estimation.py         # Red-ratio depth heuristic
│   ├── metrics.py                  # UCIQE, Entropy
│   ├── histogram_analysis.py       # RGB + luminance histograms
│   └── white_balance.py            # (standalone, tested separately)
├── templates/
│   └── index.html                  # Full single-page frontend
├── static/
│   ├── css/styles.css
│   ├── js/main.js                  # Alpine.js app logic
│   └── images/
│       ├── uploads/                # User images (auto-created)
│       └── results/                # Pipeline outputs (auto-created)
└── tests/
    ├── test_pipeline.py
    └── test_detection_init.py
```
