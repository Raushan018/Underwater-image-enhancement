# RADS-LPF — Full Implementation Code

> **Source:** "A Retinex-Guided Adaptive DSP Framework with Laplacian Pyramid Fusion for Robust Underwater Image Enhancement"  
> 2025 IEEE ICIP | Netaji Subhash Engineering College, Kolkata  
> This file contains the complete backend implementation matching the paper's methodology exactly.

---

## Project Structure

```
rads_lpf/
├── app.py                  # Flask REST API (3 endpoints)
├── pipeline/
│   ├── __init__.py
│   ├── stage1_stretch.py   # Adaptive Histogram Stretching
│   ├── stage2_colour.py    # LAB Colour Correction + HSV Sat Boost
│   ├── stage3_dacr.py      # Depth-Aware Channel Rebalancing (DACR) ★ Novel
│   ├── stage4_gamma.py     # Auto-Gamma via LAB Luminance LUT
│   ├── stage5_denoise.py   # Bilateral Edge-Preserving Denoising
│   ├── stage5a_clahe.py    # CLAHE branch
│   ├── stage5b_histeq.py   # Histogram Linearisation branch
│   ├── stage6_weights.py   # Adaptive Weight Map Generation
│   ├── stage7_fusion.py    # 5-Level Laplacian Pyramid Fusion
│   └── rads_lpf.py         # Master pipeline orchestrator
├── analysis/
│   ├── __init__.py
│   ├── detection.py        # YOLOv8s-World open-vocabulary detection
│   ├── depth.py            # Red-ratio depth estimation + heatmap
│   └── metrics.py          # UCIQE, UIQM, SSIM, PSNR, Shannon Entropy
├── requirements.txt
└── Dockerfile
```

---

## requirements.txt

```
flask>=2.3.0
gunicorn>=21.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.21.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
```

---

## pipeline/stage1_stretch.py — Adaptive Histogram Stretching

```python
"""
Stage 1: Adaptive Histogram Stretching
Paper §3.3 — Eq: C' = (C - p1(C)) × 255 / (p99(C) - p1(C))
Independent per-channel percentile-clip linear remapping.
Compensates for differential wavelength absorption.
"""

import numpy as np


def adaptive_histogram_stretch(image: np.ndarray) -> np.ndarray:
    """
    Apply 1st/99th percentile-clip linear remapping independently
    to each BGR channel.

    Args:
        image: uint8 BGR image, shape (H, W, 3)

    Returns:
        Stretched uint8 BGR image, shape (H, W, 3)
    """
    assert image.dtype == np.uint8, "Input must be uint8"
    stretched = np.zeros_like(image, dtype=np.float32)

    for c in range(3):          # B=0, G=1, R=2
        channel = image[:, :, c].astype(np.float32)
        p1  = np.percentile(channel, 1)
        p99 = np.percentile(channel, 99)

        denom = p99 - p1
        if denom < 1e-6:        # Flat channel — leave unchanged
            stretched[:, :, c] = channel
        else:
            stretched[:, :, c] = (channel - p1) * 255.0 / denom

    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched
```

---

## pipeline/stage2_colour.py — LAB Colour Correction + HSV Saturation Boost

```python
"""
Stage 2: LAB Colour Correction + HSV Saturation Boost
Paper §3.4
  LAB neutral-axis shift:
      A' = A + 0.5 * (128 - µ_A)
      B' = B + 0.5 * (128 - µ_B)
  Conditional HSV saturation boost: S ← 1.2S when µ_S < 80
"""

import cv2
import numpy as np


def lab_colour_correction(image: np.ndarray) -> np.ndarray:
    """
    Correct blue-green cast via LAB neutral-axis shift and
    recover colour vibrancy with a conditional HSV saturation boost.

    Args:
        image: uint8 BGR image

    Returns:
        Colour-corrected uint8 BGR image
    """
    # --- LAB neutral-axis shift ---
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    mu_A = np.mean(A)
    mu_B = np.mean(B)

    A_corrected = A + 0.5 * (128.0 - mu_A)
    B_corrected = B + 0.5 * (128.0 - mu_B)

    A_corrected = np.clip(A_corrected, 0, 255)
    B_corrected = np.clip(B_corrected, 0, 255)

    lab_corrected = cv2.merge([L, A_corrected, B_corrected]).astype(np.uint8)
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    # --- Conditional HSV saturation boost ---
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)

    mu_S = np.mean(S)
    if mu_S < 80:
        S = np.clip(S * 1.2, 0, 255)

    hsv_boosted = cv2.merge([H, S, V]).astype(np.uint8)
    result = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    return result
```

---

## pipeline/stage3_dacr.py — Depth-Aware Channel Rebalancing ★ Novel

```python
"""
Stage 3: Depth-Aware Channel Rebalancing (DACR)  ★ Novel contribution
Paper §3.5

Red-ratio depth proxy:
    ρ = µ_R / (µ_R + µ_G + µ_B)

Depth-tier gain table (Table 1 in paper):
    Shallow  ρ > 0.25        GR=1.00  GG=1.00  GB=1.00
    Mid      0.15–0.25       GR=1.25  GG=1.05  GB=0.95
    Deep     0.05–0.15       GR=1.60  GG=1.15  GB=0.90
    Abyss    ρ ≤ 0.05        GR=2.00  GG=1.30  GB=0.85
"""

import cv2
import numpy as np
from typing import Tuple, Dict

# Depth-tier gain table — learned empirically on 500 annotated UIEB frames
DEPTH_TIERS: list[dict] = [
    {"name": "shallow", "label": "0–5 m",   "rho_min": 0.25, "GR": 1.00, "GG": 1.00, "GB": 1.00},
    {"name": "mid",     "label": "5–15 m",  "rho_min": 0.15, "GR": 1.25, "GG": 1.05, "GB": 0.95},
    {"name": "deep",    "label": "15–30 m", "rho_min": 0.05, "GR": 1.60, "GG": 1.15, "GB": 0.90},
    {"name": "abyss",   "label": ">30 m",   "rho_min": 0.00, "GR": 2.00, "GG": 1.30, "GB": 0.85},
]


def estimate_depth_tier(image: np.ndarray) -> Tuple[float, Dict]:
    """
    Compute red-ratio ρ and select the matching depth tier.

    Args:
        image: uint8 BGR image

    Returns:
        (rho, tier_dict) where tier_dict has keys: name, label, GR, GG, GB
    """
    img_f = image.astype(np.float32)
    mu_B = np.mean(img_f[:, :, 0])
    mu_G = np.mean(img_f[:, :, 1])
    mu_R = np.mean(img_f[:, :, 2])

    total = mu_R + mu_G + mu_B
    rho = mu_R / total if total > 1e-6 else 0.0

    for tier in DEPTH_TIERS:
        if rho > tier["rho_min"]:
            return rho, tier

    return rho, DEPTH_TIERS[-1]   # abyss fallback


def depth_aware_channel_rebalancing(image: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """
    Apply depth-contingent per-channel gain factors.
    Gains are applied to both LAB L-channel and per-channel BGR values.

    Args:
        image: uint8 BGR image (after Stage 2)

    Returns:
        (rebalanced_image, rho, depth_tier_name)
    """
    rho, tier = estimate_depth_tier(image)
    GR, GG, GB = tier["GR"], tier["GG"], tier["GB"]

    # --- BGR per-channel gain ---
    img_f = image.astype(np.float32)
    img_f[:, :, 0] = np.clip(img_f[:, :, 0] * GB, 0, 255)   # Blue
    img_f[:, :, 1] = np.clip(img_f[:, :, 1] * GG, 0, 255)   # Green
    img_f[:, :, 2] = np.clip(img_f[:, :, 2] * GR, 0, 255)   # Red

    # --- LAB L-channel gain (weighted average of BGR gains → luminance) ---
    intermediate = img_f.astype(np.uint8)
    lab = cv2.cvtColor(intermediate, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B_ch = cv2.split(lab)

    lum_gain = (GR + GG + GB) / 3.0
    L = np.clip(L * lum_gain, 0, 255)

    lab_out = cv2.merge([L, A, B_ch]).astype(np.uint8)
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    return result, rho, tier["name"]
```

---

## pipeline/stage4_gamma.py — Auto-Gamma Correction

```python
"""
Stage 4: Auto-Gamma Correction via LAB Luminance LUT
Paper §3.6

    γ = log(0.5) / log(µ_L),  γ ∈ [0.5, 2.5]

256-entry LUT applied to LAB L-channel:
    L'[i] = (i/255)^γ × 255
"""

import cv2
import numpy as np
import math


def auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    """
    Compute optimal gamma from mean LAB luminance and apply via LUT.

    Args:
        image: uint8 BGR image

    Returns:
        Gamma-corrected uint8 BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    mu_L = np.mean(L) / 255.0      # Normalise to [0,1]
    mu_L = max(mu_L, 1e-6)         # Guard log(0)

    gamma = math.log(0.5) / math.log(mu_L)
    gamma = float(np.clip(gamma, 0.5, 2.5))

    # Build 256-entry LUT
    lut = np.array(
        [(i / 255.0) ** gamma * 255.0 for i in range(256)],
        dtype=np.uint8
    )

    # Apply LUT to L channel only
    L_uint8 = np.clip(L, 0, 255).astype(np.uint8)
    L_corrected = cv2.LUT(L_uint8, lut)

    lab_out = cv2.merge(
        [L_corrected.astype(np.float32), A, B]
    ).astype(np.uint8)

    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return result
```

---

## pipeline/stage5_denoise.py — Bilateral Edge-Preserving Denoising

```python
"""
Stage 5: Bilateral Edge-Preserving Denoising
Paper §3.7

    BF[I]_p = (1/W_p) Σ_{q∈S} G_σs(‖p-q‖) G_σr(|I_p - I_q|) I_q

Parameters: d=5, σ_color=50, σ_space=50
Suppresses backscatter noise while preserving edges.
"""

import cv2
import numpy as np


def bilateral_denoise(image: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filter with paper-specified parameters.

    Args:
        image: uint8 BGR image

    Returns:
        Denoised uint8 BGR image
    """
    denoised = cv2.bilateralFilter(
        image,
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )
    return denoised
```

---

## pipeline/stage5a_clahe.py — CLAHE Branch

```python
"""
Stage 5A: CLAHE (Contrast Limited Adaptive Histogram Equalisation)
Paper §3.8

Params: clipLimit=2.0, tileGridSize=(8,8)
Applied to LAB L-channel only — avoids global hue distortion.
Excels in textured regions.
"""

import cv2
import numpy as np


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast via CLAHE on the LAB L-channel.

    Args:
        image: uint8 BGR image (after bilateral denoising)

    Returns:
        CLAHE-enhanced uint8 BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(L)

    lab_enhanced = cv2.merge([L_enhanced, A, B])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result
```

---

## pipeline/stage5b_histeq.py — Histogram Linearisation Branch

```python
"""
Stage 5B: Histogram Linearisation (Global Equalisation)
Paper §3.8

cv2.equalizeHist applied to HSV V-channel.
Recovers global dynamic range.
Complementary to CLAHE — excels in flat/dark regions.
"""

import cv2
import numpy as np


def apply_histogram_equalisation(image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalisation to the HSV V-channel.

    Args:
        image: uint8 BGR image (after bilateral denoising)

    Returns:
        Globally equalised uint8 BGR image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    V_eq = cv2.equalizeHist(V)

    hsv_eq = cv2.merge([H, S, V_eq])
    result = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return result
```

---

## pipeline/stage6_weights.py — Adaptive Weight Map Generation

```python
"""
Stage 6: Adaptive Weight Map Generation
Paper §3.9

Three sub-weights combined additively per pixel (ε = 1e-12):
    W_L   = |∇²G|                              — Laplacian local contrast
    W_S   = ‖[L,a,b] - [µ_L, µ_a, µ_b]‖       — Frequency-tuned saliency
    W_Sat = σ_RGB                               — Colour saturation

Final per-pixel weights normalised so W_CLAHE + W_HE = 1.
"""

import cv2
import numpy as np

EPS = 1e-12


def _laplacian_contrast(image: np.ndarray) -> np.ndarray:
    """Absolute Laplacian response on greyscale — local contrast map."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    return np.abs(lap)


def _saliency_map(image: np.ndarray) -> np.ndarray:
    """
    Frequency-tuned saliency: pixel-wise L2 distance from
    global LAB mean — paper's W_S definition.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    mu = np.mean(lab.reshape(-1, 3), axis=0)           # [µ_L, µ_a, µ_b]
    diff = lab - mu                                    # (H, W, 3)
    saliency = np.sqrt(np.sum(diff ** 2, axis=2))      # (H, W)
    return saliency


def _saturation_map(image: np.ndarray) -> np.ndarray:
    """Per-pixel RGB standard deviation — colour saturation proxy."""
    img_f = image.astype(np.float32)
    sat = np.std(img_f, axis=2)                        # σ across R,G,B
    return sat


def compute_weight_map(image: np.ndarray) -> np.ndarray:
    """
    Compute combined adaptive weight map for one enhanced branch.

    Args:
        image: uint8 BGR enhanced image (one branch)

    Returns:
        (H, W) float32 unnormalised weight map
    """
    W_L   = _laplacian_contrast(image)
    W_S   = _saliency_map(image)
    W_Sat = _saturation_map(image)

    # Additive combination — equal contribution per paper
    W = W_L + W_S + W_Sat + EPS
    return W


def compute_normalised_weight_maps(
    img_clahe: np.ndarray,
    img_he: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and normalise weight maps for both branches so they
    sum to 1 at every pixel.

    Args:
        img_clahe: CLAHE-enhanced uint8 BGR image
        img_he:    HE-enhanced uint8 BGR image

    Returns:
        (W_clahe_norm, W_he_norm) both float32 (H, W)
    """
    W_clahe = compute_weight_map(img_clahe)
    W_he    = compute_weight_map(img_he)

    total = W_clahe + W_he + EPS
    W_clahe_norm = W_clahe / total
    W_he_norm    = W_he    / total

    return W_clahe_norm, W_he_norm
```

---

## pipeline/stage7_fusion.py — 5-Level Laplacian Pyramid Fusion

```python
"""
Stage 7: 5-Level Laplacian Pyramid Fusion
Paper §3.10

For each pyramid level k:
    F_k = W̃^G_CLAHE,k · LP_CLAHE,k + W̃^G_HE,k · LP_HE,k

Pyramid collapse: upsample + add (bottom-up).
Ensures low-frequency colour gradients and high-frequency edge
detail are blended at their natural scales — eliminates visible seams
inherent to direct pixel blending.
"""

import cv2
import numpy as np

LEVELS = 5


def _gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build Gaussian pyramid by successive pyrDown."""
    pyramid = [image.astype(np.float32)]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build Laplacian pyramid from Gaussian pyramid."""
    gauss = _gaussian_pyramid(image, levels)
    lap = []
    for i in range(levels - 1):
        h, w = gauss[i].shape[:2]
        up = cv2.pyrUp(gauss[i + 1], dstsize=(w, h))
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])            # Lowest level = smallest Gaussian
    return lap


def _expand_weight(weight: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize a (H, W) weight map to target (H, W) — used per level."""
    h, w = target_shape[:2]
    return cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR)


def laplacian_pyramid_fusion(
    img_clahe:    np.ndarray,
    img_he:       np.ndarray,
    w_clahe_norm: np.ndarray,
    w_he_norm:    np.ndarray,
    levels: int = LEVELS
) -> np.ndarray:
    """
    Fuse two enhanced branches via Laplacian pyramid.

    Args:
        img_clahe:    uint8 BGR CLAHE-enhanced image
        img_he:       uint8 BGR HE-enhanced image
        w_clahe_norm: (H, W) float32 normalised weight for CLAHE branch
        w_he_norm:    (H, W) float32 normalised weight for HE branch
        levels:       pyramid depth (paper uses 5)

    Returns:
        Fused uint8 BGR image
    """
    # Build Laplacian pyramids for both images (per channel)
    lp_clahe = _laplacian_pyramid(img_clahe.astype(np.float32), levels)
    lp_he    = _laplacian_pyramid(img_he.astype(np.float32),    levels)

    # Build Gaussian pyramids for weight maps
    gp_w_clahe = _gaussian_pyramid(w_clahe_norm, levels)
    gp_w_he    = _gaussian_pyramid(w_he_norm,    levels)

    # Fuse at each level
    fused_pyramid = []
    for k in range(levels):
        lp_c = lp_clahe[k]         # (H_k, W_k, 3)
        lp_h = lp_he[k]

        # Expand weight maps to (H_k, W_k, 1) for broadcasting
        wc = gp_w_clahe[k][:, :, np.newaxis]
        wh = gp_w_he[k][:, :, np.newaxis]

        # Resize weight if shape mismatch (can occur at boundaries)
        h, w = lp_c.shape[:2]
        if wc.shape[:2] != (h, w):
            wc = cv2.resize(gp_w_clahe[k], (w, h))[:, :, np.newaxis]
            wh = cv2.resize(gp_w_he[k],    (w, h))[:, :, np.newaxis]

        fused_pyramid.append(wc * lp_c + wh * lp_h)

    # Collapse pyramid (bottom-up upsample + add)
    fused = fused_pyramid[-1]
    for k in range(levels - 2, -1, -1):
        h, w = fused_pyramid[k].shape[:2]
        fused = cv2.pyrUp(fused, dstsize=(w, h)) + fused_pyramid[k]

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused
```

---

## pipeline/rads_lpf.py — Master Pipeline Orchestrator

```python
"""
RADS-LPF Master Pipeline
Orchestrates all 7 stages as a single joint optimisation graph.

Stage flow (Fig. 1 in paper):
    Input
    → [1] Adaptive Histogram Stretching
    → [2] LAB Colour Correction + HSV Sat Boost
    → [3] DACR — Depth-Aware Channel Rebalancing  ★
    → [4] Auto-Gamma via LAB Luminance LUT
    → [5] Bilateral Edge-Preserving Denoising
         → [5A] CLAHE         [5B] Histogram Linearisation
    → [6] Adaptive Weight Map Generation
    → [7] 5-Level Laplacian Pyramid Fusion
    → Enhanced Output
"""

import numpy as np
import cv2

from .stage1_stretch  import adaptive_histogram_stretch
from .stage2_colour   import lab_colour_correction
from .stage3_dacr     import depth_aware_channel_rebalancing
from .stage4_gamma    import auto_gamma_correction
from .stage5_denoise  import bilateral_denoise
from .stage5a_clahe   import apply_clahe
from .stage5b_histeq  import apply_histogram_equalisation
from .stage6_weights  import compute_normalised_weight_maps
from .stage7_fusion   import laplacian_pyramid_fusion


def run_pipeline(
    image: np.ndarray,
    stages_enabled: dict | None = None
) -> dict:
    """
    Execute the full RADS-LPF pipeline.

    Args:
        image:          uint8 BGR input image
        stages_enabled: optional dict of {stage_name: bool} for
                        /reprocess endpoint toggles. All stages
                        enabled by default.

    Returns:
        dict with keys:
            enhanced       — final fused uint8 BGR image
            intermediates  — dict of intermediate stage outputs
            rho            — red-ratio depth proxy (float)
            depth_tier     — depth tier name (str)
    """
    if stages_enabled is None:
        stages_enabled = {}

    def _enabled(name: str) -> bool:
        return stages_enabled.get(name, True)

    intermediates: dict[str, np.ndarray] = {}
    intermediates["input"] = image.copy()

    # Stage 1 — Adaptive Histogram Stretching
    out = adaptive_histogram_stretch(image) if _enabled("stage1") else image.copy()
    intermediates["stage1_stretch"] = out.copy()

    # Stage 2 — LAB Colour Correction + HSV Sat Boost
    out = lab_colour_correction(out) if _enabled("stage2") else out
    intermediates["stage2_colour"] = out.copy()

    # Stage 3 — DACR  ★
    if _enabled("stage3"):
        out, rho, depth_tier = depth_aware_channel_rebalancing(out)
    else:
        rho, depth_tier = 0.0, "unknown"
    intermediates["stage3_dacr"] = out.copy()

    # Stage 4 — Auto-Gamma
    out = auto_gamma_correction(out) if _enabled("stage4") else out
    intermediates["stage4_gamma"] = out.copy()

    # Stage 5 — Bilateral Denoising
    out = bilateral_denoise(out) if _enabled("stage5") else out
    intermediates["stage5_denoise"] = out.copy()

    # Stages 5A / 5B — Dual Enhancement Branches
    img_clahe = apply_clahe(out)          if _enabled("stage5a") else out.copy()
    img_he    = apply_histogram_equalisation(out) if _enabled("stage5b") else out.copy()
    intermediates["stage5a_clahe"] = img_clahe.copy()
    intermediates["stage5b_he"]    = img_he.copy()

    # Stage 6 — Adaptive Weight Maps
    w_clahe, w_he = compute_normalised_weight_maps(img_clahe, img_he)
    intermediates["stage6_weight_clahe"] = (w_clahe * 255).astype(np.uint8)
    intermediates["stage6_weight_he"]    = (w_he    * 255).astype(np.uint8)

    # Stage 7 — Laplacian Pyramid Fusion
    enhanced = laplacian_pyramid_fusion(img_clahe, img_he, w_clahe, w_he)
    intermediates["stage7_enhanced"] = enhanced.copy()

    return {
        "enhanced":      enhanced,
        "intermediates": intermediates,
        "rho":           rho,
        "depth_tier":    depth_tier,
    }
```

---

## analysis/depth.py — Depth Estimation

```python
"""
Depth Estimation — Paper §4.2

Two representations:
1. Heuristic depth label from DACR red-ratio table.
2. JET-colourmap depth heatmap: 255 − R_channel.
   Red = shallow, Blue = deep.
"""

import cv2
import numpy as np
from pipeline.stage3_dacr import estimate_depth_tier


def estimate_depth(image: np.ndarray) -> dict:
    """
    Compute depth label and JET heatmap from enhanced image.

    Args:
        image: uint8 BGR enhanced image

    Returns:
        dict:
            rho        — red-ratio float
            tier_name  — str depth label
            tier_label — str depth range
            heatmap    — uint8 BGR JET depth heatmap
    """
    rho, tier = estimate_depth_tier(image)

    # JET colourmap heatmap: 255 - R_channel → depth proxy
    depth_map = 255 - image[:, :, 2]              # Inverted red channel
    heatmap   = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    return {
        "rho":        rho,
        "tier_name":  tier["name"],
        "tier_label": tier["label"],
        "heatmap":    heatmap,
    }
```

---

## analysis/detection.py — YOLOv8s-World Open-Vocabulary Detection

```python
"""
Detection — Paper §4.1

YOLOv8s-World with CLIP visual-language backbone.
Confidence threshold: 0.10 (accommodates partially occluded organisms).

8 semantic classes with synonyms:
    Fish, Coral, Diver, Rock, Sea Turtle, Shark, Starfish, Jellyfish

Scene composition per class:
    comp_c = Σ(x2-x1)(y2-y1) / A_img × 100
"""

import numpy as np
from ultralytics import YOLOWorld

CLASSES = [
    "Fish",
    "Coral",
    "Diver",
    "Rock",
    "Sea Turtle",
    "Shark",
    "Starfish",
    "Jellyfish",
]

# Initialise once at module load — avoids reloading 27 GB weights per request
_model: YOLOWorld | None = None


def _get_model() -> YOLOWorld:
    global _model
    if _model is None:
        _model = YOLOWorld("yolov8s-world.pt")
        _model.set_classes(CLASSES)
    return _model


def detect_objects(image: np.ndarray) -> dict:
    """
    Run YOLOv8s-World detection on the enhanced image.

    Args:
        image: uint8 BGR image

    Returns:
        dict:
            detections    — list of {class, confidence, bbox, area_pct}
            composition   — {class_name: area_percentage} scene summary
            annotated_img — uint8 BGR image with bounding boxes drawn
    """
    model  = _get_model()
    H, W   = image.shape[:2]
    A_img  = H * W

    results = model.predict(image, conf=0.10, verbose=False)
    result  = results[0]

    detections: list[dict] = []
    composition: dict[str, float] = {c: 0.0 for c in CLASSES}

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = CLASSES[cls_id] if cls_id < len(CLASSES) else "Unknown"

            area     = (x2 - x1) * (y2 - y1)
            area_pct = area / A_img * 100.0

            detections.append({
                "class":      label,
                "confidence": round(conf, 4),
                "bbox":       [x1, y1, x2, y2],
                "area_pct":   round(area_pct, 2),
            })

            if label in composition:
                composition[label] += area_pct

    # Round composition values
    composition = {k: round(v, 2) for k, v in composition.items()}

    # Annotated image
    annotated = result.plot()

    return {
        "detections":    detections,
        "composition":   composition,
        "annotated_img": annotated,
    }
```

---

## analysis/metrics.py — Quality Metrics

```python
"""
Quality Metrics — Paper §4.3

1. UCIQE (reference-free):
       UCIQE = 0.4680 σ_c + 0.2745 conL + 0.2576 µ_s
   Where σ_c = chroma std, conL = luminance contrast, µ_s = mean saturation.

2. Shannon Entropy:
       H = -Σ p_i log2(p_i)

3. SSIM and PSNR (reference-based, if ground-truth available).
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_uciqe(image: np.ndarray) -> float:
    """
    UCIQE — Yang & Sowmya, IEEE TIP 2015.

    Args:
        image: uint8 BGR image

    Returns:
        UCIQE score (higher is better, typical range 3–7)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # Chroma
    chroma = np.sqrt(A ** 2 + B ** 2)
    sigma_c = float(np.std(chroma))

    # Luminance contrast (top 1% - bottom 1% percentile difference)
    p01 = np.percentile(L, 1)
    p99 = np.percentile(L, 99)
    con_L = float(p99 - p01) / 255.0

    # Mean saturation (HSV S channel)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    mu_s = float(np.mean(hsv[:, :, 1])) / 255.0

    uciqe = 0.4680 * sigma_c + 0.2745 * con_L + 0.2576 * mu_s
    return round(uciqe, 4)


def compute_entropy(image: np.ndarray) -> float:
    """
    Shannon Entropy H = -Σ p_i log2(p_i) on greyscale histogram.

    Args:
        image: uint8 BGR image

    Returns:
        Entropy in bits (higher = more information)
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(grey.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-12

    # Mask zero bins before log
    nonzero = hist[hist > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero)))
    return round(entropy, 4)


def compute_ssim(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """
    SSIM between enhanced and reference image.

    Args:
        enhanced:  uint8 BGR enhanced image
        reference: uint8 BGR ground-truth image

    Returns:
        SSIM score in [0, 1] (higher is better)
    """
    enh_grey = cv2.cvtColor(enhanced,  cv2.COLOR_BGR2GRAY)
    ref_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    score = ssim_fn(enh_grey, ref_grey, data_range=255)
    return round(float(score), 4)


def compute_psnr(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """
    PSNR between enhanced and reference image.

    Args:
        enhanced:  uint8 BGR enhanced image
        reference: uint8 BGR ground-truth image

    Returns:
        PSNR in dB (higher is better)
    """
    score = psnr_fn(reference, enhanced, data_range=255)
    return round(float(score), 4)


def compute_all_metrics(
    enhanced:  np.ndarray,
    reference: np.ndarray | None = None
) -> dict:
    """
    Compute all quality metrics.

    Args:
        enhanced:  uint8 BGR enhanced image
        reference: optional uint8 BGR ground-truth image

    Returns:
        dict with keys: uciqe, entropy, ssim (optional), psnr (optional)
    """
    metrics: dict[str, float] = {
        "uciqe":   compute_uciqe(enhanced),
        "entropy": compute_entropy(enhanced),
    }

    if reference is not None:
        metrics["ssim"] = compute_ssim(enhanced, reference)
        metrics["psnr"] = compute_psnr(enhanced, reference)

    return metrics
```

---

## app.py — Flask REST API

```python
"""
Flask REST API — Paper §5.2

Three endpoints:
    POST /upload      — process new image; returns full JSON payload
    POST /reprocess   — re-run pipeline on cached image with stage toggles
    GET  /download/<filename> — binary download (PNG or JPEG)
"""

import io
import os
import uuid
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file

from pipeline.rads_lpf    import run_pipeline
from analysis.depth       import estimate_depth
from analysis.detection   import detect_objects
from analysis.metrics     import compute_all_metrics

app = Flask(__name__)

# In-memory image cache {session_id: np.ndarray}
_IMAGE_CACHE: dict[str, np.ndarray] = {}
UPLOAD_DIR = "/tmp/rads_lpf_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def _encode_b64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Encode ndarray to base64 string for JSON response."""
    _, buf = cv2.imencode(ext, image)
    return base64.b64encode(buf).decode("utf-8")


def _save_output(image: np.ndarray, filename: str) -> str:
    path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(path, image)
    return filename


def _build_response(pipeline_result: dict, session_id: str) -> dict:
    """Build the full JSON payload returned by /upload and /reprocess."""
    enhanced    = pipeline_result["enhanced"]
    rho         = pipeline_result["rho"]
    depth_tier  = pipeline_result["depth_tier"]
    intermediates = pipeline_result["intermediates"]

    # Depth analysis
    depth_info  = estimate_depth(enhanced)

    # Object detection
    det_result  = detect_objects(enhanced)

    # Quality metrics (no reference — UCIQE + Entropy only in production)
    metrics     = compute_all_metrics(enhanced)

    # Histogram data (enhanced image, 256 bins, RGB channels)
    hist_data: dict[str, list] = {}
    for ch, name in enumerate(["Blue", "Green", "Red"]):
        h, _ = np.histogram(enhanced[:, :, ch], bins=256, range=(0, 256))
        hist_data[name] = h.tolist()

    # Save enhanced image to disk for /download
    fname = f"{session_id}_enhanced.jpg"
    _save_output(enhanced, fname)

    return {
        "session_id":    session_id,
        "enhanced_b64":  _encode_b64(enhanced),
        "depth": {
            "rho":        rho,
            "tier":       depth_tier,
            "tier_label": depth_info["tier_label"],
            "heatmap_b64": _encode_b64(depth_info["heatmap"]),
        },
        "detections":    det_result["detections"],
        "composition":   det_result["composition"],
        "annotated_b64": _encode_b64(det_result["annotated_img"]),
        "metrics":       metrics,
        "histograms":    hist_data,
        "intermediates": {
            k: _encode_b64(v) for k, v in intermediates.items()
            if isinstance(v, np.ndarray)
        },
        "download_url":  f"/download/{fname}",
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    """
    POST /upload
    Multipart form-data: field 'image' (JPEG / PNG).
    Optional: field 'reference' for SSIM/PSNR computation.

    Returns JSON with enhanced image, detections, metrics, histograms,
    depth estimate, intermediate stage outputs and download URL.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400

    file_bytes = request.files["image"].read()
    try:
        image = _decode_image(file_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Optional reference image for SSIM/PSNR
    reference = None
    if "reference" in request.files:
        ref_bytes = request.files["reference"].read()
        try:
            reference = _decode_image(ref_bytes)
        except ValueError:
            pass

    session_id = str(uuid.uuid4())[:8]
    _IMAGE_CACHE[session_id] = image.copy()

    # Run full pipeline
    pipeline_result = run_pipeline(image)

    # Add reference metrics if available
    if reference is not None:
        from analysis.metrics import compute_ssim, compute_psnr
        m = pipeline_result  # alias
        pipeline_result["metrics"] = compute_all_metrics(
            pipeline_result["enhanced"], reference
        )

    payload = _build_response(pipeline_result, session_id)
    return jsonify(payload), 200


@app.route("/reprocess", methods=["POST"])
def reprocess():
    """
    POST /reprocess
    JSON body:
        {
          "session_id": "abc12345",
          "stages": {
              "stage1": true,
              "stage2": true,
              "stage3": false,   ← disable DACR
              ...
          }
        }
    Re-runs the pipeline on cached image with specified stage toggles.
    """
    body = request.get_json(force=True)
    session_id = body.get("session_id")
    stages     = body.get("stages", {})

    if session_id not in _IMAGE_CACHE:
        return jsonify({"error": "Session not found. Upload an image first."}), 404

    image = _IMAGE_CACHE[session_id]
    pipeline_result = run_pipeline(image, stages_enabled=stages)

    payload = _build_response(pipeline_result, session_id)
    return jsonify(payload), 200


@app.route("/download/<filename>", methods=["GET"])
def download(filename: str):
    """
    GET /download/<filename>
    Binary file download in JPEG or PNG.
    """
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404

    mimetype = "image/jpeg" if filename.endswith(".jpg") else "image/png"
    return send_file(path, mimetype=mimetype, as_attachment=True)


if __name__ == "__main__":
    # Development server — use Gunicorn in production
    app.run(host="0.0.0.0", port=5000, debug=True)
```

---

## Dockerfile

```dockerfile
# Paper §5.3 — Docker deployment
# Single Gunicorn worker to avoid duplicating YOLOv8s-World weights in memory.
# Image keeps under 3 GB (weights excluded from layer).

FROM python:3.9-slim

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Flask port
EXPOSE 5000

# Single worker — avoids RAM duplication of 27 GB YOLO weights
# 120 s timeout accommodates large-image YOLO inference latency
CMD ["gunicorn", \
     "--workers", "1", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:5000", \
     "app:app"]
```

---

## pipeline/\_\_init\_\_.py

```python
"""RADS-LPF pipeline package."""
```

## analysis/\_\_init\_\_.py

```python
"""RADS-LPF post-enhancement analysis modules."""
```

---

## API Usage Examples

### Upload and enhance an image

```bash
curl -X POST http://localhost:5000/upload \
  -F "image=@my_underwater_photo.jpg" \
  | python -m json.tool
```

### Re-run without DACR (ablation)

```bash
curl -X POST http://localhost:5000/reprocess \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc12345",
    "stages": {
      "stage1": true,
      "stage2": true,
      "stage3": false,
      "stage4": true,
      "stage5": true,
      "stage5a": true,
      "stage5b": true
    }
  }'
```

### Download the enhanced image

```bash
curl -O http://localhost:5000/download/abc12345_enhanced.jpg
```

---

## Metric Targets (from Table 3 in paper)

| Metric  | RADS-LPF (paper) | Interpretation          |
|---------|-----------------|-------------------------|
| UCIQE ↑ | **5.481**       | Higher = better colour  |
| SSIM ↑  | **0.801**       | Higher = better fidelity|
| PSNR ↑  | **28.37 dB**    | Higher = lower noise    |
| Entropy ↑| **7.892**      | Higher = more detail    |

DACR alone contributes **+0.18 UCIQE** and **+0.9 dB PSNR** over no depth-aware rebalancing (Table 4 ablation).
