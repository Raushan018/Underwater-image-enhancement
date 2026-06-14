"""
adaptive_enhancement.py
Stages 1-4 of the underwater image enhancement pipeline.
Stage 1: Adaptive histogram stretching
Stage 2: Adaptive colour correction (white balance) - float32 safe
Stage 3: Adaptive gamma correction via luminance LUT
Stage 4: Edge-preserving bilateral denoising
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — Adaptive Histogram Stretching
# ---------------------------------------------------------------------------
def adaptive_histogram_stretch(image: np.ndarray) -> np.ndarray:
    """
    Expand compressed dynamic range using per-channel percentile stretch.
    Uses 1st/99th percentile to avoid extreme pixels dominating the remap.

    Why: Water scattering compresses all channel values into a narrow band.
    Stretching each channel independently compensates for differential
    wavelength absorption (red absorbed most, blue least).
    """
    if image is None or image.size == 0:
        return image

    result = np.zeros_like(image, dtype=np.float32)

    for ch in range(3):
        channel = image[:, :, ch].astype(np.float32)
        low = float(np.percentile(channel, 1))
        high = float(np.percentile(channel, 99))

        if high - low < 1.0:
            # Nearly uniform channel — leave unchanged
            result[:, :, ch] = channel
        else:
            stretched = (channel - low) * 255.0 / (high - low)
            result[:, :, ch] = np.clip(stretched, 0, 255)

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2 — Adaptive Colour Correction (White Balance)
# ---------------------------------------------------------------------------
def adaptive_color_correction(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Remove blue-green cast caused by water absorption.

    Step A: LAB neutral shift — push A and B channels toward neutral (128).
            Uses float32 arithmetic to prevent silent uint8 clipping.
    Step B: HSV saturation boost — only if image is dull (mean S < 80).

    alpha: strength of LAB correction (0–1). Default 0.5 = 50% correction.
    """
    if image is None or image.size == 0:
        return image

    # --- Step A: LAB neutral shift ---
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Convert to float32 BEFORE arithmetic — prevents uint8 wrapping
    a_f = a_ch.astype(np.float32)
    b_f = b_ch.astype(np.float32)

    # Shift toward neutral (128) by alpha fraction
    a_f = a_f + (128.0 - np.mean(a_f)) * alpha
    b_f = b_f + (128.0 - np.mean(b_f)) * alpha

    # Clip explicitly before converting back
    a_corrected = np.clip(a_f, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b_f, 0, 255).astype(np.uint8)

    lab_corrected = cv2.merge([l_ch, a_corrected, b_corrected])
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    # --- Step B: HSV saturation boost (conditional) ---
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    mean_sat = float(np.mean(s_ch))
    if mean_sat < 80:
        factor = 1.2
        # float32 multiply then clip — prevents uint8 overflow
        s_boosted = np.clip(s_ch.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        hsv_boosted = cv2.merge([h_ch, s_boosted, v_ch])
        result = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    return result


# ---------------------------------------------------------------------------
# Stage 3 — Adaptive Gamma Correction
# ---------------------------------------------------------------------------
def adaptive_gamma_correction(image: np.ndarray) -> np.ndarray:
    """
    Automatically correct brightness based on measured image luminance.

    Formula: gamma = log(0.5) / log(mean_L)
    This targets a mean luminance of 0.5 (mid-tone).

    gamma < 1 → brightens dark images
    gamma > 1 → darkens overexposed images
    Clamped to [0.5, 2.5] to prevent extreme corrections.

    Applied via a 256-entry LUT for speed (avoids per-pixel power op).
    """
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    l_norm = l_ch.astype(np.float32) / 255.0
    mean_l = float(np.mean(l_norm))
    mean_l = np.clip(mean_l, 0.01, 0.99)  # avoid log(0)

    gamma = np.log(0.5) / np.log(mean_l)
    gamma = float(np.clip(gamma, 0.5, 2.5))

    # Build 256-entry LUT
    lut = np.array(
        [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8
    )

    l_corrected = cv2.LUT(l_ch, lut)
    lab_corrected = cv2.merge([l_corrected, a_ch, b_ch])
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    logger.debug(f"Gamma correction: mean_L={mean_l:.3f}, gamma={gamma:.3f}")
    return result


# ---------------------------------------------------------------------------
# Stage 4 — Edge-Preserving Bilateral Denoising
# ---------------------------------------------------------------------------
def edge_preserving_sharpen(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from water particles without blurring structural edges.

    Bilateral filter weights each neighbour by:
      - Spatial closeness (sigmaSpace)
      - Colour similarity (sigmaColor)
    Edges (high colour difference) are NOT smoothed.
    Noise (low colour difference, uniform regions) IS smoothed.

    After denoising, apply mild unsharp masking to recover
    any softness introduced by the bilateral pass.
    """
    if image is None or image.size == 0:
        return image

    # Bilateral denoising
    denoised = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    # Unsharp mask: enhance = original + alpha*(original - blurred)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.3, blurred, -0.3, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full sequential pipeline: Stages 1–4
# ---------------------------------------------------------------------------
def apply_enhancement_pipeline(
    image: np.ndarray,
    do_stretch: bool = True,
    do_wb: bool = True,
    do_gamma: bool = True,
    do_sharp: bool = True,
) -> dict:
    """
    Run stages 1–4 in sequence. Each stage can be toggled independently.
    Returns dict with image after each stage for intermediate display.
    """
    if image is None:
        return {}

    current = image.copy()
    stages = {"original": image.copy()}

    if do_stretch:
        current = adaptive_histogram_stretch(current)

    if do_wb:
        current = adaptive_color_correction(current)
        stages["wb"] = current.copy()

    if do_gamma:
        current = adaptive_gamma_correction(current)
        stages["gamma"] = current.copy()

    if do_sharp:
        current = edge_preserving_sharpen(current)
        stages["sharp"] = current.copy()

    stages["after_stage4"] = current.copy()
    return stages
