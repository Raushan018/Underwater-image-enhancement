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
