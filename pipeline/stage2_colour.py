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
