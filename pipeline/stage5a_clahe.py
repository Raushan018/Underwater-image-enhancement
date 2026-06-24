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
