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
