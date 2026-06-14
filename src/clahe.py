"""
clahe.py
Stage 5A — CLAHE (Contrast Limited Adaptive Histogram Equalization)

Enhances local contrast without blowing out highlights.
Operates ONLY on the L (luminance) channel of LAB to avoid hue shifts.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the luminance channel only.

    Why LAB L-channel only:
      Applying contrast enhancement to BGR channels separately causes
      colour shifts. Working in LAB separates brightness (L) from colour
      (A, B) so we can stretch contrast without altering hue.

    clip_limit: redistribution ceiling per tile. Values above this are
                clipped and redistributed uniformly across the histogram.
                Lower = less noise amplification. 2.0 is standard.

    tile_grid_size: image is divided into this grid. Each tile gets its
                    own histogram equalized independently, then tiles are
                    blended using bilinear interpolation.
    """
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_ch)

    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    logger.debug(f"CLAHE applied: clip={clip_limit}, tiles={tile_grid_size}")
    return result
