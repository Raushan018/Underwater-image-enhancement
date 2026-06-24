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
