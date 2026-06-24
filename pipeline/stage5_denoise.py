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
