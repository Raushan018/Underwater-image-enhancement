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
