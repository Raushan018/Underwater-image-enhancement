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
