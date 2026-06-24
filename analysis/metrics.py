"""
Quality Metrics — Paper §4.3

1. UCIQE (reference-free):
       UCIQE = 0.4680 σ_c + 0.2745 conL + 0.2576 µ_s
   Where σ_c = chroma std, conL = luminance contrast, µ_s = mean saturation.

2. Shannon Entropy:
       H = -Σ p_i log2(p_i)

3. SSIM and PSNR (reference-based, if ground-truth available).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_uciqe(image: np.ndarray) -> float:
    """
    UCIQE — Yang & Sowmya, IEEE TIP 2015.

    Args:
        image: uint8 BGR image

    Returns:
        UCIQE score (higher is better, typical range 3–7)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # Chroma
    chroma = np.sqrt(A ** 2 + B ** 2)
    sigma_c = float(np.std(chroma))

    # Luminance contrast (top 1% - bottom 1% percentile difference)
    p01 = np.percentile(L, 1)
    p99 = np.percentile(L, 99)
    con_L = float(p99 - p01) / 255.0

    # Mean saturation (HSV S channel)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    mu_s = float(np.mean(hsv[:, :, 1])) / 255.0

    uciqe = 0.4680 * sigma_c + 0.2745 * con_L + 0.2576 * mu_s
    return round(uciqe, 4)


def compute_entropy(image: np.ndarray) -> float:
    """
    Shannon Entropy H = -Σ p_i log2(p_i) on greyscale histogram.

    Args:
        image: uint8 BGR image

    Returns:
        Entropy in bits (higher = more information)
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(grey.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-12

    # Mask zero bins before log
    nonzero = hist[hist > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero)))
    return round(entropy, 4)


def compute_ssim(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """
    SSIM between enhanced and reference image.

    Args:
        enhanced:  uint8 BGR enhanced image
        reference: uint8 BGR ground-truth image

    Returns:
        SSIM score in [0, 1] (higher is better)
    """
    enh_grey = cv2.cvtColor(enhanced,  cv2.COLOR_BGR2GRAY)
    ref_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    score = ssim_fn(enh_grey, ref_grey, data_range=255)
    return round(float(score), 4)


def compute_psnr(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """
    PSNR between enhanced and reference image.

    Args:
        enhanced:  uint8 BGR enhanced image
        reference: uint8 BGR ground-truth image

    Returns:
        PSNR in dB (higher is better)
    """
    score = psnr_fn(reference, enhanced, data_range=255)
    return round(float(score), 4)


def compute_all_metrics(
    enhanced:  np.ndarray,
    reference: np.ndarray | None = None
) -> dict:
    """
    Compute all quality metrics.

    Args:
        enhanced:  uint8 BGR enhanced image
        reference: optional uint8 BGR ground-truth image

    Returns:
        dict with keys: uciqe, entropy, ssim (optional), psnr (optional)
    """
    metrics: dict[str, float] = {
        "uciqe":   compute_uciqe(enhanced),
        "entropy": compute_entropy(enhanced),
    }

    if reference is not None:
        metrics["ssim"] = compute_ssim(enhanced, reference)
        metrics["psnr"] = compute_psnr(enhanced, reference)

    return metrics
