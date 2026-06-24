"""
RADS-LPF Master Pipeline
Orchestrates all 7 stages as a single joint optimisation graph.

Stage flow (Fig. 1 in paper):
    Input
    → [1] Adaptive Histogram Stretching
    → [2] LAB Colour Correction + HSV Sat Boost
    → [3] DACR — Depth-Aware Channel Rebalancing  ★
    → [4] Auto-Gamma via LAB Luminance LUT
    → [5] Bilateral Edge-Preserving Denoising
         → [5A] CLAHE         [5B] Histogram Linearisation
    → [6] Adaptive Weight Map Generation
    → [7] 5-Level Laplacian Pyramid Fusion
    → Enhanced Output
"""

from __future__ import annotations

import numpy as np
import cv2

from .stage1_stretch  import adaptive_histogram_stretch
from .stage2_colour   import lab_colour_correction
from .stage3_dacr     import depth_aware_channel_rebalancing
from .stage4_gamma    import auto_gamma_correction
from .stage5_denoise  import bilateral_denoise
from .stage5a_clahe   import apply_clahe
from .stage5b_histeq  import apply_histogram_equalisation
from .stage6_weights  import compute_normalised_weight_maps
from .stage7_fusion   import laplacian_pyramid_fusion


def run_pipeline(
    image: np.ndarray,
    stages_enabled: dict | None = None
) -> dict:
    """
    Execute the full RADS-LPF pipeline.

    Args:
        image:          uint8 BGR input image
        stages_enabled: optional dict of {stage_name: bool} for
                        /reprocess endpoint toggles. All stages
                        enabled by default.

    Returns:
        dict with keys:
            enhanced       — final fused uint8 BGR image
            intermediates  — dict of intermediate stage outputs
            rho            — red-ratio depth proxy (float)
            depth_tier     — depth tier name (str)
    """
    if stages_enabled is None:
        stages_enabled = {}

    def _enabled(name: str) -> bool:
        return stages_enabled.get(name, True)

    intermediates: dict[str, np.ndarray] = {}
    intermediates["input"] = image.copy()

    # Stage 1 — Adaptive Histogram Stretching
    out = adaptive_histogram_stretch(image) if _enabled("stage1") else image.copy()
    intermediates["stage1_stretch"] = out.copy()

    # Stage 2 — LAB Colour Correction + HSV Sat Boost
    out = lab_colour_correction(out) if _enabled("stage2") else out
    intermediates["stage2_colour"] = out.copy()

    # Stage 3 — DACR  ★
    if _enabled("stage3"):
        out, rho, depth_tier = depth_aware_channel_rebalancing(out)
    else:
        rho, depth_tier = 0.0, "unknown"
    intermediates["stage3_dacr"] = out.copy()

    # Stage 4 — Auto-Gamma
    out = auto_gamma_correction(out) if _enabled("stage4") else out
    intermediates["stage4_gamma"] = out.copy()

    # Stage 5 — Bilateral Denoising
    out = bilateral_denoise(out) if _enabled("stage5") else out
    intermediates["stage5_denoise"] = out.copy()

    # Stages 5A / 5B — Dual Enhancement Branches
    img_clahe = apply_clahe(out)          if _enabled("stage5a") else out.copy()
    img_he    = apply_histogram_equalisation(out) if _enabled("stage5b") else out.copy()
    intermediates["stage5a_clahe"] = img_clahe.copy()
    intermediates["stage5b_he"]    = img_he.copy()

    # Stage 6 — Adaptive Weight Maps
    w_clahe, w_he = compute_normalised_weight_maps(img_clahe, img_he)
    intermediates["stage6_weight_clahe"] = (w_clahe * 255).astype(np.uint8)
    intermediates["stage6_weight_he"]    = (w_he    * 255).astype(np.uint8)

    # Stage 7 — Laplacian Pyramid Fusion
    enhanced = laplacian_pyramid_fusion(img_clahe, img_he, w_clahe, w_he)
    intermediates["stage7_enhanced"] = enhanced.copy()

    return {
        "enhanced":      enhanced,
        "intermediates": intermediates,
        "rho":           rho,
        "depth_tier":    depth_tier,
    }
