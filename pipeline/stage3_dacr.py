"""
Stage 3: Depth-Aware Channel Rebalancing (DACR)  ★ Novel contribution
Paper §3.5

Red-ratio depth proxy:
    ρ = µ_R / (µ_R + µ_G + µ_B)

Depth-tier gain table (Table 1 in paper):
    Shallow  ρ > 0.25        GR=1.00  GG=1.00  GB=1.00
    Mid      0.15–0.25       GR=1.25  GG=1.05  GB=0.95
    Deep     0.05–0.15       GR=1.60  GG=1.15  GB=0.90
    Abyss    ρ ≤ 0.05        GR=2.00  GG=1.30  GB=0.85
"""

import cv2
import numpy as np
from typing import Tuple, Dict

# Depth-tier gain table — learned empirically on 500 annotated UIEB frames
DEPTH_TIERS: list[dict] = [
    {"name": "shallow", "label": "0–5 m",   "rho_min": 0.25, "GR": 1.00, "GG": 1.00, "GB": 1.00},
    {"name": "mid",     "label": "5–15 m",  "rho_min": 0.15, "GR": 1.25, "GG": 1.05, "GB": 0.95},
    {"name": "deep",    "label": "15–30 m", "rho_min": 0.05, "GR": 1.60, "GG": 1.15, "GB": 0.90},
    {"name": "abyss",   "label": ">30 m",   "rho_min": 0.00, "GR": 2.00, "GG": 1.30, "GB": 0.85},
]


def estimate_depth_tier(image: np.ndarray) -> Tuple[float, Dict]:
    """
    Compute red-ratio ρ and select the matching depth tier.

    Args:
        image: uint8 BGR image

    Returns:
        (rho, tier_dict) where tier_dict has keys: name, label, GR, GG, GB
    """
    img_f = image.astype(np.float32)
    mu_B = np.mean(img_f[:, :, 0])
    mu_G = np.mean(img_f[:, :, 1])
    mu_R = np.mean(img_f[:, :, 2])

    total = mu_R + mu_G + mu_B
    rho = mu_R / total if total > 1e-6 else 0.0

    for tier in DEPTH_TIERS:
        if rho > tier["rho_min"]:
            return rho, tier

    return rho, DEPTH_TIERS[-1]   # abyss fallback


def depth_aware_channel_rebalancing(image: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """
    Apply depth-contingent per-channel gain factors.
    Gains are applied to both LAB L-channel and per-channel BGR values.

    Args:
        image: uint8 BGR image (after Stage 2)

    Returns:
        (rebalanced_image, rho, depth_tier_name)
    """
    rho, tier = estimate_depth_tier(image)
    GR, GG, GB = tier["GR"], tier["GG"], tier["GB"]

    # --- BGR per-channel gain ---
    img_f = image.astype(np.float32)
    img_f[:, :, 0] = np.clip(img_f[:, :, 0] * GB, 0, 255)   # Blue
    img_f[:, :, 1] = np.clip(img_f[:, :, 1] * GG, 0, 255)   # Green
    img_f[:, :, 2] = np.clip(img_f[:, :, 2] * GR, 0, 255)   # Red

    # --- LAB L-channel gain (weighted average of BGR gains → luminance) ---
    intermediate = img_f.astype(np.uint8)
    lab = cv2.cvtColor(intermediate, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B_ch = cv2.split(lab)

    lum_gain = (GR + GG + GB) / 3.0
    L = np.clip(L * lum_gain, 0, 255)

    lab_out = cv2.merge([L, A, B_ch]).astype(np.uint8)
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    return result, rho, tier["name"]
