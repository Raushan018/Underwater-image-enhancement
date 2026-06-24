"""
Depth Estimation — Paper §4.2

Two representations:
1. Heuristic depth label from DACR red-ratio table.
2. JET-colourmap depth heatmap: 255 − R_channel.
   Red = shallow, Blue = deep.
"""

import cv2
import numpy as np
from pipeline.stage3_dacr import estimate_depth_tier


def estimate_depth(image: np.ndarray) -> dict:
    """
    Compute depth label and JET heatmap from enhanced image.

    Args:
        image: uint8 BGR enhanced image

    Returns:
        dict:
            rho        — red-ratio float
            tier_name  — str depth label
            tier_label — str depth range
            heatmap    — uint8 BGR JET depth heatmap
    """
    rho, tier = estimate_depth_tier(image)

    # JET colourmap heatmap: 255 - R_channel → depth proxy
    depth_map = 255 - image[:, :, 2]              # Inverted red channel
    heatmap   = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    return {
        "rho":        rho,
        "tier_name":  tier["name"],
        "tier_label": tier["label"],
        "heatmap":    heatmap,
    }
