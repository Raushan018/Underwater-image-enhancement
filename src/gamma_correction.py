import numpy as np
import cv2

def apply_gamma_correction(image, gamma=1.2):
    """
    Applies gamma correction to improve global contrast.
    
    Args:
        image: Input image (BGR, uint8).
        gamma: Gamma value (default 1.2, >1 brightens, <1 darkens).
        
    Returns:
        Gamma corrected image.
    """
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
                      
    return cv2.LUT(image, table)
