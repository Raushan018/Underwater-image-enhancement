import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the Luminance channel.
    
    Args:
        image: Input image (BGR, uint8).
        
    Returns:
        Luminance enhanced image.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    # Merge channels
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convert back to BGR
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result
