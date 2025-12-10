import cv2
import numpy as np

def apply_sharpening(image, N=1.5):
    """
    Applies Unsharp Masking sharpening.
    S = (I + N(I - G * I)) / 2
    
    Args:
        image: Input image (BGR, uint8).
        N: Strength of sharpening (default 1.5).
        
    Returns:
        Sharpened image.
    """
    # Convert to float for processing
    image_f = image.astype(np.float64) / 255.0
    
    # 3x3 Gaussian Blur
    # The user specified "Use a 3x3 Gaussian kernel."
    # Standard sigma for 3x3 is usually small, e.g., 0.5 to 1.0. 
    # OpenCV's GaussianBlur calculates sigma from kernel size if 0.
    blurred = cv2.GaussianBlur(image_f, (3, 3), 0)
    
    # Unsharp Masking Formula: S = (I + N(I - G*I)) / 2
    # detail = I - blurred
    detail = image_f - blurred
    sharpened = (image_f + N * detail) / 2.0
    
    # Clip and convert back
    sharpened = np.clip(sharpened, 0, 1)
    return (sharpened * 255).astype(np.uint8)
