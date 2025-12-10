import cv2
import numpy as np

def apply_histogram_linearization(image):
    """
    Applies Histogram Linearization (Equalization) to the image.
    Ideally applied to the Intensity/V channel to preserve chroma, but
    the prompt asks to apply it to "sharpened output" which is color.
    
    If applied to each channel independently, it may shift colors.
    However, often 'Histogram Linearization' implies preserving color balance, so we usually apply it to HSV-Value or LAB-L.
    BUT, looking at step 6: "Fuse luminance-enhanced + histogram-linearized images".
    Luminance enhanced is CLAHE (local). Histogram Linearized is global.
    
    Let's apply it to RGB channels independently or Value channel?
    Standard approach for "Histogram Linearization of Color Image" usually means RGB independent or Value.
    
    Given the separate CLAHE path (which handles local contrast), 
    Histogram Linearization here likely acts as the global contrast enhancement path.
    
    Let's apply to RGB independently as a naive interpretation or HSV-V as a safer one.
    Let's stick to HSV-V to avoid color shifts.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply global histogram equalization to V channel
    v_eq = cv2.equalizeHist(v)
    
    hsv_eq = cv2.merge([h, s, v_eq])
    result = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    
    return result
