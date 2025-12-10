import cv2
import numpy as np

def calculate_histogram(image):
    """
    Calculates RGB and Luminance histograms for an image.
    Args:
        image: BGR numpy array (OpenCV format)
    Returns:
        Dictionary with 'r', 'g', 'b', 'y' (luminance) lists of values.
    """
    if image is None:
        return {}

    # Calculate histograms for B, G, R
    # cv2.calcHist(images, channels, mask, histSize, ranges)
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten().tolist()
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten().tolist()
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten().tolist()

    # Calculate Luminance (Y)
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hist_y = cv2.calcHist([ycrcb], [0], None, [256], [0, 256]).flatten().tolist()

    # Normalize? 
    # To display nicely in charts likely unrelated to image size, min-max normalization or percentage fits well.
    # However, raw counts are also fine if Y-axis is auto.
    # Let's normalize to 0-1 range for easier comparison between different resolution images if needed?
    # No, keep raw counts but maybe cap them if there's a huge spike at 0 or 255?
    # Actually, for visualization, raw counts are honest. Chart.js handles scaling.
    
    return {
        'b': hist_b,
        'g': hist_g,
        'r': hist_r,
        'y': hist_y
    }
