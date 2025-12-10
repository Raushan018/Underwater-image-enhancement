import cv2
import numpy as np

def compute_laplacian_weight(image):
    """
    Computes Laplacian Contrast Weight.
    W_L = |Laplacian(Luminance)|
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.abs(laplacian)

def compute_saliency_weight(image):
    """
    Computes Saliency Weight using Achanta et al. method (Frequency-tuned).
    S = |I_mean - I_gaussian_blur|
    """
    # Convert to LAB for better perceptual difference
    # Or simply use RGB mean. The paper suggests LAB usually.
    # Let's use the simple implementation: || I_mean - I_whc || in Lab space.
    
    gfrgb = cv2.GaussianBlur(image, (3, 3), 3) # "Gaussian smooth"
    lab = cv2.cvtColor(gfrgb, cv2.COLOR_BGR2LAB)
    
    l, a, b = cv2.split(lab)
    
    l_mean = np.mean(l)
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    # Euclidian distance in Lab space
    saliency = np.sqrt(
        (l - l_mean)**2 + 
        (a - a_mean)**2 + 
        (b - b_mean)**2
    )
    
    return saliency

def compute_saturation_weight(image):
    """
    Computes Saturation Weight.
    W_sat = std(R, G, B)
    """
    image_f = image.astype(np.float64)
    r, g, b = cv2.split(image_f)
    
    # Standard deviation across channels at each pixel
    # axis=2 if we had (H,W,3), but here we split.
    # We can just do:
    mean = (r + g + b) / 3.0
    saturation = np.sqrt(
        ((r - mean)**2 + (g - mean)**2 + (b - mean)**2) / 3.0
    )
    
    return saturation

def generate_weight_map(image):
    """
    Generates combined weight map for a single image.
    W = W_L + W_S + W_Sat
    """
    w_l = compute_laplacian_weight(image)
    w_s = compute_saliency_weight(image)
    w_sat = compute_saturation_weight(image)
    
    return w_l + w_s + w_sat
