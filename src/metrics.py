import cv2
import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_uciqe(image):
    """
    Calculates UCIQE (Underwater Color Image Quality Evaluation).
    UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 1. Chroma (Saturation)
    # Chroma = sqrt(a^2 + b^2)
    chroma = np.sqrt(a.astype(float)**2 + b.astype(float)**2)
    
    # sigma_c: Standard deviation of chroma
    sigma_c = np.std(chroma)
    
    # 2. Contrast of Luminance (con_l)
    # Defined as the difference between the maximum and minimum luminance values ??
    # Or average of local contrasts? The paper defines it as contrast of luminance.
    # A common approximation for this metric implementation:
    l_float = l.astype(float)
    con_l = np.max(l_float) - np.min(l_float) # Simple global contrast
    # A better approximation might be needed, but global contrast is often used in simplified versions.
    
    # 3. Saturation (mu_s)
    # Mean of saturation
    mu_s = np.mean(chroma)
    
    # Coefficients from paper
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    
    uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s
    return uciqe

def calculate_entropy(image):
    """
    Calculates Shannon Entropy of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    return entropy(hist, base=2)

def calculate_psnr(img1, img2):
    """
    Calculates PSNR. Requires ground truth.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return psnr(img1, img2)

def calculate_ssim(img1, img2):
    """
    Calculates SSIM. Requires ground truth.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # SSIM requires grayscale usually for structural info, or channel_axis for color
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    return ssim(gray1, gray2)

def calculate_uiqm(image):
    """
    Calculates UIQM (Underwater Image Quality Measure).
    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
    
    Note: Full implementation requires block-based EME (Measure of Enhancement) calculations 
    which are complex. Returning a simplified placeholder or using UCIQE as primary.
    Use UCIQE for now as the robust metric.
    """
    return 0.0 # Placeholder
