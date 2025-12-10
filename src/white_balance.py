import numpy as np
import cv2

def apply_white_balance(image, alpha=1.0):
    """
    Applies white balancing using Red/Blue channel compensation and Grey-World algorithm.
    
    Args:
        image: Input image (BGR format, 0-255).
        alpha: Compensation factor (default 1.0).
        
    Returns:
        White-balanced image.
    """
    image = image.astype(np.float64) / 255.0
    b, g, r = cv2.split(image)
    
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    
    # Red Channel Compensation
    # I_rc(x) = I_r(x) + alpha * (I_g_bar - I_r_bar) * (1 - I_r(x)) * I_g(x)
    r_c = r + alpha * (mean_g - mean_r) * (1 - r) * g
    
    # Blue Channel Compensation
    # I_bc(x) = I_b(x) + alpha * (I_g_bar - I_b_bar) * (1 - I_b(x)) * I_g(x)
    # Note: The prompt explicitly asked for "Red and Blue channel compensation as described".
    # Assuming the symmetric formula for Blue channel using Green as reference.
    b_c = b + alpha * (mean_g - mean_b) * (1 - b) * g
    
    # Clip values to [0, 1]
    r_c = np.clip(r_c, 0, 1)
    b_c = np.clip(b_c, 0, 1)
    
    # Re-merge to apply Grey-World
    compensated = cv2.merge([b_c, g, r_c])
    
    # Grey-World Algorithm
    # Scale each channel so that their means are equal to the mean of the Green channel 
    # (or the gray value, but aligning to Green is common in underwater to preserve the dominant channel).
    # Alternatively, standard Grey-World averages all channels to a common gray value.
    # The prompt says "Apply Grey-World Algorithm after compensation."
    
    # Let's use the standard "Illuminant normalization" version of Grey-World:
    # Scale R, G, B such that average of each is equal to the average of the gray channel (0.5 or sum/3).
    
    c_b, c_g, c_r = cv2.split(compensated)
    c_mean_r = np.mean(c_r)
    c_mean_g = np.mean(c_g)
    c_mean_b = np.mean(c_b)
    
    # Target mean (Gray value). Often 0.5 or the average of the means.
    target_mean = (c_mean_r + c_mean_g + c_mean_b) / 3.0
    
    # Avoid division by zero
    c_mean_r = c_mean_r if c_mean_r > 0 else 1.0
    c_mean_g = c_mean_g if c_mean_g > 0 else 1.0
    c_mean_b = c_mean_b if c_mean_b > 0 else 1.0
    
    gw_r = c_r * (target_mean / c_mean_r)
    gw_g = c_g * (target_mean / c_mean_g)
    gw_b = c_b * (target_mean / c_mean_b)
    
    gw_out = cv2.merge([gw_b, gw_g, gw_r])
    gw_out = np.clip(gw_out, 0, 1)
    
    return (gw_out * 255).astype(np.uint8)
