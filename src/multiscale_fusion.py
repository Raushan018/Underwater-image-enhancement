import cv2
import numpy as np

def gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def laplacian_pyramid(image, levels):
    gaussian_pyr = gaussian_pyramid(image, levels)
    laplacian_pyr = []
    
    for i in range(levels - 1):
        expanded = cv2.pyrUp(gaussian_pyr[i+1], dstsize=(gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        lapacian = cv2.subtract(gaussian_pyr[i], expanded)
        laplacian_pyr.append(lapacian)
        
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

def collapse_pyramid(laplacian_pyr):
    image = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        expanded = cv2.pyrUp(image, dstsize=(laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0]))
        image = cv2.add(expanded, laplacian_pyr[i])
    return image

def apply_fusion(img1, img2, w1, w2, levels=5):
    """
    Performs Laplacian Pyramid Fusion.
    
    Args:
        img1: Input image 1 (e.g., CLAHE output).
        img2: Input image 2 (e.g., Histogram Linearized output).
        w1: Weight map for img1.
        w2: Weight map for img2.
        levels: Number of pyramid levels.
        
    Returns:
        Fused image.
    """
    # Normalize weights
    sum_w = cv2.add(w1, w2)
    # Avoid division by zero
    sum_w[sum_w == 0] = 1e-5
    
    w1_norm = cv2.divide(w1, sum_w)
    w2_norm = cv2.divide(w2, sum_w)
    
    # Convert images to float64 for precision during fusion
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 1. Generate Gaussian Pyramids for normalized weights
    w1_pyr = gaussian_pyramid(w1_norm, levels)
    w2_pyr = gaussian_pyramid(w2_norm, levels)
    
    # 2. Generate Laplacian Pyramids for input images
    img1_pyr = laplacian_pyramid(img1, levels)
    img2_pyr = laplacian_pyramid(img2, levels)
    
    # 3. Fuse pyramids
    # R_l(x) = sum(G_l(W_k) * L_l(I_k))
    fused_pyr = []
    for i in range(levels):
        # We need to replicate weights for 3 channels to multiply with color images
        w1_level = cv2.merge([w1_pyr[i], w1_pyr[i], w1_pyr[i]])
        w2_level = cv2.merge([w2_pyr[i], w2_pyr[i], w2_pyr[i]])
        
        # Calculate fused level
        fused_level = (w1_level * img1_pyr[i]) + (w2_level * img2_pyr[i])
        fused_pyr.append(fused_level)
        
    # 4. Reconstruct image
    fused_image = collapse_pyramid(fused_pyr)
    
    # Clip and convert
    fused_image = np.clip(fused_image, 0, 255)
    return fused_image.astype(np.uint8)
