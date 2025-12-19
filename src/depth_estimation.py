import numpy as np
import cv2

class DepthEstimator:
    def __init__(self):
        # Heuristics for depth estimation based on Red channel attenuation
        # Red light is absorbed first (approx 5m), then Green, then Blue.
        pass

    def estimate_depth(self, image):
        """
        Estimates the depth based on color channel ratios.

        Args:
            image: BGR image (numpy array)

        Returns:
            dict: { 'depth_range': str, 'confidence': float }
        """
        if image is None:
            return {'depth_range': 'Unknown', 'confidence': 0.0}

        # Calculate mean channel values
        b, g, r = cv2.split(image)
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)
        total_intensity = mean_r + mean_g + mean_b + 1e-6

        # Calculate ratios
        r_ratio = mean_r / total_intensity
        g_ratio = mean_g / total_intensity
        b_ratio = mean_b / total_intensity
        
        # Max channel
        max_channel = max(mean_r, mean_g, mean_b)
        
        # Heuristic Logic
        # 1. Shallow (< 5m): Red is still significant.
        # 2. Medium (5-15m): Red is attenuated, Green/Blue dominate.
        # 3. Deep (> 15m): Mostly Blue/Green, Red is minimal.
        
        depth_range = "Unknown"
        confidence = 0.5
        
        # Metrics to determine confidence (variance/spread)
        # Using simple distance from thresholds for confidence
        
        if r_ratio > 0.25 and mean_r > 50:
            depth_range = "0–5 meters"
            # Higher red ratio = higher confidence for shallow
            confidence = min(0.95, 0.6 + (r_ratio - 0.25) * 2) 
        elif r_ratio > 0.15:
            depth_range = "5–15 meters"
            # Centered around 0.20?
            dist = 1.0 - abs(r_ratio - 0.20) * 10 # Rough bell
            confidence = max(0.6, min(0.9, dist))
        else:
            # Deep water
            # Distinguish 15-30m vs >30m based on Red almost zero and Blue dominance
            if r_ratio < 0.05 and b_ratio > 0.5:
                 depth_range = "> 30 meters"
                 confidence = min(0.95, 0.7 + (b_ratio - 0.5) * 2)
            else:
                 depth_range = "15–30 meters"
                 confidence = 0.75
        
        return {
            'depth_range': depth_range,
            'confidence': round(confidence, 2)
        }

    def generate_depth_map(self, image):
        """
        Generates a visualization of relative depth.
        Heuristic: Red channel intensity is inversely proportional to depth.
        
        Args:
            image: BGR image
            
        Returns:
            dict: {
                'raw_depth': numpy array (grayscale, 0=Close, 255=Far),
                'heatmap': numpy array (BGR colorized, Red=Close, Blue=Far)
            }
        """
        if image is None:
            return None
            
        # Ensure input is uint8 for bitwise_not and applyColorMap
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Extract Red Channel
        b, g, r = cv2.split(image)
        
        # Smooth the red channel to reduce noise in the depth map
        r_blur = cv2.GaussianBlur(r, (15, 15), 0)
        
        # 1. Raw Depth Map (Distance)
        # Red is high for close objects, low for far objects.
        # Depth/Distance is the inverse.
        # 255 (Max Distance) = 0 Red
        # 0 (Min Distance) = 255 Red
        raw_depth = cv2.bitwise_not(r_blur)
        
        # 2. Visualization (Heatmap)
        # User wants: Red = Near, Blue = Far.
        # Standard cv2.COLORMAP_JET: Low(0)=Blue, High(255)=Red.
        # So we want High Values to be "Near".
        # We can pass the 'r_blur' (Proximity Map) directly to JET.
        # High Red(255) -> Near -> Jet(255) -> Red Color.
        # Low Red(0) -> Far -> Jet(0) -> Blue Color.
        heatmap = cv2.applyColorMap(r_blur, cv2.COLORMAP_JET)
        
        return {
            'raw_depth': raw_depth,
            'heatmap': heatmap
        }
