import cv2
import numpy as np

class AdaptiveEnhancer:
    def __init__(self):
        pass

    def adaptive_histogram_stretching(self, img, clip_percent=1.0):
        """
        Performs adaptive histogram stretching by clipping extreme values.
        Args:
            img: Input image (BGR)
            clip_percent: Percentage of low/high values to clip.
        Returns:
            Stretched image.
        """
        # Split channels
        channels = cv2.split(img)
        out_channels = []

        for channel in channels:
            # Calculate percentiles
            low_val = np.percentile(channel, clip_percent)
            high_val = np.percentile(channel, 100 - clip_percent)

            # Stretch
            # Apply linear stretching: (x - low) * 255 / (high - low)
            # Clip values before scaling to avoid wrapping
            stretched = cv2.normalize(channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Using actual min/max from percentiles for robust stretching
            if high_val > low_val:
                stretched = np.clip((channel - low_val) * (255.0 / (high_val - low_val)), 0, 255).astype(np.uint8)
            else:
                stretched = channel
                
            out_channels.append(stretched)

        return cv2.merge(out_channels)

    def adaptive_color_correction(self, img):
        """
        Applies adaptive color correction in LAB and HSV spaces.
        1. LAB: Gray World assumption adaptation (balancing A and B channels).
        2. HSV: Saturation enhancement.
        """
        # --- LAB Color Correction ---
        result = img.copy()
        result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(result)

        # Simple Gray World assumption: average of a and b should be 128 (neutral)
        # We shift the mean of a and b towards 128
        def adjust_channel(channel):
            mean = np.mean(channel)
            # Adjust so mean becomes 128
            # shift = 128 - mean
            # But naive shifting might clip. Let's use gain.
            # gain = 128 / mean. This assumes 0 is fixed point. 
            # Gray world normally operates on linear RGB. 
            # In LAB, 128 is neutral for A and B.
            return cv2.add(channel, (128 - mean))

        # Only adjust if significantly off? Let's apply standard correction.
        a = cv2.add(a, (128 - np.mean(a)))
        b = cv2.add(b, (128 - np.mean(b)))
        
        # Clip to valid range
        # Note: cv2.add with uint8 handles saturation automatically, but let's be safe if using float
        # Here we rely on OpenCV uint8 saturation behavior
        
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        # --- HSV Saturation Enhancement ---
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Adaptive saturation
        # If image is dull (low mean saturation), boost it more.
        mean_s = np.mean(s)
        if mean_s < 80: # Arbitrary threshold for "dull"
             # Boost saturation curve
             # S_new = S ^ (log(0.5) / log(mean/255)) ? Or simple multiplication
             # Let's use simple linear scaling with limit
             factor = 1.2
             s = cv2.multiply(s, factor)
        
        hsv_enhanced = cv2.merge([h, s, v])
        final_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

        return final_img

    def adaptive_gamma_correction(self, img):
        """
        Calculates gamma based on log mean luminance.
        """
        # Convert to HSV to get Value (Luminance approximation) or LAB L
        # HSV Value is just max(R,G,B). Lab L is perceptual.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate mean luminance normalized [0,1]
        mean_l = np.mean(l) / 255.0
        
        # We want mean_l to be around 0.5.
        # gamma = log(0.5) / log(mean_l)
        # If mean_l is very low (dark), log(mean) is large negative, gamma < 1 (brighten)
        # If mean_l is high (bright), gamma > 1 (darken)
        
        # Clamp mean_l to avoid div by zero or extreme values
        mean_l = max(0.01, min(0.99, mean_l))
        gamma = np.log(0.5) / np.log(mean_l)
        
        # Limit gamma range to avoid crazy washing out or darkening
        gamma = max(0.5, min(2.5, gamma))
        
        # Apply gamma
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(img, table)

    def apply_edge_preserving_filter(self, img):
        """
        Uses Guided Filter or Bilateral Filter.
        Guided filter is generally faster and preserves edges well.
        """
        # Guided Filter usage
        # radius: radius of window
        # eps: regularization parameter
        try:
            # Using OpenCV's guidedFilter
            # Guide is the image itself
            radius = 8
            eps = 50 * 50 # eps is usually variance^2? In cv2.ximgproc it's diff.
            # cv2.guidedFilter exists in contrib or main depending on version. 
            # Safer to use Bilateral for standard opencv-python if contrib not guaranteed.
            # requirements claims 'opencv-python', which might not have ximgproc.
            # Let's use Bilateral Filter which is standard.
            
            # Adaptive parameters?
            # If noisy via std dev estimation, increase sigmaColor
            
            # Optimization: Use small explicit diameter (d=5) for speed
            # Large sigmaSpace with d=-1 causes very slow performance
            sigmaColor = 50
            sigmaSpace = 50
            d = 5 
            
            filtered = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
            return filtered
        except:
             return img

    def process(self, img):
        """
        Full adaptive pipeline.
        """
        # 1. Histogram Stretching (Contrast)
        step1 = self.adaptive_histogram_stretching(img)
        
        # 2. Color Correction (Balance)
        step2 = self.adaptive_color_correction(step1)
        
        # 3. Gamma Correction (Brightness)
        step3 = self.adaptive_gamma_correction(step2)
        
        # 4. Edge Preserving Smoothing (De-noise/clean)
        final = self.apply_edge_preserving_filter(step3)
        
        return final.astype(np.uint8) if final is not None else None
