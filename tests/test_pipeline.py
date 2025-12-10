import unittest
import numpy as np
import cv2
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.white_balance import apply_white_balance
from src.gamma_correction import apply_gamma_correction
from src.sharpening import apply_sharpening
from src.clahe import apply_clahe
from src.histogram_linearization import apply_histogram_linearization
from src.weight_maps import generate_weight_map
from src.multiscale_fusion import apply_fusion
from src.metrics import calculate_entropy, calculate_uciqe

class TestDSPPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy image (100x100 RGB)
        self.image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_white_balance(self):
        result = apply_white_balance(self.image)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)
        
    def test_gamma_correction(self):
        result = apply_gamma_correction(self.image, gamma=1.2)
        self.assertEqual(result.shape, self.image.shape)
        
    def test_sharpening(self):
        result = apply_sharpening(self.image)
        self.assertEqual(result.shape, self.image.shape)
        
    def test_clahe(self):
        result = apply_clahe(self.image)
        self.assertEqual(result.shape, self.image.shape)
        
    def test_histogram_linearization(self):
        result = apply_histogram_linearization(self.image)
        self.assertEqual(result.shape, self.image.shape)
        
    def test_weight_map(self):
        w = generate_weight_map(self.image)
        self.assertEqual(w.shape, (100, 100)) # Single channel
        
    def test_fusion(self):
        img1 = self.image
        img2 = self.image
        w1 = np.ones((100, 100), dtype=float)
        w2 = np.ones((100, 100), dtype=float)
        
        result = apply_fusion(img1, img2, w1, w2, levels=3)
        self.assertEqual(result.shape, self.image.shape)
        
    def test_metrics(self):
        uciqe = calculate_uciqe(self.image)
        entropy = calculate_entropy(self.image)
        
        self.assertTrue(isinstance(uciqe, (float, np.floating)), f"UCIQE type: {type(uciqe)}")
        self.assertTrue(isinstance(entropy, (float, np.floating)), f"Entropy type: {type(entropy)}")

if __name__ == '__main__':
    unittest.main()
