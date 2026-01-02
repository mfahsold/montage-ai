import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock cv2 and mediapipe before importing smart_reframing
sys.modules["cv2"] = MagicMock()
sys.modules["mediapipe"] = MagicMock()

from montage_ai.smart_reframing import SmartReframer, CropWindow

class TestSmartReframing(unittest.TestCase):
    def setUp(self):
        self.reframer = SmartReframer(target_aspect=9/16)

    def test_calculate_crop_dims(self):
        # 1920x1080 (16:9) -> Target 9:16
        # Should crop width to fit height * 9/16
        # 1080 * 9/16 = 607.5 -> 607
        w, h = self.reframer._calculate_crop_dims(1920, 1080)
        self.assertEqual(h, 1080)
        self.assertAlmostEqual(w, 607, delta=1)

    def test_calculate_crop_dims_already_vertical(self):
        # 1080x1920 (9:16) -> Target 9:16
        w, h = self.reframer._calculate_crop_dims(1080, 1920)
        self.assertEqual(w, 1080)
        self.assertEqual(h, 1920)

    def test_smoothing(self):
        crops = [
            CropWindow(0, 100, 0, 100, 100, 1.0),
            CropWindow(1, 200, 0, 100, 100, 1.0),
            CropWindow(2, 100, 0, 100, 100, 1.0),
        ]
        # Window size 15, so it should average them heavily
        smoothed = self.reframer._smooth_crops(crops)
        self.assertEqual(len(smoothed), 3)
        # Middle value should be pulled down/up
        self.assertNotEqual(smoothed[1].x, 200)

    def test_segment_crops(self):
        # Create crops that stay at x=100 for 2s, then jump to x=500 for 2s
        crops = []
        # 0-2s: x=100
        for i in range(60): # 30fps * 2s
            crops.append(CropWindow(i/30.0, 100, 0, 100, 100, 1.0))
        # 2-4s: x=500
        for i in range(60, 120):
            crops.append(CropWindow(i/30.0, 500, 0, 100, 100, 1.0))
            
        segments = self.reframer._segment_crops(crops, min_segment_duration=1.0)
        
        # Should have 2 segments
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]['x'], 100)
        self.assertEqual(segments[1]['x'], 500)
        self.assertAlmostEqual(segments[0]['end'], 2.0, delta=0.1)
        self.assertAlmostEqual(segments[1]['start'], 2.0, delta=0.1)

if __name__ == "__main__":
    unittest.main()
