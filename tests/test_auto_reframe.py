import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# We need to patch 'montage_ai.auto_reframe.mp' because the module imports it at top level.
# However, since we are importing the class, we can patch it during setUp.

from montage_ai.auto_reframe import AutoReframeEngine, CropWindow

class TestAutoReframe(unittest.TestCase):
    def setUp(self):
        # Patch mp inside the module
        self.mp_patcher = patch('montage_ai.auto_reframe.mp')
        self.mock_mp = self.mp_patcher.start()
        self.mock_mp.solutions.face_detection = MagicMock()
        
        self.reframer = AutoReframeEngine(target_aspect=9/16)

    def tearDown(self):
        self.mp_patcher.stop()

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
        # Create enough frames to exceed the window size (15)
        raw_centers = [100] * 20
        raw_centers[10] = 200 # Spike in the middle
        
        # Window size 15, so it should average them heavily
        smoothed = self.reframer.path_planner.solve(raw_centers)
        self.assertEqual(len(smoothed), 20)
        # Middle value should be pulled down (smoothed)
        self.assertNotEqual(smoothed[10], 200)
        self.assertTrue(smoothed[10] < 200)

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

    @patch('montage_ai.auto_reframe.run_command')
    def test_apply_center(self, mock_run):
        # Test with None
        # We need to mock cv2.VideoCapture inside apply since it opens the file
        # And ensure constants are set
        import cv2
        cv2.CAP_PROP_FRAME_WIDTH = 3
        cv2.CAP_PROP_FRAME_HEIGHT = 4
        
        with patch('montage_ai.auto_reframe.cv2.VideoCapture') as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                3: 1920, # WIDTH
                4: 1080, # HEIGHT
            }.get(prop, 0)
            mock_cap_cls.return_value = mock_cap
            
            self.reframer.apply(None, "in.mp4", "out.mp4")
            
            # Should call ffmpeg with simple crop
            self.assertTrue(mock_run.called)
            cmd = mock_run.call_args[0][0]
            # Check for crop filter
            self.assertTrue(any("crop=" in arg for arg in cmd))

if __name__ == "__main__":
    unittest.main()
