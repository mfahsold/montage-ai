"""
Tests for high-resolution scene detection with downsampling optimization.

Tests PHASE 4 features:
- Automatic downsampling for 6K+ videos
- Resolution-aware batch sizing
- Proxy file creation for scene analysis
"""
import os
import tempfile
import subprocess
from unittest.mock import patch, MagicMock, call
import pytest

from src.montage_ai.scene_analysis import SceneDetector, Scene
from src.montage_ai.video_metadata import VideoMetadata


class TestSceneDetectionDownsampling:
    """Test suite for scene detection with high-resolution downsampling."""
    
    @patch('src.montage_ai.scene_analysis.probe_metadata')
    @patch('src.montage_ai.scene_analysis.open_video')
    @patch('src.montage_ai.scene_analysis.SceneManager')
    def test_6k_video_triggers_downsampling(
        self, 
        mock_manager_class, 
        mock_open_video, 
        mock_probe
    ):
        """6K video should trigger automatic downsampling to 1080p."""
        # Setup: 6K video metadata
        mock_metadata = VideoMetadata(
            width=6144,
            height=3160,
            fps=30.0,
            duration=10.0,
            codec="prores",
            pix_fmt="yuv422p10le",
            bitrate=500_000_000,  # 500 Mbps
            filesize=625_000_000
        )
        mock_probe.return_value = mock_metadata
        
        # Setup: Scene detection returns empty list
        mock_scene_manager = MagicMock()
        mock_scene_manager.get_scene_list.return_value = []
        mock_manager_class.return_value = mock_scene_manager
        
        # Mock VideoFileClip for fallback
        with patch('src.montage_ai.scene_analysis.VideoFileClip') as mock_clip:
            mock_clip_instance = MagicMock()
            mock_clip_instance.duration = 10.0
            mock_clip.return_value = mock_clip_instance
            
            detector = SceneDetector(threshold=30.0)
            
            # Mock _create_downsampled_proxy to avoid actual FFmpeg call
            with patch.object(detector, '_create_downsampled_proxy', return_value='/tmp/proxy.mp4') as mock_downsample:
                scenes = detector.detect("/fake/6k_video.mp4")
                
                # Verify downsampling was triggered
                mock_downsample.assert_called_once_with("/fake/6k_video.mp4", 1080)
                
                # Verify open_video was called with proxy path
                mock_open_video.assert_called_once_with('/tmp/proxy.mp4')
    
    @patch('src.montage_ai.scene_analysis.probe_metadata')
    @patch('src.montage_ai.scene_analysis.open_video')
    @patch('src.montage_ai.scene_analysis.SceneManager')
    def test_1080p_video_no_downsampling(
        self, 
        mock_manager_class, 
        mock_open_video, 
        mock_probe
    ):
        """1080p video should NOT trigger downsampling."""
        # Setup: 1080p video metadata
        mock_metadata = VideoMetadata(
            width=1920,
            height=1080,
            fps=30.0,
            duration=10.0,
            codec="h264",
            pix_fmt="yuv420p",
            bitrate=8_000_000,
            filesize=10_000_000
        )
        mock_probe.return_value = mock_metadata
        
        # Setup: Scene detection returns 2 scenes
        mock_scene_manager = MagicMock()
        mock_scene1 = (MagicMock(), MagicMock())
        mock_scene1[0].get_seconds.return_value = 0.0
        mock_scene1[1].get_seconds.return_value = 5.0
        mock_scene2 = (MagicMock(), MagicMock())
        mock_scene2[0].get_seconds.return_value = 5.0
        mock_scene2[1].get_seconds.return_value = 10.0
        mock_scene_manager.get_scene_list.return_value = [mock_scene1, mock_scene2]
        mock_manager_class.return_value = mock_scene_manager
        
        detector = SceneDetector(threshold=30.0)
        
        # Mock _create_downsampled_proxy to verify it's NOT called
        with patch.object(detector, '_create_downsampled_proxy') as mock_downsample:
            scenes = detector.detect("/fake/1080p_video.mp4")
            
            # Verify downsampling was NOT triggered
            mock_downsample.assert_not_called()
            
            # Verify open_video was called with original path
            mock_open_video.assert_called_once_with('/fake/1080p_video.mp4')
            
            # Verify scenes were detected
            assert len(scenes) == 2
            assert scenes[0].start == 0.0
            assert scenes[0].end == 5.0
    
    @patch('src.montage_ai.scene_analysis.probe_metadata')
    @patch('src.montage_ai.scene_analysis.open_video')
    @patch('src.montage_ai.scene_analysis.SceneManager')
    def test_custom_max_resolution_threshold(
        self, 
        mock_manager_class, 
        mock_open_video, 
        mock_probe
    ):
        """Custom max_resolution parameter should override default 1080p."""
        # Setup: 4K video metadata
        mock_metadata = VideoMetadata(
            width=3840,
            height=2160,
            fps=30.0,
            duration=10.0,
            codec="h264",
            pix_fmt="yuv420p",
            bitrate=50_000_000,
            filesize=62_500_000
        )
        mock_probe.return_value = mock_metadata
        
        # Setup: Scene detection returns empty list
        mock_scene_manager = MagicMock()
        mock_scene_manager.get_scene_list.return_value = []
        mock_manager_class.return_value = mock_scene_manager
        
        # Mock VideoFileClip for fallback
        with patch('src.montage_ai.scene_analysis.VideoFileClip') as mock_clip:
            mock_clip_instance = MagicMock()
            mock_clip_instance.duration = 10.0
            mock_clip.return_value = mock_clip_instance
            
            detector = SceneDetector(threshold=30.0)
            
            # Set custom max_resolution=720 (should downsample 4K to 720p)
            with patch.object(detector, '_create_downsampled_proxy', return_value='/tmp/proxy_720.mp4') as mock_downsample:
                scenes = detector.detect("/fake/4k_video.mp4", max_resolution=720)
                
                # Verify downsampling was triggered with custom resolution
                mock_downsample.assert_called_once_with("/fake/4k_video.mp4", 720)
    
    @patch('subprocess.run')
    @patch('src.montage_ai.scene_analysis.tempfile.mkstemp')
    def test_create_downsampled_proxy_ffmpeg_command(
        self, 
        mock_mkstemp, 
        mock_subprocess
    ):
        """_create_downsampled_proxy should generate correct FFmpeg command."""
        # Setup: Mock temp file creation
        mock_mkstemp.return_value = (123, '/tmp/scene_detect_abc123_proxy.mp4')
        
        # Setup: Mock successful FFmpeg execution
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='', stderr='')
        
        detector = SceneDetector()
        proxy_path = detector._create_downsampled_proxy("/input/6k_video.mp4", 1080)
        
        # Verify FFmpeg command structure
        expected_cmd = [
            "ffmpeg",
            "-i", "/input/6k_video.mp4",
            "-vf", "scale=-2:1080",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-an",
            "-y",
            "/tmp/scene_detect_abc123_proxy.mp4"
        ]
        
        mock_subprocess.assert_called_once()
        actual_cmd = mock_subprocess.call_args[0][0]
        assert actual_cmd == expected_cmd
        
        # Verify proxy path returned
        assert proxy_path == '/tmp/scene_detect_abc123_proxy.mp4'
    
    @patch('subprocess.run')
    @patch('src.montage_ai.scene_analysis.tempfile.mkstemp')
    @patch('os.path.exists')
    @patch('os.remove')
    def test_create_downsampled_proxy_handles_ffmpeg_failure(
        self, 
        mock_remove, 
        mock_exists, 
        mock_mkstemp, 
        mock_subprocess
    ):
        """_create_downsampled_proxy should handle FFmpeg failures gracefully."""
        # Setup: Mock temp file creation
        mock_mkstemp.return_value = (123, '/tmp/scene_detect_failed_proxy.mp4')
        mock_exists.return_value = True
        
        # Setup: Mock FFmpeg failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, 
            'ffmpeg', 
            stderr='Error: Invalid input file'
        )
        
        detector = SceneDetector()
        proxy_path = detector._create_downsampled_proxy("/input/corrupt_video.mp4", 1080)
        
        # Verify failed proxy file was cleaned up
        mock_remove.assert_called_once_with('/tmp/scene_detect_failed_proxy.mp4')
        
        # Verify original path returned as fallback
        assert proxy_path == "/input/corrupt_video.mp4"
    
    @patch('src.montage_ai.scene_analysis.probe_metadata')
    @patch('src.montage_ai.scene_analysis.open_video')
    @patch('src.montage_ai.scene_analysis.SceneManager')
    def test_scenes_reference_original_path_not_proxy(
        self, 
        mock_manager_class, 
        mock_open_video, 
        mock_probe
    ):
        """Scene objects should reference original path, not proxy path."""
        # Setup: 6K video metadata
        mock_metadata = VideoMetadata(
            width=6144,
            height=3160,
            fps=30.0,
            duration=10.0,
            codec="prores",
            pix_fmt="yuv422p10le",
            bitrate=500_000_000,
            filesize=625_000_000
        )
        mock_probe.return_value = mock_metadata
        
        # Setup: Scene detection returns 1 scene
        mock_scene_manager = MagicMock()
        mock_scene1 = (MagicMock(), MagicMock())
        mock_scene1[0].get_seconds.return_value = 0.0
        mock_scene1[1].get_seconds.return_value = 10.0
        mock_scene_manager.get_scene_list.return_value = [mock_scene1]
        mock_manager_class.return_value = mock_scene_manager
        
        detector = SceneDetector(threshold=30.0)
        
        # Mock _create_downsampled_proxy
        with patch.object(detector, '_create_downsampled_proxy', return_value='/tmp/proxy.mp4'):
            scenes = detector.detect("/data/input/6k_original.mp4")
            
            # Verify scene path references original file
            assert len(scenes) == 1
            assert scenes[0].path == "/data/input/6k_original.mp4"
            assert scenes[0].path != "/tmp/proxy.mp4"


class TestSceneDetectionPerformance:
    """Performance validation tests for downsampling optimization."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/ffmpeg"),
        reason="FFmpeg not available"
    )
    def test_downsampling_reduces_processing_time(self):
        """Verify downsampling provides measurable speedup (manual verification)."""
        # This test requires actual video files and is marked slow
        # It's primarily for manual performance validation
        
        # Expected speedup:
        # - 6K (6144x3160) → 1080p: ~4-9x faster
        # - 4K (3840x2160) → 1080p: ~2-4x faster
        
        # Test structure (would need real files):
        # 1. Create 6K test video with FFmpeg
        # 2. Time scene detection at full resolution
        # 3. Time scene detection with downsampling
        # 4. Assert downsampling is faster
        
        pytest.skip("Manual performance test - requires real video files")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
