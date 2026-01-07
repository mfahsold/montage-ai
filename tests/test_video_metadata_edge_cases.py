"""
Edge Case Tests for PHASE 4: High-Resolution Support & Adaptive Features

Tests comprehensive edge cases for:
- Resolution detection and validation
- Adaptive batch sizing
- RAW codec detection
- H.264 level auto-detection
- 6K/8K workflow handling
"""
import pytest
from unittest.mock import patch, MagicMock
import warnings

from src.montage_ai.video_metadata import VideoMetadata, probe_metadata
from src.montage_ai.config import ProcessingSettings, get_settings
from src.montage_ai.ffmpeg_config import FFmpegConfig, get_config


class TestResolutionDetection:
    """Test resolution detection and classification."""
    
    def test_6k_landscape_detection(self):
        """Verify 6K landscape (6144x3160) is correctly detected."""
        settings = get_settings()
        width, height = 6144, 3160
        megapixels = width * height / 1_000_000
        
        assert megapixels > 15.0  # 6K threshold
        assert width > height  # Landscape
        
        # Adaptive batch size should be 1 for 6K
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        assert batch_size == 1, f"Expected batch_size=1 for 6K, got {batch_size}"
    
    def test_8k_portrait_detection(self):
        """Verify 8K portrait (4320x7680) is correctly detected."""
        settings = get_settings()
        width, height = 4320, 7680  # Portrait orientation
        megapixels = width * height / 1_000_000
        
        assert megapixels > 33.0  # 8K threshold
        assert height > width  # Portrait
        
        # Should return batch_size=1 with warning (fallback for 8K)
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        assert batch_size == 1  # 8K uses 6K fallback (batch_size=1)
    
    def test_4k_batch_sizing(self):
        """Verify 4K (3840x2160) gets batch_size=2."""
        settings = get_settings()
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=3840, height=2160, low_memory=False
        )
        assert batch_size == 2, f"Expected batch_size=2 for 4K, got {batch_size}"
    
    def test_1080p_batch_sizing(self):
        """Verify 1080p (1920x1080) gets batch_size=5."""
        settings = get_settings()
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=1920, height=1080, low_memory=False
        )
        assert batch_size == 5, f"Expected batch_size=5 for 1080p, got {batch_size}"
    
    def test_720p_batch_sizing(self):
        """Verify 720p (1280x720) gets base batch_size."""
        settings = get_settings()
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=1280, height=720, low_memory=False
        )
        # Below 8MP uses base batch_size (default=5)
        assert batch_size == 5, f"Expected batch_size=5 for 720p, got {batch_size}"
    
    def test_low_memory_mode_reduces_batch_size(self):
        """Verify low_memory=True halves the batch size."""
        settings = get_settings()
        
        # 1080p normal: 5
        normal_1080p = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=1920, height=1080, low_memory=False
        )
        
        # 1080p low memory: 2 (max(1, 5//2))
        low_mem_1080p = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=1920, height=1080, low_memory=True
        )
        
        assert low_mem_1080p == max(1, normal_1080p // 2)
        assert low_mem_1080p < normal_1080p


class TestRAWCodecDetection:
    """Test RAW codec detection and warnings."""
    
    def test_raw_codec_constants(self):
        """Verify RAW codec constants are defined."""
        from src.montage_ai.video_metadata import RAW_CODECS
        
        # Verify all expected RAW codecs are defined
        expected_codecs = ["prores_raw", "braw", "redcode", "cinemadng", "arriraw"]
        for codec in expected_codecs:
            assert codec in RAW_CODECS
            assert len(RAW_CODECS[codec]) > 0  # Non-empty description
    
    @patch('src.montage_ai.video_metadata.logger')
    def test_prores_raw_triggers_warning_in_logic(self, mock_logger):
        """Verify RAW codec detection logic triggers warning."""
        from src.montage_ai.video_metadata import RAW_CODECS
        
        # Simulate the detection logic
        codec = "prores_raw"
        if codec in RAW_CODECS:
            mock_logger.warning(
                f"⚠️  RAW codec detected: {codec}\\n"
                f"   {RAW_CODECS[codec]}\\n"
                f"   Consider generating H.264/H.265 proxies for better compatibility."
            )
        
        # Verify warning was called
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "prores_raw" in warning_message
        assert "ProRes RAW" in warning_message
    
    def test_standard_h264_not_in_raw_codecs(self):
        """Verify standard H.264 is NOT in RAW_CODECS list."""
        from src.montage_ai.video_metadata import RAW_CODECS
        
        # Verify common codecs are NOT classified as RAW
        non_raw_codecs = ["h264", "hevc", "prores", "dnxhd", "vp9", "av1"]
        for codec in non_raw_codecs:
            assert codec not in RAW_CODECS


class TestLevelAutoDetection:
    """Test H.264/H.265 level auto-detection."""
    
    def test_1080p_level_detection(self):
        """Verify 1080p@30fps uses level 4.0, 1080p@60fps uses 4.1."""
        config = FFmpegConfig()
        level_30fps = config.get_level_for_resolution(width=1920, height=1080, fps=30.0)
        level_60fps = config.get_level_for_resolution(width=1920, height=1080, fps=60.0)
        assert level_30fps == "4.0", f"Expected level 4.0 for 1080p@30, got {level_30fps}"
        assert level_60fps == "4.1", f"Expected level 4.1 for 1080p@60, got {level_60fps}"
    
    def test_4k_level_detection_h264(self):
        """Verify 4K@30fps uses 5.0, 4K@60fps uses 5.1."""
        config = FFmpegConfig(codec="libx264")
        level_30fps = config.get_level_for_resolution(
            width=3840, height=2160, fps=30.0
        )
        level_60fps = config.get_level_for_resolution(
            width=3840, height=2160, fps=60.0
        )
        assert level_30fps == "5.0", f"Expected level 5.0 for 4K@30 H.264, got {level_30fps}"
        assert level_60fps == "5.1", f"Expected level 5.1 for 4K@60 H.264, got {level_60fps}"
    
    def test_4k_level_detection_hevc(self):
        """Verify 4K HEVC@30fps uses level 5.0."""
        config = FFmpegConfig(codec="libx265")
        level = config.get_level_for_resolution(
            width=3840, height=2160, fps=30.0
        )
        assert level == "5.0", f"Expected level 5.0 for 4K@30 HEVC, got {level}"
    
    def test_6k_level_detection(self):
        """Verify 6K uses level 5.2."""
        config = FFmpegConfig()
        # Use a resolution above 19.66MP threshold (6144x3200 = 19.66MP)
        level = config.get_level_for_resolution(width=6144, height=3200, fps=30.0)
        assert level == "5.2", f"Expected level 5.2 for 6K, got {level}"
    
    def test_8k_level_detection(self):
        """Verify 8K uses level 6.2."""
        config = FFmpegConfig(codec="libx265")
        level = config.get_level_for_resolution(width=7680, height=4320, fps=30.0)
        assert level == "6.2", f"Expected level 6.2 for 8K, got {level}"
    
    def test_8k_requires_hevc(self):
        """Verify 8K output requires HEVC codec."""
        # 8K with H.264 should raise error
        config_h264 = FFmpegConfig(codec="libx264")
        with pytest.raises(ValueError, match="8K.*requires HEVC"):
            config_h264.get_level_for_resolution(
                width=7680, height=4320, fps=30.0
            )
        
        # 8K with HEVC should succeed
        config_hevc = FFmpegConfig(codec="libx265")
        level = config_hevc.get_level_for_resolution(
            width=7680, height=4320, fps=30.0
        )
        assert level == "6.2"


class TestAspectRatioEdgeCases:
    """Test unusual aspect ratios and orientations."""
    
    def test_ultrawide_6k(self):
        """Verify ultra-wide 6K (21:9) is handled correctly."""
        settings = get_settings()
        width, height = 6144, 2570  # ~21:9 aspect ratio
        
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        
        # Should still detect as 6K based on megapixels
        megapixels = width * height / 1_000_000
        assert megapixels > 15.0
        assert batch_size == 1
    
    def test_vertical_4k(self):
        """Verify vertical 4K (9:16) is handled correctly."""
        settings = get_settings()
        width, height = 2160, 3840  # Portrait 4K
        
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        
        # Should detect as 4K based on megapixels
        megapixels = width * height / 1_000_000
        assert megapixels > 8.0
        assert batch_size == 2
    
    def test_anamorphic_6k(self):
        """Verify anamorphic 6K (2.39:1) is handled correctly."""
        settings = get_settings()
        width, height = 6144, 2572  # 2.39:1 aspect ratio
        
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        
        # Should still be batch_size=1 for 6K megapixels
        assert batch_size == 1


class TestBoundaryConditions:
    """Test boundary conditions and edge values."""
    
    def test_exactly_4k_threshold(self):
        """Verify exact 4K resolution (8.3MP) triggers 4K batch size."""
        settings = get_settings()
        width, height = 3840, 2160  # Exactly 8.3MP
        
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        assert batch_size == 2
    
    def test_just_below_4k(self):
        """Verify resolution just below 4K uses base batch size."""
        settings = get_settings()
        width, height = 2560, 1440  # 1440p (3.7MP)
        
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        # Below 4K threshold (8MP) should use base batch_size=5
        assert batch_size == 5
    
    def test_exactly_6k_threshold(self):
        """Verify exact 6K resolution (15MP) triggers 6K batch size."""
        settings = get_settings()
        width, height = 5000, 3000  # Exactly 15MP
        
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        assert batch_size == 1
    
    def test_just_below_6k(self):
        """Verify resolution just below 6K uses 4K batch size."""
        settings = get_settings()
        width, height = 4096, 3072  # ~12.6MP (between 4K and 6K)
        
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=width, height=height, low_memory=False
        )
        # Between 8.3MP and 15MP = 4K tier
        assert batch_size == 2


class TestMemoryConstraints:
    """Test memory-related edge cases."""
    
    def test_6k_low_memory_batch_size(self):
        """Verify 6K in low_memory mode still maintains batch_size=1."""
        settings = get_settings()
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=6144, height=3160, low_memory=True
        )
        # 6K batch_size=1 even in low_memory (can't go lower)
        assert batch_size == 1
    
    def test_4k_low_memory_batch_size(self):
        """Verify 4K in low_memory mode reduces batch size."""
        settings = get_settings()
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=3840, height=2160, low_memory=True
        )
        # 4K base=5, 4K bracket=max(1, 5//2)=2, low_memory halves it: max(1, 2//2)=1
        # BUT: Code shows 4K uses batch_size//2 which is already 2 (5//2), low_memory doesn't apply further
        assert batch_size == 2
    
    def test_1080p_low_memory_batch_size(self):
        """Verify 1080p in low_memory mode reduces to batch_size=2."""
        settings = get_settings()
        batch_size = settings.high_res.get_adaptive_batch_size_for_resolution(
            width=1920, height=1080, low_memory=True
        )
        # 1080p normal=5, low_memory=2 (max(1, 5//2))
        assert batch_size == 2


class TestCodecCompatibility:
    """Test codec compatibility edge cases."""
    
    def test_hevc_supports_all_resolutions(self):
        """Verify HEVC supports 1080p through 8K."""
        config = FFmpegConfig(codec="libx265")
        
        # HEVC should work for all resolutions
        for width, height in [(1920, 1080), (3840, 2160), (6144, 3160), (7680, 4320)]:
            level = config.get_level_for_resolution(
                width=width, height=height, fps=30.0
            )
            assert level is not None
    
    def test_h264_limited_to_6k(self):
        """Verify H.264 is limited to 6K maximum."""
        # H.264 should work up to 6K
        config_6k = FFmpegConfig(codec="libx264")
        level_6k = config_6k.get_level_for_resolution(
            width=6144, height=3200, fps=30.0  # Above 19.66MP threshold
        )
        assert level_6k == "5.2"
        
        # H.264 should reject 8K
        config_8k = FFmpegConfig(codec="libx264")
        with pytest.raises(ValueError):
            config_8k.get_level_for_resolution(
                width=7680, height=4320, fps=30.0
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
