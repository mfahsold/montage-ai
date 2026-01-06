import pytest
from unittest.mock import patch, MagicMock
from src.montage_ai.core.montage_builder import MontageBuilder
from src.montage_ai.core.context import OutputProfile

@pytest.fixture
def mock_settings():
    with patch('src.montage_ai.core.montage_builder.get_settings') as mock_get:
        settings = MagicMock()
        # Default settings
        settings.features.shorts_mode = False
        settings.features.stabilize = True
        settings.features.upscale = True
        settings.features.enhance = True
        settings.encoding.quality_profile = "standard"
        # Mock paths
        settings.paths.input_dir = "/tmp/input"
        settings.paths.music_dir = "/tmp/music"
        settings.paths.assets_dir = "/tmp/assets"
        settings.paths.output_dir = "/tmp/output"
        settings.paths.temp_dir = "/tmp/temp"
        settings.job_id = "test_job"
        
        mock_get.return_value = settings
        yield settings

def test_preview_mode_resolution(mock_settings):
    """Test that preview mode forces low resolution."""
    mock_settings.encoding.quality_profile = "preview"
    
    builder = MontageBuilder()
    # Mock context and video files
    builder.ctx.media.video_files = ["test.mp4"]
    
    # Mock determine_output_profile from video_metadata
    with patch('src.montage_ai.video_metadata.determine_output_profile') as mock_det:
        # Return a high res profile
        mock_profile = MagicMock()
        mock_profile.width = 1920
        mock_profile.height = 1080
        mock_profile.orientation = "horizontal"
        mock_profile.fps = 30.0
        mock_profile.codec = "h264"
        mock_profile.pix_fmt = "yuv420p"
        mock_profile.profile = "high"
        mock_profile.level = "4.1"
        mock_profile.bitrate = 5000
        mock_profile.aspect_ratio = "16:9"
        mock_profile.reason = "default"
        
        mock_det.return_value = mock_profile
        
        builder._analyzer.determine_output_profile()
        
        assert builder.ctx.media.output_profile.width == 640
        assert builder.ctx.media.output_profile.height == 360
        assert builder.ctx.media.output_profile.reason == "preview_mode"

def test_preview_mode_disables_effects(mock_settings):
    """Test that preview mode disables heavy effects."""
    mock_settings.encoding.quality_profile = "preview"
    mock_settings.features.stabilize = True
    mock_settings.features.upscale = True
    mock_settings.features.enhance = True
    
    builder = MontageBuilder()
    # Mock EditingInstructions object logic or just set dictionary?
    # Implementation handles dict conversion if passed to init.
    # Here we set direct attribute, which should be on ctx.creative.editing_instructions
    # But builder._apply_creative_director_effects() checks ctx.creative.editing_instructions
    
    # We should setup builder properly
    instructions = {'effects': {'stabilization': True, 'upscale': True, 'sharpness_boost': True}}
    builder = MontageBuilder(editing_instructions=instructions)
    
    builder._apply_creative_director_effects()
    
    assert builder.ctx.features.stabilize is False
    assert builder.ctx.features.upscale is False
    assert builder.ctx.features.enhance is False

def test_preview_mode_vertical(mock_settings):
    """Test preview mode with vertical video."""
    mock_settings.encoding.quality_profile = "preview"
    
    builder = MontageBuilder()
    builder.ctx.media.video_files = ["test.mp4"]
    
    with patch('src.montage_ai.video_metadata.determine_output_profile') as mock_det:
        mock_profile = MagicMock()
        mock_profile.width = 1080
        mock_profile.height = 1920
        mock_profile.orientation = "vertical"
        mock_profile.fps = 30.0
        mock_profile.codec = "h264"
        mock_profile.pix_fmt = "yuv420p"
        mock_profile.profile = "high"
        mock_profile.level = "4.1"
        mock_profile.bitrate = 5000
        mock_profile.aspect_ratio = "9:16"
        mock_profile.reason = "default"
        
        mock_det.return_value = mock_profile
        
        builder._analyzer.determine_output_profile()
        
        assert builder.ctx.media.output_profile.width == 360
        assert builder.ctx.media.output_profile.height == 640
        assert builder.ctx.media.output_profile.reason == "preview_mode"
