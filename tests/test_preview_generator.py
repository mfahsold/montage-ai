import pytest
import os
from unittest.mock import patch, MagicMock
from src.montage_ai.preview_generator import PreviewGenerator

@pytest.fixture
def preview_gen(tmp_path):
    return PreviewGenerator(str(tmp_path))

@patch('subprocess.run')
def test_generate_shorts_preview_static(mock_run, preview_gen):
    """Test static crop generation."""
    source = "input.mp4"
    crop = {'x': 0.5, 'y': 0.5, 'width': 0.5625, 'height': 1.0}
    output = "output.mp4"
    
    preview_gen.generate_shorts_preview(source, crop, output)
    
    # Check command
    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert "-vf" in args
    vf_idx = args.index("-vf") + 1
    vf = args[vf_idx]
    assert "crop=w=iw*0.5625" in vf
    assert "sendcmd" not in vf

@patch('subprocess.run')
def test_generate_shorts_preview_dynamic(mock_run, preview_gen):
    """Test dynamic crop generation with keyframes."""
    source = "input.mp4"
    crop = {'x': 0.5, 'y': 0.5, 'width': 0.5625, 'height': 1.0}
    output = "output.mp4"
    keyframes = [
        {'time': 0.0, 'x': 100, 'y': 50, 'width': 360, 'height': 640},
        {'time': 1.0, 'x': 200, 'y': 50, 'width': 360, 'height': 640}
    ]
    
    preview_gen.generate_shorts_preview(source, crop, output, keyframes=keyframes)
    
    # Check command
    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert "-vf" in args
    vf_idx = args.index("-vf") + 1
    vf = args[vf_idx]
    assert "sendcmd=f=" in vf
    assert "crop=w=360:h=640:x=100:y=50" in vf # Initial values
