
import pytest
import os
import shutil
from unittest.mock import MagicMock, patch
from montage_ai.timeline_exporter import TimelineExporter, Timeline, Clip

@pytest.fixture
def mock_timeline(tmp_path):
    # Create dummy media files
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    
    clip1_path = media_dir / "clip1.mp4"
    clip1_path.touch()
    
    clip2_path = media_dir / "clip2.mp4"
    clip2_path.touch()
    
    audio_path = media_dir / "music.mp3"
    audio_path.touch()
    
    clips = [
        Clip(str(clip1_path), 0.0, 5.0, 0.0),
        Clip(str(clip2_path), 0.0, 5.0, 5.0)
    ]
    
    return Timeline(
        clips=clips,
        audio_path=str(audio_path),
        total_duration=10.0,
        project_name="test_project"
    )

@patch('subprocess.Popen')
def test_proxy_generation_h264(mock_popen, mock_timeline, tmp_path):
    # Setup mock process
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("stdout", "stderr")
    mock_process.returncode = 0
    mock_process.__enter__.return_value = mock_process
    mock_popen.return_value = mock_process

    exporter = TimelineExporter(output_dir=str(tmp_path))
    
    # Test H.264 (default)
    proxy_path = exporter._generate_proxy(mock_timeline.clips[0].source_path, format="h264")
    
    assert proxy_path is not None
    assert "proxy_clip1.mp4" in proxy_path
    
    # Verify ffmpeg command
    args = mock_popen.call_args[0][0]
    assert "libx264" in args
    assert "-preset" in args

@patch('subprocess.Popen')
def test_proxy_generation_prores(mock_popen, mock_timeline, tmp_path):
    # Setup mock process
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("stdout", "stderr")
    mock_process.returncode = 0
    mock_process.__enter__.return_value = mock_process
    mock_popen.return_value = mock_process

    exporter = TimelineExporter(output_dir=str(tmp_path))
    
    # Test ProRes
    proxy_path = exporter._generate_proxy(mock_timeline.clips[0].source_path, format="prores")
    
    assert proxy_path is not None
    assert "proxy_clip1.mov" in proxy_path
    
    # Verify ffmpeg command
    args = mock_popen.call_args[0][0]
    assert "prores_ks" in args
    assert "-profile:v" in args

@patch('subprocess.Popen')
def test_proxy_generation_dnxhr(mock_popen, mock_timeline, tmp_path):
    # Setup mock process
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("stdout", "stderr")
    mock_process.returncode = 0
    mock_process.__enter__.return_value = mock_process
    mock_popen.return_value = mock_process

    exporter = TimelineExporter(output_dir=str(tmp_path))
    
    # Test DNxHR
    proxy_path = exporter._generate_proxy(mock_timeline.clips[0].source_path, format="dnxhr")
    
    assert proxy_path is not None
    assert "proxy_clip1.mov" in proxy_path
    
    # Verify ffmpeg command
    args = mock_popen.call_args[0][0]
    assert "dnxhd" in args
    assert "dnxhr_lb" in args

def test_relink_readme_generation(mock_timeline, tmp_path):
    exporter = TimelineExporter(output_dir=str(tmp_path))
    package_dir = tmp_path / "test_project_PROJECT"
    package_dir.mkdir()
    
    exporter._generate_conform_guide(str(package_dir), mock_timeline, link_to_source=False)
    
    readme_path = package_dir / "HOW_TO_CONFORM.md"
    assert readme_path.exists()
    
    content = readme_path.read_text()
    assert "Fluxibri Timeline Export" in content
    assert "DaVinci Resolve" in content
    assert "Adobe Premiere Pro" in content

@patch('subprocess.run')
def test_full_export_flow(mock_run, mock_timeline, tmp_path):
    exporter = TimelineExporter(output_dir=str(tmp_path))
    
    # Mock OTIO availability if needed, or rely on fallback
    with patch('montage_ai.timeline_exporter.OTIO_AVAILABLE', False):
        result = exporter.export_timeline(
            mock_timeline,
            generate_proxies=True,
            proxy_format="prores",
            export_otio=False # Skip OTIO since we mocked it out
        )
    
    assert "package" in result
    package_path = result["package"]
    assert os.path.exists(package_path)
    assert os.path.exists(os.path.join(package_path, "HOW_TO_CONFORM.md"))
    assert os.path.exists(os.path.join(package_path, "media"))
