import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from montage_ai.timeline_exporter import TimelineExporter, Timeline, Clip

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test output."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_edl_export(temp_dir):
    """Test CMX 3600 EDL export."""
    clips = [
        Clip(
            source_path="/data/input/clip1.mp4",
            start_time=10.0,
            duration=5.0,
            timeline_start=0.0,
            metadata={}
        ),
        Clip(
            source_path="/data/input/clip2.mp4",
            start_time=20.0,
            duration=4.0,
            timeline_start=5.0,
            metadata={}
        )
    ]
    
    timeline = Timeline(
        clips=clips,
        audio_path="/data/music/track.mp3",
        total_duration=9.0,
        project_name="test_project",
        fps=30.0
    )
    
    exporter = TimelineExporter(output_dir=temp_dir)
    edl_path = exporter._export_edl(timeline)
    
    assert os.path.exists(edl_path)
    assert edl_path.endswith(".edl")
    
    with open(edl_path, 'r') as f:
        content = f.read()
        
    # Check Header
    assert "TITLE: test_project" in content
    assert "FCM: NON-DROP FRAME" in content
    
    # Check Events
    # Clip 1: 00:00:10:00 -> 00:00:15:00 (Source) | 00:00:00:00 -> 00:00:05:00 (Rec)
    assert "001  clip1    V     C        00:00:10:00 00:00:15:00 00:00:00:00 00:00:05:00" in content
    
    # Clip 2: 00:00:20:00 -> 00:00:24:00 (Source) | 00:00:05:00 -> 00:00:09:00 (Rec)
    assert "002  clip2    V     C        00:00:20:00 00:00:24:00 00:00:05:00 00:00:09:00" in content

def test_otio_export(temp_dir):
    """Test OpenTimelineIO export."""
    try:
        import opentimelineio as otio
    except ImportError:
        pytest.skip("OpenTimelineIO not installed")

    clips = [
        Clip(
            source_path="/data/input/clip1.mp4",
            start_time=0.0,
            duration=5.0,
            timeline_start=0.0,
            metadata={"face_count": 1}
        )
    ]
    
    timeline = Timeline(
        clips=clips,
        audio_path="/data/music/track.mp3",
        total_duration=5.0,
        project_name="test_otio",
        fps=24.0
    )
    
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(timeline)
    
    assert os.path.exists(otio_path)
    assert otio_path.endswith(".otio")
    
    # Verify we can read it back
    read_timeline = otio.adapters.read_from_file(otio_path)
    assert read_timeline.name == "test_otio"
    assert len(read_timeline.tracks) == 2 # Video + Audio
    
    video_track = read_timeline.tracks[0]
    assert len(video_track) == 1
    assert video_track[0].name == "clip1.mp4"
    assert video_track[0].metadata["face_count"] == 1

@patch("montage_ai.timeline_exporter.get_settings")
def test_export_timeline_integration(mock_get_settings, temp_dir):
    """Test the full export pipeline."""
    mock_settings = MagicMock()
    mock_settings.paths.output_dir = temp_dir
    mock_get_settings.return_value = mock_settings
    
    clips = [
        Clip(
            source_path="/data/input/clip1.mp4",
            start_time=0.0,
            duration=5.0,
            timeline_start=0.0,
            metadata={}
        )
    ]
    
    timeline = Timeline(
        clips=clips,
        audio_path="/data/music/track.mp3",
        total_duration=5.0,
        project_name="test_full",
        fps=30.0
    )
    
    exporter = TimelineExporter(output_dir=temp_dir)
    
    # Mock _generate_proxy to avoid ffmpeg calls
    exporter._generate_proxy = MagicMock(return_value="/tmp/proxy.mp4")
    
    result = exporter.export_timeline(
        timeline,
        generate_proxies=True,
        export_otio=True,
        export_edl=True,
        export_xml=False, # Skip XML for now as it might be complex
        export_csv=False
    )
    
    assert "edl" in result
    assert "otio" in result
    assert "metadata" in result
    assert "package" in result
    
    assert os.path.exists(result["edl"])
    assert os.path.exists(result["otio"])
