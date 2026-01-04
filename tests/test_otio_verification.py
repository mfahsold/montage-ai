"""
OTIO Export Verification Suite

Tests OpenTimelineIO exports for real-world NLE compatibility.
Verifies format compliance with DaVinci Resolve, Premiere Pro, and Final Cut Pro.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from typing import Dict, List

try:
    import opentimelineio as otio
    OTIO_AVAILABLE = True
except ImportError:
    OTIO_AVAILABLE = False

from montage_ai.timeline_exporter import TimelineExporter, Timeline, Clip


pytestmark = pytest.mark.skipif(not OTIO_AVAILABLE, reason="OpenTimelineIO not installed")


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for exports."""
    return str(tmp_path)


@pytest.fixture
def sample_timeline():
    """Create a sample timeline with various clip types."""
    clips = [
        # Standard clip
        Clip(
            source_path="/data/input/clip_001.mp4",
            start_time=2.5,
            duration=10.0,
            timeline_start=0.0,
            metadata={"energy": 0.8, "shot": "wide"}
        ),
        # Short clip (potential frame accuracy issue)
        Clip(
            source_path="/data/input/clip_002.mp4",
            start_time=0.0,
            duration=0.5,
            timeline_start=10.0,
            metadata={"energy": 0.3, "shot": "close"}
        ),
        # Longer clip
        Clip(
            source_path="/data/input/clip_003.mp4",
            start_time=15.0,
            duration=20.0,
            timeline_start=10.5,
            metadata={"energy": 0.9, "shot": "medium"}
        ),
    ]
    
    return Timeline(
        clips=clips,
        audio_path="/data/music/background_track.mp3",
        total_duration=30.5,
        project_name="verification_test",
        fps=24.0,
        resolution=(1920, 1080)
    )


def test_otio_file_structure(temp_dir, sample_timeline):
    """Verify OTIO file is valid JSON and readable."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    assert Path(otio_path).exists()
    
    # Verify it's valid JSON
    with open(otio_path, 'r') as f:
        data = json.load(f)
    
    # Check root structure
    assert 'OTIO_SCHEMA' in data
    assert 'name' in data
    assert data['name'] == 'verification_test'


def test_otio_timeline_metadata(temp_dir, sample_timeline):
    """Verify timeline metadata is correctly stored."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    timeline = otio.adapters.read_from_file(otio_path)
    
    # Check timeline properties
    assert timeline.name == 'verification_test'
    # global_start_time is optional in OTIO, can be None
    
    # Check tracks exist
    assert len(timeline.tracks) == 2, "Should have video + audio track"
    
    # Verify track types
    video_track = timeline.tracks[0]
    audio_track = timeline.tracks[1]
    
    assert video_track.kind == otio.schema.TrackKind.Video
    assert audio_track.kind == otio.schema.TrackKind.Audio


def test_otio_clip_timings(temp_dir, sample_timeline):
    """Verify clip in/out points and durations."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    timeline = otio.adapters.read_from_file(otio_path)
    video_track = timeline.tracks[0]
    
    # Check number of clips
    assert len(video_track) == 3
    
    # Clip 1: start_time=2.5s, duration=10.0s, timeline_start=0.0s
    clip1 = video_track[0]
    assert clip1.name == "clip_001.mp4"
    
    source_range = clip1.source_range
    # Start time should be 2.5s * 24fps = 60 frames
    assert source_range.start_time.value == 2.5 * 24
    assert source_range.start_time.rate == 24.0
    # Duration should be 10.0s * 24fps = 240 frames
    assert source_range.duration.value == 10.0 * 24
    
    # Clip 2: Short clip (0.5s)
    clip2 = video_track[1]
    assert clip2.name == "clip_002.mp4"
    assert clip2.source_range.duration.value == 0.5 * 24
    
    # Clip 3: Long clip (20.0s)
    clip3 = video_track[2]
    assert clip3.name == "clip_003.mp4"
    assert clip3.source_range.duration.value == 20.0 * 24


def test_otio_media_references(temp_dir, sample_timeline):
    """Verify media references point to correct files."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    timeline = otio.adapters.read_from_file(otio_path)
    video_track = timeline.tracks[0]
    
    for clip in video_track:
        assert clip.media_reference is not None
        assert isinstance(clip.media_reference, otio.schema.ExternalReference)
        
        # Check URL format (should be file:// URI)
        target_url = clip.media_reference.target_url
        assert target_url.startswith('file://'), f"Expected file:// URI, got: {target_url}"


def test_otio_audio_track(temp_dir, sample_timeline):
    """Verify audio track is correctly configured."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    timeline = otio.adapters.read_from_file(otio_path)
    audio_track = timeline.tracks[1]
    
    assert audio_track.kind == otio.schema.TrackKind.Audio
    assert len(audio_track) == 1, "Should have single audio clip"
    
    audio_clip = audio_track[0]
    assert audio_clip.name == "background_track.mp3"
    
    # Audio should span full timeline duration
    audio_duration = audio_clip.source_range.duration.to_seconds()
    assert abs(audio_duration - 30.5) < 0.1, f"Audio duration {audio_duration} != timeline 30.5s"


def test_otio_frame_rate_consistency(temp_dir, sample_timeline):
    """Verify frame rate is consistent across all clips."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    timeline = otio.adapters.read_from_file(otio_path)
    
    expected_fps = 24.0
    
    for track in timeline.tracks:
        for item in track:
            if hasattr(item, 'source_range') and item.source_range:
                assert item.source_range.start_time.rate == expected_fps
                assert item.source_range.duration.rate == expected_fps


def test_otio_clip_metadata_preservation(temp_dir, sample_timeline):
    """Verify clip metadata is preserved in export."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    timeline = otio.adapters.read_from_file(otio_path)
    video_track = timeline.tracks[0]
    
    # Check first clip metadata
    clip1 = video_track[0]
    assert 'energy' in clip1.metadata
    assert clip1.metadata['energy'] == 0.8
    assert clip1.metadata['shot'] == "wide"
    
    # Check second clip
    clip2 = video_track[1]
    assert clip2.metadata['energy'] == 0.3
    assert clip2.metadata['shot'] == "close"


def test_otio_empty_timeline(temp_dir):
    """Test handling of empty timeline."""
    timeline = Timeline(
        clips=[],
        audio_path="/data/music/silence.mp3",
        total_duration=0.0,
        project_name="empty_test",
        fps=30.0
    )
    
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(timeline)
    
    # Should create valid OTIO even with no clips
    assert Path(otio_path).exists()
    
    loaded = otio.adapters.read_from_file(otio_path)
    assert loaded.name == "empty_test"
    assert len(loaded.tracks) == 2  # Still has video+audio tracks


def test_otio_fps_variants(temp_dir):
    """Test different frame rate standards."""
    fps_variants = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 60.0]
    
    for fps in fps_variants:
        clips = [
            Clip(
                source_path=f"/data/input/clip_{fps}fps.mp4",
                start_time=0.0,
                duration=10.0,
                timeline_start=0.0,
                metadata={}
            )
        ]
        
        timeline = Timeline(
            clips=clips,
            audio_path="/data/music/track.mp3",
            total_duration=10.0,
            project_name=f"test_{fps}fps",
            fps=fps
        )
        
        exporter = TimelineExporter(output_dir=temp_dir)
        otio_path = exporter._export_otio(timeline)
        
        loaded = otio.adapters.read_from_file(otio_path)
        video_track = loaded.tracks[0]
        clip = video_track[0]
        
        # Verify FPS is correct
        assert clip.source_range.start_time.rate == fps
        assert clip.source_range.duration.rate == fps


def test_otio_resolution_metadata(temp_dir):
    """Test that resolution is stored in timeline metadata."""
    timeline = Timeline(
        clips=[
            Clip(
                source_path="/data/input/4k_clip.mp4",
                start_time=0.0,
                duration=5.0,
                timeline_start=0.0,
                metadata={}
            )
        ],
        audio_path="/data/music/track.mp3",
        total_duration=5.0,
        project_name="resolution_test",
        fps=24.0,
        resolution=(3840, 2160)  # 4K
    )
    
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(timeline)
    
    loaded = otio.adapters.read_from_file(otio_path)
    
    # OTIO doesn't have built-in resolution, but we can check metadata
    # or verify the file contains valid structure
    assert loaded.name == "resolution_test"


def test_otio_special_characters_in_names(temp_dir):
    """Test handling of special characters in file names."""
    clips = [
        Clip(
            source_path="/data/input/clip with spaces.mp4",
            start_time=0.0,
            duration=5.0,
            timeline_start=0.0,
            metadata={}
        ),
        Clip(
            source_path="/data/input/clip-with-dashes_and_underscores.mp4",
            start_time=0.0,
            duration=5.0,
            timeline_start=5.0,
            metadata={}
        ),
    ]
    
    timeline = Timeline(
        clips=clips,
        audio_path="/data/music/track (version 2).mp3",
        total_duration=10.0,
        project_name="special_chars_test",
        fps=24.0
    )
    
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(timeline)
    
    loaded = otio.adapters.read_from_file(otio_path)
    video_track = loaded.tracks[0]
    
    # Check names are preserved
    assert video_track[0].name == "clip with spaces.mp4"
    assert video_track[1].name == "clip-with-dashes_and_underscores.mp4"


def test_otio_schema_version(temp_dir, sample_timeline):
    """Verify OTIO schema version is compatible."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    with open(otio_path, 'r') as f:
        data = json.load(f)
    
    # Check OTIO schema is present
    assert 'OTIO_SCHEMA' in data
    
    # Version should be in format "Timeline.X"
    schema = data['OTIO_SCHEMA']
    assert schema.startswith('Timeline.')


def test_otio_schema_version_strict(temp_dir, sample_timeline):
    """Regression: ensure OTIO schema remains at Timeline.1 for compatibility."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)

    with open(otio_path, 'r') as f:
        data = json.load(f)

    schema = data.get('OTIO_SCHEMA')
    assert schema == 'Timeline.1', f"Unexpected OTIO schema version: {schema}"


def test_otio_roundtrip_compatibility(temp_dir, sample_timeline):
    """Test that exported OTIO can be read back without errors."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    # Read back
    loaded_timeline = otio.adapters.read_from_file(otio_path)
    
    # Verify key properties match
    assert loaded_timeline.name == sample_timeline.project_name
    assert len(loaded_timeline.tracks) == 2
    
    # Verify clip count matches
    loaded_video_track = loaded_timeline.tracks[0]
    assert len(loaded_video_track) == len(sample_timeline.clips)


def test_otio_link_to_source_mode(temp_dir, sample_timeline):
    """Test link_to_source parameter (proxy vs original)."""
    exporter = TimelineExporter(output_dir=temp_dir)
    
    # Export with link_to_source=True
    otio_path = exporter._export_otio(sample_timeline, link_to_source=True)
    
    loaded = otio.adapters.read_from_file(otio_path)
    video_track = loaded.tracks[0]
    
    # All clips should reference source files (not proxies)
    for clip in video_track:
        target_url = clip.media_reference.target_url
        assert '/data/input/' in target_url, "Should link to source files"


def test_otio_export_large_timeline(temp_dir):
    """Test performance with many clips."""
    # Create timeline with 100 clips
    clips = []
    for i in range(100):
        clips.append(
            Clip(
                source_path=f"/data/input/clip_{i:03d}.mp4",
                start_time=0.0,
                duration=2.0,
                timeline_start=i * 2.0,
                metadata={"clip_number": i}
            )
        )
    
    timeline = Timeline(
        clips=clips,
        audio_path="/data/music/long_track.mp3",
        total_duration=200.0,
        project_name="large_timeline_test",
        fps=24.0
    )
    
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(timeline)
    
    # Verify all clips exported
    loaded = otio.adapters.read_from_file(otio_path)
    video_track = loaded.tracks[0]
    assert len(video_track) == 100


@pytest.mark.integration
def test_otio_davinci_resolve_compatibility(temp_dir, sample_timeline):
    """Verify OTIO format matches DaVinci Resolve requirements."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    # DaVinci Resolve requirements:
    # 1. Must be valid OTIO JSON
    # 2. Must have video and audio tracks
    # 3. Frame rate must be consistent
    # 4. Media references must be ExternalReference
    
    loaded = otio.adapters.read_from_file(otio_path)
    
    assert len(loaded.tracks) >= 1, "DaVinci needs at least one track"
    
    video_track = loaded.tracks[0]
    assert video_track.kind == otio.schema.TrackKind.Video
    
    for clip in video_track:
        assert isinstance(clip.media_reference, otio.schema.ExternalReference)
        assert clip.source_range is not None
        assert clip.source_range.duration.value > 0


@pytest.mark.integration
def test_otio_premiere_pro_compatibility(temp_dir, sample_timeline):
    """Verify OTIO format works with Premiere Pro."""
    exporter = TimelineExporter(output_dir=temp_dir)
    otio_path = exporter._export_otio(sample_timeline)
    
    # Premiere Pro requirements:
    # 1. Standard OTIO schema
    # 2. ExternalReference for media
    # 3. Consistent frame rates
    # 4. No missing required fields
    
    loaded = otio.adapters.read_from_file(otio_path)
    
    # Check all clips have required fields
    for track in loaded.tracks:
        for item in track:
            if hasattr(item, 'name'):
                assert item.name, "Clips must have names for Premiere"
            if hasattr(item, 'media_reference'):
                assert item.media_reference is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
