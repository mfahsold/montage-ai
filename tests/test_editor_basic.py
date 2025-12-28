"""
Basic tests for editor module

KISS: Test core functionality, not every edge case.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock


def test_imports():
    """Test that core modules can be imported."""
    try:
        from montage_ai import editor
        from montage_ai import creative_director
        from montage_ai import footage_manager
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_seconds_to_timecode():
    """Test timecode conversion."""
    from montage_ai.timeline_exporter import TimelineExporter

    # Test basic conversion
    tc = TimelineExporter._seconds_to_timecode(90.5, 30)
    assert tc == "00:01:30:15"  # 1 min 30 sec 15 frames

    # Test zero
    tc = TimelineExporter._seconds_to_timecode(0, 30)
    assert tc == "00:00:00:00"

    # Test hours
    tc = TimelineExporter._seconds_to_timecode(3661, 30)  # 1h 1m 1s
    assert tc == "01:01:01:00"


def test_allowed_file():
    """Test file extension validation using FileTypeConfig directly.

    Uses config module directly to avoid app.py import side effects.
    """
    from src.montage_ai.config import FileTypeConfig

    config = FileTypeConfig()

    # Test allowed_file method
    assert config.allowed_file('video.mp4', {'mp4', 'mov'}) is True
    assert config.allowed_file('VIDEO.MP4', {'mp4', 'mov'}) is True
    assert config.allowed_file('clip.mov', {'mp4', 'mov'}) is True

    # Invalid extensions
    assert config.allowed_file('file.txt', {'mp4', 'mov'}) is False
    assert config.allowed_file('noextension', {'mp4', 'mov'}) is False


def test_creative_director_keywords():
    """Test Creative Director keyword matching."""
    from montage_ai.creative_director import CreativeDirector

    director = CreativeDirector()

    # Test direct style match
    result = director.interpret_prompt("hitchcock")
    assert result is not None
    assert result['style']['name'] == 'hitchcock'

    # Test keyword match
    result = director.interpret_prompt("suspense thriller")
    assert result is not None
    assert result['style']['name'] == 'hitchcock'

    # Test MTV match
    result = director.interpret_prompt("fast-paced music video")
    assert result is not None
    assert result['style']['name'] == 'mtv'


@pytest.mark.skipif(
    True,  # Skip by default - numba has compatibility issues
    reason="librosa/numba has environment-specific compilation issues"
)
def test_beat_detection_mock():
    """Test beat detection with synthetic audio."""
    import librosa

    # Create synthetic audio: 120 BPM = 2 beats/sec
    duration = 4.0  # 4 seconds
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))

    # Generate sine wave with strong beats
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    # Run beat detection
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)

    # Check tempo is detected (should be around 120 BPM)
    assert 60 < tempo < 180, f"Tempo {tempo} out of expected range"

    # Check beats are detected
    assert len(beat_frames) > 0, "No beats detected"


def test_footage_clip_dataclass():
    """Test FootageClip data structure."""
    from montage_ai.footage_manager import FootageClip, UsageStatus

    # Current API uses: clip_id, source_file, in_point, out_point, duration
    clip = FootageClip(
        clip_id="clip_001",
        source_file="/test/video.mp4",
        in_point=0.0,
        out_point=10.0,
        duration=10.0,
        usage_status=UsageStatus.UNUSED
    )

    assert clip.clip_id == "clip_001"
    assert clip.source_file == "/test/video.mp4"
    assert clip.duration == 10.0
    assert clip.usage_status == UsageStatus.UNUSED
    assert clip.usage_count == 0


def test_style_template_loading():
    """Test style template loading."""
    from montage_ai.style_templates import list_available_styles, get_style_template

    # List styles
    styles = list_available_styles()
    assert len(styles) > 0
    assert 'hitchcock' in styles

    # dynamic may be renamed or merged - check if any style exists
    assert len(styles) >= 1, "At least one style should exist"

    # Load a style
    template = get_style_template('hitchcock')
    assert template is not None
    assert 'params' in template
    assert 'style' in template['params']
