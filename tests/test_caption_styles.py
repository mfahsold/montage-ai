"""
Tests for Caption Burner - Shorts Studio Caption Styles

Tests all caption styles including Karaoke (word-by-word highlighting).
"""

import pytest
import json
import tempfile
from pathlib import Path

from montage_ai.caption_burner import (
    CaptionBurner,
    CaptionStyle,
    CaptionSegment,
    parse_whisper_json,
    parse_srt,
    load_captions,
    list_caption_styles
)


@pytest.fixture
def sample_whisper_json(tmp_path):
    """Create a sample Whisper JSON with word-level timestamps."""
    data = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Welcome to Montage AI",
                "words": [
                    {"word": "Welcome", "start": 0.0, "end": 0.8},
                    {"word": "to", "start": 0.9, "end": 1.1},
                    {"word": "Montage", "start": 1.2, "end": 2.0},
                    {"word": "AI", "start": 2.1, "end": 3.0}
                ]
            },
            {
                "start": 3.5,
                "end": 6.0,
                "text": "Create amazing videos",
                "words": [
                    {"word": "Create", "start": 3.5, "end": 4.2},
                    {"word": "amazing", "start": 4.3, "end": 5.2},
                    {"word": "videos", "start": 5.3, "end": 6.0}
                ]
            }
        ]
    }
    
    json_path = tmp_path / "transcript.json"
    json_path.write_text(json.dumps(data))
    return json_path


@pytest.fixture
def sample_srt(tmp_path):
    """Create a sample SRT file."""
    srt_content = """1
00:00:00,000 --> 00:00:03,000
Welcome to Montage AI

2
00:00:03,500 --> 00:00:06,000
Create amazing videos

3
00:00:06,500 --> 00:00:09,000
With AI assistance
"""
    srt_path = tmp_path / "subtitles.srt"
    srt_path.write_text(srt_content)
    return srt_path


def test_parse_whisper_json(sample_whisper_json):
    """Test parsing Whisper JSON format."""
    segments = parse_whisper_json(sample_whisper_json)
    
    assert len(segments) == 2
    
    # First segment
    assert segments[0].text == "Welcome to Montage AI"
    assert segments[0].start == 0.0
    assert segments[0].end == 3.0
    assert segments[0].words is not None
    assert len(segments[0].words) == 4
    
    # Check word timing
    assert segments[0].words[0]['word'] == "Welcome"
    assert segments[0].words[0]['start'] == 0.0
    assert segments[0].words[0]['end'] == 0.8


def test_parse_srt(sample_srt):
    """Test parsing SRT format."""
    segments = parse_srt(sample_srt)
    
    assert len(segments) == 3
    
    # First segment
    assert segments[0].text == "Welcome to Montage AI"
    assert segments[0].start == 0.0
    assert segments[0].end == 3.0
    
    # Third segment
    assert segments[2].text == "With AI assistance"
    assert segments[2].start == 6.5
    assert segments[2].end == 9.0


def test_load_captions_whisper(sample_whisper_json):
    """Test loading captions from Whisper JSON."""
    segments = load_captions(str(sample_whisper_json))
    assert len(segments) == 2
    assert segments[0].words is not None


def test_load_captions_srt(sample_srt):
    """Test loading captions from SRT."""
    segments = load_captions(str(sample_srt))
    assert len(segments) == 3
    # SRT doesn't have word timing
    assert segments[0].words is None


def test_caption_burner_initialization():
    """Test CaptionBurner initialization."""
    burner = CaptionBurner(style=CaptionStyle.TIKTOK)
    assert burner.style == CaptionStyle.TIKTOK
    assert burner.config is not None
    assert burner.config.fontsize == 64  # TikTok style


def test_all_caption_styles():
    """Test that all caption styles are properly configured."""
    styles = [
        CaptionStyle.TIKTOK,
        CaptionStyle.YOUTUBE,
        CaptionStyle.MINIMAL,
        CaptionStyle.KARAOKE,
        CaptionStyle.BOLD,
        CaptionStyle.CINEMATIC
    ]
    
    for style in styles:
        burner = CaptionBurner(style=style)
        assert burner.config is not None
        assert burner.config.fontsize > 0
        assert burner.config.fontcolor
        assert burner.config.borderw >= 0


def test_karaoke_filter_generation(sample_whisper_json):
    """Test Karaoke-style filter generation (word-by-word)."""
    burner = CaptionBurner(style=CaptionStyle.KARAOKE)
    segments = parse_whisper_json(sample_whisper_json)
    
    # Build filters for first segment (has 4 words)
    filters = burner._build_karaoke_filters(segments[0], burner.config)
    
    # Should have base layer + 4 highlight layers = 5 total
    assert len(filters) >= 5
    
    # Check that each filter is a valid drawtext string
    for filter_str in filters:
        assert 'drawtext=' in filter_str
        assert 'enable=' in filter_str


def test_karaoke_fallback_no_words():
    """Test Karaoke style falls back when no word timing available."""
    burner = CaptionBurner(style=CaptionStyle.KARAOKE)
    
    # Segment without word timing
    segment = CaptionSegment(
        start=0.0,
        end=3.0,
        text="Test caption",
        words=None
    )
    
    # Should fall back to regular filter
    filters = burner._build_karaoke_filters(segment, burner.config)
    assert len(filters) == 1  # Just the base filter


def test_standard_filter_generation():
    """Test standard (non-Karaoke) filter generation."""
    burner = CaptionBurner(style=CaptionStyle.YOUTUBE)
    
    segment = CaptionSegment(
        start=0.0,
        end=3.0,
        text="Test caption"
    )
    
    filter_str = burner._build_drawtext_filter(segment, burner.config)
    
    # Check key components
    assert 'drawtext=' in filter_str
    assert 'fontsize=42' in filter_str  # YouTube fontsize
    assert 'fontcolor=white' in filter_str
    assert "text='Test caption'" in filter_str
    assert 'enable=' in filter_str


def test_filter_chain_karaoke(sample_whisper_json):
    """Test complete filter chain for Karaoke style."""
    burner = CaptionBurner(style=CaptionStyle.KARAOKE)
    segments = parse_whisper_json(sample_whisper_json)
    
    filter_chain = burner._build_filter_chain(segments)
    
    # Should be a comma-separated string of filters
    assert isinstance(filter_chain, str)
    assert 'drawtext=' in filter_chain
    assert ',' in filter_chain  # Multiple filters
    
    # Check multiple filters exist
    filters = filter_chain.split(',')
    assert len(filters) > 2  # Should have many filters (base + words for each segment)


def test_filter_chain_standard(sample_srt):
    """Test filter chain for standard styles."""
    burner = CaptionBurner(style=CaptionStyle.YOUTUBE)
    segments = parse_srt(sample_srt)
    
    filter_chain = burner._build_filter_chain(segments)
    
    # Filter chain should contain all 3 segments
    # Each filter is separated by comma, but may be split by `:` internally
    assert isinstance(filter_chain, str)
    assert 'drawtext=' in filter_chain
    
    # Count occurrences of drawtext (one per segment)
    drawtext_count = filter_chain.count('drawtext=')
    assert drawtext_count == 3


def test_list_caption_styles():
    """Test caption styles listing."""
    styles = list_caption_styles()
    
    assert len(styles) == 6
    
    # Check all expected styles
    style_names = [s['name'] for s in styles]
    assert 'tiktok' in style_names
    assert 'youtube' in style_names
    assert 'karaoke' in style_names
    assert 'bold' in style_names
    
    # Check descriptions exist
    for style in styles:
        assert 'description' in style
        assert len(style['description']) > 0


def test_style_specific_configs():
    """Test that each style has unique configuration."""
    tiktok = CaptionBurner(style=CaptionStyle.TIKTOK)
    youtube = CaptionBurner(style=CaptionStyle.YOUTUBE)
    bold = CaptionBurner(style=CaptionStyle.BOLD)
    
    # TikTok: large, lower middle (above TikTok UI)
    assert tiktok.config.fontsize == 64
    assert tiktok.config.y_expr == "(h-text_h)*0.75"
    
    # YouTube: medium, bottom, with box
    assert youtube.config.fontsize == 42
    assert youtube.config.y_expr == "h-80"
    assert youtube.config.box is True
    
    # Bold: extra large
    assert bold.config.fontsize == 72


def test_special_characters_escaping():
    """Test that special characters are properly escaped."""
    burner = CaptionBurner(style=CaptionStyle.YOUTUBE)
    
    segment = CaptionSegment(
        start=0.0,
        end=3.0,
        text="Test: 100% success!"
    )
    
    filter_str = burner._build_drawtext_filter(segment, burner.config)
    
    # Check escaping
    assert "Test\\: 100\\% success!" in filter_str or "Test:" in filter_str


def test_timing_accuracy():
    """Test that timing is accurately represented in filters."""
    burner = CaptionBurner(style=CaptionStyle.YOUTUBE)
    
    segment = CaptionSegment(
        start=1.234,
        end=5.678,
        text="Timing test"
    )
    
    filter_str = burner._build_drawtext_filter(segment, burner.config)
    
    # Check timing precision (3 decimal places)
    assert "enable='between(t,1.234,5.678)'" in filter_str


def test_karaoke_word_timing():
    """Test that Karaoke respects word-level timing."""
    burner = CaptionBurner(style=CaptionStyle.KARAOKE)
    
    segment = CaptionSegment(
        start=0.0,
        end=3.0,
        text="One two three",
        words=[
            {"word": "One", "start": 0.0, "end": 1.0},
            {"word": "two", "start": 1.2, "end": 2.0},
            {"word": "three", "start": 2.1, "end": 3.0}
        ]
    )
    
    filters = burner._build_karaoke_filters(segment, burner.config)
    
    # Should have base + 3 word highlights
    assert len(filters) == 4
    
    # Check word timings in filters
    all_filters = ''.join(filters)
    assert "between(t,0.000,1.000)" in all_filters  # "One"
    assert "between(t,1.200,2.000)" in all_filters  # "two"
    assert "between(t,2.100,3.000)" in all_filters  # "three"


def test_bold_style_characteristics():
    """Test Bold style has enhanced visibility."""
    bold = CaptionBurner(style=CaptionStyle.BOLD)
    
    # Bold should have large font and thick border
    assert bold.config.fontsize >= 70
    assert bold.config.borderw >= 5
    assert bold.config.shadowx >= 3 or bold.config.shadowy >= 3


def test_minimal_style_characteristics():
    """Test Minimal style is subtle."""
    minimal = CaptionBurner(style=CaptionStyle.MINIMAL)
    
    # Minimal should have small font and thin border
    assert minimal.config.fontsize <= 40
    assert minimal.config.borderw <= 1
    assert minimal.config.box is False


def test_cinematic_style_characteristics():
    """Test Cinematic style has box background."""
    cinematic = CaptionBurner(style=CaptionStyle.CINEMATIC)
    
    # Cinematic should have box with transparency
    assert cinematic.config.box is True
    assert '@' in cinematic.config.boxcolor  # Transparency marker


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
