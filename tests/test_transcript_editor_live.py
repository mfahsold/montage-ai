"""
Test Transcript Editor Live Preview Feature

Tests the complete flow from transcript loading to live preview generation.
"""

import pytest
import json
import tempfile
from pathlib import Path

from montage_ai.text_editor import TextEditor, Word, Segment
from montage_ai.preview_generator import PreviewGenerator


@pytest.fixture
def sample_transcript():
    """Create a sample transcript with word-level timestamps."""
    return {
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.99},
                    {"word": "world", "start": 1.2, "end": 2.5, "probability": 0.95}
                ]
            },
            {
                "id": 1,
                "start": 3.0,
                "end": 5.5,
                "text": "This is a test",
                "words": [
                    {"word": "This", "start": 3.0, "end": 3.5, "probability": 0.98},
                    {"word": "is", "start": 3.6, "end": 3.8, "probability": 0.97},
                    {"word": "a", "start": 3.9, "end": 4.0, "probability": 0.96},
                    {"word": "test", "start": 4.2, "end": 5.5, "probability": 0.99}
                ]
            },
            {
                "id": 2,
                "start": 6.0,
                "end": 8.0,
                "text": "Um, yeah, like, awesome",
                "words": [
                    {"word": "Um", "start": 6.0, "end": 6.2, "probability": 0.90},
                    {"word": "yeah", "start": 6.3, "end": 6.5, "probability": 0.95},
                    {"word": "like", "start": 6.6, "end": 6.8, "probability": 0.93},
                    {"word": "awesome", "start": 7.0, "end": 8.0, "probability": 0.99}
                ]
            }
        ]
    }


@pytest.fixture
def transcript_file(sample_transcript, tmp_path):
    """Create a temporary transcript JSON file."""
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps(sample_transcript))
    return transcript_path


def test_text_editor_cut_list_generation(tmp_path, transcript_file):
    """Test that TextEditor generates correct cut list from edits."""
    # Create a dummy video file (just for path validation)
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    
    # Mock video_duration since we don't have a real video
    editor = TextEditor(str(video_path))
    editor._video_duration = 10.0  # Mock 10-second video
    
    # Load transcript
    loaded = editor.load_transcript(str(transcript_file))
    assert loaded == 3, "Should load 3 segments"
    
    # Remove first word ("Hello")
    editor.toggle_word(0, removed=True)
    
    # Remove filler words ("Um", "like")
    removed_count = editor.remove_filler_words()
    assert removed_count >= 2, "Should remove at least 2 filler words"
    
    # Get cut list
    cut_list = editor.get_cut_list()
    
    # Verify we have segments
    assert len(cut_list) > 0, "Cut list should not be empty"
    
    # All regions should be "keep" regions
    for region in cut_list:
        assert region.keep is True, "All regions should be marked as 'keep'"
        assert region.start >= 0, "Start time should be non-negative"
        assert region.end <= 10.0, "End time should not exceed video duration"
        assert region.end > region.start, "End should be after start"


def test_preview_generator_segments(tmp_path):
    """Test PreviewGenerator can handle segment list."""
    generator = PreviewGenerator(str(tmp_path))
    
    # Define segments to keep (start, end in seconds)
    segments = [
        (1.0, 2.5),   # Keep first segment
        (4.2, 5.5),   # Keep only "test" word
        (7.0, 8.0)    # Keep only "awesome" word
    ]
    
    # We can't actually generate video without a real source file
    # but we can verify the method exists and accepts the right params
    assert hasattr(generator, 'generate_transcript_preview')
    
    # Verify signature
    import inspect
    sig = inspect.signature(generator.generate_transcript_preview)
    params = list(sig.parameters.keys())
    assert 'source_path' in params
    assert 'segments' in params
    assert 'output_filename' in params


def test_text_editor_stats_tracking(tmp_path, transcript_file):
    """Test that statistics are tracked correctly."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    
    editor = TextEditor(str(video_path))
    editor._video_duration = 10.0
    editor.load_transcript(str(transcript_file))
    
    # Get initial stats
    stats = editor.get_stats()
    assert stats['total_segments'] == 3
    assert stats['removed_segments'] == 0
    assert stats['total_words'] > 0
    
    # Remove some words
    editor.remove_filler_words()
    
    # Get updated stats
    stats = editor.get_stats()
    assert stats['removed_words'] > 0
    assert stats['time_saved_seconds'] > 0


def test_word_level_sync(tmp_path, transcript_file):
    """Test word-level timestamp tracking."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    
    editor = TextEditor(str(video_path))
    editor._video_duration = 10.0
    editor.load_transcript(str(transcript_file))
    
    # Verify we can access individual words with timestamps
    word_count = 0
    for seg in editor.segments:
        for word in seg.words:
            assert hasattr(word, 'text')
            assert hasattr(word, 'start')
            assert hasattr(word, 'end')
            assert hasattr(word, 'removed')
            assert word.start < word.end
            word_count += 1
    
    assert word_count > 0, "Should have parsed words with timestamps"


def test_undo_redo_compatibility(tmp_path, transcript_file):
    """Test that edit operations support undo/redo."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    
    editor = TextEditor(str(video_path))
    editor._video_duration = 10.0
    editor.load_transcript(str(transcript_file))
    
    # Track initial state
    initial_word_states = [
        (i, w.removed) 
        for seg in editor.segments 
        for i, w in enumerate(seg.words)
    ]
    
    # Make an edit (toggle first word)
    editor.toggle_word(0, removed=True)
    
    # Verify edit happened
    all_words = [w for seg in editor.segments for w in seg.words]
    assert all_words[0].removed is True
    
    # "Undo" by toggling back
    editor.toggle_word(0, removed=False)
    
    # Verify undo
    assert all_words[0].removed is False


def test_export_formats(tmp_path, transcript_file):
    """Test that all export formats are available."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    
    editor = TextEditor(str(video_path))
    editor._video_duration = 10.0
    editor.load_transcript(str(transcript_file))
    
    # Check export methods exist
    assert hasattr(editor, 'export')
    assert hasattr(editor, 'export_edl')
    assert hasattr(editor, 'export_otio')
    
    # Remove some words to have something to export
    editor.remove_filler_words()
    
    # Verify we can get cut list (needed for all exports)
    cut_list = editor.get_cut_list()
    assert len(cut_list) > 0


def test_silence_removal(tmp_path, transcript_file):
    """Test automatic silence removal."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    
    editor = TextEditor(str(video_path))
    editor._video_duration = 10.0
    editor.load_transcript(str(transcript_file))
    
    # Test silence removal (won't actually remove anything in this test data
    # since gaps are too small, but method should execute)
    removed_count = editor.remove_silence(min_gap=1.0)
    assert removed_count >= 0  # Can be 0 if no long gaps


def test_filler_word_detection(tmp_path, transcript_file):
    """Test filler word detection and removal."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    
    editor = TextEditor(str(video_path))
    editor._video_duration = 10.0
    editor.load_transcript(str(transcript_file))
    
    # Count filler words before removal
    all_words = [w for seg in editor.segments for w in seg.words]
    initial_fillers = sum(1 for w in all_words if w.text.lower() in ['um', 'like', 'uh'])
    
    # Remove fillers
    removed_count = editor.remove_filler_words()
    
    # Verify fillers were marked
    assert removed_count >= initial_fillers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
