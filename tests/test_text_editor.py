import pytest
from unittest.mock import MagicMock, patch
from montage_ai.text_editor import TextEditor, Segment, Word, EditRegion

@pytest.fixture
def mock_editor(tmp_path):
    video_path = tmp_path / "dummy.mp4"
    video_path.touch()
    
    editor = TextEditor(str(video_path))
    # Create a dummy transcript: "Hello world this is a test"
    # 0.0-0.5: Hello
    # 0.5-1.0: world
    # 1.0-1.5: this
    # 1.5-2.0: is
    # 2.0-2.5: a
    # 2.5-3.0: test
    
    words = [
        Word("Hello", 0.0, 0.5),
        Word("world", 0.5, 1.0),
        Word("this", 1.0, 1.5),
        Word("is", 1.5, 2.0),
        Word("a", 2.0, 2.5),
        Word("test", 2.5, 3.0)
    ]
    
    editor.segments = [
        Segment(0, 0.0, 3.0, "Hello world this is a test", words)
    ]
    editor._video_duration = 3.0
    return editor

def test_get_cut_list_no_edits(mock_editor):
    cut_list = mock_editor.get_cut_list(padding=0, merge_threshold=0)
    assert len(cut_list) == 1
    assert cut_list[0].start == 0.0
    assert cut_list[0].end == 3.0

def test_get_cut_list_remove_word(mock_editor):
    # Remove "this" (1.0-1.5)
    mock_editor.segments[0].words[2].removed = True
    
    cut_list = mock_editor.get_cut_list(padding=0, merge_threshold=0)
    # Should have 2 regions: 0.0-1.0 and 1.5-3.0
    assert len(cut_list) == 2
    assert cut_list[0].start == 0.0
    assert cut_list[0].end == 1.0
    assert cut_list[1].start == 1.5
    assert cut_list[1].end == 3.0

def test_export_calls_ffmpeg_with_scaling(mock_editor):
    with patch("subprocess.run") as mock_run:
        mock_editor.export("output.mp4", width=640, height=360)
        
        args = mock_run.call_args[0][0]
        # Check if scale filter is present
        filter_complex = args[args.index("-filter_complex") + 1]
        assert "scale=640:360" in filter_complex
        assert "force_original_aspect_ratio=decrease" in filter_complex

def test_export_calls_ffmpeg_without_scaling(mock_editor):
    with patch("subprocess.run") as mock_run:
        mock_editor.export("output.mp4")
        
        args = mock_run.call_args[0][0]
        # Check if scale filter is NOT present
        filter_complex = args[args.index("-filter_complex") + 1]
        assert "scale=" not in filter_complex
