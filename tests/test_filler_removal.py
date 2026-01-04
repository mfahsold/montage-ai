import pytest
from montage_ai.audio_analysis import remove_filler_words

def test_remove_filler_words_segments():
    transcript = {
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0, "end": 1},
                    {"word": "um", "start": 1, "end": 2},
                    {"word": "world", "start": 2, "end": 3}
                ]
            }
        ]
    }
    indices = remove_filler_words(transcript)
    assert indices == [1]

def test_remove_filler_words_flat():
    transcript = {
        "words": [
            {"word": "This"},
            {"word": "is"},
            {"word": "like"},
            {"word": "cool"}
        ]
    }
    indices = remove_filler_words(transcript)
    assert indices == [2]

def test_remove_filler_words_custom():
    transcript = {
        "words": [{"word": "foo"}, {"word": "bar"}]
    }
    indices = remove_filler_words(transcript, fillers=["foo"])
    assert indices == [0]

def test_remove_filler_words_punctuation():
    transcript = {
        "words": [{"word": "Um,"}, {"word": "uh..."}]
    }
    indices = remove_filler_words(transcript)
    assert indices == [0, 1]
