"""
Tests for Transcriber wrapper module.

Tests the backward-compatible Transcriber class that now
delegates to TranscribeJob.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.montage_ai.transcriber import Transcriber, transcribe_audio


class TestTranscriber(unittest.TestCase):
    """Test cases for Transcriber class."""

    def test_init_default_model(self):
        """Transcriber initializes with default model."""
        t = Transcriber()
        self.assertEqual(t.model, "medium")

    def test_init_custom_model(self):
        """Transcriber accepts custom model."""
        t = Transcriber(model="large")
        self.assertEqual(t.model, "large")

    @patch('src.montage_ai.transcriber.is_cgpu_available')
    def test_is_available_true(self, mock_available):
        """is_available returns True when cgpu is available."""
        mock_available.return_value = True
        t = Transcriber()
        self.assertTrue(t.is_available())

    @patch('src.montage_ai.transcriber.is_cgpu_available')
    def test_is_available_false(self, mock_available):
        """is_available returns False when cgpu is unavailable."""
        mock_available.return_value = False
        t = Transcriber()
        self.assertFalse(t.is_available())

    @patch('src.montage_ai.transcriber.TranscribeJob')
    def test_transcribe_success(self, mock_job_class):
        """Transcribe returns output path on success."""
        mock_job = MagicMock()
        mock_job.execute.return_value = MagicMock(
            success=True,
            output_path="/tmp/audio.srt"
        )
        mock_job_class.return_value = mock_job

        t = Transcriber()
        result = t.transcribe("/tmp/audio.wav")

        self.assertEqual(result, "/tmp/audio.srt")
        mock_job_class.assert_called_once_with(
            audio_path="/tmp/audio.wav",
            model="medium",
            output_format="srt",
            language=None,
        )

    @patch('src.montage_ai.transcriber.TranscribeJob')
    def test_transcribe_failure(self, mock_job_class):
        """Transcribe returns None on failure."""
        mock_job = MagicMock()
        mock_job.execute.return_value = MagicMock(
            success=False,
            output_path=None,
            error="Failed"
        )
        mock_job_class.return_value = mock_job

        t = Transcriber()
        result = t.transcribe("/tmp/audio.wav")

        self.assertIsNone(result)

    @patch('src.montage_ai.transcriber.TranscribeJob')
    def test_transcribe_with_options(self, mock_job_class):
        """Transcribe passes options to TranscribeJob."""
        mock_job = MagicMock()
        mock_job.execute.return_value = MagicMock(
            success=True,
            output_path="/tmp/audio.vtt"
        )
        mock_job_class.return_value = mock_job

        t = Transcriber(model="large")
        result = t.transcribe("/tmp/audio.wav", output_format="vtt", language="de")

        mock_job_class.assert_called_once_with(
            audio_path="/tmp/audio.wav",
            model="large",
            output_format="vtt",
            language="de",
        )


class TestConvenienceFunction(unittest.TestCase):
    """Test cases for transcribe_audio convenience function."""

    @patch('src.montage_ai.transcriber.TranscribeJob')
    def test_transcribe_audio_success(self, mock_job_class):
        """transcribe_audio returns path on success."""
        mock_job = MagicMock()
        mock_job.execute.return_value = MagicMock(
            success=True,
            output_path="/tmp/audio.srt"
        )
        mock_job_class.return_value = mock_job

        result = transcribe_audio("/tmp/audio.wav")
        self.assertEqual(result, "/tmp/audio.srt")

    @patch('src.montage_ai.transcriber.TranscribeJob')
    def test_transcribe_audio_with_options(self, mock_job_class):
        """transcribe_audio passes all options."""
        mock_job = MagicMock()
        mock_job.execute.return_value = MagicMock(
            success=True,
            output_path="/tmp/audio.json"
        )
        mock_job_class.return_value = mock_job

        result = transcribe_audio(
            "/tmp/audio.wav",
            model="large-v3",
            output_format="json",
            language="en"
        )

        mock_job_class.assert_called_once_with(
            audio_path="/tmp/audio.wav",
            model="large-v3",
            output_format="json",
            language="en",
        )


if __name__ == '__main__':
    unittest.main()
