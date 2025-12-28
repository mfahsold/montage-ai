"""Tests for Transcriber module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Mock dependencies
sys.modules['openai'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from montage_ai.transcriber import Transcriber


class TestTranscriber(unittest.TestCase):
    """Test Transcriber class."""

    def setUp(self):
        # Mock cgpu functions with CORRECT names
        self.mock_is_available = patch('montage_ai.transcriber.is_cgpu_available', return_value=True)
        self.mock_run_command = patch('montage_ai.transcriber.run_cgpu_command', return_value=True)
        self.mock_copy = patch('montage_ai.transcriber.copy_to_remote', return_value=True)
        self.mock_download = patch('montage_ai.transcriber.download_via_base64', return_value=True)

        self.is_available = self.mock_is_available.start()
        self.run_command = self.mock_run_command.start()
        self.copy_to_remote = self.mock_copy.start()
        self.download = self.mock_download.start()

        self.transcriber = Transcriber()

    def tearDown(self):
        self.mock_is_available.stop()
        self.mock_run_command.stop()
        self.mock_copy.stop()
        self.mock_download.stop()

    def test_init_default_model(self):
        """Default model is 'medium'."""
        t = Transcriber()
        self.assertEqual(t.model, "medium")

    def test_init_custom_model(self):
        """Accept custom model size."""
        t = Transcriber(model="large")
        self.assertEqual(t.model, "large")

    def test_is_available_true(self):
        """Check availability when cgpu is available."""
        self.assertTrue(self.transcriber.is_available())

    def test_is_available_false(self):
        """Check availability when cgpu is not available."""
        self.is_available.return_value = False
        self.assertFalse(self.transcriber.is_available())

    def test_transcribe_no_cgpu(self):
        """Return None when cgpu is unavailable."""
        self.is_available.return_value = False
        result = self.transcriber.transcribe("test_audio.wav")
        self.assertIsNone(result)

    def test_transcribe_file_not_found(self):
        """Return None when audio file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            result = self.transcriber.transcribe("/nonexistent/audio.wav")
            self.assertIsNone(result)

    def test_transcribe_success(self):
        """Successful transcription flow."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.resolve', return_value=MagicMock(
                 exists=MagicMock(return_value=True),
                 name="test.wav",
                 stem="test",
                 parent=MagicMock(__truediv__=lambda s, x: f"/data/{x}")
             )), \
             patch('os.path.getsize', return_value=1024):

            # Mock Path behavior
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.wav"
            mock_path.stem = "test"
            mock_path.parent.__truediv__ = lambda x: f"/data/{x}"

            with patch('pathlib.Path', return_value=mock_path):
                result = self.transcriber.transcribe("test_audio.wav")

            # Verify cgpu functions were called
            self.assertTrue(self.copy_to_remote.called)
            self.assertTrue(self.run_command.called)

    def test_transcribe_upload_fails(self):
        """Return None when upload fails."""
        self.copy_to_remote.return_value = False

        with patch('pathlib.Path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024):

            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.wav"

            with patch('pathlib.Path', return_value=mock_path):
                result = self.transcriber.transcribe("test_audio.wav")
                self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
