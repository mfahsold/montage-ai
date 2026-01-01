"""Integration tests using real media files.

These tests require synthetic fixtures generated via `make test-fixtures`.
They are skipped automatically if fixtures are not present.

Run with: pytest tests/integration/ -v
"""

import subprocess

import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestRealVideoProcessing:
    """Tests using actual video files."""

    def test_ffprobe_reads_test_video(self, test_video):
        """Verify ffprobe can read the synthetic test video."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(test_video),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "format" in result.stdout

    def test_video_has_expected_duration(self, test_video):
        """Verify test video is approximately 5 seconds."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(test_video),
            ],
            capture_output=True,
            text=True,
        )
        duration = float(result.stdout.strip())
        assert 4.5 <= duration <= 5.5, f"Expected ~5s, got {duration}s"


@pytest.mark.integration
@pytest.mark.slow
class TestRealAudioProcessing:
    """Tests using actual audio files."""

    def test_ffprobe_reads_test_audio(self, test_audio):
        """Verify ffprobe can read the synthetic test audio."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(test_audio),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "format" in result.stdout

    def test_audio_has_expected_duration(self, test_audio):
        """Verify test audio is approximately 10 seconds."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(test_audio),
            ],
            capture_output=True,
            text=True,
        )
        duration = float(result.stdout.strip())
        assert 9.5 <= duration <= 10.5, f"Expected ~10s, got {duration}s"


@pytest.mark.integration
class TestBunnyTrailer:
    """Tests using the included minimal bunny trailer."""

    def test_bunny_trailer_exists(self, bunny_trailer):
        """Verify the bunny trailer file exists."""
        assert bunny_trailer.exists()
        assert bunny_trailer.stat().st_size > 0

    def test_bunny_trailer_is_valid_video(self, bunny_trailer):
        """Verify bunny trailer is a valid video file."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(bunny_trailer),
            ],
            capture_output=True,
            text=True,
        )
        assert "video" in result.stdout
