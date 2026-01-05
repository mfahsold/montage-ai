"""
Unit tests for DRY refactoring modules.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAPIDecorators:
    """Tests for web_ui/decorators.py"""

    def test_api_endpoint_success(self):
        """api_endpoint passes through successful responses."""
        from src.montage_ai.web_ui.decorators import api_endpoint

        @api_endpoint
        def success_func():
            return {"data": "test"}, 200

        result = success_func()
        assert result == ({"data": "test"}, 200)

    def test_api_endpoint_file_not_found(self):
        """api_endpoint returns 404 for FileNotFoundError."""
        from flask import Flask
        from src.montage_ai.web_ui.decorators import api_endpoint

        app = Flask(__name__)

        @api_endpoint
        def not_found_func():
            raise FileNotFoundError("test.mp4 not found")

        with app.app_context():
            response, status = not_found_func()
            assert status == 404
            assert "error" in response.json

    def test_api_endpoint_value_error(self):
        """api_endpoint returns 400 for ValueError."""
        from flask import Flask
        from src.montage_ai.web_ui.decorators import api_endpoint

        app = Flask(__name__)

        @api_endpoint
        def validation_func():
            raise ValueError("Invalid parameter")

        with app.app_context():
            response, status = validation_func()
            assert status == 400

    def test_api_endpoint_generic_error(self):
        """api_endpoint returns 500 for generic exceptions."""
        from flask import Flask
        from src.montage_ai.web_ui.decorators import api_endpoint

        app = Flask(__name__)

        @api_endpoint
        def error_func():
            raise RuntimeError("Something broke")

        with app.app_context():
            response, status = error_func()
            assert status == 500

    def test_require_json_validates_fields(self):
        """require_json rejects missing fields."""
        from flask import Flask
        from src.montage_ai.web_ui.decorators import require_json

        app = Flask(__name__)

        @require_json('filename', 'style')
        def needs_fields():
            return {"ok": True}, 200

        with app.test_request_context(json={"filename": "test.mp4"}):
            response, status = needs_fields()
            assert status == 400
            assert "style" in response.json["error"]

    def test_require_json_passes_valid(self):
        """require_json passes when all fields present."""
        from flask import Flask
        from src.montage_ai.web_ui.decorators import require_json

        app = Flask(__name__)

        @require_json('filename', 'style')
        def needs_fields():
            return {"ok": True}, 200

        with app.test_request_context(json={"filename": "test.mp4", "style": "dynamic"}):
            response, status = needs_fields()
            assert status == 200


class TestFFmpegAtomics:
    """Tests for core/ffmpeg_atomics.py"""

    def test_atomic_ffmpeg_success(self):
        """atomic_ffmpeg renames temp to final on success."""
        from src.montage_ai.core.ffmpeg_atomics import atomic_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.mp4"

            with atomic_ffmpeg(output) as temp_path:
                # Simulate FFmpeg creating the file
                Path(temp_path).write_text("test content")

            assert output.exists()
            assert not Path(temp_path).exists()
            assert output.read_text() == "test content"

    def test_atomic_ffmpeg_failure_cleanup(self):
        """atomic_ffmpeg removes temp on failure."""
        from src.montage_ai.core.ffmpeg_atomics import atomic_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.mp4"

            with pytest.raises(RuntimeError):
                with atomic_ffmpeg(output) as temp_path:
                    # Create temp file
                    Path(temp_path).write_text("partial")
                    # Simulate failure
                    raise RuntimeError("FFmpeg failed")

            assert not output.exists()
            # Temp should be cleaned up
            assert not Path(temp_path).exists()

    def test_concat_list_manager_escapes_paths(self):
        """ConcatListManager properly escapes single quotes."""
        from src.montage_ai.core.ffmpeg_atomics import ConcatListManager

        with tempfile.TemporaryDirectory() as tmpdir:
            with ConcatListManager(tmpdir, "test") as concat:
                concat.write([
                    "/path/to/file.mp4",
                    "/path/with'quote.mp4",
                ])

                content = Path(concat.path).read_text()
                assert "file '/path/to/file.mp4'" in content
                assert "'\\''" in content  # Escaped quote

            # Should be cleaned up
            assert not Path(concat.path).exists()

    def test_concat_list_manager_cleanup(self):
        """ConcatListManager cleans up on exit."""
        from src.montage_ai.core.ffmpeg_atomics import ConcatListManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConcatListManager(tmpdir, "cleanup_test")

            with manager:
                manager.write(["/test.mp4"])
                assert Path(manager.path).exists()

            assert not Path(manager.path).exists()


class TestAudioQuality:
    """Tests for AudioQuality dataclass."""

    def test_quality_level_alias(self):
        """quality_level is an alias for quality_tier."""
        from src.montage_ai.audio_analysis import AudioQuality

        quality = AudioQuality(
            snr_db=35.0,
            mean_volume_db=-15.0,
            max_volume_db=-5.0,
            is_usable=True,
            quality_tier="good"
        )

        assert quality.quality_level == "good"
        assert quality.quality_level == quality.quality_tier

    def test_quality_warning_tiers(self):
        """warning property returns appropriate messages."""
        from src.montage_ai.audio_analysis import AudioQuality

        # Unusable
        unusable = AudioQuality(5.0, -30.0, -20.0, False, "unusable")
        assert "re-recording" in unusable.warning

        # Poor
        poor = AudioQuality(10.0, -25.0, -15.0, True, "poor")
        assert "unreliable" in poor.warning

        # Good - no warning
        good = AudioQuality(30.0, -15.0, -5.0, True, "good")
        assert good.warning is None
