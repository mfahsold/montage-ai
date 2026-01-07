"""
Unit tests for DRY refactoring modules.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from flask import Flask, jsonify, Response

from src.montage_ai.web_ui.decorators import api_endpoint, require_json
from src.montage_ai.core.ffmpeg_atomics import run_ffmpeg_atomic, ConcatListManager

class TestAPIDecorators:
    """Tests for web_ui/decorators.py"""

    @pytest.fixture
    def app(self):
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app

    def test_api_endpoint_success(self, app):
        """api_endpoint passes through successful responses."""
        
        with app.app_context():
            @api_endpoint
            def success_func():
                return {"data": "test"}

            # Decorator wraps function but does not force jsonify on success path
            # unless the function explicitly does so.
            # exceptions are caught and returned as (Response, code).
            
            result = success_func()
            assert result == {"data": "test"}

    def test_api_endpoint_file_not_found(self, app):
        """api_endpoint returns 404 for FileNotFoundError."""

        with app.test_request_context():
            @api_endpoint
            def not_found_func():
                raise FileNotFoundError("test.mp4")

            # Returns (Response, 404)
            result = not_found_func()
            assert isinstance(result, tuple)
            response, code = result
            assert code == 404
            assert response.is_json
            assert "error" in response.json

    def test_api_endpoint_value_error(self, app):
        """api_endpoint returns 400 for ValueError."""
        
        with app.test_request_context():
            @api_endpoint
            def validation_func():
                raise ValueError("Invalid parameter")

            result = validation_func()
            response, code = result
            assert code == 400
            assert "error" in response.json

    def test_api_endpoint_generic_error(self, app):
        """api_endpoint returns 500 for generic exceptions."""
        
        with app.test_request_context():
            @api_endpoint
            def error_func():
                raise RuntimeError("Something broke")

            result = error_func()
            response, code = result
            assert code == 500
            assert "error" in response.json

    def test_require_json_validates_fields(self, app):
        """require_json rejects missing fields."""
        
        # Pass fields as varargs, NOT list
        @require_json('filename', 'style')
        def needs_fields(data):
            return {"ok": True}

        with app.test_request_context(json={"filename": "test.mp4"}):
            # require_json decorates. If validation fails, it returns (Response, 400)?
            # Let's check decorator source:
            # return jsonify({"error": ...}), 400
            result = needs_fields()
            
            assert isinstance(result, tuple)
            response, code = result
            assert code == 400
            assert "style" in response.json["error"]

    def test_require_json_passes_valid(self, app):
        """require_json passes when all fields present."""
        
        @require_json('filename')
        def needs_fields():
            # Signature of decorated function matches wrapper
            # wrapper takes *args, **kwargs.
            # But here `needs_fields` takes no args? 
            # In decorator source: return f(*args, **kwargs)
            return {"ok": True}

        with app.test_request_context(json={"filename": "success.mp4"}):
            result = needs_fields()
            assert result == {"ok": True}


class TestFFmpegAtomics:
    """Tests for core/ffmpeg_atomics.py"""

    def test_atomic_ffmpeg_success(self, tmp_path):
        """Test that run_ffmpeg_atomic renames file on success."""
        output_file = tmp_path / "final.mp4"
        
        with patch("src.montage_ai.core.ffmpeg_atomics.run_command") as mock_run, \
             patch("src.montage_ai.core.ffmpeg_atomics.os.rename") as mock_rename, \
             patch("src.montage_ai.core.ffmpeg_atomics.os.path.exists", return_value=True):
             
            mock_run.return_value.returncode = 0
            
            result = run_ffmpeg_atomic(["-i", "input"], output_file)
            
            assert result is True
            mock_rename.assert_called_once()
            
    def test_atomic_ffmpeg_failure_cleanup(self, tmp_path):
        """Test that atomic_ffmpeg cleans up temp file on failure."""
        output_file = tmp_path / "final.mp4"
        
        with patch("src.montage_ai.core.ffmpeg_atomics.run_command") as mock_run, \
             patch("src.montage_ai.core.ffmpeg_atomics.os.remove") as mock_remove, \
             patch("src.montage_ai.core.ffmpeg_atomics.os.path.exists", return_value=True):
             
            mock_run.return_value.returncode = 1
            
            result = run_ffmpeg_atomic(["-i", "input"], output_file)
            
            assert result is False
            mock_remove.assert_called_once()
            
    def test_concat_list_manager_escapes_paths(self, tmp_path):
        """Test that ConcatListManager correctly escapes filenames."""
        files = [
            Path("/data/clip's.mp4"),
            Path("/data/normal.mp4")
        ]
        
        with ConcatListManager(tmp_path) as manager:
            manager.write(files)
            list_path = Path(manager.path)
            
            content = list_path.read_text()
            assert "file '/data/clip'\\''s.mp4'" in content
            assert "file '/data/normal.mp4'" in content
            
    def test_concat_list_manager_cleanup(self, tmp_path):
        """Test that ConcatListManager cleans up list file."""
        files = [Path("/data/clip.mp4")]
        path_to_check = None
        
        with ConcatListManager(tmp_path) as manager:
            manager.write(files)
            path_to_check = Path(manager.path)
            assert path_to_check.exists()
            
        assert not path_to_check.exists()
