"""
Tests for centralized configuration module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.montage_ai.config import (
    Settings,
    PathConfig,
    FeatureConfig,
    LLMConfig,
    FileTypeConfig,
    get_settings,
    reload_settings,
)


class TestPathConfig:
    """Tests for PathConfig."""

    def test_default_paths(self):
        """Default paths are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            config = PathConfig()
            assert config.input_dir == Path("/data/input")
            assert config.output_dir == Path("/data/output")
            assert config.music_dir == Path("/data/music")
            # OPTIMIZATION: temp_dir now uses /dev/shm (RAM disk) when available
            assert config.temp_dir in [Path("/dev/shm"), Path("/tmp")]

    def test_custom_paths_from_env(self):
        """Paths can be overridden via environment."""
        with patch.dict(os.environ, {
            "INPUT_DIR": "/custom/input",
            "OUTPUT_DIR": "/custom/output",
        }):
            config = PathConfig()
            assert config.input_dir == Path("/custom/input")
            assert config.output_dir == Path("/custom/output")

    def test_ensure_directories(self):
        """ensure_directories creates all required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {
                "INPUT_DIR": os.path.join(tmpdir, "input"),
                "OUTPUT_DIR": os.path.join(tmpdir, "output"),
                "MUSIC_DIR": os.path.join(tmpdir, "music"),
                "ASSETS_DIR": os.path.join(tmpdir, "assets"),
            }):
                config = PathConfig()
                config.ensure_directories()

                assert config.input_dir.exists()
                assert config.output_dir.exists()
                assert config.music_dir.exists()
                assert config.assets_dir.exists()

    def test_get_log_path(self):
        """get_log_path returns correct path."""
        config = PathConfig()
        log_path = config.get_log_path("job123")
        assert log_path == config.output_dir / "render_job123.log"


class TestFeatureConfig:
    """Tests for FeatureConfig."""

    def test_default_features(self):
        """Default feature flags are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            config = FeatureConfig()
            assert config.stabilize is False
            assert config.upscale is False
            assert config.enhance is True  # Default enabled
            assert config.verbose is True  # Default enabled

    def test_features_from_env(self):
        """Feature flags can be set via environment."""
        with patch.dict(os.environ, {
            "STABILIZE": "true",
            "UPSCALE": "TRUE",
            "ENHANCE": "false",
        }):
            config = FeatureConfig()
            assert config.stabilize is True
            assert config.upscale is True
            assert config.enhance is False


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_llm_config(self):
        """Default LLM config is set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()
            assert config.openai_api_base == ""
            assert config.ollama_host == "http://host.docker.internal:11434"
            assert config.cgpu_enabled is False

    def test_has_openai_backend(self):
        """has_openai_backend property works correctly."""
        with patch.dict(os.environ, {
            "OPENAI_API_BASE": "http://api.example.com",
            "OPENAI_MODEL": "gpt-4",
        }):
            config = LLMConfig()
            assert config.has_openai_backend is True

        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()
            assert config.has_openai_backend is False

    def test_has_google_backend(self):
        """has_google_backend property works correctly."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            config = LLMConfig()
            assert config.has_google_backend is True

        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()
            assert config.has_google_backend is False


class TestFileTypeConfig:
    """Tests for FileTypeConfig."""

    def test_is_video(self):
        """is_video correctly identifies video files."""
        config = FileTypeConfig()
        assert config.is_video("video.mp4") is True
        assert config.is_video("VIDEO.MP4") is True
        assert config.is_video("clip.mov") is True
        assert config.is_video("file.txt") is False
        assert config.is_video("noextension") is False

    def test_is_audio(self):
        """is_audio correctly identifies audio files."""
        config = FileTypeConfig()
        assert config.is_audio("song.mp3") is True
        assert config.is_audio("audio.wav") is True
        assert config.is_audio("video.mp4") is False

    def test_is_image(self):
        """is_image correctly identifies image files."""
        config = FileTypeConfig()
        assert config.is_image("photo.png") is True
        assert config.is_image("logo.jpg") is True
        assert config.is_image("video.mp4") is False

    def test_allowed_file(self):
        """allowed_file checks against provided extension set."""
        config = FileTypeConfig()
        assert config.allowed_file("video.mp4", {"mp4", "mov"}) is True
        assert config.allowed_file("file.txt", {"mp4", "mov"}) is False
        assert config.allowed_file("noext", {"mp4"}) is False


class TestSettings:
    """Tests for main Settings class."""

    def test_settings_initialization(self):
        """Settings initializes all sub-configs."""
        settings = Settings()
        assert settings.paths is not None
        assert settings.features is not None
        assert settings.llm is not None
        assert settings.encoding is not None
        assert settings.processing is not None
        assert settings.creative is not None
        assert settings.file_types is not None

    def test_to_env_dict(self):
        """to_env_dict returns correct environment dict."""
        settings = Settings()
        env_dict = settings.to_env_dict()

        assert "INPUT_DIR" in env_dict
        assert "OUTPUT_DIR" in env_dict
        assert "STABILIZE" in env_dict
        assert "CGPU_ENABLED" in env_dict
        assert "JOB_ID" in env_dict

    def test_reload(self):
        """reload creates new settings instance."""
        settings = Settings()
        new_settings = settings.reload()
        assert new_settings is not settings


class TestGlobalSettings:
    """Tests for global settings functions."""

    def test_get_settings_singleton(self):
        """get_settings returns same instance."""
        # Reset global
        import src.montage_ai.config as config_module
        config_module._settings = None

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_settings(self):
        """reload_settings creates new instance."""
        import src.montage_ai.config as config_module
        config_module._settings = None

        s1 = get_settings()
        s2 = reload_settings()
        assert s1 is not s2
