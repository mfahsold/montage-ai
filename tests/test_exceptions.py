"""Tests for unified exception hierarchy and compatibility wrappers."""

import importlib
import sys

import pytest

from src.montage_ai.exceptions import FFmpegError, MontageError
from src.montage_ai.redis_exceptions import RedisConnectionError, RedisTimeoutError


def _reload_with_warning(module_name: str, warning_match: str):
    """Reload a module and assert that a deprecation warning is emitted."""
    sys.modules.pop(module_name, None)
    with pytest.warns(DeprecationWarning, match=warning_match):
        return importlib.import_module(module_name)


def test_montage_error_prefers_user_message() -> None:
    """User-facing message should be the primary exception string."""
    err = MontageError(
        message="technical fallback",
        user_message="friendly message",
        technical_details="trace details",
        suggestion="retry with preview profile",
    )

    assert str(err) == "friendly message"
    assert err.user_message == "friendly message"
    assert err.technical_details == "trace details"
    assert err.suggestion == "retry with preview profile"


def test_montage_error_defaults() -> None:
    """Defaults remain backward compatible when only message is provided."""
    err = MontageError("plain message")

    assert str(err) == "plain message"
    assert err.user_message == "plain message"
    assert err.technical_details == ""
    assert err.suggestion == ""


def test_ffmpeg_error_retains_command_and_stderr() -> None:
    """FFmpegError should preserve process context fields for debugging."""
    err = FFmpegError(
        "encode failed",
        command="ffmpeg -i in.mp4 out.mp4",
        stderr="invalid argument",
    )

    assert str(err) == "encode failed"
    assert err.command == "ffmpeg -i in.mp4 out.mp4"
    assert err.stderr == "invalid argument"


def test_redis_exceptions_use_unified_base_fields() -> None:
    """Redis exceptions should expose enhanced MontageError fields."""
    err = RedisConnectionError("localhost", 6379, "connection refused")

    assert isinstance(err, MontageError)
    assert "localhost:6379" in err.user_message
    assert "connection refused" in err.technical_details
    assert "REDIS_HOST" in err.suggestion


def test_redis_timeout_error_contains_operation_context() -> None:
    """Timeout exception should include operation and timeout metadata in text."""
    err = RedisTimeoutError("enqueue", 5.0)

    assert isinstance(err, MontageError)
    assert "enqueue" in err.user_message
    assert "enqueue" in err.technical_details
    assert "REDIS_SOCKET_TIMEOUT" in err.suggestion


def test_exceptions_custom_import_emits_deprecation_warning() -> None:
    """Deprecated compatibility module should warn on import."""
    module = _reload_with_warning(
        "src.montage_ai.exceptions_custom",
        "exceptions_custom.py is deprecated",
    )
    assert hasattr(module, "OpticalFlowTimeout")
