"""
Common utility functions for Montage AI.

This module provides shared helper functions to avoid code duplication across
the codebase. Following the DRY (Don't Repeat Yourself) principle.

Usage:
    from montage_ai.utils import clamp, file_size_mb, file_exists_and_valid
"""

import os
from pathlib import Path
from typing import Union


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """
    Clamp a value between low and high bounds.

    Args:
        value: The value to clamp.
        low: Lower bound (default: 0.0).
        high: Upper bound (default: 1.0).

    Returns:
        The clamped value.

    Example:
        >>> clamp(1.5, 0.0, 1.0)
        1.0
        >>> clamp(-0.5, 0.0, 1.0)
        0.0
        >>> clamp(0.5, 0.0, 1.0)
        0.5
    """
    if value < low:
        return low
    if value > high:
        return high
    return value


def coerce_float(value: object) -> float | None:
    """
    Safely convert a value to float, returning None on failure.

    Args:
        value: Any value to convert.

    Returns:
        Float value or None if conversion fails.

    Example:
        >>> coerce_float("3.14")
        3.14
        >>> coerce_float("not a number")
        None
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def file_size_mb(path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB, or 0.0 if file doesn't exist.

    Example:
        >>> file_size_mb("/path/to/video.mp4")
        125.5
    """
    path = Path(path)
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def file_exists_and_valid(path: Union[str, Path]) -> bool:
    """
    Check if file exists and has non-zero size.

    Args:
        path: Path to check.

    Returns:
        True if file exists and has content, False otherwise.

    Example:
        >>> file_exists_and_valid("/path/to/video.mp4")
        True
        >>> file_exists_and_valid("/nonexistent.mp4")
        False
    """
    path_obj = Path(path) if isinstance(path, str) else path
    return path_obj.exists() and path_obj.stat().st_size > 0


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize a path to absolute, resolved Path object.

    Args:
        path: Path string or Path object.

    Returns:
        Resolved absolute Path.

    Example:
        >>> normalize_path("./video.mp4")
        PosixPath('/current/dir/video.mp4')
    """
    return Path(path).resolve()


def ensure_parent_dir(path: Union[str, Path]) -> Path:
    """
    Ensure the parent directory of a path exists.

    Args:
        path: Path whose parent should be created.

    Returns:
        The input path as a Path object.

    Example:
        >>> ensure_parent_dir("/new/directory/file.txt")
        PosixPath('/new/directory/file.txt')
        # Parent directory '/new/directory' is created
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "clamp",
    "coerce_float",
    "file_size_mb",
    "file_exists_and_valid",
    "normalize_path",
    "ensure_parent_dir",
]
