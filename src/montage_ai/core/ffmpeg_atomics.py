"""
FFmpeg Atomic Write Utilities for DRY operations.

Provides context managers and helpers for atomic file operations with FFmpeg,
ensuring temp files are cleaned up even on failure.

Usage:
    from montage_ai.core.ffmpeg_atomics import atomic_ffmpeg, ConcatListManager

    # Atomic write pattern
    with atomic_ffmpeg(output_path) as temp_path:
        cmd = build_ffmpeg_cmd(["-i", input, temp_path])
        run_command(cmd, check=True)
    # File is automatically renamed to output_path on success

    # Concat list management
    with ConcatListManager(output_dir, "segment") as concat:
        concat.write(clip_paths)
        cmd = build_ffmpeg_cmd(["-f", "concat", "-i", concat.path, output])
        run_command(cmd)
    # Concat file is automatically cleaned up
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Union

from ..logger import logger
from ..ffmpeg_utils import build_ffmpeg_cmd
from .cmd_runner import run_command


@contextmanager
def atomic_ffmpeg(
    output_path: Union[str, Path],
    suffix: str = ".tmp",
    cleanup_on_failure: bool = True
) -> Generator[str, None, None]:
    """
    Context manager for atomic FFmpeg operations.

    Creates a temporary file path, yields it for use in FFmpeg commands,
    then atomically renames to the final path on success. On failure,
    the temp file is cleaned up.

    Args:
        output_path: Final destination path
        suffix: Suffix for temp file (default: ".tmp")
        cleanup_on_failure: Whether to remove temp file on error

    Yields:
        Temporary file path to use in FFmpeg commands

    Example:
        with atomic_ffmpeg("/output/video.mp4") as temp:
            cmd = build_ffmpeg_cmd(["-i", input, temp])
            result = run_command(cmd, check=False)
            if result.returncode != 0:
                raise RuntimeError("FFmpeg failed")
        # temp is now renamed to /output/video.mp4
    """
    output_path = Path(output_path)
    ext = output_path.suffix
    temp_path = str(output_path) + suffix + ext

    try:
        yield temp_path

        # Success: atomically rename temp to final
        if os.path.exists(temp_path):
            os.rename(temp_path, str(output_path))
            logger.debug(f"Atomic write: {temp_path} -> {output_path}")
        else:
            raise FileNotFoundError(f"FFmpeg did not create output: {temp_path}")

    except Exception:
        # Failure: cleanup temp file
        if cleanup_on_failure and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up failed temp: {temp_path}")
            except OSError:
                pass
        raise


def run_ffmpeg_atomic(
    cmd_args: List[str],
    output_path: Union[str, Path],
    timeout: int = 300,
    log_output: bool = False
) -> bool:
    """
    Execute FFmpeg command with atomic write (temp -> rename).

    Convenience function that combines atomic_ffmpeg context manager
    with command execution.

    Args:
        cmd_args: FFmpeg arguments (excluding output path)
        output_path: Final output file path
        timeout: Command timeout in seconds
        log_output: Whether to log FFmpeg output

    Returns:
        True on success, False on failure
    """
    output_path = Path(output_path)
    ext = output_path.suffix
    temp_path = str(output_path) + ".tmp" + ext

    try:
        # Build command with temp path
        full_args = list(cmd_args) + [temp_path]
        cmd = build_ffmpeg_cmd(full_args)

        result = run_command(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
            log_output=log_output
        )

        if result.returncode == 0 and os.path.exists(temp_path):
            os.rename(temp_path, str(output_path))
            return True
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    except Exception as e:
        logger.error(f"FFmpeg atomic write failed: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


class ConcatListManager:
    """
    Manages FFmpeg concat demuxer list files with automatic cleanup.

    Handles proper escaping of file paths and ensures temp files
    are cleaned up even on error.

    Usage:
        with ConcatListManager(output_dir, "segment_0") as concat:
            concat.write(["/path/to/clip1.mp4", "/path/to/clip2.mp4"])
            cmd = build_ffmpeg_cmd([
                "-f", "concat", "-safe", "0",
                "-i", concat.path,
                "-c", "copy", "output.mp4"
            ])
            run_command(cmd)
    """

    def __init__(self, output_dir: Union[str, Path], prefix: str = "concat"):
        """
        Initialize concat list manager.

        Args:
            output_dir: Directory to create concat list file in
            prefix: Prefix for the concat list filename
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self._path: Optional[Path] = None

    @property
    def path(self) -> str:
        """Get the path to the concat list file."""
        if self._path is None:
            self._path = self.output_dir / f"{self.prefix}_concat.txt"
        return str(self._path)

    def write(self, clip_paths: List[Union[str, Path]]) -> str:
        """
        Write clip paths to concat list file with proper escaping.

        Args:
            clip_paths: List of paths to video/audio clips

        Returns:
            Path to the created concat list file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.path, 'w') as f:
            for clip_path in clip_paths:
                # FFmpeg concat demuxer requires escaping single quotes
                escaped_path = str(clip_path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        logger.debug(f"Created concat list with {len(clip_paths)} files: {self.path}")
        return self.path

    def cleanup(self) -> None:
        """Remove the concat list file."""
        if self._path and self._path.exists():
            try:
                self._path.unlink()
                logger.debug(f"Cleaned up concat list: {self._path}")
            except OSError:
                pass

    def __enter__(self) -> 'ConcatListManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


@contextmanager
def temp_audio_file(
    base_name: str = "audio",
    suffix: str = ".wav",
    cleanup: bool = True
) -> Generator[str, None, None]:
    """
    Context manager for temporary audio files.

    Creates a temp file for intermediate audio processing
    and cleans it up automatically.

    Args:
        base_name: Base name for the temp file
        suffix: File extension
        cleanup: Whether to delete on exit

    Yields:
        Path to temporary file
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=f"{base_name}_")
    os.close(fd)

    try:
        yield temp_path
    finally:
        if cleanup and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


__all__ = [
    'atomic_ffmpeg',
    'run_ffmpeg_atomic',
    'ConcatListManager',
    'temp_audio_file',
]
