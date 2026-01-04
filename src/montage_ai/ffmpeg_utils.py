"""
FFmpeg/FFprobe command helpers.

Centralizes command building to reduce duplication across the codebase.
This module provides DRY helpers to avoid repeating codec/profile/level
parameter construction across clip_enhancement.py, segment_writer.py,
caption_burner.py, text_editor.py, etc.
"""

from dataclasses import dataclass
from typing import List, Optional


def _has_flag(args: List[str], flags: List[str]) -> bool:
    return any(flag in args for flag in flags)


def build_ffmpeg_cmd(
    args: List[str],
    *,
    overwrite: bool = True,
    hide_banner: bool = False,
    loglevel: Optional[str] = None,
) -> List[str]:
    """
    Build a ffmpeg command list with optional standard flags.
    """
    cmd = ["ffmpeg"]
    if overwrite and not _has_flag(args, ["-y", "-n"]):
        cmd.append("-y")
    if hide_banner and "-hide_banner" not in args:
        cmd.append("-hide_banner")
    if loglevel and "-loglevel" not in args:
        cmd.extend(["-loglevel", loglevel])
    return cmd + args


def build_ffprobe_cmd(args: List[str], *, verbosity: Optional[str] = None) -> List[str]:
    """
    Build a ffprobe command list with optional verbosity.
    """
    cmd = ["ffprobe"]
    if verbosity and "-v" not in args:
        cmd.extend(["-v", verbosity])
    return cmd + args


# =============================================================================
# Video Encoding Parameters (DRY Helper)
# =============================================================================

@dataclass
class VideoEncodingParams:
    """
    Encapsulates video encoding parameters to avoid repetition.

    Usage:
        params = VideoEncodingParams(codec="libx264", preset="medium", crf=23)
        cmd.extend(params.to_args())
        cmd.extend([output_path])
    """
    codec: str = "libx264"
    preset: str = "medium"
    crf: int = 23
    profile: Optional[str] = None
    level: Optional[str] = None
    pix_fmt: str = "yuv420p"

    def to_args(self, *, include_audio_copy: bool = True) -> List[str]:
        """
        Convert parameters to FFmpeg command-line arguments.

        Args:
            include_audio_copy: If True, adds '-c:a copy' for audio passthrough.

        Returns:
            List of FFmpeg arguments like ['-c:v', 'libx264', '-preset', ...]
        """
        args = [
            "-c:v", self.codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
        ]
        if self.profile:
            args.extend(["-profile:v", self.profile])
        if self.level:
            args.extend(["-level", self.level])
        args.extend(["-pix_fmt", self.pix_fmt])
        if include_audio_copy:
            args.extend(["-c:a", "copy"])
        return args

    def to_args_no_audio(self) -> List[str]:
        """Convert to args without audio handling (for video-only operations)."""
        return self.to_args(include_audio_copy=False)


def build_video_encoding_args(
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 23,
    profile: Optional[str] = None,
    level: Optional[str] = None,
    pix_fmt: str = "yuv420p",
    audio_copy: bool = True,
) -> List[str]:
    """
    Build video encoding arguments for FFmpeg.

    Consolidates the repeated pattern across the codebase:
        cmd.extend(["-c:v", codec, "-preset", preset, "-crf", str(crf)])
        if profile:
            cmd.extend(["-profile:v", profile])
        if level:
            cmd.extend(["-level", level])
        cmd.extend(["-pix_fmt", pix_fmt, "-c:a", "copy"])

    Usage:
        cmd = build_ffmpeg_cmd(["-i", input, "-vf", filters])
        cmd.extend(build_video_encoding_args(
            codec="libx264", preset="fast", crf=18,
            profile="high", level="4.0", pix_fmt="yuv420p"
        ))
        cmd.append(output_path)

    Args:
        codec: Video codec (libx264, libx265, h264_nvenc, etc.)
        preset: Encoding preset (ultrafast, fast, medium, slow, etc.)
        crf: Constant Rate Factor (0-51, lower = better quality)
        profile: H.264/H.265 profile (baseline, main, high, etc.)
        level: H.264/H.265 level (3.0, 4.0, 4.1, etc.)
        pix_fmt: Pixel format (yuv420p for max compatibility)
        audio_copy: If True, adds '-c:a copy' for audio passthrough

    Returns:
        List of FFmpeg arguments
    """
    params = VideoEncodingParams(
        codec=codec,
        preset=preset,
        crf=crf,
        profile=profile,
        level=level,
        pix_fmt=pix_fmt,
    )
    return params.to_args(include_audio_copy=audio_copy)


# =============================================================================
# Audio Parameters Helper
# =============================================================================

def build_audio_encoding_args(
    codec: str = "aac",
    bitrate: str = "192k",
) -> List[str]:
    """
    Build audio encoding arguments for FFmpeg.

    Usage:
        cmd.extend(build_audio_encoding_args(codec="aac", bitrate="256k"))

    Returns:
        List like ["-c:a", "aac", "-b:a", "192k"]
    """
    return ["-c:a", codec, "-b:a", bitrate]


# =============================================================================
# Filter Chain Helper
# =============================================================================

def build_filter_chain(filters: List[str], separator: str = ",") -> str:
    """
    Join filter expressions into a single -vf chain.

    Usage:
        chain = build_filter_chain([
            "scale=1920:1080",
            "eq=brightness=0.05",
            "unsharp=3:3:0.5"
        ])
        # Returns: "scale=1920:1080,eq=brightness=0.05,unsharp=3:3:0.5"

    Args:
        filters: List of filter expressions
        separator: Separator between filters (usually ",")

    Returns:
        Joined filter string for -vf argument
    """
    # Filter out empty strings
    valid_filters = [f for f in filters if f and f.strip()]
    return separator.join(valid_filters)
