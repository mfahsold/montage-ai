"""
FFmpeg Configuration - Central place for all FFmpeg encoding defaults.

DRY principle: All FFmpeg parameters defined once, imported everywhere.
Env vars allow runtime override without code changes.

Usage:
    from .ffmpeg_config import FFmpegConfig, get_ffmpeg_video_params
    
    # Get params for FFmpeg subprocess
    params = get_ffmpeg_video_params(crf=18, preset="fast")
    
    # Or use the config object directly
    config = FFmpegConfig()
    cmd = ["ffmpeg", "-i", input_file, *config.video_params(), output_file]
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# Standard defaults (H.264 High Profile for broad compatibility)
# =============================================================================

# Resolution presets
STANDARD_WIDTH_HORIZONTAL = 1920
STANDARD_HEIGHT_HORIZONTAL = 1080
STANDARD_WIDTH_VERTICAL = 1080
STANDARD_HEIGHT_VERTICAL = 1920

# Encoding defaults
STANDARD_FPS = 30
STANDARD_PIX_FMT = "yuv420p"
STANDARD_CODEC = "libx264"
STANDARD_PROFILE = "high"
STANDARD_LEVEL = "4.1"  # Max 1080p60 or 4K30
STANDARD_PRESET = "medium"
STANDARD_CRF = 18  # Visually lossless for most content
STANDARD_AUDIO_CODEC = "aac"
STANDARD_AUDIO_BITRATE = "192k"


# =============================================================================
# Environment variable overrides
# =============================================================================

def _env_or_default(key: str, default: Any) -> Any:
    """Get env var, returning default if not set or empty."""
    val = os.environ.get(key, "")
    return val if val else default


@dataclass
class FFmpegConfig:
    """
    FFmpeg encoding configuration with env var overrides.
    
    All parameters can be overridden via environment variables:
    - OUTPUT_CODEC, OUTPUT_PROFILE, OUTPUT_LEVEL, OUTPUT_PIX_FMT
    - FFMPEG_PRESET, FFMPEG_CRF, FFMPEG_THREADS
    - OUTPUT_AUDIO_CODEC, OUTPUT_AUDIO_BITRATE
    """
    # Video encoding
    codec: str = field(default_factory=lambda: _env_or_default("OUTPUT_CODEC", STANDARD_CODEC))
    profile: str = field(default_factory=lambda: _env_or_default("OUTPUT_PROFILE", STANDARD_PROFILE))
    level: str = field(default_factory=lambda: _env_or_default("OUTPUT_LEVEL", STANDARD_LEVEL))
    pix_fmt: str = field(default_factory=lambda: _env_or_default("OUTPUT_PIX_FMT", STANDARD_PIX_FMT))
    preset: str = field(default_factory=lambda: _env_or_default("FFMPEG_PRESET", STANDARD_PRESET))
    crf: int = field(default_factory=lambda: int(_env_or_default("FFMPEG_CRF", STANDARD_CRF)))
    threads: str = field(default_factory=lambda: _env_or_default("FFMPEG_THREADS", "0"))  # 0 = auto
    
    # Audio encoding
    audio_codec: str = field(default_factory=lambda: _env_or_default("OUTPUT_AUDIO_CODEC", STANDARD_AUDIO_CODEC))
    audio_bitrate: str = field(default_factory=lambda: _env_or_default("OUTPUT_AUDIO_BITRATE", STANDARD_AUDIO_BITRATE))
    
    # Resolution (typically set by heuristics, not env)
    width: int = STANDARD_WIDTH_HORIZONTAL
    height: int = STANDARD_HEIGHT_HORIZONTAL
    fps: int = STANDARD_FPS
    
    def video_params(self, crf: Optional[int] = None, preset: Optional[str] = None) -> List[str]:
        """
        Generate FFmpeg video encoding parameters.
        
        Args:
            crf: Override CRF (quality). Lower = better quality, bigger file.
            preset: Override preset (speed vs compression tradeoff).
        
        Returns:
            List of FFmpeg CLI arguments for video encoding.
        """
        params = [
            "-c:v", self.codec,
            "-pix_fmt", self.pix_fmt,
        ]
        
        # Only add profile/level for H.264/H.265
        if self.codec in ("libx264", "libx265", "h264", "hevc"):
            if self.profile:
                params.extend(["-profile:v", self.profile])
            if self.level:
                params.extend(["-level:v", self.level])
        
        params.extend([
            "-preset", preset or self.preset,
            "-crf", str(crf if crf is not None else self.crf),
        ])
        
        if self.threads != "0":
            params.extend(["-threads", self.threads])
        
        return params
    
    def audio_params(self) -> List[str]:
        """Generate FFmpeg audio encoding parameters."""
        return [
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
        ]
    
    def full_params(self, crf: Optional[int] = None, preset: Optional[str] = None) -> List[str]:
        """Generate full FFmpeg encoding parameters (video + audio)."""
        return self.video_params(crf, preset) + self.audio_params()
    
    def moviepy_params(self, crf: Optional[int] = None) -> List[str]:
        """
        Generate parameters suitable for MoviePy's write_videofile().
        
        MoviePy passes these directly to FFmpeg via ffmpeg_params argument.
        Note: MoviePy handles codec separately, so we omit -c:v here.
        """
        params = ["-pix_fmt", self.pix_fmt]
        
        if self.codec in ("libx264", "libx265", "h264", "hevc"):
            if self.profile:
                params.extend(["-profile:v", self.profile])
            if self.level:
                params.extend(["-level:v", self.level])
        
        params.extend([
            "-preset", self.preset,
            "-crf", str(crf if crf is not None else self.crf),
        ])
        
        return params


# =============================================================================
# Convenience functions (for simpler imports)
# =============================================================================

# Singleton config instance
_config: Optional[FFmpegConfig] = None

def get_config() -> FFmpegConfig:
    """Get singleton FFmpegConfig instance."""
    global _config
    if _config is None:
        _config = FFmpegConfig()
    return _config


def get_ffmpeg_video_params(crf: Optional[int] = None, preset: Optional[str] = None) -> List[str]:
    """
    Get FFmpeg video encoding parameters.
    
    Convenience function that uses the singleton config.
    
    Example:
        params = get_ffmpeg_video_params(crf=20, preset="fast")
        # ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high', ...]
    """
    return get_config().video_params(crf, preset)


def get_moviepy_params(crf: Optional[int] = None) -> List[str]:
    """
    Get FFmpeg parameters for MoviePy's write_videofile().
    
    Example:
        clip.write_videofile(output, ffmpeg_params=get_moviepy_params(crf=18))
    """
    return get_config().moviepy_params(crf)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "STANDARD_WIDTH_HORIZONTAL",
    "STANDARD_HEIGHT_HORIZONTAL", 
    "STANDARD_WIDTH_VERTICAL",
    "STANDARD_HEIGHT_VERTICAL",
    "STANDARD_FPS",
    "STANDARD_PIX_FMT",
    "STANDARD_CODEC",
    "STANDARD_PROFILE",
    "STANDARD_LEVEL",
    "STANDARD_PRESET",
    "STANDARD_CRF",
    "STANDARD_AUDIO_CODEC",
    "STANDARD_AUDIO_BITRATE",
    # Config class
    "FFmpegConfig",
    "get_config",
    # Convenience functions
    "get_ffmpeg_video_params",
    "get_moviepy_params",
]
