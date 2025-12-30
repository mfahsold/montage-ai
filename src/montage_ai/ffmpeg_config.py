"""
FFmpeg Configuration - Central place for all FFmpeg encoding defaults.

DRY principle: All FFmpeg parameters defined once, imported everywhere.
Env vars allow runtime override without code changes.

GPU Acceleration Support:
- NVENC (NVIDIA): h264_nvenc, hevc_nvenc - via CUDA
- VAAPI (AMD/Intel Linux): h264_vaapi, hevc_vaapi - via /dev/dri/renderD128
- VideoToolbox (macOS): h264_videotoolbox, hevc_videotoolbox
- QSV (Intel): h264_qsv, hevc_qsv - via Intel Media SDK

Usage:
    from .ffmpeg_config import FFmpegConfig, get_ffmpeg_video_params
    
    # Get params for FFmpeg subprocess
    params = get_ffmpeg_video_params(crf=18, preset="fast")
    
    # Or use the config object directly
    config = FFmpegConfig()
    cmd = ["ffmpeg", "-i", input_file, *config.video_params(), output_file]
    
    # With GPU acceleration (auto-detected or forced):
    config = FFmpegConfig(hwaccel="auto")  # Auto-detect available GPU
    config = FFmpegConfig(hwaccel="nvenc") # Force NVIDIA
    config = FFmpegConfig(hwaccel="vaapi") # Force AMD/Intel Linux
"""

from __future__ import annotations
import os
import subprocess
import shutil
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .core import hardware


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
# Hardware Acceleration Detection & Configuration
# =============================================================================

def get_best_gpu_encoder() -> Optional[str]:
    """
    Get the best available GPU encoder type.
    """
    preferred = _normalize_codec_preference(_env_or_default("OUTPUT_CODEC", STANDARD_CODEC))
    from .core import hardware
    config = hardware.get_best_hwaccel(preferred_codec=preferred)
    if config.is_gpu:
        return config.type
    return None

# =============================================================================
# Environment variable overrides
# =============================================================================

def _env_or_default(key: str, default: Any) -> Any:
    """Get env var, returning default if not set or empty."""
    val = os.environ.get(key, "")
    return val if val else default


def _normalize_codec_preference(codec: str) -> str:
    """Normalize codec preference to 'h264' or 'hevc'."""
    c = (codec or "").lower()
    if "265" in c or "hevc" in c or "h265" in c:
        return "hevc"
    return "h264"


@dataclass
class FFmpegConfig:
    """
    FFmpeg encoding configuration with env var overrides and GPU acceleration.
    
    All parameters can be overridden via environment variables:
    - OUTPUT_CODEC, OUTPUT_PROFILE, OUTPUT_LEVEL, OUTPUT_PIX_FMT
    - FFMPEG_PRESET, FFMPEG_CRF, FFMPEG_THREADS
    - OUTPUT_AUDIO_CODEC, OUTPUT_AUDIO_BITRATE
    - FFMPEG_HWACCEL: "auto", "nvenc", "vaapi", "qsv", "videotoolbox", "none"
    
    GPU Acceleration:
    - hwaccel="auto": Auto-detect best available GPU encoder
    - hwaccel="nvenc": Force NVIDIA NVENC
    - hwaccel="vaapi": Force AMD/Intel VAAPI (Linux)
    - hwaccel="none": Force CPU encoding (default for compatibility)
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
    
    # Hardware acceleration
    hwaccel: str = field(default_factory=lambda: _env_or_default("FFMPEG_HWACCEL", "auto"))
    _hw_config: Optional["hardware.HWConfig"] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Resolve hardware acceleration setting."""
        preferred = _normalize_codec_preference(self.codec)
        from .core import hardware  # Lazy import to avoid circular dependency
        if self.hwaccel == "auto":
            self._hw_config = hardware.get_best_hwaccel(preferred_codec=preferred)
        elif self.hwaccel == "none":
            self._hw_config = hardware.HWConfig(
                type="cpu",
                encoder=self.codec,
                decoder_args=[],
                encoder_args=["-c:v", self.codec],
                is_gpu=False
            )
        else:
            # Try to force specific HW accel
            # This is a simplified fallback, ideally we'd check availability
            self._hw_config = hardware.get_best_hwaccel(preferred_codec=preferred)
            if self._hw_config.type != self.hwaccel:
                print(f"⚠️ Requested GPU encoder '{self.hwaccel}' not available/detected, using {self._hw_config.type}")
        
        if self._hw_config and self._hw_config.is_gpu:
            actual = _normalize_codec_preference(self._hw_config.encoder)
            if actual != preferred:
                print(f"⚠️ Requested codec '{self.codec}' not supported by {self._hw_config.type}, using {self._hw_config.encoder}")

    @property
    def is_gpu_accelerated(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._hw_config.is_gpu if self._hw_config else False
    
    @property
    def gpu_encoder_type(self) -> Optional[str]:
        """Get the active GPU encoder type (nvenc, vaapi, etc.)."""
        return self._hw_config.type if self._hw_config and self._hw_config.is_gpu else None
    
    @property
    def effective_codec(self) -> str:
        """Get the effective video codec to use."""
        return self._hw_config.encoder if self._hw_config else self.codec

    @property
    def hwupload_filter(self) -> Optional[str]:
        """Return required hwupload filter for the active GPU encoder (if any)."""
        if self._hw_config and self._hw_config.is_gpu:
            return self._hw_config.hwupload_filter
        return None
    
    def hwaccel_input_params(self) -> List[str]:
        """Generate FFmpeg input parameters for hardware-accelerated decoding."""
        return self._hw_config.decoder_args if self._hw_config else []
    
    def _gpu_quality_args(self, crf_value: int) -> List[str]:
        """Map CRF-like value to GPU encoder quality flags."""
        crf_clamped = max(0, min(51, int(crf_value)))
        if not self._hw_config:
            return []
        if self._hw_config.type == "nvenc":
            return ["-rc", "vbr_hq", "-cq", str(crf_clamped), "-b:v", "0"]
        if self._hw_config.type == "vaapi":
            return ["-qp", str(crf_clamped)]
        if self._hw_config.type == "qsv":
            return ["-global_quality", str(crf_clamped)]
        if self._hw_config.type == "videotoolbox":
            return ["-q:v", str(crf_clamped)]
        return []

    def _strip_codec_args(self, params: List[str]) -> List[str]:
        """Remove -c:v and its value from an FFmpeg args list."""
        stripped: List[str] = []
        skip_next = False
        for item in params:
            if skip_next:
                skip_next = False
                continue
            if item == "-c:v":
                skip_next = True
                continue
            stripped.append(item)
        return stripped

    def video_params(
        self,
        crf: Optional[int] = None,
        preset: Optional[str] = None,
        codec_override: Optional[str] = None,
        profile_override: Optional[str] = None,
        level_override: Optional[str] = None,
        pix_fmt_override: Optional[str] = None,
    ) -> List[str]:
        """
        Generate FFmpeg video encoding parameters.
        """
        crf_value = crf if crf is not None else self.crf
        preset_value = preset or self.preset
        profile_value = self.profile if profile_override is None else profile_override
        level_value = self.level if level_override is None else level_override
        pix_fmt_value = self.pix_fmt if pix_fmt_override is None else pix_fmt_override

        if self._hw_config and self._hw_config.is_gpu:
            args = list(self._hw_config.encoder_args)
            if self._hw_config.type in ("nvenc", "qsv") and preset_value:
                args.extend(["-preset", preset_value])
            args.extend(self._gpu_quality_args(crf_value))
            if profile_value:
                args.extend(["-profile:v", profile_value])
            if level_value:
                args.extend(["-level", str(level_value)])
            if pix_fmt_value and self._hw_config.type != "vaapi":
                args.extend(["-pix_fmt", pix_fmt_value])
            return args

        # CPU encoding path
        codec_value = codec_override or self.codec
        args = ["-c:v", codec_value]
        if preset_value:
            args.extend(["-preset", preset_value])
        args.extend(["-crf", str(crf_value)])
        if profile_value:
            args.extend(["-profile:v", profile_value])
        if level_value:
            args.extend(["-level", str(level_value)])
        if pix_fmt_value:
            args.extend(["-pix_fmt", pix_fmt_value])
        return args

    def moviepy_params(
        self,
        crf: Optional[int] = None,
        preset: Optional[str] = None,
        codec_override: Optional[str] = None,
        profile_override: Optional[str] = None,
        level_override: Optional[str] = None,
        pix_fmt_override: Optional[str] = None,
    ) -> List[str]:
        """
        Get parameters for MoviePy's write_videofile.
        MoviePy expects a list of strings for 'ffmpeg_params'.
        """
        # MoviePy handles codec via 'codec' argument, so we only need extra params
        # But if we want to use HW accel with MoviePy, we need to pass the encoder as codec
        # and extra args via ffmpeg_params.
        
        # This is tricky because MoviePy's write_videofile takes 'codec' arg.
        # We should return the codec name separately.
        params = self.video_params(
            crf=crf,
            preset=preset,
            codec_override=codec_override,
            profile_override=profile_override,
            level_override=level_override,
            pix_fmt_override=pix_fmt_override,
        )
        return self._strip_codec_args(params)

    def audio_params(self) -> List[str]:
        """Generate FFmpeg audio encoding parameters."""
        return [
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
        ]
    
    def full_params(self, crf: Optional[int] = None, preset: Optional[str] = None) -> List[str]:
        """Generate full FFmpeg encoding parameters (video + audio)."""
        return self.video_params(crf, preset) + self.audio_params()


# =============================================================================
# Convenience functions (for simpler imports)
# =============================================================================

# Singleton config instance
_config: Optional[FFmpegConfig] = None

def get_config(hwaccel: Optional[str] = None) -> FFmpegConfig:
    """
    Get singleton FFmpegConfig instance.
    
    Args:
        hwaccel: Override hardware acceleration. Use "auto" to auto-detect GPU.
    """
    global _config
    if _config is None or hwaccel is not None:
        if hwaccel:
            _config = FFmpegConfig(hwaccel=hwaccel)
        else:
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


def get_moviepy_params(crf: Optional[int] = None, preset: Optional[str] = None) -> List[str]:
    """
    Get FFmpeg parameters for MoviePy's write_videofile().
    
    Example:
        clip.write_videofile(output, ffmpeg_params=get_moviepy_params(crf=18))
    """
    return get_config().moviepy_params(crf=crf, preset=preset)


def print_gpu_status():
    """
    Print GPU encoder availability status.
    
    Useful for debugging and configuration.
    """
    print("\n🎮 GPU Encoder Status:")
    print("=" * 50)
    
    # Use hardware module to get status
    from .core import hardware
    hw_config = hardware.get_best_hwaccel()
    
    if hw_config.is_gpu:
        print(f"  ✅ GPU Acceleration Detected: {hw_config.type.upper()}")
        print(f"     Encoder: {hw_config.encoder}")
        print(f"     Decoder Args: {' '.join(hw_config.decoder_args)}")
    else:
        print(f"  ❌ No GPU Acceleration Detected (using CPU)")
        print(f"     Encoder: {hw_config.encoder}")
    
    print("=" * 50)


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
    # GPU detection
    "get_best_gpu_encoder",
    "print_gpu_status",
    # Config class
    "FFmpegConfig",
    "get_config",
    # Convenience functions
    "get_ffmpeg_video_params",
    "get_moviepy_params",
]
