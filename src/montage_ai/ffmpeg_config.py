"""
FFmpeg Configuration - Central place for all FFmpeg encoding defaults.

DRY principle: All FFmpeg parameters defined once, imported everywhere.
Env vars allow runtime override without code changes.

GPU Acceleration Support:
    - NVENC (NVIDIA): h264_nvenc, hevc_nvenc - via CUDA
    - NVMPI (Jetson): h264_nvmpi, hevc_nvmpi
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

from .config import get_settings
from .logger import logger

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

# High-Resolution presets (6K/8K)
STANDARD_WIDTH_6K_HORIZONTAL = 6144
STANDARD_HEIGHT_6K_HORIZONTAL = 3160
STANDARD_WIDTH_6K_VERTICAL = 3160
STANDARD_HEIGHT_6K_VERTICAL = 6144

STANDARD_WIDTH_8K_HORIZONTAL = 7680
STANDARD_HEIGHT_8K_HORIZONTAL = 4320
STANDARD_WIDTH_8K_VERTICAL = 4320
STANDARD_HEIGHT_8K_VERTICAL = 7680

# Preview presets (Low latency, low res for fast feedback)
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360  # 360p
PREVIEW_CRF = 28      # Lower quality
PREVIEW_PRESET = "ultrafast"

# Encoding defaults
STANDARD_FPS = 30
STANDARD_PIX_FMT = "yuv420p"
STANDARD_CODEC = "libx264"
STANDARD_PROFILE = "high"
STANDARD_LEVEL = "4.1"  # Max 1080p60 or 4K30
LEVEL_6K = "5.2"  # Max 6K60
LEVEL_8K = "6.2"  # Max 8K60 (HEVC only)
STANDARD_PRESET = "medium"
STANDARD_CRF = 18  # Visually lossless for most content
STANDARD_AUDIO_CODEC = "aac"
STANDARD_AUDIO_BITRATE = "192k"


# =============================================================================
# Color Grading Presets
# =============================================================================

COLOR_PRESETS = {
    # === CLASSIC FILM LOOKS ===
    "cinematic": "colorbalance=rs=0.1:gs=-0.05:bs=0.1:rm=0.1:bm=0.05,curves=preset=cross_process",
    "teal_orange": "colorbalance=rs=-0.15:gs=-0.05:bs=0.2:rm=0.1:bm=-0.1:rh=0.15:bh=-0.15,eq=saturation=1.1:contrast=1.05",
    "blockbuster": "colorbalance=rs=-0.1:bs=0.15:rh=0.12:bh=-0.1,curves=m='0/0 0.25/0.22 0.5/0.5 0.75/0.78 1/1',eq=contrast=1.08",
    
    # === VINTAGE / RETRO ===
    "vintage": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,curves=preset=vintage",
    "film_fade": "curves=m='0/0.05 0.5/0.5 1/0.95',colorbalance=rs=0.1:gs=0.05:bs=-0.05,eq=saturation=0.85",
    "70s": "colortemperature=temperature=5500,colorbalance=rs=0.15:gs=0.1:rh=0.1,eq=saturation=1.2:contrast=0.95",
    "polaroid": "curves=r='0/0.1 0.5/0.55 1/0.9':g='0/0.05 0.5/0.5 1/0.95':b='0/0 0.5/0.45 1/0.85',eq=saturation=0.9",
    
    # === TEMPERATURE ===
    "cold": "colortemperature=temperature=7500,colorbalance=bs=0.1:bm=0.05,eq=saturation=0.9",
    "warm": "colortemperature=temperature=4000,colorbalance=rs=0.08:rh=0.05,eq=saturation=1.1",
    "golden_hour": "colortemperature=temperature=3500,colorbalance=rs=0.12:gs=0.05:rh=0.1:gh=0.05,eq=saturation=1.15:brightness=0.03",
    "blue_hour": "colortemperature=temperature=8000,colorbalance=bs=0.15:bm=0.08,eq=saturation=0.95:contrast=1.05",
    
    # === MOOD / GENRE ===
    "noir": "hue=s=0,curves=preset=darker,eq=contrast=1.2",
    "horror": "colorbalance=gs=-0.1:bs=0.05:bm=0.1,curves=m='0/0 0.4/0.35 0.6/0.65 1/1',eq=saturation=0.7:contrast=1.15",
    "sci_fi": "colorbalance=bs=0.2:bm=0.1:gs=-0.05,eq=saturation=0.85:contrast=1.1,curves=m='0/0 0.3/0.25 1/1'",
    "dreamy": "gblur=sigma=0.5,colorbalance=rs=0.05:bs=0.05,eq=saturation=0.9:brightness=0.05,curves=m='0/0.05 0.5/0.55 1/1'",
    
    # === PROFESSIONAL ===
    "vivid": "eq=saturation=1.4:contrast=1.1,unsharp=5:5:1.0",
    "muted": "eq=saturation=0.7:contrast=0.95,curves=m='0/0.05 0.5/0.5 1/0.95'",
    "high_contrast": "curves=m='0/0 0.25/0.15 0.5/0.5 0.75/0.85 1/1',eq=contrast=1.15",
    "low_contrast": "curves=m='0/0.1 0.5/0.5 1/0.9',eq=contrast=0.9",
    "desaturated": "eq=saturation=0.5",
    "punch": "eq=saturation=1.25:contrast=1.1,unsharp=3:3:0.8,curves=m='0/0 0.2/0.15 0.8/0.85 1/1'"
}

# =============================================================================
# Audio Processing Chains (DRY Definition)
# =============================================================================

def _get_audio_filters() -> Dict[str, str]:
    """
    Build audio filter dictionary with dynamic threshold values.
    
    Ducking thresholds come from ThresholdConfig to allow environment-based tuning.
    This function is called at runtime to pick up any environment variable overrides.
    
    Returns:
        Dict with audio filter definitions
    """
    # Lazy import to avoid circular dependency
    from .config_thresholds import ThresholdConfig
    
    # Get dynamic thresholds
    core_threshold = ThresholdConfig.ducking_core_threshold()
    soft_threshold = ThresholdConfig.ducking_soft_threshold()
    
    return {
        # Voice Polish: Rumble removal -> Denoise -> Compress -> EQ -> Limit
        "voice_polish": (
            "highpass=f=80,"
            "afftdn=nf=-25,"
            "acompressor=threshold=-12dB:ratio=4:attack=20:release=250,"
            "equalizer=f=3000:t=q:w=1:g=2,"
            "alimiter=limit=-1dB"
        ),

        # Fast noise reduction for web previews (lighter CPU/latency)
        "noise_reduction_fast": (
            "afftdn=nf=-25:nr=10:nt=w,"
            "highpass=f=80,"
            "lowpass=f=14000,"
            "compand=attacks=0.3:decays=0.8:points=-80/-900|-45/-15|-27/-9|0/-7:soft-knee=6:gain=3"
        ),
        
        # Auto-Ducking: Duck [1] based on [0]
        # Note: Requires complex filter graph construction in code, this is the core effect params
        "ducking_core": f"sidechaincompress=threshold={core_threshold}:ratio=5:attack=50:release=300:link=average",

        # Softer ducking for general B-roll/music mixing
        "ducking_soft": f"sidechaincompress=threshold={soft_threshold}:ratio=4:attack=200:release=1000"
    }

# Static audio filters (legacy access)
AUDIO_FILTERS = {
    # Voice Polish: Rumble removal -> Denoise -> Compress -> EQ -> Limit
    "voice_polish": (
        "highpass=f=80,"
        "afftdn=nf=-25,"
        "acompressor=threshold=-12dB:ratio=4:attack=20:release=250,"
        "equalizer=f=3000:t=q:w=1:g=2,"
        "alimiter=limit=-1dB"
    ),

    # Fast noise reduction for web previews (lighter CPU/latency)
    "noise_reduction_fast": (
        "afftdn=nf=-25:nr=10:nt=w,"
        "highpass=f=80,"
        "lowpass=f=14000,"
        "compand=attacks=0.3:decays=0.8:points=-80/-900|-45/-15|-27/-9|0/-7:soft-knee=6:gain=3"
    ),
    
    # Auto-Ducking: Duck [1] based on [0]
    # Note: Requires complex filter graph construction in code, this is the core effect params
    "ducking_core": "sidechaincompress=threshold=0.1:ratio=5:attack=50:release=300:link=average",

    # Softer ducking for general B-roll/music mixing
    "ducking_soft": "sidechaincompress=threshold=0.03:ratio=4:attack=200:release=1000"
}

LUT_FILES = {
    "cinematic_lut": "cinematic.cube",
    "teal_orange_lut": "teal_orange.cube",
    "film_emulation": "kodak_2383.cube",
    "log_to_rec709": "log_to_rec709.cube",
    "bleach_bypass": "bleach_bypass.cube"
}


# =============================================================================
# Hardware Acceleration Detection & Configuration
# =============================================================================

def get_best_gpu_encoder() -> Optional[str]:
    """
    Get the best available GPU encoder type.
    """
    preferred = _normalize_codec_preference(get_settings().gpu.output_codec)
    from .core import hardware
    config = hardware.get_best_hwaccel(preferred_codec=preferred)
    if config.is_gpu:
        return config.type
    return None

# =============================================================================
# Environment variable overrides
# =============================================================================

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
    - FFMPEG_HWACCEL: "auto", "nvenc", "nvmpi", "vaapi", "qsv", "videotoolbox", "none"
    
    GPU Acceleration:
    - hwaccel="auto": Auto-detect best available GPU encoder
    - hwaccel="nvenc": Force NVIDIA NVENC
    - hwaccel="nvmpi": Force Jetson NVMPI
    - hwaccel="vaapi": Force AMD/Intel VAAPI (Linux)
    - hwaccel="none": Force CPU encoding (default for compatibility)
    """
    # Video encoding
    codec: str = field(default_factory=lambda: get_settings().encoding.codec)
    profile: str = field(default_factory=lambda: get_settings().encoding.profile)
    level: str = field(default_factory=lambda: get_settings().encoding.level)
    pix_fmt: str = field(default_factory=lambda: get_settings().encoding.pix_fmt)
    preset: str = field(default_factory=lambda: get_settings().encoding.preset)
    crf: int = field(default_factory=lambda: get_settings().encoding.crf)
    threads: str = field(default_factory=lambda: str(get_settings().encoding.threads))  # 0 = auto
    
    # Audio encoding
    audio_codec: str = field(default_factory=lambda: get_settings().encoding.audio_codec)
    audio_bitrate: str = field(default_factory=lambda: get_settings().encoding.audio_bitrate)
    
    # Resolution (typically set by heuristics, not env)
    width: int = STANDARD_WIDTH_HORIZONTAL
    height: int = STANDARD_HEIGHT_HORIZONTAL
    fps: int = STANDARD_FPS
    
    # Hardware acceleration
    hwaccel: str = field(default_factory=lambda: get_settings().encoding.hwaccel)
    _hw_config: Optional["hardware.HWConfig"] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Resolve hardware acceleration setting."""
        preferred = _normalize_codec_preference(self.codec)
        from .core import hardware  # Lazy import to avoid circular dependency
        if self.hwaccel == "auto":
            self._hw_config = hardware.get_best_hwaccel(preferred_codec=preferred)
        elif self.hwaccel in ("none", "cpu"):
            self._hw_config = hardware.get_hwaccel_by_type("cpu", preferred_codec=preferred)
        else:
            requested = hardware.get_hwaccel_by_type(self.hwaccel, preferred_codec=preferred)
            if requested:
                self._hw_config = requested
            else:
                self._hw_config = hardware.get_best_hwaccel(preferred_codec=preferred)
                logger.warning(
                    f"Requested GPU encoder '{self.hwaccel}' not available/detected, using {self._hw_config.type}"
                )
        
        if self._hw_config and self._hw_config.is_gpu:
            actual = _normalize_codec_preference(self._hw_config.encoder)
            if actual != preferred:
                logger.warning(f"Requested codec '{self.codec}' not supported by {self._hw_config.type}, using {self._hw_config.encoder}")

    @property
    def is_gpu_accelerated(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._hw_config.is_gpu if self._hw_config else False
    
    @property
    def gpu_encoder_type(self) -> Optional[str]:
        """Get the active GPU encoder type (nvenc, vaapi, etc.)."""
        return self._hw_config.type if self._hw_config and self._hw_config.is_gpu else None
    
    def get_level_for_resolution(self, width: int, height: int, fps: float) -> str:
        """Determine H.264/H.265 level based on resolution and FPS.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frame rate
            
        Returns:
            H.264/H.265 level string (e.g., "5.2")
            
        Raises:
            ValueError: If resolution requires HEVC but H.264 is configured
        """
        pixels = width * height
        
        # 8K (7680x4320): Level 6.2 (HEVC only)
        if pixels > 33_177_600:
            if "265" not in self.effective_codec and "hevc" not in self.effective_codec:
                raise ValueError(
                    f"8K resolution ({width}x{height}) requires HEVC/H.265. "
                    f"Current codec: {self.effective_codec}. Set OUTPUT_CODEC=hevc or use proxy workflow."
                )
            return LEVEL_8K
        
        # 6K (6144x3160): Level 5.2
        elif pixels > 19_660_800:
            logger.info(f"6K resolution detected ({width}x{height}), using Level 5.2")
            return LEVEL_6K
        
        # 4K (3840x2160): Level 5.1 (4K60) or 5.0 (4K30)
        elif pixels > 8_294_400:
            return "5.1" if fps > 30 else "5.0"
        
        # 1080p: Level 4.1 (1080p60) or 4.0 (1080p30)
        elif pixels > 2_073_600:
            return "4.1" if fps > 30 else "4.0"
        
        # SD/720p: Level 3.1
        else:
            return "3.1"
    
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
        if self._hw_config.type == "nvmpi":
            return ["-qp", str(crf_clamped)]
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


def get_preview_video_params() -> List[str]:
    """
    Get FFmpeg video encoding parameters for preview generation.
    Uses lower quality and faster preset for rapid feedback.
    """
    return get_config().video_params(crf=PREVIEW_CRF, preset=PREVIEW_PRESET)


def print_gpu_status():
    """
    Print GPU encoder availability status.
    
    Useful for debugging and configuration.
    """
    logger.info("\n🎮 GPU Encoder Status:")
    logger.info("=" * 50)
    
    # Use hardware module to get status
    from .core import hardware
    hw_config = hardware.get_best_hwaccel()
    
    if hw_config.is_gpu:
        logger.info(f"  ✅ GPU Acceleration Detected: {hw_config.type.upper()}")
        logger.info(f"     Encoder: {hw_config.encoder}")
        logger.info(f"     Decoder Args: {' '.join(hw_config.decoder_args)}")
    else:
        logger.info(f"  ❌ No GPU Acceleration Detected (using CPU)")
        logger.info(f"     Encoder: {hw_config.encoder}")
    
    logger.info("=" * 50)


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
    # Preview Constants
    "PREVIEW_WIDTH",
    "PREVIEW_HEIGHT",
    "PREVIEW_CRF",
    "PREVIEW_PRESET",
    # GPU detection
    "get_best_gpu_encoder",
    "print_gpu_status",
    # Config class
    "FFmpegConfig",
    "get_config",
    # Convenience functions
    "get_ffmpeg_video_params",
    "get_moviepy_params",
    "get_preview_video_params",
]
