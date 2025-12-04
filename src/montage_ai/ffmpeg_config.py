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

import os
import subprocess
import shutil
from typing import List, Optional, Dict, Any, Tuple
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
# Hardware Acceleration Detection & Configuration
# =============================================================================

# GPU encoder mapping: hwaccel_type -> (encoder, decoder, hwaccel_flag, device)
GPU_ENCODERS = {
    "nvenc": {
        "h264": "h264_nvenc",
        "hevc": "hevc_nvenc",
        "decoder": "h264_cuvid",
        "hwaccel": "cuda",
        "device": None,
        "preset_map": {
            # Map standard presets to NVENC presets (p1=fastest, p7=quality)
            "ultrafast": "p1", "superfast": "p2", "veryfast": "p3",
            "faster": "p4", "fast": "p5", "medium": "p5",
            "slow": "p6", "slower": "p7", "veryslow": "p7",
        },
        # NVENC uses -cq instead of -crf (similar scale)
        "quality_param": "-cq",
    },
    "vaapi": {
        "h264": "h264_vaapi",
        "hevc": "hevc_vaapi",
        "decoder": None,  # Use generic hwaccel
        "hwaccel": "vaapi",
        "device": "/dev/dri/renderD128",
        "preset_map": {},  # VAAPI doesn't use presets
        "quality_param": "-qp",  # QP mode for VAAPI
    },
    "videotoolbox": {
        "h264": "h264_videotoolbox",
        "hevc": "hevc_videotoolbox",
        "decoder": None,
        "hwaccel": "videotoolbox",
        "device": None,
        "preset_map": {},
        "quality_param": "-q:v",  # Quality scale 1-100
    },
    "qsv": {
        "h264": "h264_qsv",
        "hevc": "hevc_qsv",
        "decoder": "h264_qsv",
        "hwaccel": "qsv",
        "device": "/dev/dri/renderD128",
        "preset_map": {
            "ultrafast": "veryfast", "superfast": "veryfast",
            "veryfast": "veryfast", "faster": "faster", "fast": "fast",
            "medium": "medium", "slow": "slow", "slower": "slower",
            "veryslow": "veryslow",
        },
        "quality_param": "-global_quality",
    },
}

# Cache for detected GPU capabilities
_gpu_cache: Optional[Dict[str, bool]] = None


def detect_gpu_encoders() -> Dict[str, bool]:
    """
    Detect available GPU encoders by probing FFmpeg.
    
    Returns dict like: {"nvenc": True, "vaapi": False, "qsv": False, ...}
    """
    global _gpu_cache
    if _gpu_cache is not None:
        return _gpu_cache
    
    _gpu_cache = {}
    
    # Check if ffmpeg exists
    if not shutil.which("ffmpeg"):
        return _gpu_cache
    
    # Get list of available encoders
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        encoders_output = result.stdout.lower()
    except Exception:
        return _gpu_cache
    
    # Check each GPU encoder type with runtime verification
    for hwaccel_type, config in GPU_ENCODERS.items():
        encoder = config["h264"].lower()
        
        # First check: encoder exists in ffmpeg
        if encoder not in encoders_output:
            _gpu_cache[hwaccel_type] = False
            continue

        # Second check: runtime test - actually try to use the encoder
        # This catches cases where encoder is compiled in but GPU is unavailable
        try:
            test_result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-f", "lavfi",
                    "-i", "color=black:s=64x64:d=0.1",
                    "-c:v", config["h264"], "-f", "null", "-"
                ],
                capture_output=True, text=True, timeout=5
            )
            # Encoder works if return code is 0
            _gpu_cache[hwaccel_type] = (test_result.returncode == 0)
        except (subprocess.TimeoutExpired, Exception):
            _gpu_cache[hwaccel_type] = False
    
    # Additional VAAPI check: verify device exists
    if _gpu_cache.get("vaapi"):
        vaapi_device = GPU_ENCODERS["vaapi"]["device"]
        if vaapi_device and not os.path.exists(vaapi_device):
            _gpu_cache["vaapi"] = False
    
    return _gpu_cache


def get_best_gpu_encoder() -> Optional[str]:
    """
    Get the best available GPU encoder.
    
    Priority: nvenc > vaapi > qsv > videotoolbox
    (NVENC is fastest, VAAPI is most common on Linux)
    
    Returns: "nvenc", "vaapi", "qsv", "videotoolbox", or None
    """
    available = detect_gpu_encoders()
    
    # Priority order
    for encoder_type in ["nvenc", "vaapi", "qsv", "videotoolbox"]:
        if available.get(encoder_type):
            return encoder_type
    
    return None


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
    hwaccel: str = field(default_factory=lambda: _env_or_default("FFMPEG_HWACCEL", "none"))
    _resolved_hwaccel: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Resolve hardware acceleration setting."""
        if self.hwaccel == "auto":
            self._resolved_hwaccel = get_best_gpu_encoder()
        elif self.hwaccel in GPU_ENCODERS:
            # Verify requested encoder is available
            available = detect_gpu_encoders()
            if available.get(self.hwaccel):
                self._resolved_hwaccel = self.hwaccel
            else:
                print(f"⚠️ Requested GPU encoder '{self.hwaccel}' not available, falling back to CPU")
                self._resolved_hwaccel = None
        else:
            self._resolved_hwaccel = None
    
    @property
    def is_gpu_accelerated(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._resolved_hwaccel is not None
    
    @property
    def gpu_encoder_type(self) -> Optional[str]:
        """Get the active GPU encoder type (nvenc, vaapi, etc.)."""
        return self._resolved_hwaccel
    
    @property
    def effective_codec(self) -> str:
        """
        Get the effective video codec to use.
        
        Returns GPU encoder (e.g., h264_nvenc) if HW accel is active,
        otherwise returns the configured CPU codec (e.g., libx264).
        
        Use this for MoviePy's codec= parameter.
        """
        gpu_config = self._get_gpu_config()
        if gpu_config:
            # Determine H.264 vs HEVC based on configured codec
            if "265" in self.codec or "hevc" in self.codec.lower():
                return gpu_config["hevc"]
            return gpu_config["h264"]
        return self.codec
    
    def _get_gpu_config(self) -> Optional[Dict]:
        """Get GPU encoder configuration if available."""
        if self._resolved_hwaccel:
            return GPU_ENCODERS.get(self._resolved_hwaccel)
        return None
    
    def hwaccel_input_params(self) -> List[str]:
        """
        Generate FFmpeg input parameters for hardware-accelerated decoding.
        
        These go BEFORE the -i input flag.
        
        Example: ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        """
        gpu_config = self._get_gpu_config()
        if not gpu_config:
            return []
        
        params = []
        
        # Add hwaccel flag
        if gpu_config.get("hwaccel"):
            params.extend(["-hwaccel", gpu_config["hwaccel"]])
            
            # Keep frames in GPU memory if possible (NVENC/VAAPI)
            if self._resolved_hwaccel in ("nvenc", "vaapi"):
                params.extend(["-hwaccel_output_format", gpu_config["hwaccel"]])
        
        # Add device if needed (VAAPI, QSV)
        if gpu_config.get("device"):
            if self._resolved_hwaccel == "vaapi":
                params.extend(["-vaapi_device", gpu_config["device"]])
            elif self._resolved_hwaccel == "qsv":
                params.extend(["-init_hw_device", f"qsv=hw:{gpu_config['device']}"])
        
        return params
    
    def video_params(self, crf: Optional[int] = None, preset: Optional[str] = None) -> List[str]:
        """
        Generate FFmpeg video encoding parameters.
        
        Automatically uses GPU encoder if available and configured.
        
        Args:
            crf: Override CRF (quality). Lower = better quality, bigger file.
            preset: Override preset (speed vs compression tradeoff).
        
        Returns:
            List of FFmpeg CLI arguments for video encoding.
        """
        gpu_config = self._get_gpu_config()
        
        if gpu_config:
            return self._gpu_video_params(gpu_config, crf, preset)
        else:
            return self._cpu_video_params(crf, preset)
    
    def _cpu_video_params(self, crf: Optional[int] = None, preset: Optional[str] = None) -> List[str]:
        """Generate CPU (software) encoding parameters."""
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
    
    def _gpu_video_params(
        self, 
        gpu_config: Dict, 
        crf: Optional[int] = None, 
        preset: Optional[str] = None
    ) -> List[str]:
        """Generate GPU (hardware) encoding parameters."""
        # Determine codec based on output format
        # Default to H.264 for compatibility
        codec_key = "hevc" if "265" in self.codec or "hevc" in self.codec.lower() else "h264"
        encoder = gpu_config[codec_key]
        
        params = ["-c:v", encoder]
        
        # Pixel format - GPU encoders may need different formats
        if self._resolved_hwaccel == "vaapi":
            # VAAPI needs frames uploaded to GPU surface
            params.extend(["-vf", "format=nv12,hwupload"])
        elif self._resolved_hwaccel == "nvenc":
            params.extend(["-pix_fmt", self.pix_fmt])
        else:
            params.extend(["-pix_fmt", self.pix_fmt])
        
        # Profile (most GPU encoders support this)
        if self.profile and self._resolved_hwaccel != "vaapi":
            params.extend(["-profile:v", self.profile])
        
        # Preset mapping
        preset_to_use = preset or self.preset
        if gpu_config.get("preset_map"):
            mapped_preset = gpu_config["preset_map"].get(preset_to_use, preset_to_use)
            params.extend(["-preset", mapped_preset])
        
        # Quality parameter (different per encoder)
        quality_param = gpu_config.get("quality_param", "-crf")
        quality_value = crf if crf is not None else self.crf
        
        # VideoToolbox uses 1-100 scale, not CRF
        if self._resolved_hwaccel == "videotoolbox":
            # Convert CRF (0-51, lower=better) to VT quality (1-100, higher=better)
            quality_value = max(1, min(100, 100 - (quality_value * 2)))
        
        params.extend([quality_param, str(quality_value)])
        
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
        Note: MoviePy handles codec separately via codec= argument.

        Automatically uses GPU-appropriate quality parameters:
        - NVENC: -cq instead of -crf
        - VAAPI: -qp instead of -crf (+ hwupload filter)
        - QSV: -global_quality instead of -crf
        - CPU: -crf (standard)
        """
        gpu_config = self._get_gpu_config()
        quality_value = crf if crf is not None else self.crf

        # GPU encoding path
        if gpu_config:
            params = []

            # VAAPI needs special handling: hwupload filter for surface upload
            if self._resolved_hwaccel == "vaapi":
                # Note: -vf filter added via params, pix_fmt handled by filter chain
                params.extend(["-vf", "format=nv12,hwupload"])
            else:
                params.extend(["-pix_fmt", self.pix_fmt])

            # Profile (most GPU encoders support this, except VAAPI)
            if self.profile and self._resolved_hwaccel != "vaapi":
                params.extend(["-profile:v", self.profile])

            # Preset mapping
            if gpu_config.get("preset_map"):
                mapped_preset = gpu_config["preset_map"].get(self.preset, self.preset)
                params.extend(["-preset", mapped_preset])

            # Quality parameter (encoder-specific)
            quality_param = gpu_config.get("quality_param", "-crf")

            # VideoToolbox uses 1-100 scale instead of CRF (0-51)
            if self._resolved_hwaccel == "videotoolbox":
                # Convert CRF (0-51, lower=better) to VT quality (1-100, higher=better)
                quality_value = max(1, min(100, 100 - (quality_value * 2)))

            params.extend([quality_param, str(quality_value)])

            return params

        # CPU encoding path (standard)
        params = ["-pix_fmt", self.pix_fmt]

        if self.codec in ("libx264", "libx265", "h264", "hevc"):
            if self.profile:
                params.extend(["-profile:v", self.profile])
            if self.level:
                params.extend(["-level:v", self.level])

        params.extend([
            "-preset", self.preset,
            "-crf", str(quality_value),
        ])

        return params


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


def get_moviepy_params(crf: Optional[int] = None) -> List[str]:
    """
    Get FFmpeg parameters for MoviePy's write_videofile().
    
    Example:
        clip.write_videofile(output, ffmpeg_params=get_moviepy_params(crf=18))
    """
    return get_config().moviepy_params(crf)


def print_gpu_status():
    """
    Print GPU encoder availability status.
    
    Useful for debugging and configuration.
    """
    print("\n🎮 GPU Encoder Status:")
    print("=" * 50)
    
    available = detect_gpu_encoders()
    
    for encoder_type, is_available in available.items():
        status = "✅ Available" if is_available else "❌ Not available"
        config = GPU_ENCODERS[encoder_type]
        encoder = config["h264"]
        print(f"  {encoder_type.upper():12} ({encoder}): {status}")
    
    best = get_best_gpu_encoder()
    if best:
        print(f"\n  🏆 Best available: {best.upper()}")
        print(f"     Use FFMPEG_HWACCEL={best} to enable")
    else:
        print(f"\n  ℹ️  No GPU encoders detected, using CPU (libx264)")
    
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
    "GPU_ENCODERS",
    "detect_gpu_encoders",
    "get_best_gpu_encoder",
    "print_gpu_status",
    # Config class
    "FFmpegConfig",
    "get_config",
    # Convenience functions
    "get_ffmpeg_video_params",
    "get_moviepy_params",
]
