"""
Hardware Detection Module - Bare Metal Performance Optimization

Detects available hardware acceleration (GPU) for FFmpeg to ensure
we are running as close to the metal as possible.

Supported Accelerators:
- NVIDIA NVENC / NVMPI (Linux/Windows/Jetson)
- VAAPI (Intel/AMD Linux)
- VideoToolbox (macOS Apple Silicon/Intel)
- QSV (Intel Linux/Windows)

Usage:
    from .hardware import get_best_hwaccel, get_ffmpeg_hw_args

    hw_config = get_best_hwaccel()
    # Returns: {"type": "nvenc", "args": ["-hwaccel", "cuda", ...]}
"""

import os
import sys
import subprocess
import shutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..ffmpeg_utils import build_ffmpeg_cmd

@dataclass
class HWConfig:
    type: str  # 'cpu', 'nvenc', 'nvmpi', 'vaapi', 'videotoolbox', 'qsv'
    encoder: str  # 'libx264', 'h264_nvenc', etc.
    decoder_args: List[str]  # Input args (before -i)
    encoder_args: List[str]  # Output args
    is_gpu: bool
    hwupload_filter: Optional[str] = None  # e.g. "format=nv12,hwupload" for VAAPI

def _check_command(cmd: List[str]) -> bool:
    """Run a command and return True if exit code is 0."""
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def _has_nvidia() -> bool:
    """Check for NVIDIA GPU via nvidia-smi or Jetson device nodes."""
    visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if visible is not None and visible.strip().lower() in {"none", "void"}:
        return False
    if shutil.which("nvidia-smi") is not None:
        return True
    nvidia_paths = (
        "/dev/nvidia0",
        "/dev/nvhost-ctrl",
        "/dev/nvhost-nvenc",
        "/proc/driver/nvidia/version",
        "/etc/nv_tegra_release",
        "/sys/bus/platform/drivers/tegra-fuse",  # Jetson specific
    )
    # Also check for jtop/jetson_stats
    if shutil.which("jtop") is not None:
        return True
        
    return any(os.path.exists(path) for path in nvidia_paths)

def _has_vaapi() -> bool:
    """
    Check for VAAPI device node and working driver.
    Tests actual VAAPI initialization, not just file existence.
    """
    if not os.path.exists("/dev/dri/renderD128"):
        return False

    # Test actual VAAPI initialization with vainfo or ffmpeg
    env = os.environ.copy()
    # Set driver hints for AMD and Intel
    if not env.get("LIBVA_DRIVER_NAME"):
        # Try iHD for Intel first (more common), then radeonsi for AMD
        for driver in ["iHD", "radeonsi", "i965"]:
            env["LIBVA_DRIVER_NAME"] = driver
            try:
                # Note: VAAPI requires hwupload filter to transfer frames to GPU
                result = subprocess.run(
                    build_ffmpeg_cmd(
                        [
                            "-init_hw_device", "vaapi=va:/dev/dri/renderD128",
                            "-filter_hw_device", "va",
                            "-f", "lavfi",
                            "-i", "color=black:s=64x64:d=0.1",
                            "-vf", "format=nv12,hwupload",
                            "-c:v", "h264_vaapi", "-f", "null", "-",
                        ],
                        overwrite=False,
                        hide_banner=True,
                    ),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    timeout=10
                )
                if result.returncode == 0:
                    # Set the environment variable for subsequent calls
                    os.environ["LIBVA_DRIVER_NAME"] = driver
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return False

    # If LIBVA_DRIVER_NAME is already set, just check device exists
    return True

def _has_macos() -> bool:
    """Check if running on macOS."""
    return sys.platform == "darwin"

def _check_ffmpeg_encoder(encoder_name: str) -> bool:
    """Check if local ffmpeg supports a specific encoder."""
    try:
        result = subprocess.run(
            build_ffmpeg_cmd(["-encoders"], overwrite=False, hide_banner=True),
            capture_output=True, 
            text=True
        )
        return encoder_name in result.stdout
    except FileNotFoundError:
        return False

def _normalize_codec(codec: Optional[str]) -> str:
    """Normalize codec preference to 'h264' or 'hevc'."""
    c = (codec or "").lower()
    if "265" in c or "hevc" in c or "h265" in c:
        return "hevc"
    return "h264"

def _pick_encoder(h264_name: str, hevc_name: str, preferred: str) -> Optional[str]:
    """Pick the best available encoder with preference for H.264 or HEVC."""
    if preferred == "hevc" and _check_ffmpeg_encoder(hevc_name):
        return hevc_name
    if _check_ffmpeg_encoder(h264_name):
        return h264_name
    if _check_ffmpeg_encoder(hevc_name):
        return hevc_name
    return None

def get_hwaccel_by_type(hwaccel: str, preferred_codec: str = "h264") -> Optional[HWConfig]:
    """
    Resolve a specific hardware acceleration type to an HWConfig.

    Returns None when the requested accelerator isn't available.
    """
    codec_pref = _normalize_codec(preferred_codec)
    accel = (hwaccel or "").lower()

    if accel in ("none", "cpu"):
        cpu_encoder = "libx264"
        if codec_pref == "hevc" and _check_ffmpeg_encoder("libx265"):
            cpu_encoder = "libx265"
        return HWConfig(
            type="cpu",
            encoder=cpu_encoder,
            decoder_args=[],
            encoder_args=["-c:v", cpu_encoder],
            is_gpu=False
        )

    if accel == "nvenc":
        if not _has_nvidia():
            return None
        encoder = _pick_encoder("h264_nvenc", "hevc_nvenc", codec_pref)
        if not encoder:
            return None
        return HWConfig(
            type="nvenc",
            encoder=encoder,
            decoder_args=["-hwaccel", "cuda"],
            encoder_args=["-c:v", encoder],
            is_gpu=True
        )

    if accel == "nvmpi":
        if not _has_nvidia():
            return None
        encoder = _pick_encoder("h264_nvmpi", "hevc_nvmpi", codec_pref)
        if not encoder:
            return None
        return HWConfig(
            type="nvmpi",
            encoder=encoder,
            decoder_args=[],
            encoder_args=["-c:v", encoder],
            is_gpu=True
        )

    if accel == "videotoolbox":
        if not _has_macos():
            return None
        encoder = _pick_encoder("h264_videotoolbox", "hevc_videotoolbox", codec_pref)
        if not encoder:
            return None
        return HWConfig(
            type="videotoolbox",
            encoder=encoder,
            decoder_args=["-hwaccel", "videotoolbox"],
            encoder_args=["-c:v", encoder, "-allow_sw", "1"],
            is_gpu=True
        )

    if accel == "vaapi":
        if not _has_vaapi():
            return None
        encoder = _pick_encoder("h264_vaapi", "hevc_vaapi", codec_pref)
        if not encoder:
            return None
        return HWConfig(
            type="vaapi",
            encoder=encoder,
            decoder_args=["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"],
            encoder_args=["-c:v", encoder],
            is_gpu=True,
            hwupload_filter="format=nv12,hwupload"
        )

    if accel == "qsv":
        # QSV requires Intel GPU
        if not _is_intel_gpu():
            return None
        encoder = _pick_encoder("h264_qsv", "hevc_qsv", codec_pref)
        if not encoder:
            return None
        return HWConfig(
            type="qsv",
            encoder=encoder,
            decoder_args=["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"],
            encoder_args=["-c:v", encoder],
            is_gpu=True
        )

    return None


def _is_jetson() -> bool:
    """Detect if running on NVIDIA Jetson platform."""
    jetson_indicators = [
        "/etc/nv_tegra_release",
        "/sys/bus/platform/drivers/tegra-fuse",
        "/dev/nvhost-ctrl",
        "/dev/nvhost-nvenc",
    ]
    return any(os.path.exists(p) for p in jetson_indicators)


def _is_intel_gpu() -> bool:
    """Detect if Intel GPU is present (required for QSV encoding)."""
    # Must have DRI device
    if not os.path.exists("/dev/dri/renderD128"):
        return False

    # Check vendor ID in sysfs (Intel = 0x8086)
    vendor_paths = [
        "/sys/class/drm/card0/device/vendor",
        "/sys/class/drm/renderD128/device/vendor",
    ]
    for path in vendor_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    vendor = f.read().strip()
                    if "0x8086" in vendor or "8086" in vendor:
                        return True
            except (IOError, PermissionError):
                pass

    # Fallback: check lspci for Intel VGA
    try:
        result = subprocess.run(
            ["lspci", "-nn"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "Intel" in result.stdout and ("VGA" in result.stdout or "Display" in result.stdout):
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def check_encoder_works(hw_config: HWConfig) -> bool:
    """Test if an encoder actually works by doing a minimal encode."""
    import subprocess
    import tempfile
    
    # CPU always works
    if hw_config.type == "cpu":
        return True
        
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.mp4")
            
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            cmd.extend(hw_config.decoder_args)
            cmd.extend(["-f", "lavfi", "-i", "color=black:s=64x64:d=0.1"])
            
            if hw_config.hwupload_filter:
                cmd.extend(["-vf", hw_config.hwupload_filter])
                
            cmd.extend(hw_config.encoder_args)
            
            # Simple quality args to satisfy encoder requirements
            if hw_config.type in ("nvenc", "nvmpi"):
                cmd.extend(["-cq", "35"])
            elif hw_config.type == "vaapi":
                cmd.extend(["-qp", "35"])
            elif hw_config.type == "qsv":
                cmd.extend(["-global_quality", "35"])
            
            cmd.append(output_path)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            else:
                # logger might not be available or circular, use simple print in debug/dev
                return False
                
    except Exception:
        return False


def get_best_hwaccel(preferred_codec: str = "h264") -> HWConfig:
    """
    Detect the best available hardware acceleration.
    Returns a HWConfig object with FFmpeg arguments.

    Priority order (platform-aware):
    - Jetson: NVMPI first (NVENC doesn't work on Jetson!)
    - Desktop: NVENC → VideoToolbox → VAAPI → QSV
    - Fallback: CPU
    """
    # Jetson-specific: NVMPI is the only working encoder
    if _is_jetson():
        config = get_hwaccel_by_type("nvmpi", preferred_codec=preferred_codec)
        if config and check_encoder_works(config):
            return config
        # Fallback to CPU on Jetson if NVMPI unavailable
        return get_hwaccel_by_type("cpu", preferred_codec=preferred_codec)

    # Desktop/Server: Standard priority order
    for accel in ("nvenc", "videotoolbox", "vaapi", "qsv"):
        config = get_hwaccel_by_type(accel, preferred_codec=preferred_codec)
        if config and check_encoder_works(config):
            return config
            
    # Fallback to CPU
    return get_hwaccel_by_type("cpu", preferred_codec=preferred_codec)

def get_ffmpeg_hw_args(mode: str = "encode", preferred_codec: str = "h264") -> List[str]:
    """
    Get flat list of FFmpeg arguments for the current hardware.
    mode: 'encode' (output args) or 'decode' (input args)
    """
    config = get_best_hwaccel(preferred_codec=preferred_codec)
    if mode == "decode":
        return config.decoder_args
    return config.encoder_args
