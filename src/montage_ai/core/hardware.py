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
    )
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
        # Try radeonsi for AMD, then iHD for Intel
        for driver in ["radeonsi", "iHD", "i965"]:
            env["LIBVA_DRIVER_NAME"] = driver
            try:
                result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-init_hw_device",
                     "vaapi=va:/dev/dri/renderD128", "-f", "lavfi",
                     "-i", "color=black:s=64x64:d=0.1",
                     "-c:v", "h264_vaapi", "-f", "null", "-"],
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
            ["ffmpeg", "-encoders"], 
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
        encoder = _pick_encoder("h264_qsv", "hevc_qsv", codec_pref)
        if not encoder:
            return None
        return HWConfig(
            type="qsv",
            encoder=encoder,
            decoder_args=["-hwaccel", "qsv"],
            encoder_args=["-c:v", encoder],
            is_gpu=True
        )

    return None


def get_best_hwaccel(preferred_codec: str = "h264") -> HWConfig:
    """
    Detect the best available hardware acceleration.
    Returns a HWConfig object with FFmpeg arguments.
    """
    for accel in ("nvenc", "nvmpi", "videotoolbox", "vaapi", "qsv"):
        config = get_hwaccel_by_type(accel, preferred_codec=preferred_codec)
        if config:
            return config

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
