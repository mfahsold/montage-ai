#!/usr/bin/env python3
"""
GPU Encoding Diagnostic Script

Tests all available hardware encoders and reports their status.
Use this to verify GPU encoding works on each node in the cluster.

Usage:
    python scripts/test_gpu_encoding.py
    python scripts/test_gpu_encoding.py --cgpu  # Also test CGPU
    python scripts/test_gpu_encoding.py --full  # Full test with actual encoding
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from montage_ai.core.hardware import (
    get_best_hwaccel,
    get_hwaccel_by_type,
    _has_nvidia,
    _has_vaapi,
    _is_jetson,
    _is_intel_gpu,
    _check_ffmpeg_encoder,
)
from montage_ai.core.encoder_router import (
    EncoderRouter,
    EncoderTier,
    _is_amd_gpu,
)


def print_header(text: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(name: str, status: bool, detail: str = ""):
    """Print a status line."""
    icon = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    detail_str = f" ({detail})" if detail else ""
    print(f"  {color}{icon}{reset} {name}{detail_str}")


def test_platform_detection():
    """Test platform detection functions."""
    print_header("Platform Detection")

    print_status("NVIDIA GPU", _has_nvidia())
    print_status("Jetson Platform", _is_jetson())
    print_status("AMD GPU", _is_amd_gpu())
    print_status("Intel GPU", _is_intel_gpu())
    print_status("VAAPI Available", _has_vaapi())

    import platform
    print(f"\n  Architecture: {platform.machine()}")
    print(f"  System: {platform.system()}")


def test_encoder_availability():
    """Test which FFmpeg encoders are available."""
    print_header("FFmpeg Encoder Availability")

    encoders = [
        ("h264_nvenc", "NVIDIA NVENC H.264"),
        ("hevc_nvenc", "NVIDIA NVENC HEVC"),
        ("h264_nvmpi", "Jetson NVMPI H.264"),
        ("hevc_nvmpi", "Jetson NVMPI HEVC"),
        ("h264_vaapi", "VAAPI H.264"),
        ("hevc_vaapi", "VAAPI HEVC"),
        ("h264_qsv", "Intel QSV H.264"),
        ("hevc_qsv", "Intel QSV HEVC"),
        ("h264_videotoolbox", "macOS VideoToolbox H.264"),
        ("libx264", "Software x264"),
        ("libx265", "Software x265"),
    ]

    for encoder, name in encoders:
        available = _check_ffmpeg_encoder(encoder)
        print_status(name, available, encoder)


def test_hwaccel_configs():
    """Test hardware acceleration configurations."""
    print_header("Hardware Acceleration Configs")

    configs = ["nvenc", "nvmpi", "vaapi", "qsv", "videotoolbox", "cpu"]

    for accel in configs:
        config = get_hwaccel_by_type(accel)
        if config:
            print_status(f"{accel.upper()}", True, config.encoder)
        else:
            print_status(f"{accel.upper()}", False, "not available")

    print("\n  Best available:")
    best = get_best_hwaccel()
    print(f"    Type: {best.type}")
    print(f"    Encoder: {best.encoder}")
    print(f"    Is GPU: {best.is_gpu}")


def test_encoder_router():
    """Test the encoder router selection."""
    print_header("Encoder Router")

    router = EncoderRouter()

    scenarios = [
        ("Preview (small)", 50, 30, "preview"),
        ("Standard (medium)", 200, 60, "standard"),
        ("High (large)", 500, 180, "high"),
        ("Master (cloud)", 1000, 300, "master"),
    ]

    for name, size, duration, profile in scenarios:
        config = router.get_best_encoder(
            file_size_mb=size,
            duration_sec=duration,
            quality_profile=profile,
        )
        print(f"\n  {name}:")
        print(f"    Tier: {config.tier.value}")
        print(f"    Encoder: {config.encoder_name}")
        print(f"    Speed: {config.estimated_speed}x realtime")
        print(f"    CGPU: {config.use_cgpu}")


def test_cgpu_availability():
    """Test CGPU availability."""
    print_header("CGPU (Cloud GPU)")

    try:
        from montage_ai.cgpu_utils import is_cgpu_available

        available = is_cgpu_available(require_gpu=False)
        print_status("CGPU CLI", available)

        if available:
            gpu_available = is_cgpu_available(require_gpu=True)
            print_status("CGPU GPU", gpu_available, "Tesla T4")
    except ImportError as e:
        print_status("CGPU Module", False, str(e))


def test_actual_encoding(encoder_type: str = "auto") -> None:
    """Test actual video encoding with specified encoder."""
    print_header(f"Actual Encoding Test ({encoder_type})")

    # Create test input
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_input.mp4")
        output_path = os.path.join(tmpdir, "test_output.mp4")

        # Generate test video (1 second, 64x64)
        gen_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=1:size=64x64:rate=30",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            input_path
        ]

        print("  Generating test input...")
        result = subprocess.run(gen_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print_status("Generate input", False, result.stderr[:100])
            pytest.fail("Failed to generate test input video")
        print_status("Generate input", True)

        # Select encoder
        if encoder_type == "auto":
            hw = get_best_hwaccel()
        else:
            hw = get_hwaccel_by_type(encoder_type)
            if not hw:
                print_status(f"Encoder {encoder_type}", False, "not available")
                pytest.skip(f"Encoder {encoder_type} not available on this host")

        # Build encode command
        cmd = ["ffmpeg", "-y"]
        cmd.extend(hw.decoder_args)
        cmd.extend(["-i", input_path])

        if hw.hwupload_filter:
            cmd.extend(["-vf", hw.hwupload_filter])

        cmd.extend(hw.encoder_args)

        # Quality settings
        if hw.type in ("nvenc", "nvmpi"):
            cmd.extend(["-cq", "28"])
        elif hw.type == "vaapi":
            cmd.extend(["-qp", "28"])
        elif hw.type == "qsv":
            cmd.extend(["-global_quality", "28"])
        else:
            cmd.extend(["-crf", "28"])

        cmd.append(output_path)

        print(f"  Testing {hw.type.upper()} ({hw.encoder})...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print_status(f"Encode with {hw.encoder}", True, f"{size} bytes")
            return

        error = result.stderr[-200:] if result.stderr else "Unknown error"
        print_status(f"Encode with {hw.encoder}", False, error[:50])
        pytest.fail(f"Encoding failed for {hw.encoder}: {error}")


def main():
    parser = argparse.ArgumentParser(description="GPU Encoding Diagnostic")
    parser.add_argument("--cgpu", action="store_true", help="Test CGPU availability")
    parser.add_argument("--full", action="store_true", help="Run actual encoding tests")
    parser.add_argument("--encoder", type=str, default="auto",
                        help="Test specific encoder (nvenc/nvmpi/vaapi/qsv/cpu)")
    args = parser.parse_args()

    print("\n🎬 Montage AI - GPU Encoding Diagnostic\n")

    test_platform_detection()
    test_encoder_availability()
    test_hwaccel_configs()

    if args.cgpu:
        test_cgpu_availability()

    test_encoder_router()

    if args.full:
        test_actual_encoding(args.encoder)

    print_header("Summary")
    best = get_best_hwaccel()
    print(f"\n  Recommended encoder: {best.type.upper()} ({best.encoder})")
    print(f"  GPU acceleration: {'Yes' if best.is_gpu else 'No'}")

    if _is_jetson():
        print("\n  ⚠️  Jetson detected - using NVMPI encoder")

    print()


if __name__ == "__main__":
    main()
