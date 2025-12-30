"""
cgpu Cloud GPU Upscaler - Real-ESRGAN via Google Colab T4/A100

Thin wrapper around UpscaleJob for backward compatibility.
All heavy lifting is now in cgpu_jobs/upscale.py.

Usage:
    from montage_ai.cgpu_upscaler import upscale_with_cgpu, is_cgpu_available

    if is_cgpu_available():
        output_path = upscale_with_cgpu("input.mp4", "output.mp4", scale=2)

Version: 3.0.0 - Refactored to use UpscaleJob
"""

from typing import Optional

from .cgpu_utils import is_cgpu_available, check_cgpu_gpu
from .cgpu_jobs import UpscaleJob
from .cgpu_jobs.upscale import reset_session_cache
from .config import get_settings

__all__ = [
    "upscale_with_cgpu",
    "upscale_image_with_cgpu",
    "is_cgpu_available",
    "check_cgpu_gpu",
    "reset_session_cache",
]

VERSION = "3.0.0"


def upscale_with_cgpu(
    input_path: str,
    output_path: str,
    scale: Optional[int] = None,
    model: Optional[str] = None,
    frame_format: Optional[str] = None,
    crf: Optional[int] = None,
    tile_size: Optional[int] = None,
) -> Optional[str]:
    """
    Upscale video using Real-ESRGAN on cgpu cloud GPU.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        scale: Upscale factor (2 or 4)
        model: Real-ESRGAN model name

    Returns:
        output_path on success, None on failure
    """
    settings = get_settings()
    scale_value = scale if scale is not None else settings.upscale.scale
    model_value = model if model is not None else settings.upscale.model
    frame_format_value = frame_format if frame_format is not None else settings.upscale.frame_format
    crf_value = crf if crf is not None else settings.upscale.crf
    tile_value = tile_size if tile_size is not None else settings.upscale.tile_size

    job = UpscaleJob(
        input_path=input_path,
        output_path=output_path,
        scale=scale_value,
        model=model_value,
        frame_format=frame_format_value,
        crf=crf_value,
        tile_size=tile_value,
    )
    result = job.execute()
    return result.output_path if result.success else None


def upscale_image_with_cgpu(
    input_path: str,
    output_path: str,
    scale: int = 4
) -> Optional[str]:
    """
    Upscale a single image using Real-ESRGAN on cgpu cloud GPU.

    Args:
        input_path: Path to input image (PNG/JPG)
        output_path: Path for output image
        scale: Upscale factor (2 or 4)

    Returns:
        output_path on success, None on failure
    """
    job = UpscaleJob(
        input_path=input_path,
        output_path=output_path,
        scale=scale,
        model="realesrgan-x4plus",
    )
    result = job.execute()
    return result.output_path if result.success else None


# =============================================================================
# CLI Interface (KISS)
# =============================================================================
if __name__ == "__main__":
    import sys

    def print_usage():
        print(f"cgpu Cloud GPU Upscaler v{VERSION}")
        print()
        print("Usage:")
        print("  python -m montage_ai.cgpu_upscaler <input> [output] [scale]")
        print()
        print("Examples:")
        print("  python -m montage_ai.cgpu_upscaler video.mp4")
        print("  python -m montage_ai.cgpu_upscaler video.mp4 upscaled.mp4 4")
        print("  python -m montage_ai.cgpu_upscaler image.png output.png 2")
        print()
        print("Options:")
        print("  input   Input video or image file")
        print("  output  Output file (default: input_upscaled.ext)")
        print("  scale   Upscale factor: 2 or 4 (default: 4)")

    def main():
        # Check for help
        if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
            print_usage()
            sys.exit(0)

        # Check cgpu availability
        print(f"cgpu Upscaler v{VERSION}")
        print()

        if not is_cgpu_available():
            print("❌ cgpu not available")
            print("   Set CGPU_GPU_ENABLED=true and ensure cgpu is installed")
            sys.exit(1)

        print("✅ cgpu available")

        # Check GPU
        success, gpu_info = check_cgpu_gpu()
        if success:
            print(f"✅ GPU: {gpu_info}")
        else:
            print(f"⚠️ GPU check: {gpu_info}")
        print()

        # Parse args
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        scale = int(sys.argv[3]) if len(sys.argv) > 3 else 4

        # Auto-generate output path if not provided
        if not output_file:
            import os
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_upscaled{ext}"

        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print(f"Scale:  {scale}x")
        print()

        # Run upscaling
        result = upscale_with_cgpu(input_file, output_file, scale=scale)

        if result:
            import os
            size_mb = os.path.getsize(result) / (1024 * 1024)
            print()
            print(f"✅ Success: {result} ({size_mb:.1f} MB)")
            sys.exit(0)
        else:
            print()
            print("❌ Upscaling failed")
            sys.exit(1)

    main()
