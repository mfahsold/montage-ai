"""
VideoEncodingJob - Hardware-accelerated video encoding via NVENC on cgpu.

Offloads video encoding to Tesla T4 GPU when local hardware encoding
is unavailable or fails. Useful for:
- QSV/VAAPI failures on local machine
- Large batch encoding jobs
- High-quality master renders

Usage:
    job = VideoEncodingJob(
        input_path="/data/input.mp4",
        output_path="/data/output.mp4",
        codec="h264",
        quality=18,
    )
    result = job.execute()
    print(result.output_path)  # /data/output.mp4

Decision Logic:
    Use should_use_cgpu_encoding() to decide when CGPU is beneficial:
    - Local GPU unavailable or failed
    - Large files where GPU speed amortizes upload time
    - Master quality renders requiring best encoder
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64
from ..logger import logger
from ..utils import file_size_mb, clamp


class VideoEncodingJob(CGPUJob):
    """
    NVENC video encoding job for cgpu (Tesla T4).

    Features:
        - H.264/HEVC encoding with NVENC
        - Quality presets (p1-p7)
        - Audio passthrough or re-encoding
        - Filter chain support (color grading, scaling)

    Attributes:
        timeout: 1200 seconds (20 minutes)
        job_type: "encoding"
    """

    timeout: int = 1200  # 20 minutes
    max_retries: int = 2
    job_type: str = "encoding"
    requires_gpu: bool = True

    # NVENC quality presets (p1=fastest, p7=best quality)
    QUALITY_PRESETS = {
        "ultrafast": "p1",
        "fast": "p2",
        "medium": "p4",
        "slow": "p5",
        "veryslow": "p7",
    }

    # CRF to CQ mapping (NVENC uses -cq instead of -crf)
    # CQ range 0-51, similar semantics to CRF
    DEFAULT_CQ = 23

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        codec: str = "h264",
        quality: int = 18,
        preset: str = "medium",
        filters: Optional[str] = None,
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Initialize encoding job.

        Args:
            input_path: Path to input video
            output_path: Output path (default: input_encoded.mp4)
            codec: Video codec ("h264" or "hevc")
            quality: Quality level 0-51 (lower = better, 18 = visually lossless)
            preset: Speed/quality preset (ultrafast/fast/medium/slow/veryslow)
            filters: Optional FFmpeg filter string (e.g., color grading)
            audio_codec: Audio codec ("aac", "copy", or "none")
            audio_bitrate: Audio bitrate (e.g., "192k")
            width: Optional output width (None = keep original)
            height: Optional output height (None = keep original)
        """
        super().__init__()
        self.input_path = Path(input_path).resolve()

        # Output path
        if output_path:
            self.output_path = Path(output_path).resolve()
        else:
            stem = self.input_path.stem
            self.output_path = self.input_path.parent / f"{stem}_encoded.mp4"

        # Codec settings
        self.codec = "hevc" if "265" in codec or "hevc" in codec.lower() else "h264"
        self.encoder = f"{self.codec}_nvenc"
        self.quality = int(clamp(quality, 0, 51))
        self.preset = self.QUALITY_PRESETS.get(preset, "p4")

        # Filters
        self.filters = filters

        # Audio settings
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate

        # Resolution
        self.width = width
        self.height = height

        # Remote paths
        self.remote_input = f"{self.remote_work_dir}/input{self.input_path.suffix}"
        self.remote_output = f"{self.remote_work_dir}/output.mp4"

    def prepare_local(self) -> bool:
        """Validate input file exists and check size."""
        if not self.input_path.exists():
            self._error = f"Input file not found: {self.input_path}"
            return False

        # Check file size
        size = file_size_mb(self.input_path)
        if size > 2000:  # 2GB
            logger.warning(f"Very large file ({size:.1f} MB) - encoding may take a long time")
        elif size > 500:
            logger.info(f"Large file ({size:.1f} MB)")

        return True

    def get_requirements(self) -> List[str]:
        """No additional Python packages needed - uses system FFmpeg."""
        return []

    def upload(self) -> bool:
        """Upload input video to cgpu."""
        self.log_upload_start(self.input_path)

        success = copy_to_remote(str(self.input_path), self.remote_input)
        if not success:
            self._error = "Failed to upload input video"
            return False

        self.log_upload_complete()
        return True

    def _build_ffmpeg_command(self) -> str:
        """Build FFmpeg command for NVENC encoding."""
        cmd_parts = [
            "ffmpeg", "-y",
            # Hardware acceleration for decoding
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            # Input
            "-i", self.remote_input,
        ]

        # Video encoding
        cmd_parts.extend([
            "-c:v", self.encoder,
            "-preset", self.preset,
            "-cq", str(self.quality),  # Constant quality mode
            "-b:v", "0",  # Let CQ control quality
        ])

        # Add profile and level for compatibility
        if self.codec == "h264":
            cmd_parts.extend(["-profile:v", "high", "-level", "4.1"])
        else:
            cmd_parts.extend(["-profile:v", "main", "-level", "5.1"])

        # Scaling if requested
        if self.width and self.height:
            # Use CUDA scaling for GPU acceleration
            cmd_parts.extend([
                "-vf", f"scale_cuda={self.width}:{self.height}"
            ])
        elif self.filters:
            # Custom filters (need hwdownload/hwupload for CPU filters)
            cmd_parts.extend([
                "-vf", f"hwdownload,format=nv12,{self.filters},hwupload_cuda"
            ])

        # Audio handling
        if self.audio_codec == "copy":
            cmd_parts.extend(["-c:a", "copy"])
        elif self.audio_codec == "none":
            cmd_parts.extend(["-an"])
        else:
            cmd_parts.extend([
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
            ])

        # Output format
        cmd_parts.extend([
            "-movflags", "+faststart",
            self.remote_output,
        ])

        return " ".join(cmd_parts)

    def run_remote(self) -> bool:
        """Execute NVENC encoding on cgpu."""
        # Build and run FFmpeg command
        ffmpeg_cmd = self._build_ffmpeg_command()

        logger.info(f"Encoding with {self.encoder} (quality={self.quality}, preset={self.preset})")

        # Run encoding with extended timeout for large files
        size = file_size_mb(self.input_path)
        # Estimate: ~10x realtime for NVENC, plus overhead
        estimated_timeout = max(300, int(size * 0.5))  # At least 5 minutes

        success, stdout, stderr = run_cgpu_command(
            ffmpeg_cmd,
            timeout=min(estimated_timeout, self.timeout)
        )

        if not success:
            # Parse FFmpeg error
            if "NVENC" in stderr or "nvenc" in stderr:
                self._error = f"NVENC encoder error: {stderr[-500:]}"
            elif "No such file" in stderr:
                self._error = "Input file not found on remote"
            else:
                self._error = f"FFmpeg encoding failed: {stderr[-500:]}"
            return False

        # Verify output exists
        check_cmd = f"test -f {self.remote_output} && echo 'OK'"
        success, stdout, _ = run_cgpu_command(check_cmd, timeout=30)

        if not success or "OK" not in stdout:
            self._error = "Output file not created"
            return False

        return True

    def download(self) -> JobResult:
        """Download encoded video from cgpu."""
        self.log_download_start("encoded video")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        success = download_via_base64(self.remote_output, str(self.output_path))

        if not success:
            return JobResult(
                success=False,
                error="Failed to download encoded video",
            )

        if not self.output_path.exists():
            return JobResult(
                success=False,
                error="Downloaded file not found",
            )

        self.log_output_size(self.output_path)

        return JobResult(
            success=True,
            output_path=str(self.output_path),
            metadata={
                "encoder": self.encoder,
                "quality": self.quality,
                "preset": self.preset,
                "codec": self.codec,
            }
        )


def should_use_cgpu_encoding(
    file_size_mb: float,
    duration_sec: float,
    local_gpu_available: bool = False,
    quality_profile: str = "standard",
) -> bool:
    """
    Decide if CGPU encoding is beneficial over local encoding.

    Decision factors:
    1. Local GPU unavailable -> use CGPU
    2. Master quality profile -> use CGPU (best encoder)
    3. Large files -> CGPU if GPU speed amortizes upload

    Args:
        file_size_mb: Input file size in MB
        duration_sec: Video duration in seconds
        local_gpu_available: Whether local GPU encoding works
        quality_profile: Quality profile (preview/standard/high/master)

    Returns:
        True if CGPU encoding is recommended
    """
    # Always use CGPU for master quality (best possible encoder)
    if quality_profile == "master":
        logger.info("Using CGPU for master quality encode")
        return True

    # If local GPU works, prefer it (lower latency)
    if local_gpu_available:
        return False

    # Estimate times
    # Upload: ~10 MB/s average
    # CPU encode: ~0.3x realtime (slow)
    # NVENC encode: ~10x realtime (fast)
    # Download: ~10 MB/s average

    upload_time = file_size_mb / 10
    cpu_encode_time = duration_sec / 0.3  # CPU is slow
    nvenc_encode_time = duration_sec / 10  # NVENC is fast

    # Estimate output size (similar to input for same quality)
    download_time = file_size_mb / 10

    cgpu_total = upload_time + nvenc_encode_time + download_time
    cpu_total = cpu_encode_time

    # Use CGPU if it's faster (with 20% margin for overhead)
    if cgpu_total * 1.2 < cpu_total:
        logger.info(
            f"Using CGPU encoding: estimated {cgpu_total:.0f}s vs CPU {cpu_total:.0f}s"
        )
        return True

    return False


def encode_with_cgpu_fallback(
    input_path: str,
    output_path: str,
    codec: str = "h264",
    quality: int = 18,
    preset: str = "medium",
    filters: Optional[str] = None,
) -> Optional[str]:
    """
    Encode video with CGPU, with graceful fallback to local.

    This is the main entry point for encoding with automatic
    CGPU/local selection.

    Args:
        input_path: Input video path
        output_path: Output video path
        codec: Video codec (h264/hevc)
        quality: Quality level 0-51
        preset: Speed preset
        filters: Optional FFmpeg filters

    Returns:
        Output path if successful, None if failed
    """
    from ..cgpu_utils import is_cgpu_available

    # Try CGPU first if available
    if is_cgpu_available(require_gpu=True):
        job = VideoEncodingJob(
            input_path=input_path,
            output_path=output_path,
            codec=codec,
            quality=quality,
            preset=preset,
            filters=filters,
        )
        result = job.execute()

        if result.success:
            logger.info(f"CGPU encoding complete: {result.output_path}")
            return result.output_path
        else:
            logger.warning(f"CGPU encoding failed: {result.error}, falling back to local")

    # Fallback to local CPU encoding
    logger.info("Using local CPU encoding")
    import subprocess
    from ..ffmpeg_utils import build_ffmpeg_cmd

    cmd = build_ffmpeg_cmd([
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(quality),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            return output_path
        else:
            logger.error(f"Local encoding failed: {result.stderr[-500:]}")
            return None
    except subprocess.TimeoutExpired:
        logger.error("Local encoding timed out")
        return None
