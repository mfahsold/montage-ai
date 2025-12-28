"""
StabilizeJob - Video stabilization via FFmpeg vidstab on cgpu.

Stabilizes shaky video footage using FFmpeg's vidstab filter.
Uses a two-pass approach: detect → transform.

Usage:
    job = StabilizeJob(
        video_path="/data/shaky.mp4",
        smoothing=10,
        shakiness=5
    )
    result = job.execute()
    print(result.output_path)  # /data/shaky_stabilized.mp4
"""

import os
from pathlib import Path
from typing import List, Optional

from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64


class StabilizeJob(CGPUJob):
    """
    FFmpeg vidstab stabilization job for cgpu.

    Uses two-pass stabilization:
        1. vidstabdetect - Analyze motion and create transform file
        2. vidstabtransform - Apply stabilization transforms

    Attributes:
        timeout: 900 seconds (15 minutes)
        job_type: "stabilize"
    """

    timeout: int = 900  # 15 minutes
    max_retries: int = 2
    job_type: str = "stabilize"

    def __init__(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        smoothing: int = 10,
        shakiness: int = 5,
        accuracy: int = 15,
        stepsize: int = 6,
        zoom: float = 0.0,
        optzoom: int = 1,
        crop: str = "black",
    ):
        """
        Initialize stabilization job.

        Args:
            video_path: Path to input video
            output_path: Output path (default: input_stabilized.mp4)
            smoothing: Smoothing strength (1-30, default: 10)
            shakiness: How shaky the video is (1-10, default: 5)
            accuracy: Detection accuracy (1-15, default: 15)
            stepsize: Step size for detection (1-32, default: 6)
            zoom: Static zoom percentage (0 = no zoom, default: 0)
            optzoom: Optimal zoom (0=off, 1=static, 2=adaptive, default: 1)
            crop: How to fill borders (black, keep, default: black)
        """
        super().__init__()
        self.video_path = Path(video_path).resolve()

        # Determine output path
        if output_path:
            self.output_path = Path(output_path).resolve()
        else:
            stem = self.video_path.stem
            self.output_path = self.video_path.parent / f"{stem}_stabilized.mp4"

        # Clamp parameters to valid ranges
        self.smoothing = max(1, min(smoothing, 30))
        self.shakiness = max(1, min(shakiness, 10))
        self.accuracy = max(1, min(accuracy, 15))
        self.stepsize = max(1, min(stepsize, 32))
        self.zoom = max(-100.0, min(zoom, 100.0))
        self.optzoom = max(0, min(optzoom, 2))
        self.crop = crop if crop in ["black", "keep"] else "black"

    def prepare_local(self) -> bool:
        """Validate video file exists."""
        if not self.video_path.exists():
            self._error = f"Video file not found: {self.video_path}"
            return False

        # Check it's a video file
        valid_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
        if self.video_path.suffix.lower() not in valid_extensions:
            self._error = f"Invalid video format: {self.video_path.suffix}"
            return False

        return True

    def get_requirements(self) -> List[str]:
        """FFmpeg with vidstab is usually pre-installed on Colab."""
        return []  # ffmpeg should be available

    def upload(self) -> bool:
        """Upload video to remote."""
        remote_path = f"{self.remote_work_dir}/{self.video_path.name}"

        size_mb = self.video_path.stat().st_size / (1024 * 1024)
        print(f"   ⬆️ Uploading {self.video_path.name} ({size_mb:.1f} MB)...")

        if not copy_to_remote(str(self.video_path), remote_path):
            self._error = "Failed to upload video file"
            return False

        return True

    def run_remote(self) -> bool:
        """Run two-pass vidstab stabilization."""
        input_name = self.video_path.name
        output_name = self.output_path.name
        transform_file = "transforms.trf"

        # Pass 1: Detect motion
        detect_cmd = (
            f"cd {self.remote_work_dir} && "
            f"ffmpeg -y -i '{input_name}' "
            f"-vf vidstabdetect="
            f"stepsize={self.stepsize}:"
            f"shakiness={self.shakiness}:"
            f"accuracy={self.accuracy}:"
            f"result={transform_file} "
            f"-f null -"
        )

        print(f"   🔍 Pass 1/2: Analyzing motion (shakiness={self.shakiness})...")
        success, stdout, stderr = run_cgpu_command(detect_cmd, timeout=self.timeout // 2)

        if not success:
            self._error = f"Motion detection failed: {stderr}"
            return False

        # Pass 2: Apply stabilization
        transform_cmd = (
            f"cd {self.remote_work_dir} && "
            f"ffmpeg -y -i '{input_name}' "
            f"-vf vidstabtransform="
            f"input={transform_file}:"
            f"smoothing={self.smoothing}:"
            f"zoom={self.zoom}:"
            f"optzoom={self.optzoom}:"
            f"crop={self.crop} "
            f"-c:v libx264 -preset medium -crf 18 "
            f"-c:a copy "
            f"'{output_name}'"
        )

        print(f"   🎬 Pass 2/2: Applying stabilization (smoothing={self.smoothing})...")
        success, stdout, stderr = run_cgpu_command(transform_cmd, timeout=self.timeout // 2)

        if not success:
            self._error = f"Stabilization transform failed: {stderr}"
            return False

        # Verify output exists
        check_cmd = f"test -f {self.remote_work_dir}/{output_name} && echo EXISTS"
        success, stdout, _ = run_cgpu_command(check_cmd, timeout=10)

        if "EXISTS" not in stdout:
            self._error = "Output file not created"
            return False

        return True

    def download(self) -> JobResult:
        """Download stabilized video."""
        remote_output = f"{self.remote_work_dir}/{self.output_path.name}"

        print(f"   ⬇️ Downloading stabilized video...")

        if download_via_base64(remote_output, str(self.output_path)):
            # Get output size
            if self.output_path.exists():
                size_mb = self.output_path.stat().st_size / (1024 * 1024)
                print(f"   📦 Output: {size_mb:.1f} MB")

            return JobResult(
                success=True,
                output_path=str(self.output_path),
                metadata={
                    "smoothing": self.smoothing,
                    "shakiness": self.shakiness,
                    "zoom": self.zoom,
                    "optzoom": self.optzoom,
                }
            )
        else:
            return JobResult(
                success=False,
                error="Failed to download stabilized video"
            )


# Convenience function for quick stabilization
def stabilize_video(
    video_path: str,
    output_path: Optional[str] = None,
    smoothing: int = 10,
    shakiness: int = 5,
) -> Optional[str]:
    """
    Quick stabilization helper.

    Args:
        video_path: Input video path
        output_path: Output path (optional)
        smoothing: Smoothing strength 1-30
        shakiness: Video shakiness 1-10

    Returns:
        Path to stabilized video, or None if failed
    """
    job = StabilizeJob(
        video_path=video_path,
        output_path=output_path,
        smoothing=smoothing,
        shakiness=shakiness,
    )
    result = job.execute()
    return result.output_path if result.success else None
