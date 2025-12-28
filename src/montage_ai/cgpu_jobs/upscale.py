"""
UpscaleJob - AI video/image upscaling via Real-ESRGAN on cgpu.

Upscales video or image files using Real-ESRGAN with GPU acceleration.

Usage:
    job = UpscaleJob(
        input_path="/data/video.mp4",
        scale=4,
        model="realesrgan-x4plus"
    )
    result = job.execute()
    print(result.output_path)  # /data/video_upscaled.mp4
"""

import os
from pathlib import Path
from typing import List, Optional

from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64


class UpscaleJob(CGPUJob):
    """
    Real-ESRGAN upscaling job for cgpu.

    Attributes:
        timeout: 1800 seconds (30 minutes) - video upscaling is slow
        job_type: "upscale"
    """

    timeout: int = 1800  # 30 minutes for video
    max_retries: int = 2
    job_type: str = "upscale"

    # Available models
    VALID_MODELS = [
        "realesrgan-x4plus",
        "realesrgan-x4plus-anime",
        "realesrnet-x4plus",
    ]

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        scale: int = 4,
        model: str = "realesrgan-x4plus",
        denoise_strength: float = 0.5,
    ):
        """
        Initialize upscaling job.

        Args:
            input_path: Path to video or image file
            output_path: Output path (default: input_upscaled.ext)
            scale: Upscale factor (2, 3, or 4)
            model: Real-ESRGAN model name
            denoise_strength: Denoise strength 0-1 (only for video)
        """
        super().__init__()
        self.input_path = Path(input_path).resolve()
        self.scale = max(2, min(scale, 4))  # Clamp 2-4
        self.model = model if model in self.VALID_MODELS else "realesrgan-x4plus"
        self.denoise_strength = max(0.0, min(denoise_strength, 1.0))

        # Determine output path
        if output_path:
            self.output_path = Path(output_path).resolve()
        else:
            stem = self.input_path.stem
            suffix = self.input_path.suffix
            self.output_path = self.input_path.parent / f"{stem}_upscaled{suffix}"

        # Detect if video or image
        self.is_video = self.input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    def prepare_local(self) -> bool:
        """Validate input file exists."""
        if not self.input_path.exists():
            self._error = f"Input file not found: {self.input_path}"
            return False

        # Check file size
        size_mb = self.input_path.stat().st_size / (1024 * 1024)
        if size_mb > 1000:
            print(f"   ⚠️ Large file ({size_mb:.1f} MB) - upscaling may take a long time")

        return True

    def get_requirements(self) -> List[str]:
        """Real-ESRGAN requirements."""
        return ["realesrgan"]  # pip install realesrgan

    def upload(self) -> bool:
        """Upload input file to remote."""
        remote_path = f"{self.remote_work_dir}/{self.input_path.name}"

        size_mb = self.input_path.stat().st_size / (1024 * 1024)
        print(f"   ⬆️ Uploading {self.input_path.name} ({size_mb:.1f} MB)...")

        if not copy_to_remote(str(self.input_path), remote_path):
            self._error = "Failed to upload input file"
            return False

        return True

    def run_remote(self) -> bool:
        """Run Real-ESRGAN upscaling on cgpu."""
        input_name = self.input_path.name
        output_name = self.output_path.name

        if self.is_video:
            return self._run_video_upscale(input_name, output_name)
        else:
            return self._run_image_upscale(input_name, output_name)

    def _run_image_upscale(self, input_name: str, output_name: str) -> bool:
        """Upscale a single image."""
        # Use realesrgan-ncnn-vulkan or python realesrgan
        script = f'''
import sys
sys.path.insert(0, '/content')

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2
import numpy as np

# Load model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale={self.scale})
upsampler = RealESRGANer(
    scale={self.scale},
    model_path=None,  # Will download automatically
    dni_weight=None,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True,
    gpu_id=0
)

# Read image
img = cv2.imread("{self.remote_work_dir}/{input_name}", cv2.IMREAD_UNCHANGED)
output, _ = upsampler.enhance(img, outscale={self.scale})

# Save
cv2.imwrite("{self.remote_work_dir}/{output_name}", output)
print("UPSCALE_SUCCESS")
'''
        # Write and execute script
        script_path = f"{self.remote_work_dir}/upscale.py"
        script_escaped = script.replace("'", "'\\''")

        cmd = f"echo '{script_escaped}' > {script_path} && python3 {script_path}"

        print(f"   🔍 Upscaling image ({self.scale}x)...")
        success, stdout, stderr = run_cgpu_command(cmd, timeout=self.timeout)

        if not success or "UPSCALE_SUCCESS" not in stdout:
            self._error = f"Image upscaling failed: {stderr or stdout}"
            return False

        return True

    def _run_video_upscale(self, input_name: str, output_name: str) -> bool:
        """Upscale video frame by frame."""
        # For video, we use ffmpeg + realesrgan in a pipeline
        # This is more complex - extract frames, upscale, reassemble

        script = f'''
import os
import subprocess
import glob

work_dir = "{self.remote_work_dir}"
input_video = f"{{work_dir}}/{input_name}"
output_video = f"{{work_dir}}/{output_name}"
frames_dir = f"{{work_dir}}/frames"
upscaled_dir = f"{{work_dir}}/upscaled"

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(upscaled_dir, exist_ok=True)

# Get video info
fps_result = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", input_video],
    capture_output=True, text=True
)
fps = eval(fps_result.stdout.strip())

# Extract frames
print("Extracting frames...")
subprocess.run([
    "ffmpeg", "-i", input_video, "-qscale:v", "2",
    f"{{frames_dir}}/frame_%05d.png"
], check=True, capture_output=True)

# Upscale frames with realesrgan-ncnn-vulkan (faster than python)
print("Upscaling frames...")
subprocess.run([
    "realesrgan-ncnn-vulkan",
    "-i", frames_dir,
    "-o", upscaled_dir,
    "-s", "{self.scale}",
    "-n", "{self.model}"
], check=True, capture_output=True)

# Reassemble video
print("Reassembling video...")
subprocess.run([
    "ffmpeg", "-y", "-framerate", str(fps),
    "-i", f"{{upscaled_dir}}/frame_%05d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-crf", "18", output_video
], check=True, capture_output=True)

# Copy audio if exists
subprocess.run([
    "ffmpeg", "-y", "-i", output_video, "-i", input_video,
    "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?",
    f"{{work_dir}}/final_{output_name}"
], capture_output=True)

# Move final to output
if os.path.exists(f"{{work_dir}}/final_{output_name}"):
    os.rename(f"{{work_dir}}/final_{output_name}", output_video)

print("UPSCALE_SUCCESS")
'''
        script_path = f"{self.remote_work_dir}/upscale_video.py"

        # Write script
        write_cmd = f"cat > {script_path} << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF"
        run_cgpu_command(write_cmd, timeout=30)

        # Install realesrgan-ncnn-vulkan if needed
        run_cgpu_command(
            "which realesrgan-ncnn-vulkan || "
            "(wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip -O /tmp/rg.zip && "
            "unzip -q /tmp/rg.zip -d /usr/local/bin/ && chmod +x /usr/local/bin/realesrgan-ncnn-vulkan)",
            timeout=120
        )

        print(f"   🎬 Upscaling video ({self.scale}x)...")
        success, stdout, stderr = run_cgpu_command(
            f"python3 {script_path}",
            timeout=self.timeout
        )

        if not success or "UPSCALE_SUCCESS" not in stdout:
            self._error = f"Video upscaling failed: {stderr or stdout}"
            return False

        return True

    def download(self) -> JobResult:
        """Download upscaled file."""
        remote_output = f"{self.remote_work_dir}/{self.output_path.name}"

        print(f"   ⬇️ Downloading upscaled file...")

        if download_via_base64(remote_output, str(self.output_path)):
            # Get output size
            if self.output_path.exists():
                size_mb = self.output_path.stat().st_size / (1024 * 1024)
                print(f"   📦 Output: {size_mb:.1f} MB")

            return JobResult(
                success=True,
                output_path=str(self.output_path),
                metadata={
                    "scale": self.scale,
                    "model": self.model,
                    "is_video": self.is_video,
                }
            )
        else:
            return JobResult(
                success=False,
                error="Failed to download upscaled file"
            )
