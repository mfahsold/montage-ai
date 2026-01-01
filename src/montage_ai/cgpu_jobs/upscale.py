"""
UpscaleJob - AI video/image upscaling via Real-ESRGAN on cgpu.

Production-ready implementation with:
- CUDA diagnostics and error analysis
- Torchvision compatibility patches
- Optimized JPEG-based frame extraction
- Detailed progress logging with ETA
- Session-aware environment caching

Usage:
    job = UpscaleJob(
        input_path="/data/video.mp4",
        scale=4,
        model="realesr-animevideov3"
    )
    result = job.execute()
    print(result.output_path)  # /data/video_upscaled.mp4
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64
from ..logger import logger


# Session state - shared across all UpscaleJob instances
_session_env_ready = False


class UpscaleJob(CGPUJob):
    """
    Real-ESRGAN upscaling job for cgpu.

    Features:
        - Video upscaling via frame extraction + Real-ESRGAN + reassembly
        - Image upscaling with GPU acceleration
        - CUDA diagnostics for debugging
        - Torchvision compatibility patches (v0.18+ support)
        - Session caching for multi-job efficiency

    Attributes:
        timeout: 1800 seconds (30 minutes) - video upscaling is slow
        job_type: "upscale"
    """

    timeout: int = 1800  # 30 minutes for video
    max_retries: int = 2
    job_type: str = "upscale"

    # Available models with their configurations
    MODEL_CONFIGS = {
        "realesr-animevideov3": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            "arch": "SRVGGNetCompact",
            "init": "SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')",
            "scale": 4,
        },
        "realesrgan-x4plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "arch": "RRDBNet",
            "init": "RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)",
            "scale": 4,
        },
        "realesrgan-x4plus-anime": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "arch": "RRDBNet",
            "init": "RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)",
            "scale": 4,
        },
    }

    # Legacy model name mapping
    VALID_MODELS = list(MODEL_CONFIGS.keys())

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        scale: int = 4,
        model: str = "realesr-animevideov3",
        denoise_strength: float = 0.5,
        tile_size: int = 512,
        frame_format: str = "jpg",
        crf: int = 18,
    ):
        """
        Initialize upscaling job.

        Args:
            input_path: Path to video or image file
            output_path: Output path (default: input_upscaled.ext)
            scale: Upscale factor (2, 3, or 4)
            model: Real-ESRGAN model name
            denoise_strength: Denoise strength 0-1 (for future use)
            tile_size: GPU tile size for memory management (default: 512)
            frame_format: Frame cache format for video upscaling (jpg or png)
            crf: CRF value for output encoding (lower = higher quality)
        """
        super().__init__()
        self.input_path = Path(input_path).resolve()
        self.scale = max(2, min(scale, 4))  # Clamp 2-4
        self.denoise_strength = max(0.0, min(denoise_strength, 1.0))
        self.tile_size = tile_size
        self.crf = max(0, min(int(crf), 51))
        normalized_format = (frame_format or "jpg").strip().lower()
        self.frame_format = normalized_format if normalized_format in ("jpg", "png") else "jpg"

        # Normalize model name
        if model in self.MODEL_CONFIGS:
            self.model = model
        elif "anime" in model.lower():
            self.model = "realesr-animevideov3"
        else:
            self.model = "realesrgan-x4plus"

        # Determine output path
        if output_path:
            self.output_path = Path(output_path).resolve()
        else:
            stem = self.input_path.stem
            suffix = self.input_path.suffix if self.input_path.suffix else ".mp4"
            self.output_path = self.input_path.parent / f"{stem}_upscaled{suffix}"

        # Detect if video or image
        self.is_video = self.input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]

    def prepare_local(self) -> bool:
        """Validate input file exists."""
        if not self.input_path.exists():
            self._error = f"Input file not found: {self.input_path}"
            return False

        # Check file size and warn
        self.warn_large_file(self.input_path)

        return True

    def get_requirements(self) -> List[str]:
        """Real-ESRGAN requirements (installed via setup script)."""
        return []  # We handle installation in setup_remote_env

    def setup_remote_env(self) -> bool:
        """Set up Colab environment with Real-ESRGAN and patches."""
        global _session_env_ready

        # Create work directory
        success, _, stderr = run_cgpu_command(f"mkdir -p {self.remote_work_dir}")
        if not success:
            self._error = f"Failed to create remote directory: {stderr}"
            return False

        # Check if environment is already set up (session caching)
        if _session_env_ready:
            check_success, check_out, _ = run_cgpu_command(
                "python3 -c 'import realesrgan; print(\"ENV_OK\")'",
                timeout=30
            )
            if check_success and "ENV_OK" in check_out:
                logger.info(f"Reusing cached environment")
                return True
            else:
                logger.warning(f"Session expired, re-initializing...")
                _session_env_ready = False

        logger.info(f"Setting up Real-ESRGAN environment...")

        # Install dependencies with torchvision compatibility patch
        setup_script = '''
# Install Real-ESRGAN dependencies
pip install -q opencv-python-headless 2>/dev/null
pip install -q realesrgan basicsr 2>/dev/null || true

# Create torchvision compatibility patch for v0.18+
python3 << 'PATCH_SCRIPT'
import os
import site

patch_code = """
# Compatibility shim for torchvision.transforms.functional_tensor
# This module was removed in torchvision 0.18+
from torchvision.transforms import functional as F

def rgb_to_grayscale(img, num_output_channels=1):
    return F.rgb_to_grayscale(img, num_output_channels)
"""

for sp in site.getsitepackages():
    tv_path = os.path.join(sp, "torchvision", "transforms")
    if os.path.exists(tv_path):
        patch_file = os.path.join(tv_path, "functional_tensor.py")
        with open(patch_file, "w") as f:
            f.write(patch_code)
        print(f"PATCHED: {patch_file}")
        break
else:
    print("WARN: torchvision not found")
PATCH_SCRIPT

# Verify CUDA and print diagnostics
python3 -c "
import torch
print('=' * 50)
print('CUDA DIAGNOSTICS')
print('=' * 50)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Memory: {props.total_memory / 1024**3:.1f} GB')
print('=' * 50)
"

echo "ENV_READY"
'''
        success, stdout, stderr = run_cgpu_command(setup_script, timeout=180)

        if not success or "ENV_READY" not in stdout:
            self._error = f"Environment setup failed: {stderr or stdout}"
            return False

        # Show diagnostics
        if stdout and "CUDA DIAGNOSTICS" in stdout:
            for line in stdout.split('\n'):
                if line.strip() and not line.startswith('Authenticated'):
                    if 'CUDA' in line or 'GPU' in line or 'PyTorch' in line or 'Memory' in line:
                        logger.info(f"[GPU] {line}")

        _session_env_ready = True
        logger.info(f"Environment ready")
        return True

    def upload(self) -> bool:
        """Upload input file to remote."""
        from ..utils import file_size_mb
        remote_path = f"{self.remote_work_dir}/{self.input_path.name}"

        self.log_upload_start(self.input_path)

        # Dynamic timeout based on file size
        size_mb = file_size_mb(self.input_path)
        timeout = max(600, int(size_mb / 10 * 60))

        if not copy_to_remote(str(self.input_path), remote_path, timeout=timeout):
            self._error = "Failed to upload input file"
            return False

        self.log_upload_complete()
        return True

    def run_remote(self) -> bool:
        """Run Real-ESRGAN upscaling on cgpu."""
        if self.is_video:
            return self._run_video_upscale()
        else:
            return self._run_image_upscale()

    def _run_image_upscale(self) -> bool:
        """Upscale a single image using Python Real-ESRGAN."""
        input_name = self.input_path.name
        output_name = self.output_path.name
        config = self.MODEL_CONFIGS[self.model]

        script = f'''
import cv2
import torch
import urllib.request
import os

from realesrgan import RealESRGANer
if "{config['arch']}" == "SRVGGNetCompact":
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
else:
    from basicsr.archs.rrdbnet_arch import RRDBNet

# Download model if needed
model_path = "/content/esrgan_model.pth"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve("{config['url']}", model_path)

# Initialize model
print("Initializing Real-ESRGAN...")
model = {config['init']}
upsampler = RealESRGANer(
    scale={config['scale']},
    model_path=model_path,
    model=model,
    tile={self.tile_size},
    tile_pad=10,
    pre_pad=0,
    half=True,
    device="cuda"
)

# Process image
img = cv2.imread("{self.remote_work_dir}/{input_name}", cv2.IMREAD_UNCHANGED)
print(f"Input: {{img.shape[1]}}x{{img.shape[0]}}")

output, _ = upsampler.enhance(img, outscale={self.scale})
print(f"Output: {{output.shape[1]}}x{{output.shape[0]}}")

cv2.imwrite("{self.remote_work_dir}/{output_name}", output)
print("UPSCALE_SUCCESS")
'''
        # Write script to temp file and upload
        script_path = f"{self.remote_work_dir}/upscale_image.py"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            local_script = f.name

        try:
            if not copy_to_remote(local_script, script_path, timeout=30):
                self._error = "Failed to upload processing script"
                return False

            logger.info(f"Upscaling image ({self.scale}x, {self.model})...")
            success, stdout, stderr = run_cgpu_command(
                f"python3 {script_path}",
                timeout=self.timeout
            )
        finally:
            os.unlink(local_script)

        if not success or "UPSCALE_SUCCESS" not in stdout:
            self._error = self._analyze_cuda_error(stdout, stderr)
            return False

        return True

    def _run_video_upscale(self) -> bool:
        """Upscale video using optimized frame-by-frame pipeline."""
        input_name = self.input_path.name
        output_name = self.output_path.name
        config = self.MODEL_CONFIGS[self.model]

        # Optimized pipeline script with frame caching and progress logging
        script = f'''
import os
import glob
import subprocess
import urllib.request
import time
import torch
import cv2

from realesrgan import RealESRGANer
if "{config['arch']}" == "SRVGGNetCompact":
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
else:
    from basicsr.archs.rrdbnet_arch import RRDBNet

work_dir = "{self.remote_work_dir}"
input_video = f"{{work_dir}}/{input_name}"
output_video = f"{{work_dir}}/{output_name}"

start_time = time.time()
frame_ext = "{self.frame_format}"

# Create directories
os.makedirs(f"{{work_dir}}/frames", exist_ok=True)
os.makedirs(f"{{work_dir}}/upscaled", exist_ok=True)

# Get FPS
fps_cmd = subprocess.run([
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=r_frame_rate",
    "-of", "default=noprint_wrappers=1:nokey=1", input_video
], capture_output=True, text=True)
fps_str = fps_cmd.stdout.strip()
fps = eval(fps_str) if "/" in fps_str else float(fps_str or 30)
print(f"FPS: {{fps}}")

# Extract frames
print("Extracting frames...")
extract_start = time.time()
extract_cmd = ["ffmpeg", "-y", "-i", input_video]
if frame_ext == "jpg":
    extract_cmd.extend(["-q:v", "2"])
extract_cmd.append(f"{{work_dir}}/frames/f%06d.{{frame_ext}}")
subprocess.run(extract_cmd, check=True, capture_output=True)
frames = sorted(glob.glob(f"{{work_dir}}/frames/*.{{frame_ext}}"))
print(f"Frames: {{len(frames)}} ({{time.time()-extract_start:.1f}}s)")

if not frames:
    print("ERROR: No frames extracted")
    exit(1)

# Check resolution
test_img = cv2.imread(frames[0])
print(f"Resolution: {{test_img.shape[1]}}x{{test_img.shape[0]}}")

# Download model
model_path = "/content/esrgan_model.pth"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve("{config['url']}", model_path)

# Initialize Real-ESRGAN
print("Initializing Real-ESRGAN ({config['arch']})...")
model = {config['init']}
upsampler = RealESRGANer(
    scale={config['scale']},
    model_path=model_path,
    model=model,
    tile={self.tile_size},
    tile_pad=10,
    pre_pad=0,
    half=True,
    device="cuda"
)

# Upscale frames with progress
print("=" * 50)
print("UPSCALING FRAMES")
print("=" * 50)
upscale_start = time.time()
frame_times = []

for i, frame_path in enumerate(frames):
    frame_start = time.time()
    img = cv2.imread(frame_path)
    out, _ = upsampler.enhance(img, outscale={self.scale})
    out_path = frame_path.replace("/frames/", "/upscaled/")
    cv2.imwrite(out_path, out)
    frame_time = time.time() - frame_start
    frame_times.append(frame_time)

    if (i + 1) % 10 == 0 or i == 0:
        avg = sum(frame_times[-10:]) / len(frame_times[-10:])
        eta = (len(frames) - i - 1) * avg
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  {{i+1}}/{{len(frames)}} | {{frame_time:.2f}}s | ETA={{eta:.0f}}s | GPU={{gpu_mem:.0f}}MB")

total_time = time.time() - upscale_start
print(f"Upscaling: {{len(frames)}} frames in {{total_time:.1f}}s ({{sum(frame_times)/len(frame_times):.2f}}s/frame)")

# Reassemble video
print("Encoding output...")
encode_start = time.time()
subprocess.run([
    "ffmpeg", "-y", "-framerate", str(fps),
    "-i", f"{{work_dir}}/upscaled/f%06d.{{frame_ext}}",
    "-c:v", "libx264", "-preset", "fast", "-crf", "{self.crf}", "-pix_fmt", "yuv420p",
    f"{{work_dir}}/temp_output.mp4"
], check=True, capture_output=True)
print(f"Encoding: {{time.time()-encode_start:.1f}}s")

# Add audio back
subprocess.run([
    "ffmpeg", "-y",
    "-i", f"{{work_dir}}/temp_output.mp4",
    "-i", input_video,
    "-c:v", "copy", "-c:a", "aac",
    "-map", "0:v:0", "-map", "1:a:0?", "-shortest",
    output_video
], capture_output=True)

# Fallback if audio muxing failed
if not os.path.exists(output_video):
    os.rename(f"{{work_dir}}/temp_output.mp4", output_video)

# Summary
elapsed = time.time() - start_time
size_mb = os.path.getsize(output_video) / 1024 / 1024
print("=" * 50)
print(f"COMPLETE: {{len(frames)}} frames in {{elapsed:.1f}}s")
print(f"Output: {{size_mb:.1f}} MB")
print(f"Speed: {{len(frames)/elapsed:.1f}} fps")
print("=" * 50)
print("UPSCALE_SUCCESS")
'''
        # Write and upload script
        script_path = f"{self.remote_work_dir}/upscale_video.py"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            local_script = f.name

        try:
            if not copy_to_remote(local_script, script_path, timeout=30):
                self._error = "Failed to upload processing script"
                return False

            logger.info(f"Upscaling video ({self.scale}x, {self.model})...")
            success, stdout, stderr = run_cgpu_command(
                f"python3 {script_path}",
                timeout=self.timeout
            )

            # Show progress output
            if stdout:
                for line in stdout.split('\n'):
                    if line.strip() and not line.startswith('Authenticated'):
                        if any(kw in line for kw in ['/', 'COMPLETE', 'FPS', 'frames', 'Upscaling', 'Resolution']):
                            logger.info(f"[GPU] {line}")

        finally:
            os.unlink(local_script)

        if not success or "UPSCALE_SUCCESS" not in stdout:
            self._error = self._analyze_cuda_error(stdout, stderr)
            return False

        return True

    def _analyze_cuda_error(self, stdout: str, stderr: str) -> str:
        """Analyze CUDA errors and provide helpful diagnostics."""
        combined = (stdout or "") + (stderr or "")
        errors = []

        if "CUDA out of memory" in combined or "OutOfMemoryError" in combined:
            errors.append("CUDA OUT OF MEMORY - Video too large for GPU")
            errors.append("Try: smaller tile_size, shorter clips, or lower resolution input")

        if "No CUDA" in combined or "CUDA not available" in combined:
            errors.append("CUDA NOT AVAILABLE - GPU not detected")
            errors.append("Colab session may have lost GPU allocation")

        if "ModuleNotFoundError" in combined:
            errors.append("MISSING MODULE - Dependencies not installed")

        if "RuntimeError" in combined and "torch" in combined.lower():
            errors.append("PYTORCH ERROR - Model loading or execution failed")

        if errors:
            return " | ".join(errors)

        # Generic error with last output
        return f"Upscaling failed: {(stderr or stdout or 'Unknown error')[-500:]}"

    def download(self) -> JobResult:
        """Download upscaled file."""
        remote_output = f"{self.remote_work_dir}/{self.output_path.name}"

        self.log_download_start("upscaled file")

        if download_via_base64(remote_output, str(self.output_path)):
            self.log_output_size(self.output_path)

            return JobResult(
                success=True,
                output_path=str(self.output_path),
                metadata={
                    "scale": self.scale,
                    "model": self.model,
                    "is_video": self.is_video,
                    "tile_size": self.tile_size,
                    "frame_format": self.frame_format,
                    "crf": self.crf,
                }
            )
        else:
            return JobResult(
                success=False,
                error="Failed to download upscaled file"
            )


def reset_session_cache():
    """Reset the session cache (useful for testing or after errors)."""
    global _session_env_ready
    _session_env_ready = False
