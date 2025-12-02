"""
cgpu Cloud GPU Upscaler - Real-ESRGAN via Google Colab T4/A100

Uses cgpu (github.com/RohanAdwankar/cgpu) to run Real-ESRGAN on free cloud GPUs.
This offloads computationally expensive AI upscaling to Google Colab's CUDA GPUs.

Architecture (optimized v2.2):
  1. Upload video directly to Colab (cgpu copy) - skip frame extraction locally
  2. Extract + upscale + reassemble ALL on Colab GPU
  3. Download result video via base64 chunks
  
This approach:
  - Reduces upload size (video is ~10x smaller than PNG frames)
  - Leverages T4 GPU for frame extraction too
  - Single upload/download per video
  - Reuses Colab environment across clips (session caching)

Requirements:
  - cgpu installed: npm i -g cgpu
  - First-time setup: cgpu connect (interactive wizard)
  - CGPU_GPU_ENABLED=true environment variable

Usage:
    from montage_ai.cgpu_upscaler import upscale_with_cgpu, is_cgpu_available
    
    if is_cgpu_available():
        output_path = upscale_with_cgpu("input.mp4", "output.mp4", scale=2)

Version: 2.2.0 - Direct video upload, Colab-side processing
"""

import os
import subprocess
import shutil
import tempfile
import time
import base64
from typing import Optional, Tuple
from pathlib import Path

# Import shared cgpu utilities
from .cgpu_utils import (
    CGPUConfig,
    is_cgpu_available,
    check_cgpu_gpu,
    run_cgpu_command,
    cgpu_copy_to_remote,
    cgpu_download_base64,
)

VERSION = "2.2.0"

# Configuration from shared utils
_cgpu_config = CGPUConfig()
CGPU_GPU_ENABLED = _cgpu_config.gpu_enabled
CGPU_TIMEOUT = _cgpu_config.timeout

# Remote working directory on Colab
REMOTE_WORK_DIR = "/content/upscale_work"

# Session state - track if Colab environment is already set up
_colab_env_ready = False


# Internal wrappers for backward compatibility
def _run_cgpu_command(cmd: str, timeout: int = None) -> Tuple[bool, str, str]:
    """Wrapper for shared run_cgpu_command."""
    if timeout is None:
        timeout = CGPU_TIMEOUT
    return run_cgpu_command(cmd, timeout)


def _cgpu_copy_to_remote(local_path: str, remote_path: str) -> bool:
    """Wrapper for shared cgpu_copy_to_remote."""
    return cgpu_copy_to_remote(local_path, remote_path)


def _get_video_fps(video_path: str) -> float:
    """Get video framerate using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        return num / den
    return float(fps_str) if fps_str else 30.0


def _get_video_audio(video_path: str, output_path: str) -> bool:
    """Extract audio track from video."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "copy",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def upscale_with_cgpu(
    input_path: str,
    output_path: str,
    scale: int = 2,
    model: str = "realesr-animevideov3"
) -> Optional[str]:
    """
    Upscale video using Real-ESRGAN on cgpu cloud GPU (Tesla T4/A100).
    
    Optimized workflow (v2.2):
    1. Upload video directly to Colab (much smaller than frames)
    2. Run entire pipeline on Colab: extract → upscale → reassemble
    3. Download result video via chunked base64
    
    Args:
        input_path: Path to input video
        output_path: Path for output video  
        scale: Upscale factor (2 or 4)
        model: Real-ESRGAN model name
        
    Returns:
        output_path on success, None on failure
    """
    global _colab_env_ready
    
    print(f"   ☁️ Upscaling via cgpu cloud GPU (CUDA)...")
    
    if not is_cgpu_available():
        print(f"   ❌ cgpu not available (set CGPU_GPU_ENABLED=true)")
        return None
    
    input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"   📹 Input: {os.path.basename(input_path)} ({input_size_mb:.1f} MB)")
    
    try:
        # 1. Setup remote environment (once per session)
        # Check if environment is ready by testing if work directory exists
        need_setup = not _colab_env_ready
        if _colab_env_ready:
            # Verify directory still exists (session may have restarted)
            check_success, check_out, _ = _run_cgpu_command(
                f"test -d {REMOTE_WORK_DIR} && echo EXISTS || echo MISSING",
                timeout=30
            )
            if "MISSING" in check_out or not check_success:
                print(f"   ⚠️ Colab session restarted, re-initializing environment...")
                need_setup = True
                
        if need_setup:
            print(f"   🔧 Setting up Colab environment (first clip)...")
            # Install basicsr with torchvision compatibility patch
            # The patch creates a fake module that redirects the deprecated import
            setup_cmd = f"""
mkdir -p {REMOTE_WORK_DIR} && \
pip install -q opencv-python-headless 2>/dev/null && \
pip install -q realesrgan basicsr 2>/dev/null || true && \
python3 << 'PATCH_SCRIPT'
# Create torchvision compatibility patch
import os
patch_code = '''
# Compatibility shim for torchvision.transforms.functional_tensor
# This module was removed in torchvision 0.18+
from torchvision.transforms import functional as F

def rgb_to_grayscale(img, num_output_channels=1):
    return F.rgb_to_grayscale(img, num_output_channels)
'''
# Find site-packages and create the patch
import site
for sp in site.getsitepackages():
    tv_path = os.path.join(sp, "torchvision", "transforms")
    if os.path.exists(tv_path):
        patch_file = os.path.join(tv_path, "functional_tensor.py")
        with open(patch_file, "w") as f:
            f.write(patch_code)
        print(f"PATCHED: {{patch_file}}")
        break
else:
    print("WARN: Could not find torchvision to patch")
PATCH_SCRIPT
python -c "import torch; print(f'CUDA: {{torch.cuda.is_available()}}, GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}}')" && \
echo "ENV_READY"
"""
            success, stdout, stderr = _run_cgpu_command(setup_cmd, timeout=180)
            if not success or "ENV_READY" not in stdout:
                print(f"   ❌ Environment setup failed")
                return None
            _colab_env_ready = True
            print(f"   ✅ Environment ready (will be reused)")
        else:
            print(f"   ♻️ Reusing Colab environment")
        
        # 2. Upload video directly (much smaller than PNG frames!)
        remote_video = f"{REMOTE_WORK_DIR}/input.mp4"
        print(f"   ⬆️ Uploading video ({input_size_mb:.1f} MB)...")

        # Dynamic timeout: 1 min per 10MB, minimum 10 min
        upload_timeout = max(600, int(input_size_mb / 10 * 60))

        if not cgpu_copy_to_remote(input_path, remote_video, timeout=upload_timeout):
            print(f"   ❌ Upload failed")
            print(f"   💡 Troubleshooting:")
            print(f"      1. Check cgpu connection: cgpu status")
            print(f"      2. File size: {input_size_mb:.1f}MB (may need longer timeout)")
            print(f"      3. Try manual upload: cgpu copy {input_path} {remote_video}")

            # Fallback: Base64 upload for smaller files only
            if input_size_mb < 10:
                print(f"   → Trying base64 fallback for small file...")
                return _upscale_via_base64_upload(input_path, output_path, scale, model)
            else:
                print(f"   → File too large ({input_size_mb:.1f}MB) for base64 fallback")
                return None

        print(f"   ✅ Upload complete ({upload_timeout}s timeout used)")
        
        # 3. Run full pipeline on Colab (extract → upscale → reassemble)
        print(f"   🚀 Processing on Tesla T4 (scale={scale}x)...")

        # Build Python script for Colab - model architecture depends on model type
        # animevideov3 uses SRVGGNetCompact, others use RRDBNet
        if "animevideov3" in model or "anime" in model:
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
            model_arch = "SRVGGNetCompact"
            model_init = "SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')"
            model_scale = 4
        else:
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            model_arch = "RRDBNet"
            model_init = "RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)"
            model_scale = 4

        # === IMPROVED: Upload script as file instead of inline string ===
        # This avoids quote-escaping hell and makes debugging easier
        # Full pipeline script - runs entirely on Colab GPU with detailed CUDA logging
        # Note: torchvision.transforms.functional_tensor is patched during env setup
        pipeline_script = f'''
import os, glob, subprocess, urllib.request, time
import torch, cv2
from realesrgan import RealESRGANer
# Import the correct architecture
if "{model_arch}" == "SRVGGNetCompact":
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
else:
    from basicsr.archs.rrdbnet_arch import RRDBNet

# ============ CUDA DIAGNOSTICS ============
print("=" * 60)
print("CUDA DIAGNOSTICS")
print("=" * 60)
print(f"PyTorch version: {{torch.__version__}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")
print(f"CUDA version: {{torch.version.cuda}}")
print(f"cuDNN version: {{torch.backends.cudnn.version()}}")
print(f"cuDNN enabled: {{torch.backends.cudnn.enabled}}")
print(f"GPU count: {{torch.cuda.device_count()}}")

if torch.cuda.is_available():
    gpu_id = 0
    print(f"GPU {{gpu_id}}: {{torch.cuda.get_device_name(gpu_id)}}")
    props = torch.cuda.get_device_properties(gpu_id)
    print(f"  Compute capability: {{props.major}}.{{props.minor}}")
    print(f"  Total memory: {{props.total_memory / 1024**3:.1f}} GB")
    print(f"  Multi-processor count: {{props.multi_processor_count}}")
    
    # Current memory state
    print(f"  Memory allocated: {{torch.cuda.memory_allocated(gpu_id) / 1024**2:.1f}} MB")
    print(f"  Memory reserved: {{torch.cuda.memory_reserved(gpu_id) / 1024**2:.1f}} MB")
    
    # Get nvidia-smi output for detailed GPU state
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw", 
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if smi.returncode == 0:
            vals = smi.stdout.strip().split(", ")
            print(f"  GPU Utilization: {{vals[0]}}%")
            print(f"  Memory Utilization: {{vals[1]}}%")
            print(f"  Memory Used: {{vals[2]}} MB")
            print(f"  Memory Free: {{vals[3]}} MB")
            print(f"  Temperature: {{vals[4]}}°C")
            print(f"  Power Draw: {{vals[5]}} W")
    except: pass
print("=" * 60)

start = time.time()

# Create work dirs
os.makedirs("{REMOTE_WORK_DIR}/frames", exist_ok=True)
os.makedirs("{REMOTE_WORK_DIR}/upscaled", exist_ok=True)

# Get FPS
fps_cmd = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", 
    "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
    "{remote_video}"], capture_output=True, text=True)
fps_str = fps_cmd.stdout.strip()
fps = eval(fps_str) if "/" in fps_str else float(fps_str)
print(f"FPS: {{fps}}")

# Extract frames (JPEG to save space)
print("Extracting frames...")
extract_start = time.time()
subprocess.run(["ffmpeg", "-y", "-i", "{remote_video}", "-q:v", "2", 
    "{REMOTE_WORK_DIR}/frames/f%06d.jpg"], check=True, capture_output=True)
frames = sorted(glob.glob("{REMOTE_WORK_DIR}/frames/*.jpg"))
print(f"Frames: {{len(frames)}} (extracted in {{time.time()-extract_start:.1f}}s)")

# Check frame resolution
if frames:
    test_img = cv2.imread(frames[0])
    print(f"Frame resolution: {{test_img.shape[1]}}x{{test_img.shape[0]}}")

# Download model
model_path = "/content/esrgan.pth"
if not os.path.exists(model_path):
    print("Downloading model...")
    dl_start = time.time()
    urllib.request.urlretrieve("{model_url}", model_path)
    print(f"Model downloaded in {{time.time()-dl_start:.1f}}s")
else:
    print("Model already cached")

# Init Real-ESRGAN with tiling for faster processing
print("Initializing Real-ESRGAN ({model_arch})...")
init_start = time.time()
model = {model_init}
upsampler = RealESRGANer(scale={model_scale}, model_path=model_path, model=model, tile=512, tile_pad=10, pre_pad=0, half=True, device="cuda")
print(f"Model initialized in {{time.time()-init_start:.1f}}s")

# Memory after model load
print(f"GPU Memory after model load: {{torch.cuda.memory_allocated(0) / 1024**2:.1f}} MB")

# Upscale all frames with detailed timing
print("=" * 60)
print("UPSCALING FRAMES")
print("=" * 60)
upscale_start = time.time()
frame_times = []
for i, f in enumerate(frames):
    frame_start = time.time()
    img = cv2.imread(f)
    out, _ = upsampler.enhance(img, outscale={scale})
    out_path = f.replace("/frames/", "/upscaled/")
    cv2.imwrite(out_path, out)
    frame_time = time.time() - frame_start
    frame_times.append(frame_time)
    
    if (i+1) % 10 == 0 or i == 0:
        avg_time = sum(frame_times[-10:]) / len(frame_times[-10:])
        remaining = (len(frames) - i - 1) * avg_time
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  Frame {{i+1}}/{{len(frames)}} | {{frame_time:.2f}}s | avg={{avg_time:.2f}}s | ETA={{remaining:.0f}}s | GPU={{gpu_mem:.0f}}MB")

total_upscale_time = time.time() - upscale_start
avg_frame_time = sum(frame_times) / len(frame_times)
print(f"Upscaling complete: {{len(frames)}} frames in {{total_upscale_time:.1f}}s (avg {{avg_frame_time:.2f}}s/frame)")

# Reassemble video
print("Encoding output...")
encode_start = time.time()
subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), 
    "-i", "{REMOTE_WORK_DIR}/upscaled/f%06d.jpg",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
    "{REMOTE_WORK_DIR}/output.mp4"], check=True, capture_output=True)
print(f"Encoding done in {{time.time()-encode_start:.1f}}s")

# Add audio back if exists
subprocess.run(["ffmpeg", "-y", "-i", "{REMOTE_WORK_DIR}/output.mp4", "-i", "{remote_video}",
    "-c:v", "copy", "-c:a", "aac", "-map", "0:v", "-map", "1:a?", "-shortest",
    "{REMOTE_WORK_DIR}/final.mp4"], capture_output=True)

# Check which output exists
if os.path.exists("{REMOTE_WORK_DIR}/final.mp4"):
    os.rename("{REMOTE_WORK_DIR}/final.mp4", "{REMOTE_WORK_DIR}/output.mp4")

elapsed = time.time() - start
size = os.path.getsize("{REMOTE_WORK_DIR}/output.mp4") / 1024 / 1024
print("=" * 60)
print(f"COMPLETE: {{len(frames)}} frames upscaled in {{elapsed:.1f}}s")
print(f"Output: {{size:.1f}}MB")
print(f"Performance: {{len(frames)/elapsed:.1f}} fps")
print("=" * 60)
print("PIPELINE_SUCCESS")
'''

        # === IMPROVED: Upload script as file for better reliability ===
        remote_script_path = f"{REMOTE_WORK_DIR}/upscale_pipeline.py"

        # Write script to local temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(pipeline_script)
            local_script_path = f.name

        try:
            # Upload script to Colab
            if not _cgpu_copy_to_remote(local_script_path, remote_script_path):
                print(f"   ❌ Failed to upload processing script")
                return None

            # Execute uploaded script (no quote-escaping issues!)
            start_time = time.time()
            success, stdout, stderr = _run_cgpu_command(
                f"python3 {remote_script_path}",
                timeout=CGPU_TIMEOUT
            )
            elapsed = time.time() - start_time
        finally:
            # Cleanup local temp file
            try:
                os.unlink(local_script_path)
            except:
                pass
        
        # Always show GPU diagnostics from stdout (even on success)
        if stdout:
            # Extract and display CUDA diagnostics section
            lines = stdout.split('\n')
            in_diag = False
            for line in lines:
                if 'CUDA DIAGNOSTICS' in line or 'UPSCALING FRAMES' in line or 'COMPLETE:' in line:
                    in_diag = True
                if in_diag:
                    # Filter out authentication messages
                    if not line.startswith('Authenticated'):
                        print(f"   [T4] {line}")
                if line.startswith('=' * 10) and in_diag and 'DIAGNOSTICS' not in line and 'FRAMES' not in line:
                    in_diag = False
        
        # IMPORTANT: cgpu may return exit code 1 even on success (known issue).
        # We use PIPELINE_SUCCESS marker as the primary success indicator.
        pipeline_success = "PIPELINE_SUCCESS" in stdout if stdout else False
        
        if not pipeline_success:
            print(f"   ❌ Pipeline failed after {elapsed:.0f}s")

            # === IMPROVED: Detailed CUDA error analysis ===
            cuda_errors = []

            # Check for common CUDA errors in stdout/stderr
            combined_output = (stdout or "") + (stderr or "")

            if "CUDA out of memory" in combined_output or "OutOfMemoryError" in combined_output:
                cuda_errors.append("⚠️ CUDA OUT OF MEMORY - Video too large for T4 GPU")
                cuda_errors.append("   → Try reducing video resolution or using smaller clips")
                cuda_errors.append("   → Consider chunking into multiple smaller jobs")

            if "No CUDA" in combined_output or "CUDA not available" in combined_output:
                cuda_errors.append("⚠️ CUDA NOT AVAILABLE - No GPU detected on Colab")
                cuda_errors.append("   → Colab session may have lost GPU allocation")
                cuda_errors.append("   → Try restarting cgpu connection")

            if "ModuleNotFoundError" in combined_output:
                cuda_errors.append("⚠️ MISSING PYTHON MODULE - Dependencies not installed")
                cuda_errors.append("   → Colab environment may need reinstallation")

            if "RuntimeError" in combined_output and "torch" in combined_output.lower():
                cuda_errors.append("⚠️ PYTORCH RUNTIME ERROR - Model loading or execution failed")

            if cuda_errors:
                print(f"\n   🔍 CUDA Error Diagnosis:")
                for err in cuda_errors:
                    print(f"      {err}")

            # Show relevant output sections
            if stdout:
                # Extract error-relevant sections
                lines = stdout.split('\n')
                error_lines = [l for l in lines if any(kw in l.lower() for kw in ['error', 'failed', 'exception', 'traceback'])]

                if error_lines:
                    print(f"\n   📋 Error Messages:")
                    for line in error_lines[-10:]:  # Last 10 error lines
                        print(f"      {line}")
                else:
                    print(f"\n   📋 Last output: {stdout[-600:]}")

            if stderr:
                print(f"\n   📋 stderr: {stderr[-400:]}")

            return None
        
        print(f"   ✅ GPU processing done ({elapsed:.0f}s)")
        
        # 4. Download result (chunked to avoid timeout)
        print(f"   ⬇️ Downloading upscaled video...")
        
        # Get output size first
        size_success, size_out, _ = _run_cgpu_command(
            f"stat -c%s {REMOTE_WORK_DIR}/output.mp4",
            timeout=30
        )
        if size_success:
            out_size_mb = int(size_out.strip().split()[-1]) / 1024 / 1024
            print(f"   📦 Output size: {out_size_mb:.1f} MB")
        
        # Download via base64
        dl_success, b64_data, stderr = _run_cgpu_command(
            f"base64 {REMOTE_WORK_DIR}/output.mp4",
            timeout=600  # 10 min for large files
        )
        
        if not dl_success:
            print(f"   ❌ Download failed: {stderr[:200]}")
            return None
        
        # Clean up base64 data
        b64_lines = [l for l in b64_data.split('\\n') if l and not l.startswith('Authenticated')]
        b64_clean = ''.join(b64_lines)
        
        # Decode and save
        video_data = base64.b64decode(b64_clean)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(video_data)
        
        final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ cgpu upscaling complete: {final_size_mb:.1f} MB")
        return output_path
        
    except Exception as e:
        print(f"   ❌ cgpu error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _upscale_via_base64_upload(
    input_path: str,
    output_path: str, 
    scale: int,
    model: str
) -> Optional[str]:
    """
    Fallback: Upload video via base64 encoding for smaller files.
    Used when cgpu copy fails.
    """
    print(f"   🔄 Using base64 upload fallback...")
    
    # Read and encode video
    with open(input_path, 'rb') as f:
        video_b64 = base64.b64encode(f.read()).decode()
    
    # Upload via echo (works for <20MB)
    remote_video = f"{REMOTE_WORK_DIR}/input.mp4"
    upload_cmd = f"echo '{video_b64}' | base64 -d > {remote_video}"
    
    success, _, stderr = _run_cgpu_command(upload_cmd, timeout=120)
    if not success:
        print(f"   ❌ Base64 upload failed")
        return None
    
    # Continue with same pipeline...
    # (simplified - just return None for now, full implementation would call main pipeline)
    print(f"   ⚠️ Base64 fallback not fully implemented yet")
    return None


def upscale_image_with_cgpu(
    input_path: str,
    output_path: str,
    scale: int = 4
) -> Optional[str]:
    """
    Upscale a single image using Real-ESRGAN on cgpu cloud GPU.
    
    Simpler workflow for single images:
    1. Upload image to Colab
    2. Run Real-ESRGAN
    3. Download result via base64
    
    Args:
        input_path: Path to input image (PNG/JPG)
        output_path: Path for output image
        scale: Upscale factor (2 or 4)
        
    Returns:
        output_path on success, None on failure
    """
    print(f"   ☁️ Upscaling image via cgpu cloud GPU...")
    
    if not is_cgpu_available():
        print(f"   ❌ cgpu not available")
        return None
    
    input_name = os.path.basename(input_path)
    remote_input = f"/content/{input_name}"
    remote_output = "/content/upscaled_output.png"
    
    try:
        # 1. Upload
        print(f"   ⬆️ Uploading {input_name}...")
        if not _cgpu_copy_to_remote(input_path, remote_input):
            print(f"   ❌ Upload failed")
            return None
        
        # 2. Run Real-ESRGAN with torchvision patch
        print(f"   🚀 Running Real-ESRGAN (scale={scale}x)...")
        
        # Build the script as separate lines to avoid quote escaping issues
        script_lines = [
            "import sys",
            "class _FakeFT:",
            "    @staticmethod",
            "    def rgb_to_grayscale(img, n=1):",
            "        import torchvision.transforms.functional as F",
            "        return F.rgb_to_grayscale(img, n)",
            "sys.modules['torchvision.transforms.functional_tensor'] = _FakeFT()",
            "",
            "import torch, cv2, os, urllib.request",
            "from basicsr.archs.rrdbnet_arch import RRDBNet",
            "from realesrgan import RealESRGANer",
            "",
            "model_path = '/content/RealESRGAN_x4plus.pth'",
            "if not os.path.exists(model_path):",
            "    urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', model_path)",
            "",
            f"model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale={scale})",
            f"up = RealESRGANer(scale={scale}, model_path=model_path, model=model, tile=0, half=True, device='cuda')",
            "",
            f"img = cv2.imread('{remote_input}')",
            "print('Input:', img.shape)",
            f"out, _ = up.enhance(img, outscale={scale})",
            "print('Output:', out.shape)",
            f"cv2.imwrite('{remote_output}', out)",
            "print('UPSCALE_SUCCESS')",
        ]
        
        # Write script to temp file, upload, and execute
        script_content = "\n".join(script_lines)
        
        # Create temp script file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            local_script = f.name
        
        try:
            # Upload script
            remote_script = "/content/esrgan_run.py"
            if not _cgpu_copy_to_remote(local_script, remote_script):
                print(f"   ❌ Script upload failed")
                return None
            
            # Execute script
            success, stdout, stderr = _run_cgpu_command(
                f"python3 {remote_script}",
                timeout=300
            )
        finally:
            os.unlink(local_script)
        
        if not success or "UPSCALE_SUCCESS" not in stdout:
            print(f"   ❌ Upscaling failed: {stderr[:300]}")
            print(f"   stdout: {stdout[:200]}")
            return None
        
        print(f"   ✅ Upscaling done")
        
        # 3. Download via base64
        print(f"   ⬇️ Downloading result...")
        success, b64_data, stderr = _run_cgpu_command(
            f"base64 {remote_output}",
            timeout=120
        )
        
        if not success:
            print(f"   ❌ Download failed: {stderr}")
            return None
        
        # Filter out cgpu auth message line
        b64_lines = [l for l in b64_data.split('\n') if l and not l.startswith('Authenticated')]
        b64_clean = ''.join(b64_lines)
        
        # Decode and save
        img_data = base64.b64decode(b64_clean)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(img_data)
        
        print(f"   ✅ Saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    print(f"cgpu Cloud GPU Upscaler v{VERSION}")
    print(f"CGPU_GPU_ENABLED: {CGPU_GPU_ENABLED}")
    print(f"CGPU_TIMEOUT: {CGPU_TIMEOUT}s")
    print()
    
    print("Checking cgpu availability...")
    available = is_cgpu_available()
    
    if available:
        print("✅ cgpu is available")
        
        print("\nChecking GPU...")
        success, gpu_info = check_cgpu_gpu()
        if success:
            print(f"✅ Cloud GPU: {gpu_info}")
        else:
            print(f"⚠️ GPU check: {gpu_info}")
    else:
        print("❌ cgpu not available")
        print("   Set CGPU_GPU_ENABLED=true and ensure cgpu is installed")
        sys.exit(1)
    
    # Test with file if provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "upscaled_output.mp4"
        
        # Determine scale from args or default to 4
        scale = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        
        print(f"\n{'='*50}")
        print(f"Upscaling: {input_file}")
        print(f"Output: {output_file}")
        print(f"Scale: {scale}x")
        print(f"{'='*50}\n")
        
        # Use image upscaler for single images, video upscaler otherwise
        if input_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            result = upscale_image_with_cgpu(input_file, output_file, scale=scale)
        else:
            result = upscale_with_cgpu(input_file, output_file, scale=scale)
        
        if result:
            print(f"\n✅ Success: {result}")
            sys.exit(0)
        else:
            print(f"\n❌ Upscaling failed")
            sys.exit(1)
