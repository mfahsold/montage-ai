"""
cgpu Cloud GPU Upscaler - Real-ESRGAN via Google Colab T4/A100

Uses cgpu (github.com/RohanAdwankar/cgpu) to run Real-ESRGAN on free cloud GPUs.
This offloads computationally expensive AI upscaling to Google Colab's CUDA GPUs.

Architecture:
  1. Extract frames locally (FFmpeg)
  2. Upload frames to Colab (cgpu copy)  
  3. Run Real-ESRGAN on T4/A100 GPU (cgpu run)
  4. Download upscaled frames (base64 over stdout)
  5. Reassemble video locally (FFmpeg)

Requirements:
  - cgpu installed: npm i -g cgpu
  - First-time setup: cgpu connect (interactive wizard)
  - CGPU_GPU_ENABLED=true environment variable

Usage:
    from montage_ai.cgpu_upscaler import upscale_with_cgpu, is_cgpu_available
    
    if is_cgpu_available():
        output_path = upscale_with_cgpu("input.mp4", "output.mp4", scale=2)

Version: 2.0.0
"""

import os
import subprocess
import shutil
import tempfile
import time
import base64
from typing import Optional, Tuple
from pathlib import Path

VERSION = "2.0.0"

# Configuration
CGPU_GPU_ENABLED = os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true"
CGPU_TIMEOUT = int(os.environ.get("CGPU_TIMEOUT", "600"))  # 10 minutes default

# Remote working directory on Colab
REMOTE_WORK_DIR = "/content/upscale_work"


def is_cgpu_available() -> bool:
    """
    Check if cgpu is installed and configured.
    
    Returns:
        True if cgpu is available for cloud GPU tasks
    """
    if not CGPU_GPU_ENABLED:
        return False
    
    try:
        result = subprocess.run(
            ["cgpu", "status"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0 and "Authenticated" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_cgpu_gpu() -> Tuple[bool, str]:
    """
    Check if cgpu can access a GPU.
    
    Returns:
        (success, gpu_info) tuple
    """
    try:
        result = subprocess.run(
            ["cgpu", "run", "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            # Extract just the GPU info line
            lines = [l for l in result.stdout.strip().split('\n') if 'Tesla' in l or 'T4' in l or 'A100' in l or 'MiB' in l]
            return True, lines[0] if lines else result.stdout.strip()
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def _run_cgpu_command(cmd: str, timeout: int = CGPU_TIMEOUT) -> Tuple[bool, str, str]:
    """
    Run a command on Colab via cgpu.
    
    Returns:
        (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ["cgpu", "run", cmd],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def _cgpu_copy_to_remote(local_path: str, remote_path: str) -> bool:
    """
    Copy file from local to Colab using cgpu copy.
    """
    try:
        result = subprocess.run(
            ["cgpu", "copy", local_path, remote_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 min for uploads
        )
        return result.returncode == 0
    except Exception as e:
        print(f"   ❌ Copy to remote failed: {e}")
        return False


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
    
    Workflow:
    1. Extract frames from input video locally
    2. Upload frames tarball to Colab via cgpu copy
    3. Install and run Real-ESRGAN on cloud GPU
    4. Download upscaled frames via base64 stdout
    5. Reassemble video with original audio
    
    Args:
        input_path: Path to input video
        output_path: Path for output video  
        scale: Upscale factor (2 or 4)
        model: Real-ESRGAN model name
        
    Returns:
        output_path on success, None on failure
    """
    print(f"   ☁️ Upscaling via cgpu cloud GPU (CUDA)...")
    
    if not is_cgpu_available():
        print(f"   ❌ cgpu not available (set CGPU_GPU_ENABLED=true)")
        return None
    
    # Create local temp directory
    work_dir = tempfile.mkdtemp(prefix="cgpu_upscale_")
    frames_dir = os.path.join(work_dir, "frames")
    upscaled_dir = os.path.join(work_dir, "upscaled")
    os.makedirs(frames_dir)
    os.makedirs(upscaled_dir)
    
    try:
        # 1. Get video info
        fps = _get_video_fps(input_path)
        print(f"   📹 Input: {os.path.basename(input_path)} @ {fps:.2f} FPS")
        
        # 2. Extract audio (to preserve it)
        audio_path = os.path.join(work_dir, "audio.aac")
        has_audio = _get_video_audio(input_path, audio_path)
        
        # 3. Extract frames locally
        print(f"   📸 Extracting frames...")
        extract_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-q:v", "2",
            os.path.join(frames_dir, "frame_%06d.png")
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        frame_count = len(frame_files)
        print(f"   📸 Extracted {frame_count} frames")
        
        if frame_count == 0:
            print(f"   ❌ No frames extracted")
            return None
        
        # 4. Create tarball of frames
        print(f"   📦 Compressing frames...")
        tar_path = os.path.join(work_dir, "frames.tar.gz")
        subprocess.run(
            ["tar", "-czf", tar_path, "-C", frames_dir, "."],
            check=True, capture_output=True
        )
        tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        print(f"   📦 Tarball: {tar_size_mb:.1f} MB")
        
        # 5. Setup remote environment
        print(f"   🔧 Setting up Colab environment...")
        # Note: torch/torchvision are pre-installed on Colab with CUDA support
        setup_cmd = f"""
mkdir -p {REMOTE_WORK_DIR}/frames {REMOTE_WORK_DIR}/upscaled && \
pip install -q realesrgan basicsr opencv-python-headless 2>/dev/null && \
python -c "import torch; print(f'PyTorch CUDA: {{torch.cuda.is_available()}}')"
"""
        setup_success, stdout, stderr = _run_cgpu_command(setup_cmd, timeout=180)
        
        if not setup_success:
            print(f"   ❌ Remote setup failed: {stderr}")
            return None
        print(f"   ✅ Environment ready")
        
        # 6. Upload frames
        print(f"   ⬆️ Uploading {frame_count} frames to Colab...")
        
        if not _cgpu_copy_to_remote(tar_path, f"{REMOTE_WORK_DIR}/frames.tar.gz"):
            print(f"   ❌ Failed to upload frames")
            return None
        
        # Extract on remote
        extract_success, _, stderr = _run_cgpu_command(
            f"cd {REMOTE_WORK_DIR}/frames && tar -xzf ../frames.tar.gz && ls | wc -l",
            timeout=60
        )
        if not extract_success:
            print(f"   ❌ Remote extraction failed: {stderr}")
            return None
        
        print(f"   ✅ Frames uploaded")
        
        # 7. Run Real-ESRGAN on cloud GPU
        print(f"   🚀 Running Real-ESRGAN on Tesla T4 (scale={scale}x)...")
        
        # Determine model weights URL and architecture
        # Real-ESRGAN has x4plus model - we always use 4x model and adjust final scale
        if "animevideov3" in model or "anime" in model:
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
            num_block = 6
            model_scale = 4
        else:
            # x4plus is the standard model, works for both 2x and 4x output
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            num_block = 23
            model_scale = 4  # Model is always 4x, outscale parameter handles 2x
        
        # Python script with torchvision compatibility patch
        # (Colab's torchvision removed functional_tensor module)
        esrgan_script = f'''
import sys

# Monkey-patch for torchvision compatibility (functional_tensor removed in newer versions)
class _FakeFunctionalTensor:
    @staticmethod
    def rgb_to_grayscale(img, num_output_channels=1):
        import torchvision.transforms.functional as F
        return F.rgb_to_grayscale(img, num_output_channels)
sys.modules["torchvision.transforms.functional_tensor"] = _FakeFunctionalTensor()

import os, glob, cv2, torch, urllib.request
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

print(f"CUDA: {{torch.cuda.is_available()}}, GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}")

# Download model weights if needed
model_path = "/content/esrgan_model.pth"
if not os.path.exists(model_path):
    print("Downloading model weights...")
    urllib.request.urlretrieve("{model_url}", model_path)

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block={num_block}, num_grow_ch=32, scale={model_scale})
upsampler = RealESRGANer(scale={model_scale}, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=True, device="cuda")

frames = sorted(glob.glob("{REMOTE_WORK_DIR}/frames/*.png"))
print(f"Processing {{len(frames)}} frames...")

for i, f in enumerate(frames):
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    out, _ = upsampler.enhance(img, outscale={scale})
    cv2.imwrite(f.replace("/frames/", "/upscaled/"), out)
    if (i+1) % 20 == 0 or i == 0:
        print(f"  {{i+1}}/{{len(frames)}}")

print("UPSCALE_DONE")
'''
        
        start_time = time.time()
        run_success, stdout, stderr = _run_cgpu_command(
            f"python -c '{esrgan_script}'",
            timeout=CGPU_TIMEOUT
        )
        
        elapsed = time.time() - start_time
        
        if not run_success or "UPSCALE_DONE" not in stdout:
            print(f"   ❌ Upscaling failed after {elapsed:.0f}s")
            print(f"   stderr: {stderr[:500]}")
            print(f"   stdout: {stdout[:500]}")
            return None
        
        print(f"   ✅ Upscaling complete ({elapsed:.0f}s)")
        
        # 8. Download upscaled frames via base64
        print(f"   ⬇️ Downloading upscaled frames...")
        
        # Create tarball on remote
        tar_success, _, _ = _run_cgpu_command(
            f"cd {REMOTE_WORK_DIR}/upscaled && tar -czf ../upscaled.tar.gz . && ls -la ../upscaled.tar.gz",
            timeout=120
        )
        
        # Download via base64 (cgpu copy is upload-only)
        dl_success, b64_data, stderr = _run_cgpu_command(
            f"base64 {REMOTE_WORK_DIR}/upscaled.tar.gz",
            timeout=300
        )
        
        if not dl_success:
            print(f"   ❌ Download failed: {stderr}")
            return None
        
        # Filter out non-base64 lines (cgpu adds auth messages)
        b64_lines = [l for l in b64_data.split('\n') if l and not l.startswith('Authenticated')]
        b64_clean = ''.join(b64_lines)
        
        # Decode and extract
        try:
            tar_data = base64.b64decode(b64_clean)
            dl_tar_path = os.path.join(work_dir, "upscaled.tar.gz")
            with open(dl_tar_path, 'wb') as f:
                f.write(tar_data)
            
            subprocess.run(
                ["tar", "-xzf", dl_tar_path, "-C", upscaled_dir],
                check=True, capture_output=True
            )
        except Exception as e:
            print(f"   ❌ Failed to decode/extract: {e}")
            return None
        
        upscaled_count = len([f for f in os.listdir(upscaled_dir) if f.endswith('.png')])
        print(f"   ✅ Downloaded {upscaled_count} upscaled frames")
        
        if upscaled_count == 0:
            print(f"   ❌ No upscaled frames received")
            return None
        
        # 9. Reassemble video
        print(f"   🎬 Reassembling video...")
        
        encode_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(upscaled_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ]
        
        # Add audio if present
        if has_audio and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            encode_cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])
        
        encode_cmd.append(output_path)
        
        subprocess.run(encode_cmd, check=True, capture_output=True)
        
        final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ Cloud GPU upscaling complete: {output_path} ({final_size_mb:.1f} MB)")
        return output_path
        
    except subprocess.TimeoutExpired:
        print(f"   ⏱️ cgpu operation timed out after {CGPU_TIMEOUT}s")
        return None
    except Exception as e:
        print(f"   ❌ cgpu upscaling error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup local
        shutil.rmtree(work_dir, ignore_errors=True)
        # Cleanup remote (best effort)
        try:
            _run_cgpu_command(f"rm -rf {REMOTE_WORK_DIR}", timeout=30)
        except:
            pass


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
