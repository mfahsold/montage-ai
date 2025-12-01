"""
cgpu Cloud GPU Upscaler - Real-ESRGAN via Google Colab

Uses cgpu (github.com/RohanAdwankar/cgpu) to run Real-ESRGAN on free cloud GPUs.
This offloads the computationally expensive AI upscaling to Google Colab.

Architecture:
  Local Video → Upload to Colab → Real-ESRGAN on T4/A100 → Download Result

Requirements:
  - cgpu installed: npm i -g cgpu
  - First-time setup: cgpu connect (interactive wizard)

Usage:
    from montage_ai.cgpu_upscaler import upscale_with_cgpu, is_cgpu_available
    
    if is_cgpu_available():
        output_path = upscale_with_cgpu(input_path, output_path, scale=2)

Version: 1.0.0
"""

import os
import subprocess
import shutil
import tempfile
from typing import Optional, Tuple
from pathlib import Path

VERSION = "1.0.0"

# Configuration
CGPU_GPU_ENABLED = os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true"
CGPU_TIMEOUT = int(os.environ.get("CGPU_TIMEOUT", "600"))  # 10 minutes default


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
            ["cgpu", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
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
            timeout=60
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def upscale_with_cgpu(
    input_path: str,
    output_path: str,
    scale: int = 2,
    model: str = "realesr-animevideov3"
) -> Optional[str]:
    """
    Upscale video using Real-ESRGAN on cgpu cloud GPU.
    
    This function:
    1. Extracts frames from input video
    2. Uploads frames to Colab via cgpu
    3. Runs Real-ESRGAN on cloud GPU
    4. Downloads upscaled frames
    5. Reassembles video
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        scale: Upscale factor (2 or 4)
        model: Real-ESRGAN model name
        
    Returns:
        output_path on success, None on failure
    """
    print(f"   ☁️ Upscaling via cgpu cloud GPU...")
    
    if not is_cgpu_available():
        print(f"   ❌ cgpu not available (CGPU_GPU_ENABLED={CGPU_GPU_ENABLED})")
        return None
    
    # Create temp directories
    work_dir = tempfile.mkdtemp(prefix="cgpu_upscale_")
    frames_dir = os.path.join(work_dir, "frames")
    upscaled_dir = os.path.join(work_dir, "upscaled")
    os.makedirs(frames_dir)
    os.makedirs(upscaled_dir)
    
    try:
        # 1. Get video info (fps)
        fps = _get_video_fps(input_path)
        print(f"   📹 Input FPS: {fps}")
        
        # 2. Extract frames
        print(f"   📸 Extracting frames...")
        extract_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-q:v", "2",
            os.path.join(frames_dir, "frame_%08d.png")
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        print(f"   📸 Extracted {frame_count} frames")
        
        # 3. Create upscaling script for cgpu
        upscale_script = _create_upscale_script(frames_dir, upscaled_dir, scale, model)
        script_path = os.path.join(work_dir, "upscale.py")
        with open(script_path, 'w') as f:
            f.write(upscale_script)
        
        # 4. Run upscaling on cloud GPU via cgpu
        print(f"   🚀 Running Real-ESRGAN on cloud GPU (this may take a while)...")
        
        # Upload frames and run script
        cgpu_cmd = [
            "cgpu", "run",
            f"cd {work_dir} && pip install -q realesrgan && python upscale.py"
        ]
        
        result = subprocess.run(
            cgpu_cmd,
            capture_output=True,
            text=True,
            timeout=CGPU_TIMEOUT
        )
        
        if result.returncode != 0:
            print(f"   ❌ cgpu upscaling failed: {result.stderr}")
            return None
        
        # 5. Check upscaled frames exist
        upscaled_count = len([f for f in os.listdir(upscaled_dir) if f.endswith('.png')])
        if upscaled_count == 0:
            print(f"   ❌ No upscaled frames produced")
            return None
        
        print(f"   ✅ Upscaled {upscaled_count} frames")
        
        # 6. Reassemble video
        print(f"   🎬 Reassembling video...")
        reassemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(upscaled_dir, "frame_%08d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path
        ]
        subprocess.run(reassemble_cmd, check=True, capture_output=True)
        
        print(f"   ✅ Cloud GPU upscaling complete: {output_path}")
        return output_path
        
    except subprocess.TimeoutExpired:
        print(f"   ⏱️ cgpu upscaling timed out after {CGPU_TIMEOUT}s")
        return None
    except Exception as e:
        print(f"   ❌ cgpu upscaling error: {e}")
        return None
    finally:
        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)


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


def _create_upscale_script(frames_dir: str, output_dir: str, scale: int, model: str) -> str:
    """
    Create Python script to run on cloud GPU.
    
    This script installs Real-ESRGAN and processes all frames.
    """
    return f'''
"""Real-ESRGAN upscaling script for cgpu cloud GPU."""
import os
import glob
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2
import torch

# Setup model
model_name = "{model}"
scale = {scale}

# Model configurations
if "animevideov3" in model_name:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
    netscale = scale
else:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    netscale = scale

# Initialize upsampler
upsampler = RealESRGANer(
    scale=netscale,
    model_path=None,  # Will download automatically
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True  # Use FP16 for faster processing
)

# Process frames
frames_dir = "{frames_dir}"
output_dir = "{output_dir}"

frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
print(f"Processing {{len(frame_files)}} frames...")

for i, frame_path in enumerate(frame_files):
    img = cv2.imread(frame_path)
    output, _ = upsampler.enhance(img, outscale=scale)
    
    # Save with same filename
    output_path = os.path.join(output_dir, os.path.basename(frame_path))
    cv2.imwrite(output_path, output)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {{i + 1}}/{{len(frame_files)}} frames")

print("Upscaling complete!")
'''


# Alternative: Direct cgpu run command approach (simpler but less flexible)
def upscale_with_cgpu_simple(
    input_path: str,
    output_path: str,
    scale: int = 2
) -> Optional[str]:
    """
    Simple cgpu upscaling using pre-installed Real-ESRGAN on Colab.
    
    This uses the Colab's pre-configured environment directly.
    """
    print(f"   ☁️ Simple cgpu upscale (scale={scale}x)...")
    
    if not is_cgpu_available():
        return None
    
    work_dir = tempfile.mkdtemp(prefix="cgpu_simple_")
    
    try:
        # Copy input to work dir
        input_name = os.path.basename(input_path)
        work_input = os.path.join(work_dir, input_name)
        shutil.copy2(input_path, work_input)
        
        output_name = f"upscaled_{input_name}"
        work_output = os.path.join(work_dir, output_name)
        
        # Run Real-ESRGAN directly via cgpu
        # Note: Assumes realesrgan-ncnn-vulkan or python realesrgan is available
        cmd = [
            "cgpu", "run",
            f"cd {work_dir} && "
            f"pip install -q realesrgan && "
            f"python -m realesrgan.inference_realesrgan -i {input_name} -o {output_name} -s {scale}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=CGPU_TIMEOUT)
        
        if result.returncode == 0 and os.path.exists(work_output):
            shutil.copy2(work_output, output_path)
            return output_path
        
        print(f"   ❌ Simple upscale failed: {result.stderr}")
        return None
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    # Test cgpu availability
    print(f"cgpu Upscaler v{VERSION}")
    print(f"CGPU_GPU_ENABLED: {CGPU_GPU_ENABLED}")
    print(f"cgpu available: {is_cgpu_available()}")
    
    if is_cgpu_available():
        success, gpu_info = check_cgpu_gpu()
        if success:
            print(f"Cloud GPU: {gpu_info}")
        else:
            print(f"GPU check failed: {gpu_info}")
