"""
cgpu Cloud GPU Upscaler v3 - Polling-based approach

Simplified architecture:
1. Upload video to Colab with unique job ID
2. Start processing in background (nohup)
3. Poll for completion marker file
4. Download result

This solves the "cgpu run hangs" problem by not waiting for stdout.

Version: 3.0.1
"""

import os
import subprocess
import sys
import time
import base64
import uuid
from typing import Optional, Tuple
from pathlib import Path


def _log(msg: str):
    """Print with immediate flush for real-time logging."""
    print(msg, flush=True)


# Configuration
CGPU_GPU_ENABLED = os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true"
CGPU_TIMEOUT = int(os.environ.get("CGPU_TIMEOUT", "1200"))

# Polling settings
POLL_INTERVAL = 10  # seconds
MAX_POLL_ATTEMPTS = CGPU_TIMEOUT // POLL_INTERVAL


def _run_cgpu(cmd: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """Run cgpu command with timeout. Returns (success, stdout, stderr)."""
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


def _cgpu_copy(local_path: str, remote_path: str, timeout: int = 600) -> bool:
    """Upload file to Colab."""
    try:
        result = subprocess.run(
            ["cgpu", "copy", local_path, remote_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except Exception:
        return False


def is_cgpu_available() -> bool:
    """Check if cgpu is available and authenticated."""
    if not CGPU_GPU_ENABLED:
        return False
    try:
        result = subprocess.run(
            ["cgpu", "status"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return "Authenticated" in result.stdout
    except Exception:
        return False


def upscale_with_cgpu(
    input_path: str,
    output_path: str,
    scale: int = 2,
    model: str = "realesr-animevideov3"
) -> Optional[str]:
    """
    Upscale video using Real-ESRGAN on cgpu cloud GPU.
    
    Uses polling approach to avoid hanging on cgpu run.
    """
    if not is_cgpu_available():
        print("   ❌ cgpu not available")
        return None
    
    # Unique job ID for parallel safety
    job_id = uuid.uuid4().hex[:8]
    work_dir = f"/content/upscale_{job_id}"
    
    input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    _log(f"   ☁️ cgpu upscaling [{job_id}] ({input_size_mb:.1f} MB)")
    
    try:
        # 1. Setup work directory
        _run_cgpu(f"mkdir -p {work_dir}", timeout=30)
        
        # 2. Upload video
        _log(f"   ⬆️ Uploading...")
        remote_input = f"{work_dir}/input.mp4"
        if not _cgpu_copy(input_path, remote_input, timeout=max(300, int(input_size_mb * 30))):
            _log(f"   ❌ Upload failed")
            return None
        
        # 3. Generate and upload processing script
        script = _generate_pipeline_script(work_dir, scale, model)
        script_path = f"/tmp/cgpu_script_{job_id}.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        remote_script = f"{work_dir}/pipeline.py"
        if not _cgpu_copy(script_path, remote_script, timeout=60):
            os.unlink(script_path)
            _log(f"   ❌ Script upload failed")
            return None
        os.unlink(script_path)
        
        # 4. Start processing in background (nohup)
        _log(f"   🚀 Processing on GPU...")
        bg_cmd = f"cd {work_dir} && nohup python3 pipeline.py > log.txt 2>&1 &"
        _run_cgpu(bg_cmd, timeout=30)
        
        # 5. Poll for completion
        success = _poll_for_completion(work_dir, job_id)
        
        if not success:
            # Show error log
            _, log, _ = _run_cgpu(f"cat {work_dir}/log.txt 2>/dev/null | tail -30", timeout=30)
            _log(f"   ❌ Processing failed. Log:\n{log[-500:]}")
            return None
        
        # 6. Download result
        _log(f"   ⬇️ Downloading...")
        if not _download_result(f"{work_dir}/output.mp4", output_path):
            return None
        
        # 7. Cleanup
        _run_cgpu(f"rm -rf {work_dir}", timeout=30)
        
        final_size = os.path.getsize(output_path) / (1024 * 1024)
        _log(f"   ✅ Done: {final_size:.1f} MB")
        return output_path
        
    except Exception as e:
        _log(f"   ❌ Error: {e}")
        return None


def _generate_pipeline_script(work_dir: str, scale: int, model: str) -> str:
    """Generate the Colab processing script."""
    
    # Model configuration
    if "animevideov3" in model:
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        model_init = "SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')"
        model_import = "from realesrgan.archs.srvgg_arch import SRVGGNetCompact"
        model_scale = 4
    else:
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_init = "RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)"
        model_import = "from basicsr.archs.rrdbnet_arch import RRDBNet"
        model_scale = 4
    
    return f'''#!/usr/bin/env python3
"""Auto-generated upscaling pipeline for cgpu."""
import os, sys, glob, subprocess, urllib.request, time
WORK_DIR = "{work_dir}"
SCALE = {scale}
MODEL_SCALE = {model_scale}

# Write status marker
def mark(status, msg=""):
    with open(f"{{WORK_DIR}}/status.txt", "w") as f:
        f.write(f"{{status}}\\n{{msg}}")

mark("STARTING")

try:
    # Install dependencies (quiet)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless", "realesrgan", "basicsr"], 
                   capture_output=True)
    
    # Torchvision compatibility patch
    import site
    for sp in site.getsitepackages():
        tv_path = os.path.join(sp, "torchvision", "transforms")
        if os.path.exists(tv_path):
            patch = os.path.join(tv_path, "functional_tensor.py")
            with open(patch, "w") as f:
                f.write("from torchvision.transforms import functional as F\\n")
                f.write("def rgb_to_grayscale(img, n=1): return F.rgb_to_grayscale(img, n)\\n")
            break
    
    import torch, cv2
    from realesrgan import RealESRGANer
    {model_import}
    
    mark("SETUP_DONE", f"CUDA={{torch.cuda.is_available()}}")
    
    # Extract frames
    os.makedirs(f"{{WORK_DIR}}/frames", exist_ok=True)
    os.makedirs(f"{{WORK_DIR}}/upscaled", exist_ok=True)
    
    # Get FPS
    fps_cmd = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
        f"{{WORK_DIR}}/input.mp4"], capture_output=True, text=True)
    fps_str = fps_cmd.stdout.strip()
    fps = eval(fps_str) if "/" in fps_str else float(fps_str or "30")
    
    subprocess.run(["ffmpeg", "-y", "-i", f"{{WORK_DIR}}/input.mp4", "-q:v", "2",
        f"{{WORK_DIR}}/frames/f%06d.jpg"], capture_output=True, check=True)
    
    frames = sorted(glob.glob(f"{{WORK_DIR}}/frames/*.jpg"))
    mark("FRAMES_EXTRACTED", f"{{len(frames)}} frames @ {{fps}} fps")
    
    # Download model
    model_path = "/content/esrgan_model.pth"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve("{model_url}", model_path)
    
    # Initialize upscaler
    model = {model_init}
    upsampler = RealESRGANer(scale=MODEL_SCALE, model_path=model_path, model=model,
                             tile=512, tile_pad=10, pre_pad=0, half=True, device="cuda")
    
    mark("MODEL_LOADED")
    
    # Upscale frames
    for i, f in enumerate(frames):
        img = cv2.imread(f)
        out, _ = upsampler.enhance(img, outscale=SCALE)
        cv2.imwrite(f.replace("/frames/", "/upscaled/"), out)
        if (i + 1) % 10 == 0:
            mark("UPSCALING", f"{{i+1}}/{{len(frames)}}")
    
    mark("UPSCALING_DONE", f"{{len(frames)}} frames")
    
    mark("ENCODING", "Creating video...")
    
    # Encode output video
    enc_result = subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"{{WORK_DIR}}/upscaled/f%06d.jpg",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        f"{{WORK_DIR}}/output_temp.mp4"], capture_output=True, text=True)
    
    if enc_result.returncode != 0:
        mark("FAILED", f"Encode failed: {{enc_result.stderr[:500]}}")
        sys.exit(1)
    
    # Check if input has audio
    audio_check = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0",
        f"{{WORK_DIR}}/input.mp4"], capture_output=True, text=True)
    
    has_audio = bool(audio_check.stdout.strip())
    
    if has_audio:
        # Add audio back
        audio_result = subprocess.run(["ffmpeg", "-y", 
            "-i", f"{{WORK_DIR}}/output_temp.mp4",
            "-i", f"{{WORK_DIR}}/input.mp4", 
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest", 
            f"{{WORK_DIR}}/output.mp4"], capture_output=True, text=True)
        
        if audio_result.returncode != 0:
            # Audio merge failed, use video-only
            os.rename(f"{{WORK_DIR}}/output_temp.mp4", f"{{WORK_DIR}}/output.mp4")
    else:
        # No audio in input, just rename
        os.rename(f"{{WORK_DIR}}/output_temp.mp4", f"{{WORK_DIR}}/output.mp4")
    
    size_mb = os.path.getsize(f"{{WORK_DIR}}/output.mp4") / 1024 / 1024
    mark("SUCCESS", f"{{size_mb:.1f}}MB")
    
except Exception as e:
    import traceback
    mark("FAILED", str(e) + "\\n" + traceback.format_exc())
'''


def _poll_for_completion(work_dir: str, job_id: str) -> bool:
    """Poll for job completion. Returns True if successful."""
    
    consecutive_failures = 0
    max_consecutive_failures = 5  # Abort if dir is gone
    
    for attempt in range(MAX_POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL)
        
        # Check if work directory still exists (session might have recycled)
        success, status, stderr = _run_cgpu(f"cat {work_dir}/status.txt 2>&1", timeout=30)
        
        # Detect if the directory no longer exists (session recycled)
        if "No such file or directory" in status or "No such file or directory" in stderr:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                _log(f"   ❌ Work directory gone - Colab session recycled?")
                return False
            if attempt % 6 == 0:
                _log(f"   ⏳ Waiting... ({attempt * POLL_INTERVAL}s)")
            continue
        
        if not success or not status.strip():
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                _log(f"   ❌ Lost connection to cgpu job")
                return False
            if attempt % 6 == 0:
                _log(f"   ⏳ Waiting... ({attempt * POLL_INTERVAL}s)")
            continue
        
        # Reset failure counter on successful status read
        consecutive_failures = 0
        
        # Clean up cgpu output noise
        # Remove "Authenticated as..." line and "__COLAB_CLI_EXIT__" garbage
        lines = [
            line for line in status.strip().split('\n')
            if line.strip() 
            and not line.startswith("Authenticated")
            and "__COLAB_CLI" not in line
            and "printf " not in line
        ]
        
        state = lines[0] if lines else ""
        detail = lines[1] if len(lines) > 1 else ""
        
        if state == "SUCCESS":
            _log(f"   ✅ Processing complete: {detail}")
            return True
        elif state == "FAILED":
            _log(f"   ❌ Job failed: {detail[:200]}")
            return False
        elif state == "UPSCALING":
            # Show progress
            _log(f"   🔄 {detail}")
        elif state in ("STARTING", "SETUP_DONE", "FRAMES_EXTRACTED", "MODEL_LOADED", "UPSCALING_DONE", "ENCODING"):
            # Progress markers, keep waiting
            pass
    
    _log(f"   ❌ Timeout after {CGPU_TIMEOUT}s")
    return False


def _download_result(remote_path: str, local_path: str) -> bool:
    """Download file from Colab via chunked base64 to avoid timeouts."""
    
    # First get file size
    success, size_out, _ = _run_cgpu(f"stat -c%s {remote_path} 2>/dev/null", timeout=30)
    if not success:
        _log(f"   ❌ Cannot get file size")
        return False
    
    try:
        file_size = int(size_out.strip().split()[-1])
    except:
        file_size = 0
    
    file_size_mb = file_size / (1024 * 1024)
    _log(f"   📦 File size: {file_size_mb:.1f} MB")
    
    # For small files (<15MB), use direct download
    if file_size < 15 * 1024 * 1024:
        return _download_direct(remote_path, local_path)
    
    # For larger files, use chunked download
    return _download_chunked(remote_path, local_path, file_size)


def _download_direct(remote_path: str, local_path: str) -> bool:
    """Direct base64 download for small files."""
    success, b64_data, stderr = _run_cgpu(f"base64 {remote_path}", timeout=300)
    
    if not success:
        _log(f"   ❌ Download failed: {stderr[:200]}")
        return False
    
    try:
        b64_clean = ''.join(
            line for line in b64_data.split('\n')
            if line and not line.startswith('Authenticated')
        )
        data = base64.b64decode(b64_clean)
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        _log(f"   ❌ Decode failed: {e}")
        return False


def _download_chunked(remote_path: str, local_path: str, file_size: int) -> bool:
    """Chunked download for large files - splits into 10MB chunks."""
    
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
    num_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    _log(f"   📥 Downloading in {num_chunks} chunks...")
    
    os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
    
    with open(local_path, 'wb') as out_file:
        for i in range(num_chunks):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, file_size)
            chunk_num = i + 1
            
            # Use dd to extract chunk, then base64
            # dd skip=X count=Y reads Y blocks starting at block X
            cmd = f"dd if={remote_path} bs=1 skip={start} count={end-start} 2>/dev/null | base64"
            
            success, b64_data, stderr = _run_cgpu(cmd, timeout=180)
            
            if not success:
                _log(f"   ❌ Chunk {chunk_num}/{num_chunks} failed")
                return False
            
            try:
                b64_clean = ''.join(
                    line for line in b64_data.split('\n')
                    if line and not line.startswith('Authenticated')
                )
                chunk_data = base64.b64decode(b64_clean)
                out_file.write(chunk_data)
                _log(f"   📥 Chunk {chunk_num}/{num_chunks} ({len(chunk_data)//1024}KB)")
            except Exception as e:
                _log(f"   ❌ Chunk decode failed: {e}")
                return False
    
    # Verify file size
    actual_size = os.path.getsize(local_path)
    if actual_size != file_size:
        _log(f"   ⚠️ Size mismatch: expected {file_size}, got {actual_size}")
    
    return True


# CLI for testing
if __name__ == "__main__":
    import sys
    
    print(f"cgpu Upscaler v3.0 (Polling-based)")
    print(f"CGPU_GPU_ENABLED: {CGPU_GPU_ENABLED}")
    
    if not is_cgpu_available():
        print("❌ cgpu not available")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage: python cgpu_upscaler_v3.py <input.mp4> [output.mp4] [scale]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "upscaled.mp4"
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    result = upscale_with_cgpu(input_file, output_file, scale=scale)
    sys.exit(0 if result else 1)
