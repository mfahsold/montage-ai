"""
cgpu Utilities - Shared Cloud GPU Functions

Centralized utilities for cgpu (github.com/RohanAdwankar/cgpu) integration.
All modules that need cloud GPU access should import from here.

Features:
- Unified cgpu command execution
- File upload/download via base64
- Environment checking
- Error handling and retry logic

Used by:
- cgpu_upscaler.py (Real-ESRGAN)
- wan_vace.py (Wan2.1 video generation)
- open_sora.py (Open-Sora video generation)
- video_agent.py (VQA on cloud if needed)
"""

import os
import subprocess
import base64
import tempfile
from typing import Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


# ============================================================================
# Configuration (from environment)
# ============================================================================

# GPU-related cgpu settings
CGPU_GPU_ENABLED = os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true"
CGPU_TIMEOUT = int(os.environ.get("CGPU_TIMEOUT", "1200"))  # 20 minutes default for video upscaling

# LLM-related cgpu settings (for cgpu serve)
CGPU_ENABLED = os.environ.get("CGPU_ENABLED", "false").lower() == "true"
CGPU_HOST = os.environ.get("CGPU_HOST", "127.0.0.1")
CGPU_PORT = os.environ.get("CGPU_PORT", "8090")  # Updated default port to match montage-ai.sh
CGPU_MODEL = os.environ.get("CGPU_MODEL", "gemini-2.0-flash")


@dataclass
class CGPUConfig:
    """Configuration container for cgpu settings."""
    gpu_enabled: bool = CGPU_GPU_ENABLED
    timeout: int = CGPU_TIMEOUT
    llm_enabled: bool = CGPU_ENABLED
    host: str = CGPU_HOST
    port: str = CGPU_PORT
    model: str = CGPU_MODEL


# ============================================================================
# cgpu Availability Checks
# ============================================================================

def is_cgpu_available() -> bool:
    """
    Check if cgpu is installed and authenticated.
    
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
    Check if cgpu can access a GPU on Colab.
    
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
            # Extract GPU info line
            lines = [l for l in result.stdout.strip().split('\n') 
                     if 'Tesla' in l or 'T4' in l or 'A100' in l or 'MiB' in l]
            return True, lines[0] if lines else result.stdout.strip()
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def is_cgpu_serve_available() -> bool:
    """
    Check if cgpu serve (LLM endpoint) is running.
    
    Returns:
        True if cgpu serve is accessible
    """
    if not CGPU_ENABLED:
        return False
    
    try:
        import requests
        response = requests.get(
            f"http://{CGPU_HOST}:{CGPU_PORT}/v1/models",
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


# ============================================================================
# cgpu Command Execution
# ============================================================================

def run_cgpu_command(
    cmd: str,
    timeout: int = CGPU_TIMEOUT,
    retries: int = 1,  # Reduced from 2 to avoid excessive retries
    retry_delay: int = 5
) -> Tuple[bool, str, str]:
    """
    Run a shell command on Colab via cgpu with retry logic.

    Args:
        cmd: Shell command to execute
        timeout: Timeout in seconds PER ATTEMPT
        retries: Number of retry attempts on failure (default: 1)
        retry_delay: Seconds to wait between retries

    Returns:
        (success, stdout, stderr) tuple
    """
    import time

    last_error = ""

    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                ["cgpu", "run", cmd],
                capture_output=True,
                text=True,
                timeout=timeout  # Timeout per attempt, not total
            )

            # Check for session invalidation errors
            if "session expired" in result.stderr.lower() or "not authenticated" in result.stderr.lower():
                print(f"   ⚠️ cgpu session expired, attempting reconnection...")
                # Try to reconnect (cgpu status triggers re-auth)
                subprocess.run(["cgpu", "status"], capture_output=True, timeout=30)
                if attempt < retries:
                    time.sleep(retry_delay)
                    continue  # Retry

            return result.returncode == 0, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            last_error = f"Timeout after {timeout}s"
            # For timeout, don't retry - it's likely the command will timeout again
            print(f"   ⚠️ cgpu command timed out after {timeout}s")
            break  # Don't retry on timeout
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                print(f"   ⚠️ Error on attempt {attempt+1}/{retries+1}: {e}, retrying...")
                time.sleep(retry_delay)
                continue

    return False, "", last_error


def run_cgpu_python(
    script: str,
    timeout: int = CGPU_TIMEOUT,
    script_name: str = "script.py"
) -> Tuple[bool, str, str]:
    """
    Execute a Python script on Colab via cgpu.
    
    Uploads script file and runs it, avoiding shell escaping issues.
    
    Args:
        script: Python script content
        timeout: Timeout in seconds
        script_name: Name for the remote script file
        
    Returns:
        (success, stdout, stderr) tuple
    """
    remote_script = f"/tmp/{script_name}"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        local_script = f.name
    
    try:
        # Upload script
        if not copy_to_remote(local_script, remote_script):
            return False, "", "Failed to upload script"
        
        # Execute
        return run_cgpu_command(f"python3 {remote_script}", timeout=timeout)
        
    finally:
        os.unlink(local_script)


# ============================================================================
# File Transfer
# ============================================================================

def copy_to_remote(local_path: str, remote_path: str, timeout: int = 600) -> bool:
    """
    Copy file from local to Colab using cgpu copy.

    Note: cgpu copy is upload-only. For downloads, use download_via_base64.

    Args:
        local_path: Path to local file
        remote_path: Destination path on Colab
        timeout: Upload timeout in seconds (default: 600 = 10 min)

    Returns:
        True if successful
    """
    try:
        # Get file size for logging
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)

        result = subprocess.run(
            ["cgpu", "copy", local_path, remote_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            print(f"   ❌ cgpu copy failed (file: {os.path.basename(local_path)}, size: {file_size_mb:.1f}MB)")
            if result.stderr:
                # Show first line of error
                error_line = result.stderr.strip().split('\n')[0]
                print(f"      Error: {error_line}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"   ❌ Upload timeout after {timeout}s (file: {os.path.basename(local_path)})")
        print(f"      → File may be too large ({file_size_mb:.1f}MB), consider increasing timeout")
        return False
    except Exception as e:
        print(f"   ❌ Copy to remote failed: {e}")
        return False


def download_via_base64(
    remote_path: str, 
    local_path: str,
    timeout: int = 300
) -> bool:
    """
    Download file from Colab via base64 encoding.
    
    Since cgpu copy is upload-only, we use base64 over stdout.
    
    Args:
        remote_path: Path to file on Colab
        local_path: Destination path locally
        timeout: Timeout in seconds
        
    Returns:
        True if successful
    """
    success, b64_data, stderr = run_cgpu_command(
        f"base64 {remote_path}",
        timeout=timeout
    )
    
    if not success:
        print(f"   ❌ Download failed: {stderr}")
        return False
    
    try:
        # Filter out cgpu auth message lines
        b64_lines = [l for l in b64_data.split('\n') 
                     if l and not l.startswith('Authenticated')]
        b64_clean = ''.join(b64_lines)
        
        # Decode and save
        file_data = base64.b64decode(b64_clean)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        
        with open(local_path, 'wb') as f:
            f.write(file_data)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to decode/save: {e}")
        return False


def parse_base64_output(output: str, start_marker: str, end_marker: str) -> Optional[bytes]:
    """
    Extract base64 data from cgpu output using markers.
    
    Args:
        output: Full cgpu stdout
        start_marker: Marker indicating start of base64 data
        end_marker: Marker indicating end of base64 data
        
    Returns:
        Decoded bytes or None if not found
    """
    if start_marker not in output or end_marker not in output:
        return None
    
    start_idx = output.index(start_marker) + len(start_marker)
    end_idx = output.index(end_marker)
    
    b64_data = output[start_idx:end_idx].strip()
    return base64.b64decode(b64_data)


# ============================================================================
# Environment Setup Helpers
# ============================================================================

def setup_python_packages(packages: list, quiet: bool = True) -> bool:
    """
    Install Python packages on Colab.
    
    Args:
        packages: List of package names
        quiet: Suppress pip output
        
    Returns:
        True if successful
    """
    quiet_flag = "-q" if quiet else ""
    packages_str = " ".join(packages)
    
    success, stdout, stderr = run_cgpu_command(
        f"pip install {quiet_flag} {packages_str}",
        timeout=300
    )
    
    return success


def check_cuda_available() -> Tuple[bool, str]:
    """
    Check if CUDA is available on Colab.
    
    Returns:
        (available, info) tuple
    """
    script = '''
import torch
if torch.cuda.is_available():
    print(f"CUDA_OK: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
else:
    print("CUDA_FAIL: No GPU")
'''
    
    success, stdout, stderr = run_cgpu_python(script, timeout=60)
    
    if success and "CUDA_OK" in stdout:
        info = stdout.split("CUDA_OK:")[1].strip().split('\n')[0]
        return True, info
    
    return False, stderr or "CUDA not available"


# ============================================================================
# Cleanup Helpers
# ============================================================================

def cleanup_remote(paths: list) -> None:
    """
    Clean up files/directories on Colab (best effort).
    
    Args:
        paths: List of paths to remove
    """
    for path in paths:
        try:
            run_cgpu_command(f"rm -rf {path}", timeout=30)
        except Exception:
            pass


# ============================================================================
# Aliases for backward compatibility
# ============================================================================

# cgpu_upscaler.py uses these names
cgpu_copy_to_remote = copy_to_remote
cgpu_download_base64 = download_via_base64


# ============================================================================
# Module Self-Test
# ============================================================================

if __name__ == "__main__":
    print("cgpu Utilities - Status Check")
    print("=" * 50)
    
    print(f"\nConfiguration:")
    print(f"  CGPU_GPU_ENABLED: {CGPU_GPU_ENABLED}")
    print(f"  CGPU_ENABLED (LLM): {CGPU_ENABLED}")
    print(f"  CGPU_HOST: {CGPU_HOST}")
    print(f"  CGPU_PORT: {CGPU_PORT}")
    print(f"  CGPU_MODEL: {CGPU_MODEL}")
    print(f"  CGPU_TIMEOUT: {CGPU_TIMEOUT}s")
    
    print(f"\nAvailability:")
    
    gpu_available = is_cgpu_available()
    print(f"  cgpu (GPU): {'✅' if gpu_available else '❌'}")
    
    if gpu_available:
        success, info = check_cgpu_gpu()
        print(f"  GPU Info: {info if success else '❌ ' + info}")
        
        cuda_ok, cuda_info = check_cuda_available()
        print(f"  CUDA: {'✅ ' + cuda_info if cuda_ok else '❌ ' + cuda_info}")
    
    serve_available = is_cgpu_serve_available()
    print(f"  cgpu serve (LLM): {'✅' if serve_available else '❌'}")
