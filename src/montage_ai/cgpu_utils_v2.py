"""
cgpu Utilities - Minimal Shared Functions

Provides only essential shared utilities for cgpu integration:
- LLM endpoint checking (for creative_director.py)
- Basic availability checks

The main upscaling logic is now in cgpu_upscaler_v3.py (polling-based).

Version: 2.0.0 (simplified)
"""

import os
import subprocess
from typing import Tuple
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

# GPU settings (used by cgpu_upscaler_v3.py)
CGPU_GPU_ENABLED = os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true"
CGPU_TIMEOUT = int(os.environ.get("CGPU_TIMEOUT", "1200"))

# LLM settings (used by creative_director.py)
CGPU_ENABLED = os.environ.get("CGPU_ENABLED", "false").lower() == "true"
CGPU_HOST = os.environ.get("CGPU_HOST", "127.0.0.1")
CGPU_PORT = os.environ.get("CGPU_PORT", "8080")
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
# Availability Checks
# ============================================================================

def is_cgpu_available() -> bool:
    """Check if cgpu is installed and authenticated for GPU tasks."""
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
    except Exception:
        return False


def is_cgpu_serve_available() -> bool:
    """Check if cgpu serve (LLM endpoint) is running."""
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


def check_cgpu_gpu() -> Tuple[bool, str]:
    """Check if cgpu can access a GPU on Colab."""
    try:
        result = subprocess.run(
            ["cgpu", "run", "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l for l in result.stdout.strip().split('\n') 
                     if 'Tesla' in l or 'T4' in l or 'A100' in l or 'MiB' in l]
            return True, lines[0] if lines else result.stdout.strip()
        return False, result.stderr
    except Exception as e:
        return False, str(e)


# ============================================================================
# Simple Command Execution (for backward compatibility)
# ============================================================================

def run_cgpu_command(cmd: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """Run a command on Colab via cgpu."""
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


def cgpu_copy_to_remote(local_path: str, remote_path: str, timeout: int = 600) -> bool:
    """Copy file from local to Colab."""
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


# Alias for backward compatibility
copy_to_remote = cgpu_copy_to_remote


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("cgpu Utilities - Status Check")
    print("=" * 40)
    print(f"CGPU_GPU_ENABLED: {CGPU_GPU_ENABLED}")
    print(f"CGPU_ENABLED (LLM): {CGPU_ENABLED}")
    print(f"CGPU_HOST: {CGPU_HOST}:{CGPU_PORT}")
    print()
    print(f"cgpu (GPU): {'✅' if is_cgpu_available() else '❌'}")
    print(f"cgpu serve: {'✅' if is_cgpu_serve_available() else '❌'}")
