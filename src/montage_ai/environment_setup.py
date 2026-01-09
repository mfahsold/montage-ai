"""
Environment Setup & Initialization Module

Handles idempotent environment setup:
- Cleanup of stale temporary files
- Directory creation
- Pre-flight checks (FFmpeg, disk space, etc.)
"""

import os
import glob
import time
import subprocess
from typing import Optional
from .logger import logger


def cleanup_stale_temp_files(max_age_hours: int = 1) -> int:
    """
    Clean up temporary files older than max_age_hours.
    
    Args:
        max_age_hours: Files older than this are deleted
    
    Returns:
        Number of files deleted
    """
    deleted = 0
    try:
        # Remove old scene detection proxies
        old_proxies = glob.glob("/tmp/scene_detect_*.mp4")
        for proxy in old_proxies:
            try:
                age_hours = (time.time() - os.path.getmtime(proxy)) / 3600
                if age_hours > max_age_hours:
                    os.remove(proxy)
                    logger.debug(f"Cleaned old proxy: {proxy} ({age_hours:.1f}h old)")
                    deleted += 1
            except Exception as e:
                logger.debug(f"Could not clean proxy {proxy}: {e}")
        
        # Remove old FFprobe cache
        old_caches = glob.glob("/tmp/ffprobe_cache_*.json")
        for cache in old_caches:
            try:
                age_hours = (time.time() - os.path.getmtime(cache)) / 3600
                if age_hours > max_age_hours:
                    os.remove(cache)
                    logger.debug(f"Cleaned old cache: {cache}")
                    deleted += 1
            except Exception as e:
                logger.debug(f"Could not clean cache {cache}: {e}")
    except Exception as e:
        logger.warning(f"Temp cleanup failed: {e}")
    
    return deleted


def ensure_directories(output_dir: str, input_dir: str) -> bool:
    """
    Idempotently create required directories.
    
    Args:
        output_dir: Where final videos go
        input_dir: Where source files are located
    
    Returns:
        True if successful
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        logger.debug(f"✓ Directories ready: {output_dir}, {input_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
            text=True
        )
        if result.returncode == 0:
            # Extract version from output (first line usually has it)
            version_line = result.stdout.split('\n')[0]
            logger.debug(f"✓ FFmpeg available: {version_line}")
            return True
        else:
            logger.error("FFmpeg returned non-zero exit code")
            return False
    except FileNotFoundError:
        logger.error("FFmpeg not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg check timed out")
        return False
    except Exception as e:
        logger.error(f"FFmpeg check failed: {e}")
        return False


def check_disk_space(path: str, min_gb: float = 5.0) -> bool:
    """
    Check available disk space.
    
    Args:
        path: Path to check
        min_gb: Minimum required space in GB
    
    Returns:
        True if enough space available
    """
    try:
        import shutil
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024 ** 3)
        
        if available_gb < min_gb:
            logger.warning(f"Low disk space: {available_gb:.1f}GB available (need {min_gb}GB)")
            return False
        
        logger.debug(f"✓ Disk space OK: {available_gb:.1f}GB available")
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        # Don't fail hard on disk check
        return True


def initialize_environment(
    output_dir: str,
    input_dir: str,
    min_disk_gb: float = 5.0,
    cleanup_temp: bool = True
) -> bool:
    """
    Idempotently initialize the environment for montage creation.
    
    This function is designed to be called every run without side effects:
    - Cleans up old temporary files (optional)
    - Creates necessary directories
    - Verifies FFmpeg availability
    - Checks disk space
    
    Args:
        output_dir: Where final videos should go
        input_dir: Where source footage is stored
        min_disk_gb: Minimum required disk space (GB)
        cleanup_temp: Whether to clean stale temp files
    
    Returns:
        True if all checks pass
    """
    logger.info("🔧 Initializing environment...")
    
    all_ok = True
    
    # Cleanup
    if cleanup_temp:
        deleted = cleanup_stale_temp_files(max_age_hours=1)
        if deleted > 0:
            logger.debug(f"   Cleaned {deleted} old temp files")
    
    # Directories
    if not ensure_directories(output_dir, input_dir):
        all_ok = False
        logger.warning("   Failed to create directories")
    
    # FFmpeg
    if not check_ffmpeg_available():
        all_ok = False
        logger.error("   FFmpeg not available")
    
    # Disk space
    if not check_disk_space(output_dir, min_disk_gb):
        all_ok = False
        # Don't fail hard - user might want to continue anyway
        logger.warning("   Disk space check failed")
    
    if all_ok:
        logger.info("   ✅ Environment ready")
    else:
        logger.warning("   ⚠️  Some checks failed (continuing anyway)")
    
    return all_ok
