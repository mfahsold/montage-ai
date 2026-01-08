"""
Subprocess Management & Signal Handling

Ensures graceful cleanup of FFmpeg and other subprocesses when the
Flask app receives SIGTERM (from K3s graceful shutdown).

Usage:
    from montage_ai.subprocess_manager import install_signal_handlers, track_subprocess
    
    install_signal_handlers()
    proc = track_subprocess(['ffmpeg', '-i', 'input.mp4', ...])
"""

import signal
import os
import time
import psutil
import logging
from typing import List, Optional
from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

# Global registry of active subprocesses
_active_processes: List[Popen] = []
_shutdown_timeout = int(os.environ.get('GRACEFUL_SHUTDOWN_TIMEOUT', '55'))


def track_subprocess(cmd: List[str], **kwargs) -> Popen:
    """
    Create a tracked subprocess.
    
    Args:
        cmd: Command list (e.g., ['ffmpeg', '-i', 'in.mp4', 'out.mp4'])
        **kwargs: Additional Popen arguments
    
    Returns:
        Popen instance (tracked globally)
    """
    proc = Popen(cmd, **kwargs)
    _active_processes.append(proc)
    
    logger.info(f"Tracked subprocess PID {proc.pid}: {' '.join(cmd[:3])}")
    
    return proc


def cleanup_subprocess(proc: Popen, timeout: int = 10) -> bool:
    """
    Gracefully cleanup a subprocess.
    
    1. Send SIGTERM
    2. Wait up to `timeout` seconds
    3. Send SIGKILL if still running
    
    Args:
        proc: Popen instance
        timeout: Timeout in seconds
    
    Returns:
        True if cleaned up, False if had to force kill
    """
    if proc.poll() is not None:
        # Already terminated
        logger.debug(f"Process {proc.pid} already terminated")
        return True
    
    logger.info(f"Cleaning up subprocess PID {proc.pid}")
    
    try:
        # Terminate gracefully
        proc.terminate()
        
        # Wait for graceful shutdown
        try:
            proc.wait(timeout=timeout)
            logger.info(f"Process {proc.pid} terminated gracefully")
            return True
        except:
            # Force kill
            logger.warning(f"Process {proc.pid} did not terminate, force killing")
            proc.kill()
            proc.wait(timeout=5)
            return False
    
    except Exception as e:
        logger.error(f"Error cleaning up process {proc.pid}: {e}")
        return False


def cleanup_all_subprocesses(timeout: int = 30) -> int:
    """
    Cleanup all tracked subprocesses.
    
    Args:
        timeout: Total timeout for all cleanup (seconds)
    
    Returns:
        Number of processes cleaned up
    """
    global _active_processes
    
    if not _active_processes:
        logger.debug("No active subprocesses to cleanup")
        return 0
    
    logger.info(f"Cleaning up {len(_active_processes)} active subprocesses")
    
    start_time = time.time()
    cleaned_up = 0
    
    for proc in _active_processes[:]:
        # Respect total timeout
        elapsed = time.time() - start_time
        remaining_timeout = max(1, timeout - elapsed)
        
        if cleanup_subprocess(proc, timeout=int(remaining_timeout)):
            cleaned_up += 1
        
        _active_processes.remove(proc)
    
    # Kill any orphaned children
    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        
        for child in children:
            if child.status() != psutil.STATUS_ZOMBIE:
                logger.warning(f"Killing orphaned child process PID {child.pid}")
                child.kill()
    except Exception as e:
        logger.warning(f"Error killing orphaned processes: {e}")
    
    logger.info(f"Cleaned up {cleaned_up} subprocesses")
    return cleaned_up


def signal_handler(signum: int, frame):
    """Handle SIGTERM signal from K3s graceful shutdown."""
    signal_name = signal.strsignal(signum) if hasattr(signal, 'strsignal') else f"SIG{signum}"
    logger.critical(f"🛑 Received {signal_name}, initiating graceful shutdown...")
    
    # Cleanup subprocesses
    cleanup_all_subprocesses(timeout=_shutdown_timeout - 5)
    
    # Flush logs
    logging.shutdown()
    
    # Exit
    os._exit(0)


def install_signal_handlers():
    """Install SIGTERM/SIGINT handlers for graceful shutdown."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Signal handlers installed (SIGTERM, SIGINT)")


def get_active_process_count() -> int:
    """Get number of active tracked subprocesses."""
    # Clean up completed processes
    global _active_processes
    
    _active_processes = [p for p in _active_processes if p.poll() is None]
    return len(_active_processes)


def get_process_memory_usage() -> int:
    """Get total memory used by all tracked subprocesses (bytes)."""
    total_memory = 0
    
    for proc in _active_processes:
        try:
            if proc.poll() is None:
                p = psutil.Process(proc.pid)
                total_memory += p.memory_info().rss
        except:
            pass
    
    return total_memory
