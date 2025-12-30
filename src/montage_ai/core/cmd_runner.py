import subprocess
import os
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from ..logger import logger

class CommandError(Exception):
    """Exception raised when a command fails."""
    def __init__(self, cmd: List[str], returncode: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"Command failed with return code {returncode}: {' '.join(str(x) for x in cmd)}\nStderr: {stderr}")

def run_command(
    cmd: List[str],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    check: bool = True,
    capture_output: bool = True,
    log_output: bool = False
) -> subprocess.CompletedProcess:
    """
    Run a shell command with consistent logging and error handling.
    
    Args:
        cmd: List of command arguments.
        cwd: Working directory.
        env: Environment variables (merged with os.environ).
        timeout: Timeout in seconds.
        check: If True, raise CommandError on non-zero exit code.
        capture_output: If True, capture stdout and stderr.
        log_output: If True, log stdout and stderr to debug log.
        
    Returns:
        subprocess.CompletedProcess object.
    """
    cmd_str = " ".join(str(x) for x in cmd)
    logger.debug(f"Running command: {cmd_str}")
    
    # Merge environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
        
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            timeout=timeout,
            check=False, # We handle check manually to raise custom exception
            capture_output=capture_output,
            text=True
        )
        
        if log_output:
            if result.stdout:
                logger.debug(f"Command stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"Command stderr: {result.stderr}")
                
        if check and result.returncode != 0:
            raise CommandError(cmd, result.returncode, result.stdout, result.stderr)
            
        return result
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout}s: {cmd_str}")
        raise
    except Exception as e:
        if not isinstance(e, CommandError):
            logger.error(f"Error running command {cmd_str}: {e}")
        raise
