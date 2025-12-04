"""Version information for montage-ai package.

The actual version is determined at runtime from GIT_COMMIT env var.
This file provides a fallback for pip/setuptools.
"""

import os
import subprocess
from pathlib import Path


def get_version() -> str:
    """Get version from git commit hash or fallback."""
    # Check env var first (set at Docker build time)
    git_commit = os.environ.get("GIT_COMMIT", "").strip()
    if git_commit:
        return git_commit[:8]
    
    # Try live git command
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).parent.parent.parent  # repo root
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return "dev"


VERSION = get_version()
__version__ = VERSION
