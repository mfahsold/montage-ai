"""Version information for montage-ai package.

The actual version is determined at runtime from GIT_COMMIT env var.
This file provides a fallback for pip/setuptools.
"""

import os
import subprocess
from pathlib import Path

# Base version for PEP 440 compliance (used by pip/setuptools)
BASE_VERSION = "0.4.0"


def get_git_commit() -> str:
    """Get git commit hash for display purposes."""
    # Check env var first (set at Docker build time)
    git_commit = os.environ.get("GIT_COMMIT", "").strip()
    if git_commit and git_commit != "dev":
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


def get_version() -> str:
    """Get PEP 440 compliant version string."""
    commit = get_git_commit()
    if commit and commit != "dev":
        # Format: 0.4.0+c6eea1c5 (local version identifier)
        return f"{BASE_VERSION}+{commit}"
    return BASE_VERSION


# For setuptools/pip
__version__ = BASE_VERSION  # Must be static for build tools

# For runtime display (includes git commit)
VERSION = get_version()
GIT_COMMIT = get_git_commit()
