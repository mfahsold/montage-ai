import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_vulture_smoke():
    # Attempt to find vulture executable or run it as a module
    vulture_path = shutil.which("vulture")
    
    if vulture_path:
        cmd = [vulture_path, "src", "--min-confidence", "70"]
    else:
        # Fallback to python -m vulture
        try:
            # Check if vulture is available as a module
            # We don't use -m vulture directly in check because it might not be installed
            import vulture
            cmd = [sys.executable, "-m", "vulture", "src", "--min-confidence", "70"]
        except ImportError:
            pytest.skip("vulture not installed in runtime image")
            
    # Run vulture and ensure it exits successfully (non-blocking)
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    # Optionally print findings for CI logs
    if result.stdout:
        print("Vulture output:\n", result.stdout)
