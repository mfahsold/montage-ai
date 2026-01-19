import shutil
import subprocess

import pytest


def test_vulture_smoke():
    if shutil.which("vulture") is None:
        pytest.skip("vulture not installed in runtime image")
    # Run vulture in --min-confidence=70 and ensure it exits successfully (non-blocking)
    # This test should not be strict about specific findings; it's a smoke test to ensure vulture runs.
    result = subprocess.run(["vulture", "src", "--min-confidence", "70"], capture_output=True, text=True)
    assert result.returncode == 0
    # Optionally print findings for CI logs
    if result.stdout:
        print("Vulture output:\n", result.stdout)
