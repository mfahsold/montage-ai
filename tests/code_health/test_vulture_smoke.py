import subprocess


def test_vulture_runs():
    # Non-blocking smoke test to ensure vulture is callable in dev env
    result = subprocess.run(["vulture", "src", "--min-confidence", "50"], capture_output=True, text=True)
    # Exit code may be non-zero if issues found; ensure command ran successfully (exit 0 or 1)
    assert result.returncode in (0, 1)
