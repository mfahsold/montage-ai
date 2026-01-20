import os
from pathlib import Path


def test_single_canonical_overlay_exists_and_unique():
    """Ensure there is exactly one canonical cluster overlay named 'production'.

    This prevents repository drift where multiple overlays claim to be the
    canonical cluster deployment. The test asserts there is exactly one
    directory named `production` under `deploy/k3s/overlays/`.
    """
    repo_root = Path(__file__).resolve().parents[2]
    overlays_dir = repo_root / 'deploy' / 'k3s' / 'overlays'
    assert overlays_dir.exists(), "deploy/k3s/overlays not found"

    production_dirs = [p for p in overlays_dir.iterdir() if p.is_dir() and p.name == 'production']
    assert len(production_dirs) == 1, (
        f"Expected exactly one canonical overlay named 'production', found: {[p.name for p in overlays_dir.iterdir() if p.is_dir()]}"
    )
