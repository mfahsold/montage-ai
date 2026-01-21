import os

from montage_ai.core.hardware import get_best_hwaccel


def test_hwaccel_env_none_forces_cpu(monkeypatch):
    monkeypatch.setenv('FFMPEG_HWACCEL', 'none')
    cfg = get_best_hwaccel()
    assert cfg.type == 'cpu'


def test_hwaccel_env_unknown_falls_back_cleanly(monkeypatch):
    monkeypatch.setenv('FFMPEG_HWACCEL', 'totally_unknown_accel')
    cfg = get_best_hwaccel()
    # Unknown override should fall back to auto-detect which at minimum
    # returns a CPU config in CI environments.
    assert cfg is not None
    assert hasattr(cfg, 'type')
