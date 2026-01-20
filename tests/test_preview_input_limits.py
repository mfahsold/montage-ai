import os
import tempfile
from pathlib import Path

import pytest

from montage_ai.core.montage_builder import MontageBuilder


@pytest.mark.unit
def test_preview_skips_large_files(tmp_path, monkeypatch):
    # Create small and large fake files
    small = tmp_path / "small.mp4"
    large = tmp_path / "large.mp4"
    small.write_bytes(b"0" * 1024 * 1024 * 1)   # 1 MB
    large.write_bytes(b"0" * 1024 * 1024 * 300) # 300 MB

    # Instantiate builder and inject video files
    b = MontageBuilder(variant_id=1)
    b.ctx.paths.input_dir = tmp_path
    b.ctx.media.video_files = [str(small), str(large)]

    # Force preview quality in settings (preferred over env)
    b.settings.encoding.quality_profile = "preview"
    b.settings.processing.preview_max_input_size_mb = 200
    b.settings.processing.preview_max_files = 2

    # Ensure workspace/executor are initialized (mirrors normal runtime)
    b.setup_workspace()

    # Avoid running ffmpeg in ProxyGenerator during unit test
    monkeypatch.setattr("montage_ai.core.montage_builder.ProxyGenerator.ensure_proxy", lambda self, p: None)

    # Stub out music analysis (not under test)
    b._analyzer.analyze_music = lambda *a, **k: None

    # Capture telemetry events
    events = []
    monkeypatch.setattr("montage_ai.telemetry.record_event", lambda et, data: events.append((et, data)))

    # Replace the analyzer.detect_scenes with a spy that records the file list
    called = {}

    def fake_detect_scenes(*args, **kwargs):
        called['files'] = list(b.ctx.media.video_files)
        return None

    b._analyzer.detect_scenes = fake_detect_scenes

    # Run analyze_assets (should not raise) and assert large file skipped
    b.analyze_assets()
    assert 'files' in called
    assert any('small.mp4' in f for f in called['files'])
    assert not any('large.mp4' in f for f in called['files'])

    # Telemetry: expect at least one proxy_generation and a scene_detection event
    assert any(e[0] == 'proxy_generation' for e in events), f"missing proxy telemetry: {events}"
    assert any(e[0] == 'scene_detection' for e in events), f"missing scene telemetry: {events}"

    # Context should be restored after analysis
    assert list(b.ctx.media.video_files) == [str(small), str(large)]
