import os
import pytest
from montage_ai.config import get_settings
from montage_ai.core.montage_builder import MontageBuilder

class DummyEngine:
    def __init__(self):
        self.render_called = False
    def set_renderer(self, r):
        pass
    def render_output(self):
        self.render_called = True
    def render_distributed(self):
        self.render_called = True


def make_builder(tmp_path, monkeypatch):
    # Create a MontageBuilder and patch its render engine for testing
    from montage_ai.core.montage_builder import MontageBuilder

    b = MontageBuilder(variant_id=1)
    b._render_engine = DummyEngine()
    # Ensure media list exists for tests
    b.ctx.media.video_files = []
    return b


def test_refuse_local_render_for_large_input_when_cluster_disabled(monkeypatch, tmp_path):
    b = make_builder(tmp_path, monkeypatch)
    settings = get_settings()
    # Ensure cluster mode is disabled
    monkeypatch.setattr(settings.features, "cluster_mode", False)
    # Make preview fast-path small so threshold is low (patch the builder's settings for determinism)
    monkeypatch.setattr(b.settings.processing, "preview_max_input_size_mb", 10, raising=False)

    # Add a single large input and fake its size
    b.ctx.media.video_files = [str(tmp_path / "big.mp4")]
    monkeypatch.setattr(os.path, "getsize", lambda p: 2500 * 1024 * 1024)

    with pytest.raises(RuntimeError, match="Refusing to run local encode for large input"):
        b.render_output()


def test_allow_render_when_cluster_enabled(monkeypatch, tmp_path):
    b = make_builder(tmp_path, monkeypatch)
    settings = get_settings()
    monkeypatch.setattr(settings.features, "cluster_mode", True)
    monkeypatch.setattr(b.settings.processing, "preview_max_input_size_mb", 10, raising=False)

    b.ctx.media.video_files = [str(tmp_path / "big.mp4")]
    monkeypatch.setattr(os.path, "getsize", lambda p: 2500 * 1024 * 1024)

    # Should not raise and should call the distributed render path
    b.render_output()
    assert b._render_engine.render_called
