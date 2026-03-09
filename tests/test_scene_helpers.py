"""Tests for deprecated scene helper and benchmark utility warnings."""

import importlib
import sys

import pytest


def _reload_with_warning(module_name: str, warning_match: str):
    """Reload a module and assert that a deprecation warning is emitted."""
    sys.modules.pop(module_name, None)
    with pytest.warns(DeprecationWarning, match=warning_match):
        return importlib.import_module(module_name)


def test_scene_helpers_import_emits_deprecation_warning() -> None:
    """Deprecated scene_helpers module should warn on import."""
    module = _reload_with_warning(
        "src.montage_ai.scene_helpers",
        "scene_helpers.py is deprecated",
    )
    assert hasattr(module, "SceneProcessor")


def test_scene_processor_normalize_scene_defaults() -> None:
    """Normalization should map start/end aliases and keep defaults stable."""
    module = importlib.import_module("src.montage_ai.scene_helpers")

    scene = module.SceneProcessor.normalize_scene(
        {
            "start": 1.25,
            "end": 3.75,
            "scene_id": "scene_01",
        }
    )

    assert scene.id == "scene_01"
    assert scene.start_time == 1.25
    assert scene.end_time == 3.75
    assert scene.duration == 2.5
    assert scene.visual_features == {}
    assert scene.audio_features == {}


def test_scene_similarity_stays_in_unit_interval() -> None:
    """Similarity scores should remain normalized to [0.0, 1.0]."""
    module = importlib.import_module("src.montage_ai.scene_helpers")

    score = module.SceneProcessor.calculate_scene_similarity(
        {"start": 0.0, "end": 2.0, "visual_features": {"brightness": 0.8}},
        {"start": 0.0, "end": 2.2, "visual_features": {"brightness": 0.7}},
    )

    assert 0.0 <= score <= 1.0


def test_benchmark_audio_gpu_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deprecated benchmark utility should emit warning and still return a result."""
    module = importlib.import_module("src.montage_ai.audio_analysis_gpu")

    monkeypatch.setattr(module, "_gpu_backend", "cuda")
    monkeypatch.setattr(module, "_detect_gpu_backend", lambda: "cpu")
    monkeypatch.setattr(module, "gpu_spectral_analysis", lambda _path: object())

    with pytest.warns(DeprecationWarning, match=r"benchmark_audio_gpu\(\) is deprecated"):
        result = module.benchmark_audio_gpu("dummy.wav")

    assert "cpu_seconds" in result
    assert result["gpu_available"] is False


def test_benchmark_backends_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deprecated scene backend benchmark should emit warning."""
    module = importlib.import_module("src.montage_ai.scene_detection_sota")

    monkeypatch.setattr(
        module,
        "detect_scenes_sota",
        lambda _path, backend, use_cache: [(0.0, 1.0)] if backend == "pyscenedetect" else [],
    )
    monkeypatch.setattr(module, "_check_transnetv2_available", lambda: False)
    monkeypatch.setattr(module, "_check_autoshot_available", lambda: False)

    with pytest.warns(DeprecationWarning, match=r"benchmark_backends\(\) is deprecated"):
        result = module.benchmark_backends("dummy.mp4")

    assert result["pyscenedetect"]["available"] is True
    assert result["transnetv2"]["available"] is False
    assert result["autoshot"]["available"] is False
