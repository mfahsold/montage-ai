import importlib

from src.montage_ai.ffmpeg_config import FFmpegConfig, get_config, PREVIEW_CRF, PREVIEW_PRESET


def test_cpu_video_params_respects_overrides():
    # Force CPU path to avoid hardware detection
    cfg = FFmpegConfig(hwaccel="none")
    params = cfg.video_params(crf=22, preset="fast")

    # Check key args are present
    assert "-c:v" in params
    assert "libx264" in params
    assert "-crf" in params
    assert "22" in params
    assert "-preset" in params
    assert "fast" in params


def test_get_preview_video_params_uses_preview_settings():
    # Reset singleton and force CPU encoding for deterministic behavior
    get_config(hwaccel="none")
    params = get_config().video_params(crf=PREVIEW_CRF, preset=PREVIEW_PRESET)

    assert "-crf" in params
    assert str(PREVIEW_CRF) in params
    assert "-preset" in params
    assert PREVIEW_PRESET in params
