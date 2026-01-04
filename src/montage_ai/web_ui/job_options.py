"""
Helpers for parsing and normalizing Web UI job options.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from ..config import Settings


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def apply_preview_preset(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply fast preview overrides for /api/jobs preset=fast.
    """
    if data.get("preset") != "fast":
        return data

    updated = copy.deepcopy(data)
    updated["style"] = "dynamic"
    updated["target_duration"] = min(updated.get("target_duration", 30) or 30, 30)
    updated["upscale"] = False
    updated["stabilize"] = False
    updated["enhance"] = False
    updated["cgpu"] = False
    updated["creative_loop"] = False

    options = updated.setdefault("options", {})
    is_shorts = _parse_bool(options.get("shorts_mode", updated.get("shorts_mode", False)))
    if is_shorts:
        options["export_width"] = 360
        options["export_height"] = 640
    else:
        options["export_width"] = 640
        options["export_height"] = 360

    return updated


def apply_finalize_overrides(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upgrade a preview job to a high-quality render.
    """
    updated = copy.deepcopy(options)
    updated["quality_profile"] = "high"
    updated["preview"] = False
    updated["export_width"] = 1920
    updated["export_height"] = 1080
    updated["upscale"] = True
    updated["stabilize"] = True
    updated["enhance"] = True
    return updated


def normalize_options(data: Dict[str, Any], defaults: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    """
    Normalize and validate job options from API request.

    Supports both nested "options" and flat request payloads.
    Automatically derives music_end from target_duration when missing.
    """
    opts = data.get("options", {})

    target_duration = float(opts.get("target_duration", data.get("target_duration", 0)) or 0)
    music_start = float(opts.get("music_start", data.get("music_start", 0)) or 0)
    music_end_raw = opts.get("music_end", data.get("music_end", None))

    music_end = float(music_end_raw) if music_end_raw is not None else None
    if target_duration > 0 and music_end is None:
        music_end = music_start + target_duration

    target_duration = max(0, min(target_duration, 3600))
    music_start = max(0, music_start)
    if music_end is not None:
        music_end = max(music_start + 1, music_end)

    def get_bool(key: str) -> bool:
        val = opts.get(key, data.get(key))
        if val is None:
            return bool(defaults.get(key, False))
        return _parse_bool(val)

    quality_profile = str(opts.get("quality_profile", data.get("quality_profile", "standard"))).lower()

    cloud_acceleration = get_bool("cloud_acceleration") or get_bool("cgpu") or settings.llm.cgpu_enabled

    return {
        "prompt": str(opts.get("prompt", data.get("prompt", ""))),
        "stabilize": get_bool("stabilize"),
        "upscale": get_bool("upscale"),
        "enhance": get_bool("enhance"),
        "llm_clip_selection": get_bool("llm_clip_selection"),
        "shorts_mode": get_bool("shorts_mode"),
        "export_width": int(opts.get("export_width", data.get("export_width", 0)) or 0),
        "export_height": int(opts.get("export_height", data.get("export_height", 0)) or 0),
        "creative_loop": get_bool("creative_loop"),
        "story_engine": get_bool("story_engine"),
        "captions": get_bool("captions"),
        "export_timeline": get_bool("export_timeline"),
        "generate_proxies": get_bool("generate_proxies"),
        "preserve_aspect": get_bool("preserve_aspect"),
        "target_duration": target_duration,
        "music_start": music_start,
        "music_end": music_end,
        "preview": data.get("preset") == "fast",
        "quality_profile": quality_profile,
        "cloud_acceleration": cloud_acceleration,
        "cgpu": cloud_acceleration,
        "clean_audio": get_bool("clean_audio"),
        "voice_isolation": get_bool("voice_isolation"),
        "story_arc": str(opts.get("story_arc", data.get("story_arc", ""))),
        "reframe_mode": str(opts.get("reframe_mode", data.get("reframe_mode", "auto"))),
        "caption_style": str(opts.get("caption_style", data.get("caption_style", "tiktok"))),
    }
