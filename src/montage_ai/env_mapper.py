"""
Environment Variable Mapper (DRY)

Centralizes the logic for converting user options (dict) into environment variables
for subprocess execution. This ensures consistency between CLI and Web UI.

Quality Profiles:
  - preview:  360p, no enhancements, fast iteration
  - standard: 1080p, color grading (default)
  - high:     1080p + stabilization + enhancement
  - master:   4K + all enhancements + AI upscaling

Cloud Acceleration:
  Single toggle that enables CGPU for: upscaling, transcription, LLM.
  Auto-fallback to local if cloud unavailable.
"""

import os
from typing import Dict, Any, Optional

# Quality Profile Definitions (Single Source of Truth)
QUALITY_PROFILES = {
    "preview": {
        "enhance": False,
        "stabilize": False,
        "upscale": False,
        "width": 640,
        "height": 360,
        "crf": 28,
        "preset": "ultrafast",
    },
    "standard": {
        "enhance": True,
        "stabilize": False,
        "upscale": False,
        "width": 1920,
        "height": 1080,
        "crf": 18,
        "preset": "medium",
    },
    "high": {
        "enhance": True,
        "stabilize": True,
        "upscale": False,
        "width": 1920,
        "height": 1080,
        "crf": 17,
        "preset": "slow",
    },
    "master": {
        "enhance": True,
        "stabilize": True,
        "upscale": True,
        "width": 3840,
        "height": 2160,
        "crf": 16,
        "preset": "slow",
    },
}


def bool_to_env(value: Any) -> str:
    """Convert boolean-like value to 'true'/'false' string."""
    if isinstance(value, str):
        return "true" if value.lower() == "true" else "false"
    return "true" if value else "false"


def expand_quality_profile(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand quality profile into individual settings.
    
    Quality profiles bundle enhance/stabilize/upscale/resolution settings.
    Individual overrides still take precedence if explicitly set.
    """
    profile_name = str(options.get("quality_profile", "standard")).lower()
    profile = QUALITY_PROFILES.get(profile_name, QUALITY_PROFILES["standard"])
    
    result = options.copy()
    
    # Apply profile defaults (only if not explicitly set)
    for key, value in profile.items():
        if key not in result or result[key] is None:
            result[key] = value
        # Special handling: booleans that are False might need profile override
        elif key in ("enhance", "stabilize", "upscale") and result.get(key) is None:
            result[key] = value
    
    return result


def map_options_to_env(
    style: str,
    options: Dict[str, Any],
    job_id: Optional[str] = None,
    base_env: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Maps a dictionary of options to the environment variables expected by the editor.
    
    Args:
        style: The cut style (e.g., 'dynamic', 'mtv').
        options: Dictionary of options (e.g., {'stabilize': True}).
        job_id: Optional job ID.
        base_env: Optional base environment to start from (defaults to os.environ).
        
    Returns:
        A dictionary of environment variables.
    """
    env = (base_env or os.environ).copy()
    
    # Expand quality profile first
    expanded = expand_quality_profile(options)
    
    # Core
    env["CUT_STYLE"] = style
    if job_id:
        env["JOB_ID"] = job_id
    
    # Creative Director
    env["CREATIVE_PROMPT"] = str(expanded.get("prompt", ""))
    
    # Feature Flags (from expanded profile)
    env["STABILIZE"] = bool_to_env(expanded.get("stabilize"))
    env["UPSCALE"] = bool_to_env(expanded.get("upscale"))
    env["ENHANCE"] = bool_to_env(expanded.get("enhance"))
    env["LLM_CLIP_SELECTION"] = bool_to_env(expanded.get("llm_clip_selection"))
    env["SHORTS_MODE"] = bool_to_env(expanded.get("shorts_mode"))
    env["EXPORT_TIMELINE"] = bool_to_env(expanded.get("export_timeline"))
    env["GENERATE_PROXIES"] = bool_to_env(expanded.get("generate_proxies"))
    env["PRESERVE_ASPECT"] = bool_to_env(expanded.get("preserve_aspect"))
    env["CREATIVE_LOOP"] = bool_to_env(expanded.get("creative_loop"))
    env["ENABLE_STORY_ENGINE"] = bool_to_env(expanded.get("story_engine"))
    env["CAPTIONS"] = bool_to_env(expanded.get("captions"))
    
    # Audio Polish (Clean Audio = Voice Isolation + Denoise)
    env["VOICE_ISOLATION"] = bool_to_env(expanded.get("clean_audio") or expanded.get("voice_isolation"))
    
    if "strict_cloud_compute" in expanded:
        env["STRICT_CLOUD_COMPUTE"] = bool_to_env(expanded.get("strict_cloud_compute"))

    # Story Arc (tension curve preset)
    story_arc = expanded.get("story_arc", "")
    if story_arc:
        env["STORY_ARC"] = str(story_arc)

    # Quality Profile
    quality_profile = expanded.get("quality_profile", "standard")
    env["QUALITY_PROFILE"] = str(quality_profile)
    
    # Resolution from profile
    if expanded.get("width"):
        env["EXPORT_WIDTH"] = str(expanded["width"])
    if expanded.get("height"):
        env["EXPORT_HEIGHT"] = str(expanded["height"])
    
    # Encoding quality from profile
    if expanded.get("crf"):
        env["FINAL_CRF"] = str(expanded["crf"])
    if expanded.get("preset"):
        env["FFMPEG_PRESET"] = str(expanded["preset"])
    
    # Cloud Acceleration (single toggle for all cloud features)
    cloud_enabled = expanded.get("cloud_acceleration") or expanded.get("cgpu")
    cgpu_enabled = bool_to_env(cloud_enabled)
    env["CGPU_ENABLED"] = cgpu_enabled
    env["CGPU_GPU_ENABLED"] = cgpu_enabled
    
    # Audio/Timing
    env["TARGET_DURATION"] = str(expanded.get("target_duration", 0))
    env["MUSIC_START"] = str(expanded.get("music_start", 0))
    music_end = expanded.get("music_end")
    env["MUSIC_END"] = str(music_end) if music_end is not None else ""
    
    # Legacy export resolution overrides (backwards compat)
    if "export_width" in options and options["export_width"]:
        env["EXPORT_WIDTH"] = str(options["export_width"])
    if "export_height" in options and options["export_height"]:
        env["EXPORT_HEIGHT"] = str(options["export_height"])

    # Performance / Preview mode (legacy support)
    if options.get("preview"):
        env["QUALITY_PROFILE"] = "preview"
        env["FFMPEG_PRESET"] = "ultrafast"
        env["FINAL_CRF"] = "28"
    
    # System
    env["VERBOSE"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    
    return env
