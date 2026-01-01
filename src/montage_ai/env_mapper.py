"""
Environment Variable Mapper (DRY)

Centralizes the logic for converting user options (dict) into environment variables
for subprocess execution. This ensures consistency between CLI and Web UI.
"""

import os
from typing import Dict, Any, Optional

def bool_to_env(value: Any) -> str:
    """Convert boolean-like value to 'true'/'false' string."""
    if isinstance(value, str):
        return "true" if value.lower() == "true" else "false"
    return "true" if value else "false"

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
    
    # Core
    env["CUT_STYLE"] = style
    if job_id:
        env["JOB_ID"] = job_id
    
    # Creative Director
    env["CREATIVE_PROMPT"] = str(options.get("prompt", ""))
    
    # Feature Flags
    # Using explicit mapping to ensure we catch typos in keys
    env["STABILIZE"] = bool_to_env(options.get("stabilize"))
    env["UPSCALE"] = bool_to_env(options.get("upscale"))
    env["ENHANCE"] = bool_to_env(options.get("enhance"))
    env["LLM_CLIP_SELECTION"] = bool_to_env(options.get("llm_clip_selection"))
    env["EXPORT_TIMELINE"] = bool_to_env(options.get("export_timeline"))
    env["GENERATE_PROXIES"] = bool_to_env(options.get("generate_proxies"))
    env["PRESERVE_ASPECT"] = bool_to_env(options.get("preserve_aspect"))
    env["CREATIVE_LOOP"] = bool_to_env(options.get("creative_loop"))
    env["ENABLE_STORY_ENGINE"] = bool_to_env(options.get("story_engine"))
    if "strict_cloud_compute" in options:
        env["STRICT_CLOUD_COMPUTE"] = bool_to_env(options.get("strict_cloud_compute"))
    
    # Cloud/GPU
    cgpu_enabled = bool_to_env(options.get("cgpu"))
    env["CGPU_ENABLED"] = cgpu_enabled
    env["CGPU_GPU_ENABLED"] = cgpu_enabled # Usually linked
    
    # Audio/Timing
    env["TARGET_DURATION"] = str(options.get("target_duration", 0))
    env["MUSIC_START"] = str(options.get("music_start", 0))
    music_end = options.get("music_end")
    env["MUSIC_END"] = str(music_end) if music_end is not None else ""
    
    # Performance / Preview
    if options.get("preview"):
        env["FFMPEG_PRESET"] = "ultrafast"
        env["FINAL_CRF"] = "28"
    
    # System
    env["VERBOSE"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    
    return env
