"""
Constants Configuration Module

Static constants and thresholds used throughout the system.
These rarely change and can be overridden via YAML.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from . import BaseConfig


@dataclass
class ConstantsConfig(BaseConfig):
    """System-wide constants and thresholds."""

    # Video standards
    standard_fps: int = 30
    standard_width: int = 1080
    standard_height: int = 1920
    standard_pix_fmt: str = "yuv420p"

    # Audio standards
    target_lufs: float = -14.0
    min_clip_duration: float = 0.5
    max_clip_duration: float = 10.0

    # Scene detection
    scene_threshold: float = 27.0
    min_scene_duration: float = 0.5

    # Beat detection
    beat_min_bpm: int = 60
    beat_max_bpm: int = 200
    beat_lookahead: float = 0.5

    # Cache sizes
    histogram_cache_size: int = 2000
    metadata_cache_ttl: int = 86400  # 24 hours

    # Processing limits
    max_scene_workers: int = 8
    max_batch_size: int = 50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstantsConfig":
        """Create from dict with flat key support (e.g., 'video.fps')."""
        if not data:
            return cls()

        # Flatten nested dicts
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    flat[f"{key}_{subkey}"] = subval
            else:
                flat[key] = value

        return super().from_dict(flat)
