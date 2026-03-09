"""
Feature Configuration Module

Toggle features on/off and configure feature-specific settings.
"""

from dataclasses import dataclass, field
from typing import List

from . import BaseConfig, _env_override


@dataclass
class FeatureConfig(BaseConfig):
    """Feature flags and settings."""

    # Core features
    upscale: bool = field(default_factory=lambda: _env_override("upscale", True))
    stabilize: bool = field(default_factory=lambda: _env_override("stabilize", True))
    color_grade: bool = field(
        default_factory=lambda: _env_override("color_grade", True)
    )
    auto_reframe: bool = field(
        default_factory=lambda: _env_override("auto_reframe", True)
    )

    # Audio features
    normalize_audio: bool = field(
        default_factory=lambda: _env_override("normalize_audio", True)
    )
    ducking: bool = field(default_factory=lambda: _env_override("ducking", True))
    voice_isolation: bool = field(
        default_factory=lambda: _env_override("voice_isolation", False)
    )

    # Output features
    captions: bool = field(default_factory=lambda: _env_override("captions", False))
    logo_overlay: bool = field(
        default_factory=lambda: _env_override("logo_overlay", False)
    )
    film_grain: bool = field(default_factory=lambda: _env_override("film_grain", False))

    # Analysis features
    scene_detection: bool = field(
        default_factory=lambda: _env_override("scene_detection", True)
    )
    beat_detection: bool = field(
        default_factory=lambda: _env_override("beat_detection", True)
    )
    face_detection: bool = field(
        default_factory=lambda: _env_override("face_detection", True)
    )

    # Export features
    timeline_export: bool = field(
        default_factory=lambda: _env_override("timeline_export", True)
    )
    proxy_generation: bool = field(
        default_factory=lambda: _env_override("proxy_generation", False)
    )

    # Performance features
    parallel_processing: bool = field(
        default_factory=lambda: _env_override("parallel_processing", True)
    )
    gpu_acceleration: bool = field(
        default_factory=lambda: _env_override("gpu_acceleration", True)
    )
    progressive_render: bool = field(
        default_factory=lambda: _env_override("progressive_render", True)
    )
