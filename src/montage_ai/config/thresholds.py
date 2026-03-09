"""
Thresholds Configuration Module

Detection and processing thresholds.
"""

from dataclasses import dataclass, field

from . import BaseConfig, _env_override


@dataclass
class ThresholdsConfig(BaseConfig):
    """Detection and processing thresholds."""

    # Scene detection
    scene_threshold: float = field(
        default_factory=lambda: _env_override("scene_threshold", 27.0)
    )

    # Speech detection
    speech_threshold: float = field(
        default_factory=lambda: _env_override("speech_threshold", 0.5)
    )

    # Silence detection
    silence_level_db: str = field(
        default_factory=lambda: _env_override("silence_level_db", "-50dB")
    )
    silence_duration_s: float = field(
        default_factory=lambda: _env_override("silence_duration_s", 0.3)
    )
    silence_threshold: float = field(
        default_factory=lambda: _env_override("silence_threshold", 0.02)
    )

    # Face detection
    face_confidence: float = field(
        default_factory=lambda: _env_override("face_confidence", 0.5)
    )

    # Audio ducking
    ducking_core_threshold: float = field(
        default_factory=lambda: _env_override("ducking_core_threshold", 0.3)
    )
    ducking_soft_threshold: float = field(
        default_factory=lambda: _env_override("ducking_soft_threshold", 0.1)
    )

    # Duration limits
    music_min_duration_s: float = field(
        default_factory=lambda: _env_override("music_min_duration_s", 30.0)
    )
    speech_min_duration_ms: int = field(
        default_factory=lambda: _env_override("speech_min_duration_ms", 100)
    )
    speech_min_silence_ms: int = field(
        default_factory=lambda: _env_override("speech_min_silence_ms", 50)
    )
