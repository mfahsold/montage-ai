"""
Hardware Configuration Module

GPU, encoding, and hardware acceleration settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from . import BaseConfig, _env_override


@dataclass
class GPUConfig(BaseConfig):
    """GPU and hardware acceleration configuration."""

    # Auto-detection
    auto_detect: bool = field(
        default_factory=lambda: _env_override("gpu_auto_detect", True)
    )

    # Force specific encoder
    force_encoder: Optional[str] = field(
        default_factory=lambda: _env_override("force_encoder", None)
    )  # h264_nvenc, h264_vaapi, h264_qsv, libx264

    # GPU selection
    gpu_id: int = field(default_factory=lambda: _env_override("gpu_id", 0))

    # Memory
    vram_limit_gb: Optional[float] = field(
        default_factory=lambda: _env_override("vram_limit_gb", None)
    )

    # Encoder preferences (fallback order)
    encoder_priority: List[str] = field(
        default_factory=lambda: [
            "h264_nvenc",  # NVIDIA
            "h264_vaapi",  # AMD/Intel Linux
            "h264_qsv",  # Intel QuickSync
            "libx264",  # CPU fallback
        ]
    )


@dataclass
class EncodingConfig(BaseConfig):
    """Video encoding configuration."""

    # Codec settings
    codec: str = field(default_factory=lambda: _env_override("output_codec", "libx264"))
    profile: str = field(
        default_factory=lambda: _env_override("output_profile", "high")
    )
    level: str = field(default_factory=lambda: _env_override("output_level", "4.1"))
    pixel_format: str = field(
        default_factory=lambda: _env_override("output_pix_fmt", "yuv420p")
    )

    # Rate control
    bitrate: Optional[str] = field(
        default_factory=lambda: _env_override("output_bitrate", None)
    )
    crf: Optional[int] = field(
        default_factory=lambda: _env_override("output_crf", None)
    )
    preset: str = field(
        default_factory=lambda: _env_override("output_preset", "medium")
    )

    # Audio
    audio_codec: str = field(
        default_factory=lambda: _env_override("audio_codec", "aac")
    )
    audio_bitrate: str = field(
        default_factory=lambda: _env_override("audio_bitrate", "192k")
    )
    audio_sample_rate: int = field(
        default_factory=lambda: _env_override("audio_sample_rate", 48000)
    )

    def validate(self) -> list:
        """Validate encoding configuration."""
        errors = []

        valid_presets = [
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ]
        if self.preset not in valid_presets:
            errors.append(f"Invalid preset: {self.preset}. Use: {valid_presets}")

        valid_profiles = ["baseline", "main", "high", "high10", "high422", "high444"]
        if self.profile not in valid_profiles:
            errors.append(f"Invalid profile: {self.profile}. Use: {valid_profiles}")

        return errors
