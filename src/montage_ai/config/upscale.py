"""
Upscale Configuration Module

AI upscaling settings.
"""

from dataclasses import dataclass, field
from typing import Optional

from . import BaseConfig, _env_override


@dataclass
class UpscaleConfig(BaseConfig):
    """AI upscaling configuration."""

    # Scale factor
    scale: int = field(default_factory=lambda: _env_override("upscale_scale", 2))

    # Model selection
    model: str = field(
        default_factory=lambda: _env_override("upscale_model", "RealESRGAN_x4plus")
    )

    # Quality vs Speed
    denoise_strength: float = field(
        default_factory=lambda: _env_override("upscale_denoise", 0.5)
    )

    # Face enhancement
    face_enhance: bool = field(
        default_factory=lambda: _env_override("upscale_face_enhance", False)
    )

    # Tile size for memory efficiency
    tile: int = field(
        default_factory=lambda: _env_override("upscale_tile", 0)
    )  # 0 = auto
