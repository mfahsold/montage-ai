"""
Color Grading Module - Centralized Color Grading for Montage AI

This module provides a unified interface for color grading operations:
- Built-in FFmpeg filter presets (no external dependencies)
- LUT file support (.cube, .3dl)
- Intensity/strength control
- NLE-compatible output (Rec.709)

Based on industry standards:
- 33x33x33 .cube LUTs (DaVinci Resolve, Premiere Pro compatible)
- Rec.709 color space for web/broadcast delivery
- Scene-referred workflow: normalize first, then grade

References:
- https://www.studiobinder.com/blog/what-is-lut/
- https://gabor.heja.hu/blog/2024/12/10/using-ffmpeg-to-color-correct-color-grade-a-video-lut-hald-clut/
- https://www.tobiamontanari.com/luts-in-color-grading-when-to-use-them-and-when-not-to/
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logger import logger


# =============================================================================
# Color Grading Presets (FFmpeg Filter Chains)
# =============================================================================

class ColorGradePreset(str, Enum):
    """
    Built-in color grading presets using FFmpeg filters.

    These work without external LUT files and are NLE-compatible.
    Designed for Rec.709 output (standard for HD/web delivery).
    """
    # No processing
    NONE = "none"

    # Neutral - broadcast safe, minimal processing
    NEUTRAL = "neutral"

    # Temperature adjustments
    WARM = "warm"
    COOL = "cool"
    GOLDEN_HOUR = "golden_hour"
    BLUE_HOUR = "blue_hour"

    # Cinematic looks
    TEAL_ORANGE = "teal_orange"
    CINEMATIC = "cinematic"
    BLOCKBUSTER = "blockbuster"

    # Stylized
    VIBRANT = "vibrant"
    DESATURATED = "desaturated"
    HIGH_CONTRAST = "high_contrast"
    FILMIC_WARM = "filmic_warm"
    VINTAGE = "vintage"
    NOIR = "noir"

    # Documentary/Natural
    DOCUMENTARY = "documentary"
    NATURAL = "natural"


# FFmpeg filter chains for each preset
# These are designed to be composable and work at various intensities
PRESET_FILTERS: Dict[str, str] = {
    # No grading - pass through
    "none": "",

    # Neutral - broadcast safe levels only (16-235 range)
    "neutral": "colorlevels=rimin=0.0625:gimin=0.0625:bimin=0.0625:rimax=0.92:gimax=0.92:bimax=0.92",

    # Warm - golden/sunset warmth (6500K)
    "warm": "colortemperature=6500,colorbalance=rs=0.08:gs=0.04:bs=-0.08,eq=saturation=1.08",

    # Cool - blue/teal coolness (8000K)
    "cool": "colortemperature=8000,colorbalance=rs=-0.05:gs=0:bs=0.08,eq=saturation=0.95",

    # Golden Hour - sunset warmth with lifted shadows
    "golden_hour": "colortemperature=5500,colorbalance=rs=0.12:gs=0.06:bs=-0.1,curves=m='0 0.05 0.5 0.5 1 1',eq=saturation=1.1",

    # Blue Hour - dawn/dusk with cool shadows
    "blue_hour": "colortemperature=9000,colorbalance=rs=-0.08:gs=0:bs=0.12,curves=m='0 0 0.5 0.48 1 0.95',eq=saturation=0.9",

    # Teal & Orange - Hollywood blockbuster complementary colors
    "teal_orange": "colorbalance=rs=-0.1:gs=-0.05:bs=0.15:rm=0.05:gm=0:bm=-0.05:rh=0.1:gh=0.05:bh=-0.1",

    # Cinematic - Teal & Orange with S-curve contrast
    "cinematic": "colorbalance=rs=-0.1:gs=-0.05:bs=0.15:rm=0.05:gm=0:bm=-0.05:rh=0.1:gh=0.05:bh=-0.1,curves=m='0 0 0.25 0.22 0.5 0.5 0.75 0.78 1 1'",

    # Blockbuster - high contrast cinematic
    "blockbuster": "colorbalance=rs=-0.12:gs=-0.06:bs=0.18:rm=0.06:gm=0:bm=-0.06:rh=0.12:gh=0.06:bh=-0.12,curves=m='0 0 0.2 0.15 0.5 0.5 0.8 0.85 1 1',eq=contrast=1.1",

    # Vibrant - punchy, saturated (MTV, action)
    "vibrant": "eq=saturation=1.25:contrast=1.1,unsharp=3:3:0.4",

    # Desaturated - muted, filmic (documentary, minimalist)
    "desaturated": "eq=saturation=0.75:contrast=1.05",

    # High Contrast - strong blacks/whites
    "high_contrast": "eq=contrast=1.3:brightness=0.02,curves=m='0 0 0.15 0.05 0.5 0.5 0.85 0.95 1 1'",

    # Filmic Warm - classic film warmth (wedding)
    "filmic_warm": "colorbalance=rs=0.08:gs=0.04:bs=-0.08,curves=m='0 0 0.25 0.22 0.5 0.5 0.75 0.78 1 1',eq=saturation=0.92",

    # Vintage - faded film look with lifted blacks
    "vintage": "curves=m='0 0.05 0.5 0.5 1 0.95',eq=saturation=0.8,colorbalance=rs=0.1:gs=0.05:bs=-0.05",

    # Noir - desaturated high contrast (B&W friendly)
    "noir": "eq=saturation=0.3:contrast=1.4,curves=m='0 0 0.2 0.1 0.5 0.5 0.8 0.9 1 1'",

    # Documentary - natural with slight sharpening
    "documentary": "eq=saturation=1.0:contrast=1.02,unsharp=3:3:0.3",

    # Natural - minimal processing, true to source
    "natural": "eq=saturation=1.02:contrast=1.01",
}

# Aliases for style template compatibility
PRESET_ALIASES: Dict[str, str] = {
    "cinematic_teal_orange": "cinematic",
    "film_fade": "vintage",
    "warm_vintage": "filmic_warm",
    "cold": "cool",
    "punch": "vibrant",
    "muted": "desaturated",
    "low_contrast": "neutral",
}


@dataclass
class ColorGradeConfig:
    """
    Configuration for color grading operation.

    Attributes:
        preset: Built-in preset name or "custom" for LUT
        intensity: Grade strength 0.0-1.0 (0.7 recommended, 1.0 = full)
        lut_path: Optional path to .cube/.3dl LUT file
        normalize_first: Apply broadcast safe levels before grading
    """
    preset: str = "teal_orange"
    intensity: float = 1.0
    lut_path: Optional[str] = None
    normalize_first: bool = True

    def __post_init__(self):
        # Resolve aliases
        if self.preset in PRESET_ALIASES:
            self.preset = PRESET_ALIASES[self.preset]

        # Clamp intensity
        self.intensity = max(0.0, min(1.0, self.intensity))


# =============================================================================
# LUT File Support
# =============================================================================

SUPPORTED_LUT_EXTENSIONS = {".cube", ".3dl", ".dat"}


def find_lut_file(name: str, lut_dir: Path) -> Optional[Path]:
    """
    Find a LUT file by name in the LUT directory.

    Args:
        name: LUT name (with or without extension)
        lut_dir: Directory to search

    Returns:
        Path to LUT file if found, None otherwise
    """
    if not lut_dir.exists():
        return None

    # Try exact match first
    for ext in SUPPORTED_LUT_EXTENSIONS:
        lut_path = lut_dir / f"{name}{ext}"
        if lut_path.exists():
            return lut_path

    # Try with extension already included
    lut_path = lut_dir / name
    if lut_path.exists() and lut_path.suffix.lower() in SUPPORTED_LUT_EXTENSIONS:
        return lut_path

    return None


def list_available_luts(lut_dir: Path) -> List[str]:
    """
    List all available LUT files in directory.

    Args:
        lut_dir: Directory to scan

    Returns:
        List of LUT names (without extension)
    """
    if not lut_dir.exists():
        return []

    luts = []
    for file in lut_dir.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_LUT_EXTENSIONS:
            luts.append(file.stem)

    return sorted(luts)


# =============================================================================
# Filter Chain Builder
# =============================================================================

def build_color_grade_filter(
    config: ColorGradeConfig,
    lut_dir: Optional[Path] = None
) -> str:
    """
    Build FFmpeg filter chain for color grading.

    Args:
        config: Color grading configuration
        lut_dir: Optional directory for LUT file lookup

    Returns:
        FFmpeg filter string (empty if no grading needed)
    """
    filters = []

    # 1. Normalize to broadcast safe levels first (recommended workflow)
    if config.normalize_first and config.preset != "none":
        filters.append(PRESET_FILTERS["neutral"])

    # 2. Apply LUT if specified
    if config.lut_path:
        lut_path = Path(config.lut_path)

        # Try to find LUT in directory if not absolute path
        if not lut_path.is_absolute() and lut_dir:
            found = find_lut_file(config.lut_path, lut_dir)
            if found:
                lut_path = found

        if lut_path.exists():
            # Apply LUT with intensity blending if < 1.0
            if config.intensity < 1.0:
                # Use split/blend for partial LUT application
                # This creates: original -> [split] -> lut3d -> [blend with original]
                filters.append(f"split[a][b];[a]lut3d={lut_path}[graded];[b][graded]blend=all_expr='A*{1-config.intensity}+B*{config.intensity}'")
            else:
                filters.append(f"lut3d={lut_path}")

            logger.debug(f"Applied LUT: {lut_path}")
            return ",".join(f for f in filters if f)

    # 3. Apply preset filter
    preset_filter = PRESET_FILTERS.get(config.preset, "")

    if preset_filter:
        # Apply intensity scaling for presets that support it
        if config.intensity < 1.0 and "eq=" in preset_filter:
            # Scale eq parameters by intensity
            preset_filter = _scale_eq_intensity(preset_filter, config.intensity)

        filters.append(preset_filter)
        logger.debug(f"Applied preset '{config.preset}' at {config.intensity:.0%} intensity")

    return ",".join(f for f in filters if f)


def _scale_eq_intensity(filter_str: str, intensity: float) -> str:
    """
    Scale eq filter parameters by intensity factor.

    This allows partial application of saturation/contrast adjustments.
    """
    import re

    def scale_param(match):
        param = match.group(1)
        value = float(match.group(2))

        # Scale deviation from neutral (1.0)
        if param in ("saturation", "contrast"):
            scaled = 1.0 + (value - 1.0) * intensity
            return f"{param}={scaled:.2f}"
        elif param == "brightness":
            scaled = value * intensity
            return f"{param}={scaled:.3f}"

        return match.group(0)

    return re.sub(r'(saturation|contrast|brightness)=([\d.]+)', scale_param, filter_str)


# =============================================================================
# UI Helpers
# =============================================================================

def get_preset_display_info() -> List[Dict[str, str]]:
    """
    Get display information for all presets (for UI dropdowns).

    Returns:
        List of dicts with 'id', 'name', 'description', 'category'
    """
    presets = [
        # No grading
        {"id": "none", "name": "None", "description": "No color grading applied", "category": "Basic"},
        {"id": "neutral", "name": "Neutral", "description": "Broadcast safe levels only", "category": "Basic"},
        {"id": "natural", "name": "Natural", "description": "Minimal processing, true to source", "category": "Basic"},

        # Temperature
        {"id": "warm", "name": "Warm", "description": "Golden sunset warmth", "category": "Temperature"},
        {"id": "cool", "name": "Cool", "description": "Blue/teal coolness", "category": "Temperature"},
        {"id": "golden_hour", "name": "Golden Hour", "description": "Sunset warmth with lifted shadows", "category": "Temperature"},
        {"id": "blue_hour", "name": "Blue Hour", "description": "Dawn/dusk cool tones", "category": "Temperature"},

        # Cinematic
        {"id": "teal_orange", "name": "Teal & Orange", "description": "Hollywood blockbuster look", "category": "Cinematic"},
        {"id": "cinematic", "name": "Cinematic", "description": "Teal & Orange with S-curve contrast", "category": "Cinematic"},
        {"id": "blockbuster", "name": "Blockbuster", "description": "High contrast action movie", "category": "Cinematic"},

        # Stylized
        {"id": "vibrant", "name": "Vibrant", "description": "Punchy, saturated colors", "category": "Stylized"},
        {"id": "desaturated", "name": "Desaturated", "description": "Muted, filmic look", "category": "Stylized"},
        {"id": "high_contrast", "name": "High Contrast", "description": "Strong blacks and whites", "category": "Stylized"},
        {"id": "vintage", "name": "Vintage", "description": "Faded film with lifted blacks", "category": "Stylized"},
        {"id": "filmic_warm", "name": "Filmic Warm", "description": "Classic warm film look", "category": "Stylized"},
        {"id": "noir", "name": "Noir", "description": "Desaturated high contrast", "category": "Stylized"},

        # Documentary
        {"id": "documentary", "name": "Documentary", "description": "Natural with slight sharpening", "category": "Documentary"},
    ]

    return presets


def get_preset_categories() -> Dict[str, List[str]]:
    """
    Get presets organized by category.

    Returns:
        Dict mapping category name to list of preset IDs
    """
    categories = {
        "Basic": ["none", "neutral", "natural"],
        "Temperature": ["warm", "cool", "golden_hour", "blue_hour"],
        "Cinematic": ["teal_orange", "cinematic", "blockbuster"],
        "Stylized": ["vibrant", "desaturated", "high_contrast", "vintage", "filmic_warm", "noir"],
        "Documentary": ["documentary"],
    }
    return categories


# =============================================================================
# Style Template Mapping
# =============================================================================

# Maps style template color_grading values to our presets
STYLE_TO_PRESET: Dict[str, str] = {
    # Direct mappings
    "none": "none",
    "neutral": "neutral",
    "warm": "warm",
    "cool": "cool",
    "teal_orange": "teal_orange",
    "vibrant": "vibrant",
    "desaturated": "desaturated",
    "high_contrast": "high_contrast",

    # Style template specific mappings
    "cinematic_teal_orange": "cinematic",
    "filmic_warm": "filmic_warm",
    "golden_hour": "golden_hour",
    "blue_hour": "blue_hour",
    "vintage": "vintage",
    "noir": "noir",
    "documentary": "documentary",

    # Legacy/compatibility mappings
    "film_fade": "vintage",
    "blockbuster": "blockbuster",
    "punch": "vibrant",
    "muted": "desaturated",
    "cold": "cool",
    "natural": "natural",
}


def resolve_style_color_grade(style_value: str) -> str:
    """
    Resolve style template color_grading value to our preset.

    Args:
        style_value: Value from style template's effects.color_grading

    Returns:
        Resolved preset name (defaults to "teal_orange" if unknown)
    """
    normalized = style_value.lower().strip()
    return STYLE_TO_PRESET.get(normalized, "teal_orange")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ColorGradePreset",
    # Config
    "ColorGradeConfig",
    # Constants
    "PRESET_FILTERS",
    "PRESET_ALIASES",
    "SUPPORTED_LUT_EXTENSIONS",
    # Functions
    "build_color_grade_filter",
    "find_lut_file",
    "list_available_luts",
    "get_preset_display_info",
    "get_preset_categories",
    "resolve_style_color_grade",
]
