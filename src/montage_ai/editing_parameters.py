"""
Unified editing parameter schema for AI director/editor control.

This module consolidates all tunable parameters across stabilization, color grading,
clip selection, and pacing domains. Designed to be LLM-friendly with clear field names,
documentation, and sensible defaults.

Following the "Polish, don't generate" philosophy - parameters control editing decisions,
not pixel generation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum


# ============================================================================
# STABILIZATION PARAMETERS
# ============================================================================

class StabilizationMethod(str, Enum):
    """Available stabilization methods."""
    VIDSTAB = "vidstab"
    DESHAKE = "deshake"
    CGPU = "cgpu"  # Cloud GPU acceleration
    DISABLED = "disabled"


class CropMode(str, Enum):
    """Crop behavior after stabilization."""
    BLACK = "black"  # Keep black borders
    KEEP = "keep"    # Crop to remove borders


@dataclass
class StabilizationParameters:
    """
    Stabilization tuning parameters for shake reduction.
    
    Attributes:
        method: Stabilization algorithm to use
        smoothing: Smoothness of camera motion (1-30, higher = smoother)
        shakiness: Shakiness detection sensitivity (1-10, higher = more aggressive)
        accuracy: Motion estimation accuracy (1-15, higher = slower but better)
        stepsize: Search step size for motion estimation (1-32, lower = more accurate)
        zoom: Zoom percentage to compensate motion (-100 to 100, 0 = no zoom)
        optzoom: Optimal zoom mode (0=disabled, 1=static, 2=adaptive)
        crop: How to handle borders after stabilization
    """
    method: StabilizationMethod = StabilizationMethod.VIDSTAB
    smoothing: int = 15  # Balanced default (range 1-30)
    shakiness: int = 5   # Medium sensitivity (range 1-10)
    accuracy: int = 10   # High accuracy (range 1-15)
    stepsize: int = 6    # Standard step size (range 1-32)
    zoom: int = 0        # No zoom by default (range -100 to 100)
    optzoom: int = 1     # Static optimal zoom (0-2)
    crop: CropMode = CropMode.KEEP  # Remove black borders
    
    def validate(self) -> None:
        """Validate parameter ranges."""
        assert 1 <= self.smoothing <= 30, f"smoothing must be 1-30, got {self.smoothing}"
        assert 1 <= self.shakiness <= 10, f"shakiness must be 1-10, got {self.shakiness}"
        assert 1 <= self.accuracy <= 15, f"accuracy must be 1-15, got {self.accuracy}"
        assert 1 <= self.stepsize <= 32, f"stepsize must be 1-32, got {self.stepsize}"
        assert -100 <= self.zoom <= 100, f"zoom must be -100 to 100, got {self.zoom}"
        assert 0 <= self.optzoom <= 2, f"optzoom must be 0-2, got {self.optzoom}"


# ============================================================================
# COLOR GRADING PARAMETERS
# ============================================================================

# Available presets from color_grading.py
COLOR_GRADING_PRESETS = [
    "teal_orange", "cinematic", "blockbuster", "vintage", "noir",
    "warm", "cool", "vibrant", "desaturated", "high_contrast",
    "filmic_warm", "filmic_cool", "bleach_bypass", "sepia",
    "cross_process", "technicolor", "moonlight", "golden_hour",
    "flat_log", "rec709"
]


@dataclass
class ColorGradingParameters:
    """
    Color grading parameters for cinematic look.
    
    Attributes:
        preset: Named color preset (e.g., "cinematic", "teal_orange")
        intensity: Strength of preset application (0.0-1.0)
        lut_path: Optional path to .cube or .3dl LUT file
        normalize_first: Apply normalization before grading
        temperature: Color temperature shift (-1.0 to 1.0, 0 = neutral)
        tint: Magenta/green shift (-1.0 to 1.0, 0 = neutral)
        saturation: Saturation multiplier (0.0-2.0, 1.0 = original)
        contrast: Contrast adjustment (0.0-2.0, 1.0 = original)
        brightness: Brightness shift (-1.0 to 1.0, 0 = original)
    """
    preset: Optional[str] = "cinematic"
    intensity: float = 0.8  # 80% strength by default
    lut_path: Optional[str] = None
    normalize_first: bool = False
    temperature: float = 0.0  # Neutral
    tint: float = 0.0  # Neutral
    saturation: float = 1.0  # Original saturation
    contrast: float = 1.0  # Original contrast
    brightness: float = 0.0  # Original brightness
    
    def validate(self) -> None:
        """Validate parameter ranges."""
        if self.preset and self.preset not in COLOR_GRADING_PRESETS:
            raise ValueError(f"Unknown preset: {self.preset}. Available: {COLOR_GRADING_PRESETS}")
        assert 0.0 <= self.intensity <= 1.0, f"intensity must be 0-1, got {self.intensity}"
        assert -1.0 <= self.temperature <= 1.0, f"temperature must be -1 to 1, got {self.temperature}"
        assert -1.0 <= self.tint <= 1.0, f"tint must be -1 to 1, got {self.tint}"
        assert 0.0 <= self.saturation <= 2.0, f"saturation must be 0-2, got {self.saturation}"
        assert 0.0 <= self.contrast <= 2.0, f"contrast must be 0-2, got {self.contrast}"
        assert -1.0 <= self.brightness <= 1.0, f"brightness must be -1 to 1, got {self.brightness}"


# ============================================================================
# CLIP SELECTION PARAMETERS
# ============================================================================

@dataclass
class ClipSelectionParameters:
    """
    Clip selection scoring and ranking parameters.
    
    Attributes:
        fresh_clip_bonus: Score bonus for unused clips (0-100)
        jump_cut_penalty: Score penalty for jump cuts (0-100)
        shot_variation_bonus: Score bonus for shot variety (0-50)
        shot_repetition_penalty: Score penalty for repeated shot types (0-50)
        energy_weight: Weight for energy matching (0.0-1.0)
        visual_weight: Weight for visual quality (0.0-1.0)
        semantic_weight: Weight for semantic matching (0.0-1.0)
        match_cut_threshold: Similarity threshold for match cuts (0.0-1.0)
        use_llm_ranking: Enable LLM-powered intelligent ranking
        llm_top_n: Number of candidates to send to LLM for ranking
    """
    fresh_clip_bonus: int = 50
    jump_cut_penalty: int = 50
    shot_variation_bonus: int = 10
    shot_repetition_penalty: int = 10
    energy_weight: float = 0.4
    visual_weight: float = 0.3
    semantic_weight: float = 0.3
    match_cut_threshold: float = 0.7
    use_llm_ranking: bool = True
    llm_top_n: int = 5
    
    def validate(self) -> None:
        """Validate parameter ranges."""
        assert 0 <= self.fresh_clip_bonus <= 100, f"fresh_clip_bonus must be 0-100"
        assert 0 <= self.jump_cut_penalty <= 100, f"jump_cut_penalty must be 0-100"
        assert 0 <= self.shot_variation_bonus <= 50, f"shot_variation_bonus must be 0-50"
        assert 0 <= self.shot_repetition_penalty <= 50, f"shot_repetition_penalty must be 0-50"
        assert 0.0 <= self.energy_weight <= 1.0, f"energy_weight must be 0-1"
        assert 0.0 <= self.visual_weight <= 1.0, f"visual_weight must be 0-1"
        assert 0.0 <= self.semantic_weight <= 1.0, f"semantic_weight must be 0-1"
        assert 0.0 <= self.match_cut_threshold <= 1.0, f"match_cut_threshold must be 0-1"
        assert self.llm_top_n > 0, f"llm_top_n must be positive"


# ============================================================================
# PACING PARAMETERS
# ============================================================================

class PacingSpeed(str, Enum):
    """Pacing speed presets."""
    VERY_FAST = "very_fast"  # 1 beat
    FAST = "fast"            # 2-4 beats
    MEDIUM = "medium"        # 4 beats
    SLOW = "slow"            # 8 beats
    VERY_SLOW = "very_slow"  # 16 beats
    DYNAMIC = "dynamic"      # Energy-adaptive


class PacingPattern(str, Enum):
    """Rhythmic pattern types."""
    STRAIGHT = "straight"              # Constant beat count
    FIBONACCI = "fibonacci"            # Fibonacci sequence
    SYNCOPATION = "syncopation"        # Off-beat variations
    THE_STUTTER = "the_stutter"        # Quick bursts
    THE_FADE = "the_fade"              # Gradual slowdown
    PHASE_AWARE = "phase_aware"        # Intro/build/climax/outro logic


@dataclass
class PacingParameters:
    """
    Pacing and rhythm parameters for cut timing.
    
    Attributes:
        speed: Base pacing speed (affects beats per cut)
        pattern: Rhythmic pattern to apply
        chaos_factor: Probability of random pattern injection (0.0-1.0)
        min_clip_duration: Minimum clip length in seconds
        max_clip_duration: Maximum clip length in seconds
        respect_beats: Sync cuts to audio beats
        energy_adaptive: Adjust pacing based on music energy
        phase_aware: Use intro/build/climax/outro logic
        fibonacci_sequences: Custom Fibonacci sequences for patterns
        override_beats_per_cut: Manual override for specific sections
    """
    speed: PacingSpeed = PacingSpeed.DYNAMIC
    pattern: PacingPattern = PacingPattern.PHASE_AWARE
    chaos_factor: float = 0.15  # 15% randomness
    min_clip_duration: float = 1.0  # 1 second
    max_clip_duration: float = 10.0  # 10 seconds
    respect_beats: bool = True
    energy_adaptive: bool = True
    phase_aware: bool = True
    fibonacci_sequences: List[List[int]] = field(default_factory=lambda: [[1, 1, 2, 3, 5], [8, 5, 3, 2, 1, 1]])
    override_beats_per_cut: Optional[int] = None
    
    def validate(self) -> None:
        """Validate parameter ranges."""
        assert 0.0 <= self.chaos_factor <= 1.0, f"chaos_factor must be 0-1"
        assert self.min_clip_duration > 0, f"min_clip_duration must be positive"
        assert self.max_clip_duration > self.min_clip_duration, "max > min duration"
        if self.override_beats_per_cut is not None:
            assert self.override_beats_per_cut > 0, "override_beats_per_cut must be positive"


# ============================================================================
# UNIFIED EDITING PARAMETERS
# ============================================================================

@dataclass
class EditingParameters:
    """
    Unified parameter schema for AI director/editor control.
    
    This consolidates all tunable parameters across post-production domains.
    Designed for LLM-based intelligent tuning while maintaining sensible defaults.
    
    Usage:
        params = EditingParameters()
        params.color_grading.preset = "teal_orange"
        params.color_grading.intensity = 0.9
        params.stabilization.smoothing = 20
        params.validate()  # Check all sub-parameters
    """
    stabilization: StabilizationParameters = field(default_factory=StabilizationParameters)
    color_grading: ColorGradingParameters = field(default_factory=ColorGradingParameters)
    clip_selection: ClipSelectionParameters = field(default_factory=ClipSelectionParameters)
    pacing: PacingParameters = field(default_factory=PacingParameters)
    
    def validate(self) -> None:
        """Validate all sub-parameter groups."""
        self.stabilization.validate()
        self.color_grading.validate()
        self.clip_selection.validate()
        self.pacing.validate()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stabilization": {
                "method": self.stabilization.method.value,
                "smoothing": self.stabilization.smoothing,
                "shakiness": self.stabilization.shakiness,
                "accuracy": self.stabilization.accuracy,
                "stepsize": self.stabilization.stepsize,
                "zoom": self.stabilization.zoom,
                "optzoom": self.stabilization.optzoom,
                "crop": self.stabilization.crop.value,
            },
            "color_grading": {
                "preset": self.color_grading.preset,
                "intensity": self.color_grading.intensity,
                "lut_path": self.color_grading.lut_path,
                "normalize_first": self.color_grading.normalize_first,
                "temperature": self.color_grading.temperature,
                "tint": self.color_grading.tint,
                "saturation": self.color_grading.saturation,
                "contrast": self.color_grading.contrast,
                "brightness": self.color_grading.brightness,
            },
            "clip_selection": {
                "fresh_clip_bonus": self.clip_selection.fresh_clip_bonus,
                "jump_cut_penalty": self.clip_selection.jump_cut_penalty,
                "shot_variation_bonus": self.clip_selection.shot_variation_bonus,
                "shot_repetition_penalty": self.clip_selection.shot_repetition_penalty,
                "energy_weight": self.clip_selection.energy_weight,
                "visual_weight": self.clip_selection.visual_weight,
                "semantic_weight": self.clip_selection.semantic_weight,
                "match_cut_threshold": self.clip_selection.match_cut_threshold,
                "use_llm_ranking": self.clip_selection.use_llm_ranking,
                "llm_top_n": self.clip_selection.llm_top_n,
            },
            "pacing": {
                "speed": self.pacing.speed.value,
                "pattern": self.pacing.pattern.value,
                "chaos_factor": self.pacing.chaos_factor,
                "min_clip_duration": self.pacing.min_clip_duration,
                "max_clip_duration": self.pacing.max_clip_duration,
                "respect_beats": self.pacing.respect_beats,
                "energy_adaptive": self.pacing.energy_adaptive,
                "phase_aware": self.pacing.phase_aware,
                "fibonacci_sequences": self.pacing.fibonacci_sequences,
                "override_beats_per_cut": self.pacing.override_beats_per_cut,
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EditingParameters":
        """Create from dictionary (e.g., JSON deserialization)."""
        return cls(
            stabilization=StabilizationParameters(
                method=StabilizationMethod(data["stabilization"]["method"]),
                smoothing=data["stabilization"]["smoothing"],
                shakiness=data["stabilization"]["shakiness"],
                accuracy=data["stabilization"]["accuracy"],
                stepsize=data["stabilization"]["stepsize"],
                zoom=data["stabilization"]["zoom"],
                optzoom=data["stabilization"]["optzoom"],
                crop=CropMode(data["stabilization"]["crop"]),
            ),
            color_grading=ColorGradingParameters(
                preset=data["color_grading"]["preset"],
                intensity=data["color_grading"]["intensity"],
                lut_path=data["color_grading"].get("lut_path"),
                normalize_first=data["color_grading"]["normalize_first"],
                temperature=data["color_grading"]["temperature"],
                tint=data["color_grading"]["tint"],
                saturation=data["color_grading"]["saturation"],
                contrast=data["color_grading"]["contrast"],
                brightness=data["color_grading"]["brightness"],
            ),
            clip_selection=ClipSelectionParameters(
                fresh_clip_bonus=data["clip_selection"]["fresh_clip_bonus"],
                jump_cut_penalty=data["clip_selection"]["jump_cut_penalty"],
                shot_variation_bonus=data["clip_selection"]["shot_variation_bonus"],
                shot_repetition_penalty=data["clip_selection"]["shot_repetition_penalty"],
                energy_weight=data["clip_selection"]["energy_weight"],
                visual_weight=data["clip_selection"]["visual_weight"],
                semantic_weight=data["clip_selection"]["semantic_weight"],
                match_cut_threshold=data["clip_selection"]["match_cut_threshold"],
                use_llm_ranking=data["clip_selection"]["use_llm_ranking"],
                llm_top_n=data["clip_selection"]["llm_top_n"],
            ),
            pacing=PacingParameters(
                speed=PacingSpeed(data["pacing"]["speed"]),
                pattern=PacingPattern(data["pacing"]["pattern"]),
                chaos_factor=data["pacing"]["chaos_factor"],
                min_clip_duration=data["pacing"]["min_clip_duration"],
                max_clip_duration=data["pacing"]["max_clip_duration"],
                respect_beats=data["pacing"]["respect_beats"],
                energy_adaptive=data["pacing"]["energy_adaptive"],
                phase_aware=data["pacing"]["phase_aware"],
                fibonacci_sequences=data["pacing"]["fibonacci_sequences"],
                override_beats_per_cut=data["pacing"].get("override_beats_per_cut"),
            )
        )
