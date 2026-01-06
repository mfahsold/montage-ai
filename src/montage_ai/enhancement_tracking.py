"""
Enhancement Tracking Module for Montage AI

Tracks all AI-driven enhancement decisions for professional NLE export.
Enables non-destructive workflows where editors can see and adjust AI decisions.

Key Classes:
- EnhancementDecision: Per-clip record of all enhancements applied
- EnhancementTracker: Collects decisions during pipeline execution
- Parameter classes: Detailed settings for each enhancement type

Usage:
    tracker = EnhancementTracker()
    decision = tracker.record_enhancement(
        clip_id="clip_001",
        source_path="/path/to/video.mp4",
        timeline_in=0.0,
        timeline_out=5.0,
    )
    decision.record_denoise(DenoiseConfig(spatial_strength=0.3))
    decision.record_stabilize(StabilizeParams(method="vidstab"))

    # Export with timeline
    exporter.export_with_enhancements(timeline, tracker.get_decisions())
"""

import hashlib
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from .clip_enhancement import DenoiseConfig, SharpenConfig, FilmGrainConfig


# =============================================================================
# Enums
# =============================================================================

class EnhancementMethod(Enum):
    """Enhancement method identifiers."""
    # Stabilization
    VIDSTAB = "vidstab"
    DESHAKE = "deshake"
    CGPU_STABILIZE = "cgpu_stabilize"

    # Upscaling
    REALESRGAN = "realesrgan"
    LANCZOS = "lanczos"
    CGPU_UPSCALE = "cgpu_upscale"

    # Color
    FFMPEG_FILTER = "ffmpeg_filter"
    LUT = "lut"
    COLOR_MATCH = "color_match"


# =============================================================================
# Parameter Data Classes
# =============================================================================

@dataclass
class StabilizeParams:
    """Parameters for video stabilization."""
    method: str = "vidstab"  # vidstab, deshake, cgpu
    smoothing: int = 30  # Frame window for smoothing
    crop_mode: str = "black"  # black, crop, fill
    zoom: float = 0.0  # Additional zoom to hide borders
    shakiness: int = 5  # vidstab: motion detection sensitivity (1-10)
    accuracy: int = 15  # vidstab: motion analysis accuracy (1-15)

    def to_resolve_params(self) -> Dict[str, Any]:
        """Generate DaVinci Resolve Stabilizer parameters."""
        mode_map = {"vidstab": "Perspective", "deshake": "Translation"}
        return {
            "node": "Stabilizer",
            "mode": mode_map.get(self.method, "Perspective"),
            "smoothing": self.smoothing / 100.0,  # Resolve uses 0-1
            "crop_ratio": 1.0 if self.crop_mode == "crop" else 0.0,
            "zoom": self.zoom,
        }

    def to_premiere_params(self) -> Dict[str, Any]:
        """Generate Premiere Pro Warp Stabilizer parameters."""
        return {
            "effect": "Warp Stabilizer",
            "result": "Smooth Motion" if self.smoothing > 20 else "No Motion",
            "smoothness": min(100, self.smoothing),
            "method": "Position, Scale, Rotation" if self.method == "vidstab" else "Position",
            "framing": "Stabilize, Crop" if self.crop_mode == "crop" else "Stabilize Only",
        }

    def to_recipe_text(self) -> str:
        """Human-readable recreation instructions."""
        return (
            f"Stabilization ({self.method})\n"
            f"  - Smoothing: {self.smoothing} frames\n"
            f"  - Crop Mode: {self.crop_mode}\n"
            f"  - Zoom: {self.zoom}%"
        )


@dataclass
class UpscaleParams:
    """Parameters for video upscaling."""
    method: str = "lanczos"  # realesrgan, lanczos, cgpu
    scale_factor: int = 2  # 2x or 4x
    model: str = "realesrgan-x4plus"  # AI model name
    crf: int = 18  # Output quality

    def to_resolve_params(self) -> Dict[str, Any]:
        """Generate DaVinci Resolve Super Scale parameters."""
        return {
            "node": "Super Scale",
            "scale": f"{self.scale_factor}x",
            "sharpness": "Enhanced" if self.method == "realesrgan" else "Medium",
            "noise_reduction": "Low" if self.method == "realesrgan" else "None",
        }

    def to_premiere_params(self) -> Dict[str, Any]:
        """Generate Premiere Pro scale parameters."""
        return {
            "effect": "Motion > Scale",
            "scale": self.scale_factor * 100,
            "scale_algorithm": "Bicubic Sharper",
            "note": f"Original AI model: {self.model}",
        }

    def to_recipe_text(self) -> str:
        return (
            f"Upscaling ({self.method})\n"
            f"  - Scale: {self.scale_factor}x\n"
            f"  - Model: {self.model}\n"
            f"  - Quality CRF: {self.crf}"
        )


@dataclass
class ColorGradeParams:
    """Parameters for color grading."""
    preset: str = "none"  # cinematic, teal_orange, etc.
    intensity: float = 0.7  # 0.0-1.0
    lut_path: Optional[str] = None  # Path to LUT file if used

    # Primary color wheels (for NLE recreation)
    lift: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Shadows RGB
    gamma: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Midtones RGB
    gain: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Highlights RGB
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Overall offset

    # Basic adjustments
    saturation: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    temperature: float = 0.0  # -1.0 (cool) to 1.0 (warm)
    tint: float = 0.0  # -1.0 (green) to 1.0 (magenta)

    def to_resolve_params(self) -> Dict[str, Any]:
        """Generate DaVinci Resolve Color page parameters."""
        return {
            "node": "Primary Color Wheels",
            "lift": {"r": self.lift[0], "g": self.lift[1], "b": self.lift[2]},
            "gamma": {"r": self.gamma[0], "g": self.gamma[1], "b": self.gamma[2]},
            "gain": {"r": self.gain[0], "g": self.gain[1], "b": self.gain[2]},
            "saturation": self.saturation * 100,
            "contrast": self.contrast,
            "pivot": 0.5,
            "lut": self.lut_path,
        }

    def to_premiere_params(self) -> Dict[str, Any]:
        """Generate Premiere Pro Lumetri Color parameters."""
        return {
            "effect": "Lumetri Color",
            "basic": {
                "temperature": self.temperature * 100,
                "tint": self.tint * 100,
                "saturation": self.saturation * 100,
                "contrast": (self.contrast - 1.0) * 100,
            },
            "creative": {
                "look": self.preset if self.preset != "none" else None,
                "intensity": self.intensity * 100,
            },
            "color_wheels": {
                "shadows": self.lift,
                "midtones": self.gamma,
                "highlights": self.gain,
            },
        }

    def to_recipe_text(self) -> str:
        lines = [
            f"Color Grading (Preset: {self.preset})",
            f"  - Intensity: {self.intensity:.0%}",
            f"  - Saturation: {self.saturation:.0%}",
            f"  - Contrast: {self.contrast:.0%}",
        ]
        if self.lut_path:
            lines.append(f"  - LUT: {os.path.basename(self.lut_path)}")
        if any(v != 0 for v in self.lift):
            lines.append(f"  - Lift (Shadows): R={self.lift[0]:.2f} G={self.lift[1]:.2f} B={self.lift[2]:.2f}")
        if any(v != 1 for v in self.gamma):
            lines.append(f"  - Gamma (Mids): R={self.gamma[0]:.2f} G={self.gamma[1]:.2f} B={self.gamma[2]:.2f}")
        if any(v != 1 for v in self.gain):
            lines.append(f"  - Gain (Highs): R={self.gain[0]:.2f} G={self.gain[1]:.2f} B={self.gain[2]:.2f}")
        return "\n".join(lines)


@dataclass
class ColorMatchParams:
    """Parameters for shot-to-shot color matching."""
    reference_clip: str  # Path to reference clip
    method: str = "mkl"  # mkl, hm, mvgd (color-matcher algorithms)
    r_adjustment: float = 0.0  # Red channel adjustment
    g_adjustment: float = 0.0  # Green channel adjustment
    b_adjustment: float = 0.0  # Blue channel adjustment

    def to_resolve_params(self) -> Dict[str, Any]:
        """Generate DaVinci Resolve Color Match parameters."""
        return {
            "node": "Shot Match",
            "reference": os.path.basename(self.reference_clip),
            "method": "ColorMatch" if self.method == "mkl" else "Manual",
            "adjustments": {
                "red": self.r_adjustment,
                "green": self.g_adjustment,
                "blue": self.b_adjustment,
            },
        }

    def to_recipe_text(self) -> str:
        return (
            f"Color Match (to: {os.path.basename(self.reference_clip)})\n"
            f"  - Method: {self.method}\n"
            f"  - R adjust: {self.r_adjustment:+.3f}\n"
            f"  - G adjust: {self.g_adjustment:+.3f}\n"
            f"  - B adjust: {self.b_adjustment:+.3f}"
        )


@dataclass
class DialogueDuckingParams:
    """Parameters for dialogue detection and music auto-ducking."""
    voice_track: Optional[str] = None  # Path to voice/dialogue track
    duck_level_db: float = -12.0  # Volume reduction in dB during speech
    attack_time: float = 0.15  # Ramp down time in seconds
    release_time: float = 0.30  # Ramp up time in seconds
    speech_segments_count: int = 0  # Number of detected speech segments
    total_speech_duration: float = 0.0  # Total speech time in seconds
    speech_percentage: float = 0.0  # Percentage of audio with speech
    detection_method: str = "auto"  # silero, webrtc, ffmpeg, auto
    keyframes_count: int = 0  # Number of generated automation keyframes

    def to_resolve_params(self) -> Dict[str, Any]:
        """Generate DaVinci Resolve Fairlight parameters."""
        return {
            "node": "Fairlight > Volume Automation",
            "duck_level": f"{self.duck_level_db:.1f} dB",
            "attack": f"{self.attack_time:.2f}s",
            "release": f"{self.release_time:.2f}s",
            "keyframes": self.keyframes_count,
            "note": f"Detected {self.speech_segments_count} speech segments ({self.speech_percentage:.1f}%)",
        }

    def to_premiere_params(self) -> Dict[str, Any]:
        """Generate Premiere Pro Essential Sound parameters."""
        return {
            "effect": "Essential Sound > Auto Ducking",
            "duck_amount": abs(self.duck_level_db),
            "sensitivity": "Medium" if self.detection_method == "silero" else "Low",
            "fade_time": self.attack_time * 1000,  # Convert to ms
            "note": f"Speech: {self.total_speech_duration:.1f}s ({self.speech_percentage:.1f}%)",
        }

    def to_recipe_text(self) -> str:
        return (
            f"Dialogue Ducking (Auto-detected)\n"
            f"  - Duck Level: {self.duck_level_db:.1f} dB\n"
            f"  - Attack: {self.attack_time:.2f}s\n"
            f"  - Release: {self.release_time:.2f}s\n"
            f"  - Speech Segments: {self.speech_segments_count}\n"
            f"  - Speech Duration: {self.total_speech_duration:.1f}s ({self.speech_percentage:.1f}%)\n"
            f"  - Detection Method: {self.detection_method}\n"
            f"  - Keyframes: {self.keyframes_count}"
        )


# =============================================================================
# Analysis Data (Why AI Made Decisions)
# =============================================================================

@dataclass
class ClipAnalysis:
    """Analysis results that drove enhancement decisions."""
    # Motion analysis
    shake_score: float = 0.0  # 0-1, higher = more shake
    motion_type: str = "static"  # static, pan, handheld, action

    # Image quality
    noise_level: float = 0.0  # 0-1, estimated noise
    sharpness_score: float = 0.5  # 0-1, edge sharpness
    brightness_avg: float = 128.0  # 0-255
    is_dark: bool = False
    is_bright: bool = False

    # Content
    scene_type: str = "unknown"  # establishing, action, detail, portrait
    dominant_colors: List[str] = field(default_factory=list)
    has_faces: bool = False
    has_text: bool = False


# =============================================================================
# Main Enhancement Decision Class
# =============================================================================

@dataclass
class EnhancementDecision:
    """
    Complete record of all enhancements applied to a single clip.

    This is the core tracking unit. Each clip in the timeline gets one
    EnhancementDecision that records what was done and why.
    """
    # Identification
    clip_id: str  # Unique ID (hash of source + timecodes)
    source_path: str  # Original source file
    timeline_in: float  # Start time in timeline (seconds)
    timeline_out: float  # End time in timeline (seconds)

    # Enhancement flags (quick lookup)
    stabilized: bool = False
    upscaled: bool = False
    denoised: bool = False
    sharpened: bool = False
    color_graded: bool = False
    color_matched: bool = False
    film_grain_added: bool = False
    brightness_adjusted: bool = False
    dialogue_ducking_applied: bool = False

    # Detailed parameters (for NLE recreation)
    stabilize_params: Optional[StabilizeParams] = None
    upscale_params: Optional[UpscaleParams] = None
    denoise_params: Optional[DenoiseConfig] = None
    sharpen_params: Optional[SharpenConfig] = None
    color_grade_params: Optional[ColorGradeParams] = None
    color_match_params: Optional[ColorMatchParams] = None
    film_grain_params: Optional[FilmGrainConfig] = None
    dialogue_ducking_params: Optional[DialogueDuckingParams] = None

    # Analysis (why decisions were made)
    analysis: Optional[ClipAnalysis] = None
    ai_reasoning: Optional[str] = None  # LLM explanation of decisions

    # Timestamps
    decided_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None

    # Output tracking
    enhanced_path: Optional[str] = None  # Path to enhanced file (if rendered)

    # --- Recording Methods ---

    def record_stabilize(self, params: StabilizeParams) -> None:
        """Record that stabilization was applied."""
        self.stabilized = True
        self.stabilize_params = params
        self.applied_at = datetime.now()

    def record_upscale(self, params: UpscaleParams) -> None:
        """Record that upscaling was applied."""
        self.upscaled = True
        self.upscale_params = params
        self.applied_at = datetime.now()

    def record_denoise(self, params: DenoiseConfig) -> None:
        """Record that denoising was applied."""
        self.denoised = True
        self.denoise_params = params
        self.applied_at = datetime.now()

    def record_sharpen(self, params: SharpenConfig) -> None:
        """Record that sharpening was applied."""
        self.sharpened = True
        self.sharpen_params = params
        self.applied_at = datetime.now()

    def record_color_grade(self, params: ColorGradeParams) -> None:
        """Record that color grading was applied."""
        self.color_graded = True
        self.color_grade_params = params
        self.applied_at = datetime.now()

    def record_color_match(self, params: ColorMatchParams) -> None:
        """Record that color matching was applied."""
        self.color_matched = True
        self.color_match_params = params
        self.applied_at = datetime.now()

    def record_film_grain(self, params: FilmGrainConfig) -> None:
        """Record that film grain was added."""
        self.film_grain_added = True
        self.film_grain_params = params
        self.applied_at = datetime.now()

    def record_dialogue_ducking(self, params: DialogueDuckingParams) -> None:
        """Record that dialogue ducking was applied."""
        self.dialogue_ducking_applied = True
        self.dialogue_ducking_params = params
        self.applied_at = datetime.now()

    def record_analysis(self, analysis: ClipAnalysis, reasoning: Optional[str] = None) -> None:
        """Record analysis results and AI reasoning."""
        self.analysis = analysis
        self.ai_reasoning = reasoning

    # --- Export Methods ---

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "clip_id": self.clip_id,
            "source_path": self.source_path,
            "timeline_in": self.timeline_in,
            "timeline_out": self.timeline_out,
            "enhancements": {
                "stabilized": self.stabilized,
                "upscaled": self.upscaled,
                "denoised": self.denoised,
                "sharpened": self.sharpened,
                "color_graded": self.color_graded,
                "color_matched": self.color_matched,
                "film_grain_added": self.film_grain_added,
                "dialogue_ducking_applied": self.dialogue_ducking_applied,
            },
            "params": {
                "stabilize": asdict(self.stabilize_params) if self.stabilize_params else None,
                "upscale": asdict(self.upscale_params) if self.upscale_params else None,
                "denoise": asdict(self.denoise_params) if self.denoise_params else None,
                "sharpen": asdict(self.sharpen_params) if self.sharpen_params else None,
                "color_grade": asdict(self.color_grade_params) if self.color_grade_params else None,
                "color_match": asdict(self.color_match_params) if self.color_match_params else None,
                "film_grain": asdict(self.film_grain_params) if self.film_grain_params else None,
                "dialogue_ducking": asdict(self.dialogue_ducking_params) if self.dialogue_ducking_params else None,
            },
            "analysis": asdict(self.analysis) if self.analysis else None,
            "ai_reasoning": self.ai_reasoning,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }

    def to_edl_comments(self) -> List[str]:
        """Generate EDL comment lines for this clip's enhancements."""
        comments = []
        if self.stabilized and self.stabilize_params:
            p = self.stabilize_params
            comments.append(f"* MONTAGE_AI STABILIZE: {p.method} smoothing={p.smoothing} crop={p.crop_mode}")
        if self.denoised and self.denoise_params:
            p = self.denoise_params
            comments.append(f"* MONTAGE_AI DENOISE: spatial={p.spatial_strength:.2f} temporal={p.temporal_strength:.2f}")
        if self.sharpened and self.sharpen_params:
            p = self.sharpen_params
            comments.append(f"* MONTAGE_AI SHARPEN: amount={p.amount:.2f} radius={p.radius:.1f}")
        if self.color_graded and self.color_grade_params:
            p = self.color_grade_params
            comments.append(f"* MONTAGE_AI COLOR_GRADE: {p.preset} intensity={p.intensity:.2f}")
        if self.color_matched and self.color_match_params:
            p = self.color_match_params
            comments.append(f"* MONTAGE_AI COLOR_MATCH: ref={os.path.basename(p.reference_clip)}")
        if self.upscaled and self.upscale_params:
            p = self.upscale_params
            comments.append(f"* MONTAGE_AI UPSCALE: {p.method} {p.scale_factor}x model={p.model}")
        if self.film_grain_added and self.film_grain_params:
            p = self.film_grain_params
            comments.append(f"* MONTAGE_AI FILM_GRAIN: {p.grain_type} intensity={p.intensity:.2f}")
        if self.ai_reasoning:
            # Truncate long reasoning for EDL
            reason = self.ai_reasoning[:100].replace("\n", " ")
            comments.append(f"* MONTAGE_AI REASONING: {reason}")
        return comments

    def to_recipe_markdown(self) -> str:
        """Generate human-readable recipe card section."""
        lines = [
            f"## Clip: {os.path.basename(self.source_path)}",
            f"Timeline: {self._format_timecode(self.timeline_in)} - {self._format_timecode(self.timeline_out)}",
            "",
            "### Applied Enhancements:",
        ]

        checks = [
            ("Stabilization", self.stabilized),
            ("Upscaling", self.upscaled),
            ("Denoising", self.denoised),
            ("Sharpening", self.sharpened),
            ("Color Grading", self.color_graded),
            ("Color Matching", self.color_matched),
            ("Film Grain", self.film_grain_added),
        ]
        for name, applied in checks:
            mark = "x" if applied else " "
            lines.append(f"- [{mark}] {name}")

        # DaVinci Resolve instructions
        lines.extend(["", "### DaVinci Resolve Recreation:"])
        step = 1
        if self.stabilized and self.stabilize_params:
            lines.append(f"{step}. **Stabilizer** (Color Page > Tracker)")
            p = self.stabilize_params.to_resolve_params()
            lines.append(f"   - Mode: {p['mode']}")
            lines.append(f"   - Smoothing: {p['smoothing']:.2f}")
            step += 1

        if self.denoised and self.denoise_params:
            lines.append(f"{step}. **Noise Reduction** (Color Page > Spatial NR)")
            lines.append(f"   - Luma Threshold: {self.denoise_params.spatial_strength * 10:.1f}")
            lines.append(f"   - Chroma Threshold: {self.denoise_params.chroma_strength * 10:.1f}")
            step += 1

        if self.sharpened and self.sharpen_params:
            lines.append(f"{step}. **Sharpening** (Color Page > Blur/Sharpen)")
            lines.append(f"   - Amount: {self.sharpen_params.amount:.2f}")
            lines.append(f"   - Radius: {self.sharpen_params.radius:.1f}")
            step += 1

        if self.color_graded and self.color_grade_params:
            lines.append(f"{step}. **Color Wheels** (Color Page > Primary)")
            lines.append(self.color_grade_params.to_recipe_text().replace("\n", "\n   "))
            step += 1

        # AI Reasoning
        if self.ai_reasoning:
            lines.extend([
                "",
                "### AI Reasoning:",
                f"> {self.ai_reasoning}",
            ])

        return "\n".join(lines)

    def _format_timecode(self, seconds: float, fps: float = 30.0) -> str:
        """Format seconds as SMPTE timecode."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * fps)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


# =============================================================================
# Enhancement Tracker (Collects Decisions During Pipeline)
# =============================================================================

class EnhancementTracker:
    """
    Collects and manages enhancement decisions across the montage pipeline.

    Usage:
        tracker = EnhancementTracker()

        # During clip processing
        decision = tracker.create_decision(clip_path, timeline_in, timeline_out)
        decision.record_stabilize(StabilizeParams(...))

        # At export time
        for decision in tracker.get_decisions():
            exporter.add_enhancement_metadata(decision)
    """

    def __init__(self):
        self._decisions: Dict[str, EnhancementDecision] = {}

    @staticmethod
    def _generate_clip_id(source_path: str, timeline_in: float, timeline_out: float) -> str:
        """Generate unique clip ID from source path and timecodes."""
        key = f"{source_path}:{timeline_in:.3f}:{timeline_out:.3f}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def create_decision(
        self,
        source_path: str,
        timeline_in: float,
        timeline_out: float
    ) -> EnhancementDecision:
        """Create a new enhancement decision for a clip."""
        clip_id = self._generate_clip_id(source_path, timeline_in, timeline_out)

        decision = EnhancementDecision(
            clip_id=clip_id,
            source_path=source_path,
            timeline_in=timeline_in,
            timeline_out=timeline_out,
        )
        self._decisions[clip_id] = decision
        return decision

    def get_decision(self, clip_id: str) -> Optional[EnhancementDecision]:
        """Get an existing decision by clip ID."""
        return self._decisions.get(clip_id)

    def get_decision_for_clip(
        self,
        source_path: str,
        timeline_in: float,
        timeline_out: float
    ) -> Optional[EnhancementDecision]:
        """Get decision by source path and timecodes."""
        clip_id = self._generate_clip_id(source_path, timeline_in, timeline_out)
        return self._decisions.get(clip_id)

    def get_decisions(self) -> List[EnhancementDecision]:
        """Get all recorded decisions, ordered by timeline position."""
        return sorted(self._decisions.values(), key=lambda d: d.timeline_in)

    def get_enhanced_clips_count(self) -> int:
        """Count how many clips have at least one enhancement."""
        return sum(
            1 for d in self._decisions.values()
            if any([
                d.stabilized, d.upscaled, d.denoised, d.sharpened,
                d.color_graded, d.color_matched, d.film_grain_added
            ])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export all decisions as dictionary."""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "clip_count": len(self._decisions),
            "enhanced_count": self.get_enhanced_clips_count(),
            "decisions": [d.to_dict() for d in self.get_decisions()],
        }

    def to_json(self) -> str:
        """Export all decisions as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    def generate_recipe_card(self, project_name: str = "Montage AI Export") -> str:
        """Generate complete recipe card markdown for all clips."""
        lines = [
            f"# Enhancement Recipe Card - {project_name}",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Clips: {len(self._decisions)}",
            f"Enhanced Clips: {self.get_enhanced_clips_count()}",
            "",
            "---",
            "",
        ]

        for i, decision in enumerate(self.get_decisions(), 1):
            lines.append(f"## Clip {i}")
            lines.append(decision.to_recipe_markdown())
            lines.append("")
            lines.append("---")
            lines.append("")

        lines.extend([
            "## About This Export",
            "",
            "This recipe card was generated by Montage AI to help you recreate",
            "or adjust AI-driven enhancements in your professional NLE.",
            "",
            "The parameters above can be applied manually in:",
            "- **DaVinci Resolve**: Color Page nodes, Tracker, Super Scale",
            "- **Premiere Pro**: Lumetri Color, Warp Stabilizer, Effects",
            "- **Final Cut Pro**: Color Board, Stabilization, Effects",
            "",
            "For questions: https://github.com/fluxibri/montage-ai",
        ])

        return "\n".join(lines)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "EnhancementMethod",
    # Parameter classes
    "StabilizeParams",
    "UpscaleParams",
    "ColorGradeParams",
    "ColorMatchParams",
    "DialogueDuckingParams",
    # Analysis
    "ClipAnalysis",
    # Main classes
    "EnhancementDecision",
    "EnhancementTracker",
]
