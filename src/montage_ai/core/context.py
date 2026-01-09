from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
import numpy as np
from ..config import get_settings, Settings
from .models import EditingInstructions

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AudioAnalysisResult:
    """Results from audio analysis phase."""
    music_path: str
    beat_times: np.ndarray
    tempo: float
    energy_times: np.ndarray
    energy_values: np.ndarray
    duration: float
    sections: List[Any] = field(default_factory=list)

    @property
    def beat_count(self) -> int:
        """Number of detected beats."""
        return len(self.beat_times)

    @property
    def avg_energy(self) -> float:
        """Average energy level (0-1)."""
        if len(self.energy_values) > 0:
            return float(np.mean(self.energy_values))
        return 0.5

    @property
    def energy_profile(self) -> str:
        """Categorize energy as high/mixed/low."""
        settings = get_settings()
        avg = self.avg_energy
        if avg > settings.audio.energy_high_threshold:
            return "high"
        elif avg < settings.audio.energy_low_threshold:
            return "low"
        return "mixed"


@dataclass
class SceneInfo:
    """Information about a detected scene."""
    path: str
    start: float
    end: float
    duration: float
    meta: Dict[str, Any] = field(default_factory=dict)
    deep_analysis: Optional[Dict[str, Any]] = None

    @property
    def midpoint(self) -> float:
        """Middle point of the scene."""
        return self.start + (self.duration / 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dictionary format."""
        return {
            'path': self.path,
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'meta': self.meta,
        }


@dataclass
class OutputProfile:
    """Video output profile configuration."""
    width: int
    height: int
    fps: float
    codec: str
    pix_fmt: str
    profile: Optional[str] = None
    level: Optional[str] = None
    bitrate: int = 0
    orientation: str = "vertical"
    aspect_ratio: str = "9:16"
    reason: str = "default"


@dataclass
class ClipMetadata:
    """Metadata for a placed clip in the timeline."""
    source_path: str
    start_time: float
    duration: float
    timeline_start: float
    energy: float
    action: str
    shot: str
    beat_idx: int
    beats_per_cut: float
    selection_score: float
    enhancements: Dict[str, bool] = field(default_factory=dict)
    enhancement_decision: Optional[Any] = None  # EnhancementDecision for NLE export


@dataclass
class MontagePaths:
    input_dir: Path
    music_dir: Path
    assets_dir: Path
    output_dir: Path
    temp_dir: Path


@dataclass
class MontageFeatures:
    stabilize: bool = False
    upscale: bool = False
    enhance: bool = False
    color_grade: str = "teal_orange"  # Color grading preset from style template
    color_intensity: float = 1.0  # Color grading strength (0.0-1.0)
    # New features
    denoise: bool = False  # FFmpeg hqdn3d/nlmeans noise reduction
    sharpen: bool = False  # Unsharp mask sharpening
    film_grain: str = "none"  # Film grain preset: none, 35mm, 16mm, 8mm, digital
    dialogue_duck: bool = False  # Auto-duck music during speech
    detected_scenes: list = field(default_factory=list)  # Scene detection results


@dataclass
class MontageCreative:
    editing_instructions: Optional[EditingInstructions] = None
    semantic_query: Optional[str] = None
    broll_plan: Optional[List[Dict[str, Any]]] = None
    style_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MontageMedia:
    audio_result: Optional[AudioAnalysisResult] = None
    all_scenes: List[SceneInfo] = field(default_factory=list)
    all_scenes_dicts: List[Dict[str, Any]] = field(default_factory=list)
    video_files: List[str] = field(default_factory=list)
    output_profile: Optional[OutputProfile] = None
    similarity_index: Optional[Any] = None  # K-D Tree scene similarity index (for O(log n) lookups)


@dataclass
class MontageTimeline:
    target_duration: float = 0.0
    current_time: float = 0.0
    beat_idx: int = 0
    cut_number: int = 0
    estimated_total_cuts: int = 0
    clips_metadata: List[ClipMetadata] = field(default_factory=list)
    current_pattern: Optional[List[float]] = None
    pattern_idx: int = 0
    last_used_path: Optional[str] = None
    last_shot_type: Optional[str] = None
    last_tags: List[str] = field(default_factory=list)
    last_clip_end_time: Optional[float] = None
    enable_xfade: bool = False
    xfade_duration: float = 0.3


@dataclass
class MontageRender:
    output_filename: Optional[str] = None
    render_duration: float = 0.0
    logo_path: Optional[str] = None
    exported_files: Optional[Dict[str, str]] = None


@dataclass
class MontageTiming:
    start_time: float = field(default_factory=time.time)


@dataclass
class MontageContext:
    """
    Encapsulates all state for a montage job.
    """
    job_id: str
    variant_id: int
    settings: Settings
    paths: MontagePaths
    features: MontageFeatures = field(default_factory=MontageFeatures)
    creative: MontageCreative = field(default_factory=MontageCreative)
    media: MontageMedia = field(default_factory=MontageMedia)
    timeline: MontageTimeline = field(default_factory=MontageTimeline)
    render: MontageRender = field(default_factory=MontageRender)
    timing: MontageTiming = field(default_factory=MontageTiming)

    def reset_timeline_state(self):
        """Reset timeline state for a fresh pass."""
        self.timeline.current_time = 0.0
        self.timeline.beat_idx = 0
        self.timeline.cut_number = 0
        self.timeline.clips_metadata = []
        self.timeline.current_pattern = None
        self.timeline.pattern_idx = 0
        self.timeline.last_used_path = None
        self.timeline.last_shot_type = None
        self.timeline.last_clip_end_time = None

    def get_story_position(self) -> float:
        """Get current position in story arc (0-1)."""
        if self.timeline.target_duration > 0:
            return self.timeline.current_time / self.timeline.target_duration
        return 0.0

    def get_story_phase(self) -> str:
        """Get current story phase based on position."""
        return self.map_position_to_phase(self.get_story_position())

    @staticmethod
    def map_position_to_phase(position: float) -> str:
        """Map normalized timeline position (0.0-1.0) to story phase."""
        if position < 0.15:
            return "intro"
        elif position < 0.40:
            return "build"
        elif position < 0.70:
            return "climax"
        elif position < 0.90:
            return "sustain"
        return "outro"

    def elapsed_time(self) -> float:
        """Time since job started."""
        return time.time() - self.timing.start_time


@dataclass
class MontageResult:
    """Result of a montage build operation."""
    success: bool
    output_path: Optional[str]
    duration: float
    cut_count: int
    render_time: float
    file_size_mb: float = 0.0
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    project_package_path: Optional[str] = None
