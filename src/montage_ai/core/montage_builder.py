"""
Montage Builder Module for Montage AI

Object-oriented pipeline for creating video montages.
Transforms the procedural create_montage() function into a maintainable class.

Usage:
    from montage_ai.core.montage_builder import MontageBuilder

    builder = MontageBuilder(variant_id=1)
    result = builder.build()
"""

import os
import gc
import time
import random
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np

from ..config import get_settings, Settings
from ..logger import logger
from ..resource_manager import get_resource_manager, ResourceManager
from ..style_templates import get_style_template
from ..utils import file_exists_and_valid, coerce_float
from ..ffmpeg_utils import build_ffmpeg_cmd
from .analysis_cache import get_analysis_cache, EpisodicMemoryEntry
from ..timeline_exporter import export_timeline_from_montage
from ..storytelling import StoryArc, TensionProvider, StorySolver
from ..proxy_generator import ProxyGenerator


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


@dataclass
class MontageCreative:
    editing_instructions: Optional[Dict[str, Any]] = None
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


_LEGACY_ATTR_MAP = {
    "input_dir": ("paths", "input_dir"),
    "music_dir": ("paths", "music_dir"),
    "assets_dir": ("paths", "assets_dir"),
    "output_dir": ("paths", "output_dir"),
    "temp_dir": ("paths", "temp_dir"),
    "stabilize": ("features", "stabilize"),
    "upscale": ("features", "upscale"),
    "enhance": ("features", "enhance"),
    "editing_instructions": ("creative", "editing_instructions"),
    "semantic_query": ("creative", "semantic_query"),
    "broll_plan": ("creative", "broll_plan"),
    "style_params": ("creative", "style_params"),
    "audio_result": ("media", "audio_result"),
    "all_scenes": ("media", "all_scenes"),
    "all_scenes_dicts": ("media", "all_scenes_dicts"),
    "video_files": ("media", "video_files"),
    "output_profile": ("media", "output_profile"),
    "target_duration": ("timeline", "target_duration"),
    "current_time": ("timeline", "current_time"),
    "beat_idx": ("timeline", "beat_idx"),
    "cut_number": ("timeline", "cut_number"),
    "estimated_total_cuts": ("timeline", "estimated_total_cuts"),
    "clips_metadata": ("timeline", "clips_metadata"),
    "current_pattern": ("timeline", "current_pattern"),
    "pattern_idx": ("timeline", "pattern_idx"),
    "last_used_path": ("timeline", "last_used_path"),
    "last_shot_type": ("timeline", "last_shot_type"),
    "last_clip_end_time": ("timeline", "last_clip_end_time"),
    "enable_xfade": ("timeline", "enable_xfade"),
    "xfade_duration": ("timeline", "xfade_duration"),
    "output_filename": ("render", "output_filename"),
    "render_duration": ("render", "render_duration"),
    "logo_path": ("render", "logo_path"),
    "exported_files": ("render", "exported_files"),
    "start_time": ("timing", "start_time"),
}


@dataclass
class MontageContext:
    """
    Encapsulates all state for a montage job.

    Groups related state into cohesive sub-objects, while preserving legacy
    attribute access for a gradual migration.
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

    def __getattr__(self, name: str):
        mapping = _LEGACY_ATTR_MAP.get(name)
        if mapping:
            group, field_name = mapping
            return getattr(getattr(self, group), field_name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        mapping = _LEGACY_ATTR_MAP.get(name)
        if mapping:
            group, field_name = mapping
            if group in self.__dict__:
                setattr(getattr(self, group), field_name, value)
                if name in self.__dict__:
                    object.__delattr__(self, name)
                return
        object.__setattr__(self, name, value)

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



def process_clip_task(
    scene_path: str,
    clip_start: float,
    cut_duration: float,
    temp_dir: str,
    temp_clip_name: str,
    ctx_stabilize: bool,
    ctx_upscale: bool,
    ctx_enhance: bool,
    enhancer: Any,
    output_profile: Any,
    settings: Any,
    resource_manager: Any
) -> Tuple[str, Dict[str, bool], List[str]]:
    """
    Process a single clip: extract, enhance, normalize.
    Executed in a thread pool.
    """
    import subprocess
    import shutil
    
    temp_clip_path = os.path.join(temp_dir, temp_clip_name)
    temp_files = [temp_clip_path]
    
    # 1. Extract subclip
    if settings.encoding.extract_reencode:
        target_fps = output_profile.fps if output_profile else 24.0
        cmd = build_ffmpeg_cmd([
            "-ss", str(clip_start),
            "-i", scene_path,
            "-t", str(cut_duration),
            "-vf", f"fps={target_fps}",
            "-c:v", settings.encoding.codec,
            "-preset", settings.encoding.preset,
            "-crf", str(settings.encoding.crf),
            "-pix_fmt", settings.encoding.pix_fmt,
            "-an",
            temp_clip_path,
        ])
    else:
        cmd = build_ffmpeg_cmd([
            "-ss", str(clip_start),
            "-i", scene_path,
            "-t", str(cut_duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            temp_clip_path
        ])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.processing.ffmpeg_short_timeout,
    )
    if result.returncode != 0 or not os.path.exists(temp_clip_path) or os.path.getsize(temp_clip_path) == 0:
        err_lines = (result.stderr or "").strip().splitlines()
        err = err_lines[-1] if err_lines else "unknown error"
        raise RuntimeError(f"ffmpeg extract failed: {err}")
    
    # 1.5 Auto Reframe (Shorts Mode)
    if settings.features.shorts_mode:
        try:
            from ..auto_reframe import AutoReframeEngine
            # Initialize reframer (target 9:16)
            reframer = AutoReframeEngine(target_aspect=9/16)
            
            # Analyze the extracted subclip
            crops = reframer.analyze(temp_clip_path)
            
            shorts_clip_name = f"shorts_{temp_clip_name}"
            shorts_clip_path = os.path.join(temp_dir, shorts_clip_name)
            
            # Apply dynamic crop
            reframer.apply(crops, temp_clip_path, shorts_clip_path)
            
            # Update path and track temp file
            if os.path.exists(shorts_clip_path) and os.path.getsize(shorts_clip_path) > 0:
                temp_clip_path = shorts_clip_path
                temp_files.append(shorts_clip_path)
            else:
                logger.warning("Smart reframing failed, using original crop")
                
        except Exception as e:
            logger.warning(f"Smart reframing error: {e}")

    # 2. Enhance
    current_path = temp_clip_path
    stabilize_applied = False
    upscale_applied = False
    enhance_applied = False
    
    if enhancer:
        if ctx_stabilize:
            stab_path = os.path.join(temp_dir, f"stab_{temp_clip_name}")
            result = enhancer.stabilize(current_path, stab_path)
            if result != current_path:
                current_path = result
                temp_files.append(stab_path)
                stabilize_applied = True

        if ctx_upscale:
            upscale_path = os.path.join(temp_dir, f"upscale_{temp_clip_name}")
            result = enhancer.upscale(current_path, upscale_path)
            if result != current_path:
                current_path = result
                temp_files.append(upscale_path)
                upscale_applied = True

        if ctx_enhance:
            enhance_path = os.path.join(temp_dir, f"enhance_{temp_clip_name}")
            result = enhancer.enhance(current_path, enhance_path)
            if result != current_path:
                current_path = result
                temp_files.append(enhance_path)
                enhance_applied = True

    # 3. Normalize (optional)
    final_clip_path = current_path

    if not settings.encoding.normalize_clips:
        if not file_exists_and_valid(final_clip_path):
            raise RuntimeError("source clip missing before normalize")
    elif not output_profile:
        final_clip_path = os.path.join(temp_dir, f"norm_{temp_clip_name}")
        if not file_exists_and_valid(current_path):
            raise RuntimeError("source clip missing before normalize")
        shutil.copy(current_path, final_clip_path)
        if not file_exists_and_valid(final_clip_path):
            raise RuntimeError("failed to write normalized clip")
    else:
        final_clip_path = os.path.join(temp_dir, f"norm_{temp_clip_name}")
        # Get optimal encoder
        encoder_config = None
        if resource_manager:
            encoder_config = resource_manager.get_encoder(prefer_gpu=True)
            ffmpeg_params = encoder_config.video_params(
                crf=settings.encoding.crf,
                preset=settings.encoding.preset,
                codec_override=output_profile.codec,
                profile_override=output_profile.profile,
                level_override=output_profile.level,
                pix_fmt_override=output_profile.pix_fmt,
            )
        else:
            ffmpeg_params = [
                "-c:v", output_profile.codec,
                "-pix_fmt", output_profile.pix_fmt,
                "-crf", str(settings.encoding.crf),
                "-preset", settings.encoding.preset,
            ]
            if output_profile.profile:
                ffmpeg_params.extend(["-profile:v", output_profile.profile])
            if output_profile.level:
                ffmpeg_params.extend(["-level", output_profile.level])

        vf_filters = [
            f"scale={output_profile.width}:{output_profile.height}:force_original_aspect_ratio=decrease",
            f"pad={output_profile.width}:{output_profile.height}:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",  # Ensure square pixels
        ]
        if getattr(settings.features, "colorlevels", True):
            vf_filters.append(
                "colorlevels=rimin=0.063:gimin=0.063:bimin=0.063:"
                "rimax=0.922:gimax=0.922:bimax=0.922"
            )
        
        # Ensure output pixel format matches encoder expectation
        if output_profile.pix_fmt:
            vf_filters.append(f"format={output_profile.pix_fmt}")

        # Removed 'normalize' filter to prevent RGB conversion errors
        # if getattr(settings.features, "luma_normalize", True):
        #     vf_filters.append("normalize=blackpt=black:whitept=white:smoothing=10")
        vf_chain = ",".join(vf_filters)

        def run_normalize(cfg, params, label: str) -> bool:
            vf = vf_chain
            if cfg and cfg.hwupload_filter:
                vf = f"{vf_chain},{cfg.hwupload_filter}"
            cmd = build_ffmpeg_cmd([])
            if cfg and cfg.is_gpu_accelerated:
                cmd.extend(cfg.hwaccel_input_params())
            cmd.extend([
                "-i", current_path,
                "-vf", vf,
                "-r", str(output_profile.fps),
                *params,
                "-an",
                final_clip_path
            ])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=settings.processing.ffmpeg_timeout,
            )
            if result.returncode == 0 and file_exists_and_valid(final_clip_path):
                return True
            err_lines = (result.stderr or "").strip().splitlines()
            # Log full error for debugging
            logger.warning(f"Normalize ({label}) failed details:\n{result.stderr}")
            err = err_lines[-1] if err_lines else "unknown error"
            logger.warning(f"Normalize ({label}) failed: {err}")
            if os.path.exists(final_clip_path):
                try:
                    os.remove(final_clip_path)
                except OSError:
                    pass
            return False

        label = "gpu" if encoder_config and encoder_config.is_gpu_accelerated else "cpu"
        success = run_normalize(encoder_config, ffmpeg_params, label)

        if not success and resource_manager:
            cpu_config = resource_manager.get_encoder(prefer_gpu=False, cache_key="cpu_normalize")
            cpu_params = cpu_config.video_params(
                crf=settings.encoding.crf,
                preset=settings.encoding.preset,
                codec_override=output_profile.codec,
                profile_override=output_profile.profile,
                level_override=output_profile.level,
                pix_fmt_override=output_profile.pix_fmt,
            )
            success = run_normalize(cpu_config, cpu_params, "cpu_fallback")

        if not success:
            if file_exists_and_valid(current_path):
                logger.warning("Normalization failed; using unnormalized clip")
                final_clip_path = current_path
            else:
                raise RuntimeError("Normalization failed and source clip missing")

    enhancements = {
        'stabilized': stabilize_applied,
        'upscaled': upscale_applied,
        'enhanced': enhance_applied,
    }
    
    return final_clip_path, enhancements, temp_files


# =============================================================================
# MontageBuilder Class
# =============================================================================

class MontageBuilder:
    """
    Object-oriented pipeline for creating video montages.

    Transforms the procedural create_montage() function into a maintainable class
    with clearly separated phases.

    Phases:
        1. setup_workspace()   - Initialize paths, load config
        2. analyze_assets()    - Audio beat/energy + video scene detection
        3. plan_montage()      - Clip selection & beat matching
        4. enhance_assets()    - Stabilization/upscaling via ClipEnhancer
        5. render_output()     - Final composition via SegmentWriter
    """

    def __init__(
        self,
        variant_id: int = 1,
        settings: Optional[Settings] = None,
        editing_instructions: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Any] = None
    ):
        """
        Initialize the MontageBuilder.

        Args:
            variant_id: Variant number for this montage (1-based)
            settings: Optional Settings instance (uses global if None)
            editing_instructions: Optional Creative Director instructions
            job_id: Unique job identifier
            progress_callback: Callable[[int, str], None] for status updates
        """
        self.settings = settings or get_settings()
        self.variant_id = variant_id
        self.rng = random.Random(42 + variant_id)  # Seeded RNG for determinism
        self.editing_instructions = editing_instructions
        self.progress_callback = progress_callback

        # Initialize context
        self.ctx = self._create_context()
        if job_id:
            self.ctx.job_id = job_id

        # Component references (lazy initialized)
        self._monitor = None
        self._footage_pool = None
        self._progressive_renderer = None
        self._memory_manager = None
        self._clip_enhancer = None
        self._intelligent_selector = None
        self._resource_manager: Optional[ResourceManager] = None
        
        # Parallel processing
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending_futures: List[Future] = []

    def _create_context(self) -> MontageContext:
        """Create a fresh MontageContext from settings."""
        return MontageContext(
            job_id=self.settings.job_id,
            variant_id=self.variant_id,
            settings=self.settings,
            paths=MontagePaths(
                input_dir=self.settings.paths.input_dir,
                music_dir=self.settings.paths.music_dir,
                assets_dir=self.settings.paths.assets_dir,
                output_dir=self.settings.paths.output_dir,
                temp_dir=self.settings.paths.temp_dir,
            ),
            features=MontageFeatures(
                stabilize=self.settings.features.stabilize,
                upscale=self.settings.features.upscale,
                enhance=self.settings.features.enhance,
            ),
            creative=MontageCreative(
                editing_instructions=self.editing_instructions,
            ),
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def build(self) -> MontageResult:
        """
        Execute the complete montage pipeline.

        Returns:
            MontageResult with success status and output path
        """
        try:
            logger.info(f"\n🎬 Starting Montage Variant #{self.variant_id}")

            # Phase 1: Setup
            self.setup_workspace()

            # Phase 2: Analyze
            self.analyze_assets()

            # Phase 3: Plan
            self.plan_montage()

            # Phase 4: Enhance (if enabled)
            if self.ctx.stabilize or self.ctx.upscale or self.ctx.enhance:
                self.enhance_assets()

            # Phase 5: Render
            self.render_output()

            # Phase 6: Export Timeline
            self.export_timeline()

            # Phase 7: Save episodic memory (if enabled)
            self._save_episodic_memory()

            # Phase 8: Cleanup
            self.cleanup()

            # Build result
            return MontageResult(
                success=True,
                output_path=self.ctx.output_filename,
                duration=self.ctx.current_time,
                cut_count=self.ctx.cut_number,
                render_time=self.ctx.render_duration,
                file_size_mb=self._get_output_file_size(),
                stats=self._collect_stats(),
                project_package_path=self.ctx.exported_files.get('package') if self.ctx.exported_files else None,
            )

        except Exception as e:
            logger.error(f"❌ Montage build failed: {e}")
            self.cleanup()
            return MontageResult(
                success=False,
                output_path=None,
                duration=0.0,
                cut_count=0,
                render_time=0.0,
                error=str(e),
            )

    # =========================================================================
    # Pipeline Phases
    # =========================================================================

    def setup_workspace(self):
        """
        Phase 1: Initialize workspace.

        - Initialize resource manager (detect GPUs, cluster)
        - Apply Creative Director effects overrides
        - Initialize monitoring
        - Initialize intelligent clip selector
        - Ensure directories exist
        """
        logger.info("   📁 Setting up workspace...")

        # Ensure temp directory exists
        os.makedirs(str(self.ctx.temp_dir), exist_ok=True)

        # Initialize resource manager and log status
        self._resource_manager = get_resource_manager(refresh=True)
        status_line = self._resource_manager.get_status_line()
        logger.info(f"   🖥️ Resources: {status_line}")

        # Apply Creative Director effects (ENV takes precedence)
        self._apply_creative_director_effects()

        # Initialize monitor (if available)
        self._init_monitor()

        # Initialize intelligent clip selector (if available)
        self._init_intelligent_selector()

        # Initialize thread pool for parallel clip processing
        adaptive_workers = self.settings.processing.get_adaptive_parallel_jobs(self.settings.features.low_memory_mode)
        optimal_workers = self._resource_manager.get_optimal_threads()
        max_workers = max(1, min(adaptive_workers, optimal_workers))
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"   🚀 Initialized parallel processor with {max_workers} workers")

        logger.info(f"   🎨 Effects: STABILIZE={self.ctx.stabilize}, UPSCALE={self.ctx.upscale}, ENHANCE={self.ctx.enhance}")

    def analyze_assets(self):
        """
        Phase 2: Analyze audio and video assets.

        OPTIMIZED: Runs cgpu voice isolation in parallel with scene detection
        to maximize utilization of both cloud GPU and local CPU.

        - Start voice isolation async (cgpu) if enabled
        - Detect scenes in videos (parallel, local CPU)
        - Wait for voice isolation to complete
        - Analyze music with isolated audio
        - Determine output profile
        """
        logger.info("\n   🎵 Analyzing assets...")

        # Story Engine Analysis Trigger (Phase 1)
        if self.settings.features.story_engine:
            self._trigger_story_analysis()

        # OPTIMIZATION: Start proxy generation in background
        # This ensures proxies are ready for NLE export later
        proxy_futures = []
        if self.ctx.video_files:
            logger.info("   🎞️ Checking proxies...")
            # Try to store proxies next to input for NLE linking standard
            proxy_dir = self.ctx.input_dir / "Proxies"
            try:
                proxy_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                proxy_dir = self.ctx.temp_dir / "proxies"
            
            generator = ProxyGenerator(proxy_dir)
            for video_path in self.ctx.video_files:
                proxy_futures.append(
                    self._executor.submit(generator.ensure_proxy, video_path)
                )

        # OPTIMIZATION: Start voice isolation in background while doing scene detection
        voice_isolation_future = None
        isolated_audio_path = None

        if self.settings.features.voice_isolation:
            # Get music file first
            music_files = self._get_files(self.ctx.music_dir, ('.mp3', '.wav'))
            if music_files:
                music_index = (self.variant_id - 1) % len(music_files)
                music_path = music_files[music_index]

                # Start voice isolation async (runs on cgpu while we do scene detection)
                logger.info("   🚀 Starting voice isolation async (cgpu)...")
                voice_isolation_future = self._executor.submit(
                    self._apply_voice_isolation, music_path
                )

        # Detect scenes in videos (parallel, local CPU) - runs while cgpu processes
        self._detect_scenes()

        # Wait for voice isolation to complete (if started)
        if voice_isolation_future is not None:
            try:
                # Use configured timeout (default 1200s) instead of hardcoded 300s
                timeout = self.settings.llm.cgpu_timeout
                isolated_audio_path = voice_isolation_future.result(timeout=timeout)
                logger.info(f"   ✅ Voice isolation completed")
            except Exception as e:
                if self.settings.features.strict_cloud_compute:
                    raise RuntimeError(f"Strict cloud compute enabled: Voice isolation async failed: {e}") from e
                logger.warning(f"   ⚠️ Voice isolation async failed: {e}")
                isolated_audio_path = None

        # Analyze music (use isolated audio if available)
        self._analyze_music(isolated_audio_path=isolated_audio_path)

        # Determine output profile
        self._determine_output_profile()

        # Initialize footage pool
        self._init_footage_pool()

        # Wait for proxies
        if proxy_futures:
            logger.info("   ⏳ Waiting for proxy generation to complete...")
            completed = 0
            for f in proxy_futures:
                try:
                    f.result()
                    completed += 1
                except Exception as e:
                    logger.warning(f"   ⚠️ Proxy generation warning: {e}")
            if completed > 0:
                logger.info(f"   ✅ {completed} Proxies ready")

    def _setup_output_paths(self, style_name: str):
        """Determine output filename and logo path."""
        # Sanitize style name
        style_label = "".join(c for c in style_name if c.isalnum() or c in ('-', '_')).strip()
        if not style_label:
            style_label = "dynamic"

        self.ctx.output_filename = os.path.join(
            str(self.ctx.output_dir),
            f"gallery_montage_{self.ctx.job_id}_v{self.variant_id}_{style_label}.mp4"
        )

        # Check for logo
        logo_files = self._get_files(self.ctx.assets_dir, ('.png', '.jpg'))
        self.ctx.logo_path = logo_files[0] if logo_files else None

    def plan_montage(self):
        """
        Phase 3: Plan the montage timeline.

        - Calculate target duration
        - Initialize progressive renderer
        - Run clip selection loop (beat-synced, AI-scored)
        """
        logger.info("\n   📋 Planning montage...")

        # Story Engine Planning (Phase 1)
        if self.settings.features.story_engine:
            self._run_story_assembly()
            return

        # Initialize audio clip and calculate duration
        self._init_audio_duration()

        # AI B-Roll Planning (Epic 2)
        if self.ctx.editing_instructions and "script" in self.ctx.editing_instructions:
            self._plan_broll_sequence()

        # Initialize crossfade settings
        self._init_crossfade_settings()

        # Initialize progressive renderer
        self._init_progressive_renderer()

        # Estimate total cuts for progress tracking
        tempo = self.ctx.audio_result.tempo
        avg_beats_per_cut = 4.0
        self.ctx.estimated_total_cuts = int(
            (self.ctx.target_duration * tempo / 60) / avg_beats_per_cut
        )

        # Determine output filename
        style_name = "dynamic"
        if self.ctx.editing_instructions is not None:
            style_name = self.ctx.editing_instructions.get('style', {}).get('name', 'dynamic')
        
        # Load style params
        try:
            template = get_style_template(style_name)
            self.ctx.style_params = template.get("params", {})
            logger.info(f"   🎨 Loaded style '{style_name}' params")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load style '{style_name}': {e}")
            self.ctx.style_params = {}

        self._setup_output_paths(style_name)

        # Run the main assembly loop
        logger.info("   ✂️ Assembling cuts...")
        self._run_assembly_loop()

    def enhance_assets(self):
        """
        Phase 4: Apply clip enhancements.

        - Stabilization (if enabled)
        - Upscaling (if enabled)
        - Color enhancement (if enabled)
        """
        logger.info("\n   ✨ Enhancing assets...")

        # Initialize ClipEnhancer if needed
        if self._clip_enhancer is None:
            from ..clip_enhancement import ClipEnhancer
            self._clip_enhancer = ClipEnhancer(self.settings)

        # Enhancement happens during timeline assembly
        # This phase is for pre-enhancement if doing batch processing
        pass

    def render_output(self):
        """
        Phase 5: Render final output.

        - Finalize progressive renderer (or legacy path)
        - Add audio
        - Add logo overlay (if present)
        """
        logger.info("\n   🎬 Rendering output...")

        if self.settings.processing.should_skip_output(self.ctx.output_filename):
            logger.info(f"   ♻️ Output exists, skipping render: {os.path.basename(self.ctx.output_filename)}")
            self.ctx.render_duration = 0.0
            return

        render_start_time = time.time()

        if self._progressive_renderer:
            # Progressive path: finalize with FFmpeg
            logger.info(f"   🔗 Finalizing with Progressive Renderer ({self._progressive_renderer.get_segment_count()} segments)...")

            audio_duration = self.ctx.target_duration
            success = self._progressive_renderer.finalize(
                output_path=self.ctx.output_filename,
                audio_path=self.ctx.audio_result.music_path,
                audio_duration=audio_duration,
                logo_path=self.ctx.logo_path
            )

            if success:
                method_str = "xfade" if self.ctx.enable_xfade else "-c copy"
                self.ctx.render_duration = time.time() - render_start_time
                logger.info(f"   ✅ Final video rendered via FFmpeg ({method_str}) in {self.ctx.render_duration:.1f}s")
                if self.ctx.logo_path:
                    logger.info(f"   🏷️ Logo overlay: {os.path.basename(self.ctx.logo_path)}")
            else:
                raise RuntimeError("Progressive render failed")
        else:
            # Legacy path would go here
            raise NotImplementedError("Legacy MoviePy rendering not implemented in MontageBuilder")

    def cleanup(self):
        """
        Phase 6: Cleanup resources.

        - Close clips
        - Delete temp files
        - Export timeline (if enabled)
        - Force GC if configured
        """
        logger.info("\n   🧹 Cleaning up...")

        cleanup_count = 0
        cleanup_size_mb = 0.0

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=False)

        # Cleanup progressive renderer resources
        if self._progressive_renderer:
            try:
                self._progressive_renderer.cleanup()
            except Exception as e:
                logger.warning(f"   ⚠️ Progressive renderer cleanup failed: {e}")

        # Cleanup temp directory
        temp_dir = str(self.ctx.temp_dir)
        if os.path.isdir(temp_dir):
            for f in os.listdir(temp_dir):
                if f.startswith(f"clip_{self.ctx.job_id}") or f.startswith("temp_clip_"):
                    try:
                        fpath = os.path.join(temp_dir, f)
                        size = os.path.getsize(fpath) / (1024 * 1024)
                        os.remove(fpath)
                        cleanup_count += 1
                        cleanup_size_mb += size
                    except Exception:
                        pass

        logger.info(f"   ✅ Deleted {cleanup_count} temp files ({cleanup_size_mb:.1f} MB freed)")

        if self.settings.processing.force_gc:
            gc.collect()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _apply_creative_director_effects(self):
        """Apply Creative Director effect overrides."""
        if self.ctx.editing_instructions is None:
            return

        effects = self.ctx.editing_instructions.get('effects', {})

        # ENV takes precedence over style template
        env_stabilize = self.settings.features.stabilize
        env_upscale = self.settings.features.upscale
        env_enhance = self.settings.features.enhance

        # Force disable heavy features in Preview Mode
        if self.settings.encoding.quality_profile == "preview":
            logger.info("   ⚡ Preview Mode: Disabling stabilization, upscale, and enhancement")
            env_stabilize = False
            env_upscale = False
            env_enhance = False
            # Also override instructions
            if 'effects' in self.ctx.editing_instructions:
                self.ctx.editing_instructions['effects']['stabilization'] = False
                self.ctx.editing_instructions['effects']['upscale'] = False
                self.ctx.editing_instructions['effects']['sharpness_boost'] = False

        if not env_stabilize and 'stabilization' in effects:
            self.ctx.stabilize = effects['stabilization']
        else:
            self.ctx.stabilize = env_stabilize

        if not env_upscale and 'upscale' in effects:
            self.ctx.upscale = effects['upscale']
        else:
            self.ctx.upscale = env_upscale

        if not env_enhance and 'sharpness_boost' in effects:
            self.ctx.enhance = effects['sharpness_boost']
        else:
            self.ctx.enhance = env_enhance

        # Extract semantic query for Phase 2: Semantic Storytelling
        # Priority: explicit semantic_query > content_focus > ENV
        env_semantic = os.environ.get('SEMANTIC_QUERY', '')
        if 'semantic_query' in self.ctx.editing_instructions:
            self.ctx.semantic_query = self.ctx.editing_instructions['semantic_query']
        elif 'content_focus' in self.ctx.editing_instructions:
            self.ctx.semantic_query = self.ctx.editing_instructions['content_focus']
        elif 'style' in self.ctx.editing_instructions:
            style = self.ctx.editing_instructions['style']
            # Combine mood, theme, or content hints
            semantic_parts = []
            if style.get('mood'):
                semantic_parts.append(style['mood'])
            if style.get('theme'):
                semantic_parts.append(style['theme'])
            if style.get('content'):
                semantic_parts.append(style['content'])
            if semantic_parts:
                self.ctx.semantic_query = ' '.join(semantic_parts)
        if env_semantic and not self.ctx.semantic_query:
            self.ctx.semantic_query = env_semantic

    def _plan_broll_sequence(self):
        """
        Generate B-Roll plan from script.
        
        Uses BrollPlanner to find relevant clips for script segments.
        Estimates timing based on word count if no timestamps provided.
        """
        try:
            from ..broll_planner import plan_broll
            
            script = self.ctx.editing_instructions["script"]
            logger.info(f"   📜 Planning B-Roll for script: {script[:50]}...")
            
            # Plan B-Roll (find suggestions)
            # We use input_dir from context
            plan = plan_broll(
                script, 
                input_dir=str(self.ctx.input_dir),
                top_k=3,
                analyze_first=True
            )
            
            # Estimate timing
            # Simple heuristic: Distribute segments evenly or by word count
            total_words = len(script.split())
            duration_per_word = self.ctx.target_duration / max(1, total_words)
            
            current_time = 0.0
            for segment in plan:
                seg_text = segment["segment"]
                seg_words = len(seg_text.split())
                
                # If we have a voiceover file, we could use forced alignment here
                # For now, we estimate
                seg_duration = seg_words * duration_per_word
                
                # Ensure minimum duration
                seg_duration = max(seg_duration, 2.0)
                
                segment["start_time"] = current_time
                segment["end_time"] = current_time + seg_duration
                current_time += seg_duration
                
            self.ctx.broll_plan = plan
            logger.info(f"   ✅ B-Roll Plan created with {len(plan)} segments")
            
        except Exception as e:
            logger.warning(f"   ⚠️ B-Roll planning failed: {e}")
            self.ctx.broll_plan = None

    def _init_monitor(self):
        """Initialize monitoring system."""
        try:
            from ..monitoring import get_monitor, init_monitor
            self._monitor = get_monitor()
            if self._monitor is None:
                self._monitor = init_monitor(self.ctx.job_id, self.settings.features.verbose)
        except ImportError:
            self._monitor = None

    def _init_intelligent_selector(self):
        """Initialize intelligent clip selector."""
        try:
            from ..clip_selector import IntelligentClipSelector
            style = "dynamic"
            if self.ctx.editing_instructions is not None:
                style = self.ctx.editing_instructions.get('style', {}).get('template', 'dynamic')
            self._intelligent_selector = IntelligentClipSelector(style=style)
            logger.info(f"   🧠 Intelligent Clip Selector initialized (style={style})")
        except ImportError:
            self._intelligent_selector = None
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to initialize Intelligent Clip Selector: {e}")
            self._intelligent_selector = None

    def _analyze_music(self, isolated_audio_path: Optional[str] = None):
        """Load and analyze music file with caching support.

        Args:
            isolated_audio_path: Pre-isolated audio path from async processing.
                                 If provided, skips voice isolation step.
        """
        from ..audio_analysis import get_beat_times, analyze_music_energy, detect_music_sections

        # Get music files
        music_files = self._get_files(self.ctx.music_dir, ('.mp3', '.wav'))
        if not music_files:
            raise ValueError("No music found in music directory")

        # Select music file (rotate through variants)
        music_index = (self.variant_id - 1) % len(music_files)
        music_path = music_files[music_index]

        # Use pre-isolated audio if provided (from async processing)
        # Otherwise apply voice isolation synchronously if enabled
        if isolated_audio_path:
            music_path = isolated_audio_path
        elif self.settings.features.voice_isolation:
            music_path = self._apply_voice_isolation(music_path)

        # Apply noise reduction if enabled (can stack with voice_isolation or run standalone)
        if self.settings.features.noise_reduction:
            music_path = self._apply_noise_reduction(music_path)

        # Try cache first
        cache = get_analysis_cache()
        cached = cache.load_audio(music_path)

        if cached:
            # Cache hit - use cached analysis
            logger.info("   ⚡ Using cached audio analysis")
            # Note: Cached object might not have sections if it's old, so we might need to re-detect
            # For now, we'll just re-detect sections from energy profile if missing
            
            energy_profile = analyze_music_energy(music_path, verbose=False) # Lightweight if cached? No, analyze_music_energy does work.
            # Actually, we can reconstruct EnergyProfile from cached data
            from ..audio_analysis import EnergyProfile
            ep = EnergyProfile(
                times=np.array(cached.energy_times),
                rms=np.array(cached.energy_values),
                sample_rate=22050, # Assumption
                hop_length=512 # Assumption
            )
            sections = detect_music_sections(ep)

            self.ctx.audio_result = AudioAnalysisResult(
                music_path=music_path,
                beat_times=np.array(cached.beat_times),
                tempo=cached.tempo,
                energy_times=np.array(cached.energy_times),
                energy_values=np.array(cached.energy_values),
                duration=cached.duration,
                sections=sections
            )
        else:
            # Cache miss - analyze and cache
            beat_info = get_beat_times(music_path, verbose=self.settings.features.verbose)
            energy_profile = analyze_music_energy(music_path, verbose=self.settings.features.verbose)
            sections = detect_music_sections(energy_profile)

            # Store in context
            self.ctx.audio_result = AudioAnalysisResult(
                music_path=music_path,
                beat_times=beat_info.beat_times,
                tempo=beat_info.tempo,
                energy_times=energy_profile.times,
                energy_values=energy_profile.rms,
                duration=beat_info.duration,
                sections=sections
            )

            # Save to cache
            cache.save_audio(music_path, beat_info, energy_profile)

        if self._monitor:
            self._monitor.log_beat_analysis(
                music_path,
                self.ctx.audio_result.tempo,
                len(self.ctx.audio_result.beat_times),
                self.ctx.audio_result.energy_profile
            )

    def _detect_scenes(self):
        """Detect scenes in all video files with caching and parallel processing."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from ..scene_analysis import detect_scenes, analyze_scene_content, clear_histogram_cache
        from .analysis_cache import get_analysis_cache

        # Clear histogram cache from previous runs
        clear_histogram_cache()

        # Get video files
        video_files = self._get_files(self.ctx.input_dir, ('.mp4', '.mov'))
        if not video_files:
            raise ValueError("No videos found in input directory")

        self.ctx.video_files = video_files
        self.ctx.all_scenes = []
        self.ctx.all_scenes_dicts = []

        # Scene detection threshold (from scenedetect defaults)
        threshold = 30.0
        cache = get_analysis_cache()
        cache_hits = 0
        uncached_videos = []

        # First pass: check cache and collect uncached videos
        cached_scenes = {}  # path -> scenes list
        for v_path in video_files:
            cached = cache.load_scenes(v_path, threshold)
            if cached:
                cache_hits += 1
                cached_scenes[v_path] = [(s["start"], s["end"]) for s in cached.scenes]
            else:
                uncached_videos.append(v_path)

        # Parallel scene detection for uncached videos
        detected_scenes = {}  # path -> scenes list

        def detect_video_scenes(v_path: str):
            """Detect scenes in a single video (thread-safe)."""
            try:
                scenes = detect_scenes(v_path, threshold=threshold)
                return v_path, scenes
            except Exception as e:
                logger.warning(f"   ⚠️ Scene detection failed for {v_path}: {e}")
                return v_path, []

        if uncached_videos:
            # Use ThreadPoolExecutor for parallel I/O
            # Use more workers based on available CPU cores for better cluster utilization
            optimal_workers = self._resource_manager.get_optimal_threads()
            adaptive_workers = self.settings.processing.get_adaptive_parallel_jobs(self.settings.features.low_memory_mode)
            max_workers = min(optimal_workers, adaptive_workers, len(uncached_videos))
            logger.info(f"   🚀 Parallel scene detection ({len(uncached_videos)} videos, {max_workers} workers)")
            
            # Progress tracking setup
            total_tasks = len(uncached_videos)
            completed_tasks = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(detect_video_scenes, v): v for v in uncached_videos}

                for future in as_completed(futures):
                    v_path, scenes = future.result()
                    detected_scenes[v_path] = scenes
                    # Save to cache
                    cache.save_scenes(v_path, threshold, scenes)
                    
                    # Update progress
                    completed_tasks += 1
                    if self.progress_callback:
                        # Map 0-100% of this phase to 10-40% of overall job
                        local_pct = int((completed_tasks / total_tasks) * 100)
                        # We arbitrarily map "Scene Detection" to the 20-50% range of the whole job
                        # But simpler is just to report "Analyzing X/Y"
                        msg = f"Detecting scenes in {os.path.basename(v_path)} ({completed_tasks}/{total_tasks})"
                        self.progress_callback(local_pct, msg)

        # Combine cached and detected scenes
        all_video_scenes = {**cached_scenes, **detected_scenes}

        # Build scene list
        for v_path in video_files:
            scenes = all_video_scenes.get(v_path, [])
            for start, end in scenes:
                duration = end - start
                if duration > 1.0:  # Ignore tiny clips
                    scene_info = SceneInfo(
                        path=v_path,
                        start=start,
                        end=end,
                        duration=duration,
                    )
                    self.ctx.all_scenes.append(scene_info)

        if cache_hits > 0:
            logger.info(f"   ⚡ Used cached scene detection for {cache_hits}/{len(video_files)} videos")

        # Parallel AI scene analysis (limit to first 20 for speed)
        logger.info("   🤖 AI Director is watching footage...")
        scenes_to_analyze = self.ctx.all_scenes[:20]

        def analyze_scene(scene):
            """Analyze a single scene (thread-safe)."""
            try:
                meta = analyze_scene_content(scene.path, scene.midpoint)
                return scene, meta
            except Exception:
                return scene, {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_scene, s) for s in scenes_to_analyze]
            for future in as_completed(futures):
                scene, meta = future.result()
                scene.meta = meta

        # Shuffle for variety
        random.shuffle(self.ctx.all_scenes)

        # Create legacy dict format (needed for footage_manager compatibility)
        self.ctx.all_scenes_dicts = [s.to_dict() for s in self.ctx.all_scenes]

        logger.info(f"   📹 Found {len(self.ctx.all_scenes)} scenes in {len(video_files)} videos")

    def _determine_output_profile(self):
        """Determine output profile from input footage."""
        from ..video_metadata import determine_output_profile as _determine_profile
        from .. import segment_writer as segment_writer_module

        profile = _determine_profile(self.ctx.video_files)

        # Override if EXPORT_WIDTH/HEIGHT are explicitly set in env
        # We check os.environ directly to avoid default values in settings.export
        import os
        env_width = os.environ.get("EXPORT_WIDTH")
        env_height = os.environ.get("EXPORT_HEIGHT")
        
        if env_width and env_height:
             w = int(env_width)
             h = int(env_height)
             logger.info(f"   🔧 Explicit resolution override: {w}x{h}")
             profile.width = w
             profile.height = h
             if w > h:
                 profile.orientation = "horizontal"
             elif w < h:
                 profile.orientation = "vertical"
             else:
                 profile.orientation = "square"
             profile.reason = "explicit_override"

        # Override for Shorts Mode
        if self.settings.features.shorts_mode:
            logger.info("   📱 Shorts Mode enabled: Forcing 9:16 vertical output")
            # Assuming 1080x1920 for shorts
            profile.width = 1080
            profile.height = 1920
            profile.orientation = "vertical"
            profile.aspect_ratio = "9:16"
            profile.reason = "shorts_mode"

        # Override for Preview Mode (Low Res)
        if self.settings.encoding.quality_profile == "preview":
            logger.info("   ⚡ Preview Mode enabled: Forcing low resolution (360p)")
            
            # Import constants
            from ..ffmpeg_config import PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_CRF, PREVIEW_PRESET
            
            if profile.orientation == "vertical" or self.settings.features.shorts_mode:
                profile.width = PREVIEW_HEIGHT  # 360
                profile.height = PREVIEW_WIDTH  # 640
            else:
                profile.width = PREVIEW_WIDTH   # 640
                profile.height = PREVIEW_HEIGHT # 360
            profile.reason = "preview_mode"
            
            # Update encoding settings for speed
            self.settings.encoding.crf = PREVIEW_CRF
            self.settings.encoding.preset = PREVIEW_PRESET

        # determine_output_profile returns an OutputProfile dataclass
        self.ctx.output_profile = OutputProfile(
            width=profile.width,
            height=profile.height,
            fps=profile.fps,
            codec=profile.codec,
            pix_fmt=profile.pix_fmt,
            profile=profile.profile,
            level=profile.level,
            bitrate=profile.bitrate,
            orientation=profile.orientation,
            aspect_ratio=profile.aspect_ratio,
            reason=getattr(profile, 'reason', 'default'),
        )

        # CRITICAL: Sync output profile to segment_writer globals
        # This ensures clip normalization uses the correct dimensions
        segment_writer_module.STANDARD_WIDTH = profile.width
        segment_writer_module.STANDARD_HEIGHT = profile.height
        segment_writer_module.STANDARD_FPS = profile.fps
        segment_writer_module.STANDARD_PIX_FMT = profile.pix_fmt
        segment_writer_module.TARGET_CODEC = profile.codec
        segment_writer_module.TARGET_PROFILE = profile.profile
        segment_writer_module.TARGET_LEVEL = profile.level

        if self.settings.features.verbose:
            logger.info(f"\n   🧭 Output profile: {self.ctx.output_profile.width}x{self.ctx.output_profile.height} @ {self.ctx.output_profile.fps:.1f}fps")

    def _init_footage_pool(self):
        """Initialize footage pool manager."""
        from ..footage_manager import integrate_footage_manager

        # Use the same dict objects that are stored in context
        # (important: id() matching requires same objects)
        self._footage_pool = integrate_footage_manager(
            self.ctx.all_scenes_dicts,
            strict_once=False,
        )

    def _apply_voice_isolation(self, audio_path: str) -> str:
        """
        Apply voice isolation to audio before analysis.

        Uses cgpu/demucs to separate vocals from music, resulting in
        cleaner beat detection and energy analysis.

        Args:
            audio_path: Path to original audio file

        Returns:
            Path to isolated vocals (or original if isolation fails)
        """
        model = self.settings.features.voice_isolation_model
        logger.info(f"\n🎤 Isolating voice ({model} model)...")

        try:
            from ..cgpu_utils import is_cgpu_available
            if not is_cgpu_available():
                if self.settings.features.strict_cloud_compute:
                    raise RuntimeError("Strict cloud compute enabled: cgpu voice isolation not available.")
                logger.warning("   Voice isolation requires cgpu - using original audio")
                return audio_path

            from ..cgpu_jobs import VoiceIsolationJob
            job = VoiceIsolationJob(
                audio_path=audio_path,
                model=model,
                two_stems=True,  # Fast mode: vocals + accompaniment
                keep_all_stems=True, # We need no_vocals for better beat detection
            )
            result = job.execute()

            if result.success:
                # Prefer no_vocals (accompaniment) for beat detection as it has clearer drums/bass
                # without vocal interference
                stems = result.metadata.get("stems", {})
                if stems.get("no_vocals"):
                    logger.info(f"   ✅ Voice isolated: Using accompaniment for analysis: {stems['no_vocals']}")
                    return stems["no_vocals"]
                
                if result.output_path:
                    logger.info(f"   ✅ Voice isolated: {result.output_path}")
                    return result.output_path
            else:
                if self.settings.features.strict_cloud_compute:
                    raise RuntimeError(f"Strict cloud compute enabled: Voice isolation failed: {result.error}")
                logger.warning(f"   Voice isolation failed: {result.error}")
                return audio_path

        except Exception as e:
            if self.settings.features.strict_cloud_compute:
                raise RuntimeError(f"Strict cloud compute enabled: Voice isolation error: {e}")
            logger.warning(f"   Voice isolation error: {e}")
            return audio_path

    def _apply_noise_reduction(self, audio_path: str) -> str:
        """
        Apply noise reduction to audio using DeepFilterNet via cgpu.

        Lighter/faster than voice_isolation - good for podcasts, interviews,
        vlogs with background noise.

        Args:
            audio_path: Path to original audio file

        Returns:
            Path to cleaned audio (or original if reduction fails)
        """
        strength = self.settings.features.noise_reduction_strength
        logger.info(f"\n🔇 Applying noise reduction (strength: {strength})...")

        try:
            from ..cgpu_utils import is_cgpu_available
            if not is_cgpu_available():
                if self.settings.features.strict_cloud_compute:
                    raise RuntimeError("Strict cloud compute enabled: cgpu noise reduction not available.")
                logger.warning("   Noise reduction requires cgpu - using original audio")
                return audio_path

            from ..cgpu_jobs import NoiseReductionJob
            job = NoiseReductionJob(
                audio_path=audio_path,
                attenuation_limit=strength,
            )
            result = job.execute()

            if result.success and result.output_path:
                logger.info(f"   ✅ Noise reduced: {result.output_path}")
                return result.output_path
            else:
                if self.settings.features.strict_cloud_compute:
                    raise RuntimeError(f"Strict cloud compute enabled: Noise reduction failed: {result.error}")
                logger.warning(f"   Noise reduction failed: {result.error}")
                return audio_path

        except Exception as e:
            if self.settings.features.strict_cloud_compute:
                raise RuntimeError(f"Strict cloud compute enabled: Noise reduction error: {e}")
            logger.warning(f"   Noise reduction error: {e}")
            return audio_path

    def _get_files(self, directory: Path, extensions: Tuple[str, ...]) -> List[str]:
        """Get files with given extensions from directory."""
        files = []
        dir_str = str(directory)
        if os.path.isdir(dir_str):
            for f in os.listdir(dir_str):
                if f.lower().endswith(extensions):
                    files.append(os.path.join(dir_str, f))
        return sorted(files)

    def _get_output_file_size(self) -> float:
        """Get output file size in MB."""
        if self.ctx.output_filename and os.path.exists(self.ctx.output_filename):
            return os.path.getsize(self.ctx.output_filename) / (1024 * 1024)
        return 0.0

    def _collect_stats(self) -> Dict[str, Any]:
        """Collect statistics for the result."""
        return {
            "job_id": self.ctx.job_id,
            "variant_id": self.variant_id,
            "cut_count": self.ctx.cut_number,
            "duration": self.ctx.current_time,
            "tempo": self.ctx.audio_result.tempo if self.ctx.audio_result else 0,
            "scene_count": len(self.ctx.all_scenes),
            "elapsed_time": self.ctx.elapsed_time(),
            "story_engine": self.settings.features.story_engine,
        }

    # =========================================================================
    # Assembly Loop Helpers
    # =========================================================================

    def _init_audio_duration(self):
        """Initialize audio and calculate target duration."""
        # Get duration settings from config
        target_duration_setting = self.settings.creative.target_duration
        music_start = self.settings.creative.music_start
        music_end = self.settings.creative.music_end

        audio_duration = self.ctx.audio_result.duration

        # Apply music trimming
        if music_end and music_end < audio_duration:
            audio_duration = music_end - music_start
        elif music_start > 0:
            audio_duration = audio_duration - music_start

        # Determine target duration
        if target_duration_setting > 0:
            self.ctx.target_duration = target_duration_setting
        else:
            self.ctx.target_duration = audio_duration

        # Apply Creative Director constraints if provided (only when env didn't override)
        if target_duration_setting <= 0 and self.ctx.editing_instructions:
            constraints = self.ctx.editing_instructions.get("constraints", {})
            target_override = coerce_float(constraints.get("target_duration_sec"))
            if target_override and target_override > 0:
                self.ctx.target_duration = min(audio_duration, target_override)

        if self.settings.features.verbose:
            logger.info(f"   ⏱️ Target duration: {self.ctx.target_duration:.1f}s")

    def _init_crossfade_settings(self):
        """Initialize crossfade settings from config and Creative Director."""
        enable_xfade = self.settings.creative.enable_xfade
        xfade_duration = self.settings.creative.xfade_duration

        if enable_xfade == "true":
            self.ctx.enable_xfade = True
        elif enable_xfade == "false":
            self.ctx.enable_xfade = False
        elif self.ctx.editing_instructions is not None:
            transitions = self.ctx.editing_instructions.get('transitions', {})
            transition_type = transitions.get('type', 'energy_aware')
            xfade_duration = transitions.get('crossfade_duration_sec', xfade_duration)
            if transition_type == 'crossfade':
                self.ctx.enable_xfade = True

        self.ctx.xfade_duration = xfade_duration

    def _init_progressive_renderer(self):
        """Initialize progressive renderer if available."""
        try:
            from ..segment_writer import ProgressiveRenderer
            from ..memory_monitor import get_memory_manager

            self._memory_manager = get_memory_manager()

            # Use adaptive batch size for low-memory environments
            low_memory = self.settings.features.low_memory_mode
            batch_size = self.settings.processing.get_adaptive_batch_size(low_memory)

            if low_memory:
                logger.info(f"   ⚠️ LOW_MEMORY_MODE: Batch size reduced to {batch_size}")

            self._progressive_renderer = ProgressiveRenderer(
                batch_size=batch_size,
                output_dir=os.path.join(str(self.ctx.temp_dir), f"segments_{self.ctx.job_id}"),
                memory_manager=self._memory_manager,
                job_id=self.ctx.job_id,
                enable_xfade=self.ctx.enable_xfade,
                xfade_duration=self.ctx.xfade_duration,
                ffmpeg_crf=self.settings.encoding.crf,
                normalize_clips=self.settings.encoding.normalize_clips,
            )
            logger.info(f"   ✅ Progressive Renderer initialized (batch={batch_size})")
        except ImportError:
            self._progressive_renderer = None
            self._memory_manager = None
            logger.warning("   ⚠️ Progressive Renderer not available")

    def _flush_pending_futures(self):
        """Wait for all pending clip processing futures to complete."""
        if not self._pending_futures:
            return

        logger.info(f"   ⏳ Waiting for {len(self._pending_futures)} pending clips to finish processing...")
        while self._pending_futures:
            fut, meta = self._pending_futures.pop(0)
            try:
                final_path, enhancements, temp_files = fut.result()
                meta.enhancements = enhancements
                
                if self._progressive_renderer:
                    self._progressive_renderer.add_clip_path(final_path)
                    
                    # Cleanup intermediate temp files
                    for tf in temp_files:
                        if tf != final_path and os.path.exists(tf):
                            try:
                                os.remove(tf)
                            except Exception:
                                pass
            except Exception as e:
                logger.error(f"Error processing clip {meta.source_path}: {e}")

    def _run_assembly_loop(self):
        """
        Main assembly loop: select clips, apply enhancements, add to timeline.

        This is the heart of the montage creation process.
        """
        from ..audio_analysis import calculate_dynamic_cut_length
        from ..scene_analysis import (
            calculate_visual_similarity,
            detect_motion_blur,
            find_best_start_point,
        )

        # Cut patterns for dynamic pacing
        default_patterns = [
            [4, 4, 4, 4],       # Steady pace
            [2, 2, 4, 8],       # Accelerate then hold
            [8, 4, 2, 2],       # Long shot into fast cuts
            [1, 1, 2, 3, 5],    # Fibonacci
            [1.5, 1.5, 5],      # Syncopated
        ]
        
        # Load patterns from style if available
        cut_patterns = default_patterns
        if self.ctx.style_params and "cut_patterns" in self.ctx.style_params:
            cut_patterns = self.ctx.style_params["cut_patterns"]
            logger.info(f"   🎨 Using {len(cut_patterns)} custom cut patterns from style")

        # Get audio data
        beat_times = self.ctx.audio_result.beat_times
        energy_times = self.ctx.audio_result.energy_times
        energy_values = self.ctx.audio_result.energy_values
        tempo = self.ctx.audio_result.tempo

        # Get unique video count for variety logic
        unique_videos = len(set(s.path for s in self.ctx.all_scenes))

        # Main loop
        while self.ctx.current_time < self.ctx.target_duration and self.ctx.all_scenes:
            # Get current energy
            current_energy = self._get_energy_at_time(
                self.ctx.current_time, energy_times, energy_values
            )

            # Calculate beats per cut
            beats_per_cut = self._calculate_beats_per_cut(
                current_energy, tempo, cut_patterns
            )

            # Calculate cut duration
            cut_duration = self._calculate_cut_duration(
                beats_per_cut, beat_times
            )

            # Early exit: prevent overrun beyond target
            remaining_time = self.ctx.target_duration - self.ctx.current_time
            if remaining_time < cut_duration * 0.3:
                # Less than 30% of a cut remaining - stop here
                logger.info(f"   🛑 Target reached ({self.ctx.current_time:.1f}s / {self.ctx.target_duration:.1f}s)")
                break

            # Trim last cut if it would significantly overshoot
            if cut_duration > remaining_time:
                cut_duration = max(remaining_time, cut_duration * 0.5)

            # Get available footage
            min_dur = cut_duration * 0.5
            available_footage = self._footage_pool.get_available_clips(min_duration=min_dur)

            if not available_footage:
                logger.warning("   ⚠️ No more footage available. Stopping.")
                break

            # Score and select clip
            selected_scene, best_score = self._select_clip(
                available_footage, current_energy, unique_videos
            )

            if selected_scene is None:
                logger.warning("   ⚠️ No valid scene found. Stopping.")
                break

            # Determine start point
            # If this scene has been used before, use random selection to ensure variety
            # Otherwise use optical flow to find the "best" moment
            usage_count = 0
            if hasattr(self, '_footage_pool') and hasattr(self._footage_pool, 'clips'):
                clip_obj = self._footage_pool.clips.get(id(selected_scene))
                if clip_obj:
                    usage_count = clip_obj.usage_count

            if usage_count > 0:
                # Random selection for variety on reuse
                max_start = selected_scene['end'] - cut_duration
                if max_start > selected_scene['start']:
                    clip_start = random.uniform(selected_scene['start'], max_start)
                    logger.info(f"   🎲 Reusing scene (count={usage_count}): Random start {clip_start:.1f}s")
                else:
                    clip_start = selected_scene['start']
            else:
                # First use: Find peak action via optical flow
                clip_start = find_best_start_point(
                    selected_scene['path'],
                    selected_scene['start'],
                    selected_scene['end'],
                    cut_duration
                )

            clip_end = clip_start + cut_duration

            # Mark clip as consumed
            self._footage_pool.consume_clip(
                clip_id=id(selected_scene),
                timeline_position=self.ctx.current_time,
                used_in_point=clip_start,
                used_out_point=clip_end
            )

            # Process and add clip to timeline
            self._process_and_add_clip(
                selected_scene, clip_start, cut_duration, current_energy, best_score
            )

            # Update state
            self.ctx.last_used_path = selected_scene['path']
            self.ctx.last_shot_type = selected_scene.get('meta', {}).get('shot', 'medium')
            self.ctx.last_clip_end_time = clip_end

            self.ctx.current_time += cut_duration
            self.ctx.beat_idx += int(beats_per_cut)
            self.ctx.cut_number += 1

            # Progress logging
            if self._monitor and self.ctx.cut_number % 5 == 0:
                self._monitor.log_progress(
                    self.ctx.cut_number,
                    self.ctx.estimated_total_cuts,
                    "cuts placed"
                )

        # Flush remaining futures
        self._flush_pending_futures()

        logger.info(f"   ✅ Assembly complete: {self.ctx.cut_number} cuts, {self.ctx.current_time:.1f}s")

    def _get_energy_at_time(
        self, time_sec: float, energy_times: np.ndarray, energy_values: np.ndarray
    ) -> float:
        """Get energy level at a specific time."""
        if len(energy_times) == 0:
            return 0.5
        idx = np.searchsorted(energy_times, time_sec)
        idx = min(idx, len(energy_values) - 1)
        return float(energy_values[idx])

    def _calculate_beats_per_cut(
        self, current_energy: float, tempo: float, cut_patterns: List[List[float]]
    ) -> float:
        """Calculate beats per cut based on pacing settings."""
        from ..audio_analysis import calculate_dynamic_cut_length

        min_beats = None
        max_beats = None
        if self.ctx.editing_instructions and tempo > 0:
            constraints = self.ctx.editing_instructions.get("constraints", {})
            min_clip = coerce_float(constraints.get("min_clip_duration_sec"))
            max_clip = coerce_float(constraints.get("max_clip_duration_sec"))
            beat_duration = 60.0 / tempo
            if min_clip:
                min_beats = min_clip / beat_duration
            if max_clip:
                max_beats = max_clip / beat_duration
            if min_beats is not None and max_beats is not None and min_beats > max_beats:
                min_beats, max_beats = max_beats, min_beats

        def clamp_beats(beats: float) -> float:
            if min_beats is not None:
                beats = max(beats, min_beats)
            if max_beats is not None:
                beats = min(beats, max_beats)
            return beats

        pacing_speed = "dynamic"
        if self.ctx.editing_instructions is not None:
            pacing_speed = self.ctx.editing_instructions.get('pacing', {}).get('speed', 'dynamic')

        # Section-based pacing override (Pacing Curves)
        # If we are in a high energy section, force faster cuts regardless of instantaneous drops
        current_section = None
        if self.ctx.audio_result and self.ctx.audio_result.sections:
            for section in self.ctx.audio_result.sections:
                if section.start_time <= self.ctx.current_time < section.end_time:
                    current_section = section
                    break
        
        if pacing_speed == "dynamic" and current_section:
            if current_section.energy_level == "high":
                # High energy section: 2 beats (fast) or 4 beats (medium-fast)
                # We still allow some variation but cap the max length
                return 2 if tempo < 130 else 4
            elif current_section.energy_level == "low":
                # Low energy section: 8 beats (slow) or 16 beats (very slow)
                return 16 if tempo < 100 else 8

        if pacing_speed == "very_fast":
            return clamp_beats(1)
        elif pacing_speed == "fast":
            return clamp_beats(2 if tempo < 130 else 4)
        elif pacing_speed == "medium":
            return clamp_beats(4)
        elif pacing_speed == "slow":
            return clamp_beats(8)
        elif pacing_speed == "very_slow":
            return clamp_beats(16 if tempo < 100 else 8)
        else:  # dynamic
            if self.ctx.current_pattern is None or self.ctx.pattern_idx >= len(self.ctx.current_pattern):
                self.ctx.current_pattern = calculate_dynamic_cut_length(
                    current_energy, tempo, self.ctx.current_time,
                    self.ctx.target_duration, cut_patterns
                )
                self.ctx.pattern_idx = 0

            beats = self.ctx.current_pattern[self.ctx.pattern_idx]
            self.ctx.pattern_idx += 1
            return clamp_beats(beats)

    def _calculate_cut_duration(
        self, beats_per_cut: float, beat_times: np.ndarray
    ) -> float:
        """Calculate cut duration from beat times."""
        target_beat_idx = self.ctx.beat_idx + beats_per_cut

        if target_beat_idx >= len(beat_times):
            return self.ctx.target_duration - self.ctx.current_time

        # Interpolate for fractional beats
        idx_int = int(self.ctx.beat_idx)
        if idx_int >= len(beat_times) - 1:
            t_start = beat_times[-1]
        else:
            frac = self.ctx.beat_idx - idx_int
            t_start = beat_times[idx_int] + (beat_times[min(idx_int+1, len(beat_times)-1)] - beat_times[idx_int]) * frac

        target_idx_int = int(target_beat_idx)
        if target_idx_int >= len(beat_times) - 1:
            t_end = beat_times[-1]
        else:
            frac = target_beat_idx - target_idx_int
            t_end = beat_times[target_idx_int] + (beat_times[min(target_idx_int+1, len(beat_times)-1)] - beat_times[target_idx_int]) * frac

        cut_duration = t_end - t_start

        # Add micro-timing jitter for humanization
        jitter = random.uniform(-0.05, 0.05)
        if cut_duration + jitter > 0.5:
            cut_duration += jitter

        return max(cut_duration, 0.5)

    def _apply_style_scoring(self, meta: Dict[str, Any], style_params: Dict[str, Any]) -> float:
        """
        Apply data-driven scoring based on style weights.
        
        Abstracts the logic so new styles can be defined purely in JSON
        without code changes.
        """
        score = 0.0
        weights = style_params.get('weights', {})
        
        # 1. Numeric Fields (Linear scaling)
        # Automatically apply weights to any matching numeric field in meta
        # e.g. "face_count": 2.0 -> +20 points per face
        # e.g. "visual_quality": 1.5 -> +15 points per quality point (0-1)
        for key, weight in weights.items():
            if key == 'action': continue  # Handled specially below
            
            val = meta.get(key)
            if isinstance(val, (int, float)) and val > 0:
                # Normalize factor: 10 points per unit * weight
                score += (val * 10.0) * weight

        # 2. Action Level (Categorical -> Scalar)
        if 'action' in weights:
            action = meta.get('action', 'medium')
            weight = weights['action']
            if action == 'high':
                score += 15.0 * weight
            elif action == 'low':
                score -= 5.0 * weight

        # 3. Preferred Lists (Binary match)
        # Automatically checks "preferred_X" against "X" in meta
        # e.g. "preferred_shots" -> checks meta["shot"]
        for param_key, param_val in style_params.items():
            if param_key.startswith('preferred_') and isinstance(param_val, list):
                # Extract field name: preferred_shots -> shot
                field_name = param_key.replace('preferred_', '').rstrip('s')
                
                meta_val = meta.get(field_name)
                if meta_val and meta_val in param_val:
                    score += 20.0

        return score

    def _get_candidate_scenes(self, available_footage) -> List[Dict[str, Any]]:
        clip_ids = {c.clip_id for c in available_footage}
        return [
            scene for scene in self.ctx.media.all_scenes_dicts
            if id(scene) in clip_ids
        ]

    def _resolve_scoring_rules(self) -> Dict[str, float]:
        scoring_rules = self.ctx.creative.style_params.get("scoring_rules", {})
        return {
            "fresh_clip_bonus": scoring_rules.get("fresh_clip_bonus", 50),
            "jump_cut_penalty": scoring_rules.get("jump_cut_penalty", 50),
            "shot_variation_bonus": scoring_rules.get("shot_variation_bonus", 10),
            "shot_repetition_penalty": scoring_rules.get("shot_repetition_penalty", 10),
        }

    def _get_current_music_section(self):
        audio = self.ctx.media.audio_result
        if not audio or not audio.sections:
            return None
        current_time = self.ctx.timeline.current_time
        for section in audio.sections:
            if section.start_time <= current_time < section.end_time:
                return section
        return None

    def _score_usage_and_story_phase(
        self,
        footage_clip: Any,
        current_energy: float,
        fresh_clip_bonus: float,
    ) -> float:
        score = 0.0
        if not footage_clip:
            return score

        if footage_clip.usage_count == 0:
            score += fresh_clip_bonus
        elif footage_clip.usage_count == 1:
            score += (fresh_clip_bonus * 0.4)
        else:
            score -= footage_clip.usage_count * 10

        phase = self.ctx.get_story_phase()
        if phase == "intro" and current_energy < 0.4:
            score += 15
        elif phase == "build" and 0.4 <= current_energy < 0.7:
            score += 15
        elif phase == "climax" and current_energy >= 0.7:
            score += 15
        elif phase == "outro" and current_energy < 0.5:
            score += 15

        return score

    def _score_jump_cut(
        self, scene_path: str, unique_videos: int, jump_cut_penalty: float
    ) -> float:
        if scene_path != self.ctx.timeline.last_used_path:
            return 0.0
        if unique_videos > 1:
            return -jump_cut_penalty
        return -(jump_cut_penalty * 0.1)

    @staticmethod
    def _score_action_energy(
        meta: Dict[str, Any],
        current_energy: float,
        current_section: Any,
    ) -> float:
        score = 0.0
        action = meta.get('action', 'medium')
        if current_energy > 0.6 and action == 'high':
            score += 20
        if current_energy < 0.4 and action == 'low':
            score += 20

        if current_section:
            if current_section.energy_level == "high" and action == "high":
                score += 25
            elif current_section.energy_level == "low" and action == "low":
                score += 25

        return score

    def _score_style_preferences(self, meta: Dict[str, Any]) -> float:
        style_params = self.ctx.creative.style_params
        if not style_params:
            return 0.0
        return self._apply_style_scoring(meta, style_params)

    def _score_shot_variation(
        self,
        shot: str,
        shot_variation_bonus: float,
        shot_repetition_penalty: float,
    ) -> float:
        if self.ctx.timeline.last_shot_type and shot == self.ctx.timeline.last_shot_type:
            return -shot_repetition_penalty
        return shot_variation_bonus

    def _score_match_cut(self, scene: Dict[str, Any], similarity_fn) -> float:
        if self.ctx.timeline.last_clip_end_time is None or not self.ctx.timeline.last_used_path:
            return 0.0

        try:
            similarity = similarity_fn(
                self.ctx.timeline.last_used_path,
                self.ctx.timeline.last_clip_end_time,
                scene['path'],
                scene['start'],
            )
            if similarity > 0.7:
                return 30.0
        except Exception:
            pass
        return 0.0

    def _score_semantic_match(self, meta: Dict[str, Any]) -> float:
        if not self.ctx.creative.semantic_query:
            return 0.0
        if not (meta.get('tags') or meta.get('caption')):
            return 0.0

        try:
            from ..semantic_matcher import get_semantic_matcher
            matcher = get_semantic_matcher()
            if matcher.is_available:
                sem_result = matcher.match_query_to_clip(self.ctx.creative.semantic_query, meta)
                return int(sem_result.overall_score * 40)
        except Exception:
            return 0.0
        return 0.0

    def _score_broll_match(self, scene: Dict[str, Any], meta: Dict[str, Any]) -> float:
        if not self.ctx.creative.broll_plan:
            return 0.0

        active_segment = None
        current_time = self.ctx.timeline.current_time
        for seg in self.ctx.creative.broll_plan:
            if seg.get("start_time", 0) <= current_time < seg.get("end_time", 99999):
                active_segment = seg
                break

        if not active_segment:
            return 0.0

        for sug in active_segment.get("suggestions", []):
            if sug.get("clip") == scene["path"]:
                return 100.0

        if not (meta.get('tags') or meta.get('caption')):
            return 0.0

        try:
            from ..semantic_matcher import get_semantic_matcher
            matcher = get_semantic_matcher()
            if matcher.is_available:
                sem_result = matcher.match_query_to_clip(active_segment["segment"], meta)
                return int(sem_result.overall_score * 50)
        except Exception:
            return 0.0
        return 0.0

    def _select_with_intelligent_selector(
        self,
        candidates: List[Dict[str, Any]],
        current_energy: float,
    ) -> Tuple[Dict[str, Any], float]:
        from ..clip_selector import ClipCandidate

        clip_candidates = [
            ClipCandidate(
                path=scene['path'],
                start_time=scene['start'],
                duration=scene['duration'],
                heuristic_score=int(scene.get('_heuristic_score', 0)),
                metadata=scene.get('meta', {}),
            )
            for scene in candidates
        ]

        prev_clips = []
        if self.ctx.timeline.last_used_path:
            prev_meta = {'shot': self.ctx.timeline.last_shot_type}
            prev_clips.append({'meta': prev_meta})

        context = {
            'current_energy': current_energy,
            'position': self.ctx.get_story_phase(),
            'previous_clips': prev_clips,
            'beat_position': self.ctx.timeline.beat_idx,
        }

        best_candidate, reason = self._intelligent_selector.select_best_clip(
            clip_candidates,
            context,
            top_n=5,
        )

        selected_scene = next(
            (
                scene for scene in candidates
                if scene['path'] == best_candidate.path
                and scene['start'] == best_candidate.start_time
            ),
            candidates[0],
        )

        if "LLM" in reason:
            logger.info(f"   🧠 {reason}")

        return selected_scene, best_candidate.heuristic_score

    @staticmethod
    def _select_probabilistic(
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        candidates.sort(key=lambda x: x.get('_heuristic_score', -1000), reverse=True)
        top_n = min(3, len(candidates))
        if top_n > 0:
            top_candidates = candidates[:top_n]
            min_score = min(c.get('_heuristic_score', 0) for c in top_candidates)
            weights = [c.get('_heuristic_score', 0) - min_score + 10 for c in top_candidates]
            selected_scene = random.choices(top_candidates, weights=weights, k=1)[0]
            best_score = selected_scene.get('_heuristic_score', 0)
            return selected_scene, best_score
        return None, 0.0

    def _select_clip(
        self,
        available_footage,
        current_energy: float,
        unique_videos: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Score and select the best clip for this cut."""
        from ..scene_analysis import calculate_visual_similarity

        valid_scenes = self._get_candidate_scenes(available_footage)
        if not valid_scenes:
            return None, 0

        candidates = valid_scenes[:20]
        scoring_rules = self._resolve_scoring_rules()
        current_section = self._get_current_music_section()
        clip_map = {c.clip_id: c for c in available_footage}

        for scene in candidates:
            meta = scene.get('meta', {})
            shot = meta.get('shot', 'medium')

            footage_clip = clip_map.get(id(scene))
            score = 0.0
            score += self._score_usage_and_story_phase(
                footage_clip,
                current_energy,
                scoring_rules["fresh_clip_bonus"],
            )
            score += self._score_jump_cut(
                scene['path'],
                unique_videos,
                scoring_rules["jump_cut_penalty"],
            )
            score += self._score_action_energy(meta, current_energy, current_section)
            score += self._score_style_preferences(meta)
            score += self._score_shot_variation(
                shot,
                scoring_rules["shot_variation_bonus"],
                scoring_rules["shot_repetition_penalty"],
            )
            score += self._score_match_cut(scene, calculate_visual_similarity)
            score += self._score_semantic_match(meta)
            score += self._score_broll_match(scene, meta)

            score += random.randint(-15, 15)
            scene['_heuristic_score'] = score

        if self._intelligent_selector:
            return self._select_with_intelligent_selector(candidates, current_energy)

        return self._select_probabilistic(candidates)

    def _process_and_add_clip(
        self,
        scene: Dict[str, Any],
        clip_start: float,
        cut_duration: float,
        current_energy: float,
        selection_score: float
    ):
        """Submit clip processing task and track metadata."""
        
        # Generate temp paths
        temp_clip_name = f"temp_clip_{self.ctx.beat_idx}_{random.randint(0, 9999)}.mp4"
        
        # Submit task
        if self._executor:
            future = self._executor.submit(
                process_clip_task,
                scene_path=scene['path'],
                clip_start=clip_start,
                cut_duration=cut_duration,
                temp_dir=str(self.ctx.temp_dir),
                temp_clip_name=temp_clip_name,
                ctx_stabilize=self.ctx.stabilize,
                ctx_upscale=self.ctx.upscale,
                ctx_enhance=self.ctx.enhance,
                enhancer=self._clip_enhancer,
                output_profile=self.ctx.output_profile,
                settings=self.settings,
                resource_manager=self._resource_manager
            )
        else:
            raise RuntimeError("Executor not initialized")

        # Create metadata (enhancements will be updated later)
        clip_meta = ClipMetadata(
            source_path=scene['path'],
            start_time=clip_start,
            duration=cut_duration,
            timeline_start=self.ctx.current_time,
            energy=current_energy,
            action=scene.get('meta', {}).get('action', 'medium'),
            shot=scene.get('meta', {}).get('shot', 'medium'),
            beat_idx=self.ctx.beat_idx,
            beats_per_cut=self.ctx.current_pattern[self.ctx.pattern_idx - 1] if self.ctx.current_pattern else 4,
            selection_score=selection_score,
            enhancements={} # Updated on completion
        )
        self.ctx.clips_metadata.append(clip_meta)
        
        self._pending_futures.append((future, clip_meta))
        
        # Process completed futures in order to keep memory usage in check
        # and feed the progressive renderer
        while self._pending_futures and self._pending_futures[0][0].done():
            fut, meta = self._pending_futures.pop(0)
            try:
                final_path, enhancements, temp_files = fut.result()
                meta.enhancements = enhancements
                
                if self._progressive_renderer:
                    self._progressive_renderer.add_clip_path(final_path)
                    
                    # Cleanup intermediate temp files
                    for tf in temp_files:
                        if tf != final_path and os.path.exists(tf):
                            try:
                                os.remove(tf)
                            except Exception:
                                pass
            except Exception as e:
                logger.error(f"Error processing clip {meta.source_path}: {e}")



    def export_timeline(self):
        """
        Phase 6: Export timeline to NLE formats (EDL, XML, OTIO).
        """
        if not self.settings.features.export_timeline:
            return

        logger.info("\n   📝 Exporting timeline...")
        
        # Convert ClipMetadata to format expected by exporter
        clips_data = []
        for clip in self.ctx.clips_metadata:
            clips_data.append({
                'source_path': clip.source_path,
                'start_time': clip.start_time,
                'duration': clip.duration,
                'timeline_start': clip.timeline_start,
                'metadata': {
                    'energy': clip.energy,
                    'action': clip.action,
                    'shot': clip.shot,
                    'selection_score': clip.selection_score,
                    'enhancements': clip.enhancements
                }
            })

        # Get audio path safely
        audio_path = self.ctx.audio_result.music_path if self.ctx.audio_result else ""
        if not audio_path:
            logger.warning("   ⚠️ No audio path found for export, using placeholder")
            audio_path = "audio_track_missing.mp3"

        try:
            self.ctx.exported_files = export_timeline_from_montage(
                clips_data=clips_data,
                audio_path=audio_path,
                total_duration=self.ctx.current_time,
                output_dir=str(self.ctx.output_dir),
                project_name=f"{self.ctx.job_id}_v{self.variant_id}",
                generate_proxies=self.settings.features.generate_proxies,
                resolution=(self.ctx.output_profile.width, self.ctx.output_profile.height),
                fps=self.ctx.output_profile.fps
            )
        except Exception as e:
            logger.error(f"   ❌ Timeline export failed: {e}")

    def _save_episodic_memory(self):
        """
        Save clip usage to episodic memory for future learning.

        Only runs if EPISODIC_MEMORY feature flag is enabled.
        Tracks which clips were used in which story phases.
        """
        if not self.settings.features.episodic_memory:
            return

        cache = get_analysis_cache()
        montage_id = f"{self.ctx.job_id}_v{self.variant_id}"
        total_duration = self.ctx.current_time or 1.0  # Avoid division by zero

        saved_count = 0
        for clip in self.ctx.clips_metadata:
            # Calculate story phase based on timeline position
            position = clip.timeline_start / total_duration
            if position < 0.15:
                phase = "intro"
            elif position < 0.40:
                phase = "build"
            elif position < 0.70:
                phase = "climax"
            elif position < 0.90:
                phase = "sustain"
            else:
                phase = "outro"

            entry = EpisodicMemoryEntry(
                clip_path=clip.source_path,
                montage_id=montage_id,
                story_phase=phase,
                timestamp_used=clip.timeline_start,
                clip_start=clip.start_time,
                clip_end=clip.start_time + clip.duration,
            )

            if cache.save_episodic_memory(entry):
                saved_count += 1

        if saved_count > 0:
            logger.info(f"   📝 Episodic memory: saved {saved_count} clip usage records")




    # =========================================================================
    # Story Engine Methods (Phase 1)
    # =========================================================================

    def _get_tension_metadata_dir(self) -> Path:
        """Resolve the directory for tension metadata (prefers cache or output)."""
        env_dir = os.environ.get("TENSION_METADATA_DIR")
        if env_dir:
            return Path(env_dir)
        cache_dir = os.environ.get("METADATA_CACHE_DIR")
        if cache_dir:
            return Path(cache_dir) / "tension"
        return self.ctx.output_dir / "tension_analysis"

    def _trigger_story_analysis(self):
        """
        Identifies clips missing tension analysis and triggers cgpu jobs.
        """
        logger.info("   📖 Story Engine: Checking for missing analysis...")
        
        # 1. Identify all input clips
        input_clips = self.ctx.video_files or self._get_files(self.ctx.input_dir, ('.mp4', '.mov', '.mkv'))
        if not input_clips:
            logger.warning("   ⚠️ Story Engine: No input clips found for analysis.")
            return
        
        # 2. Check which ones are missing metadata
        # We use TensionProvider to check existence (it has logic for ID generation)
        tension_meta_dir = self._get_tension_metadata_dir()
        tension_meta_dir.mkdir(parents=True, exist_ok=True)

        provider = TensionProvider(tension_meta_dir)
        missing_clips = []
        
        from ..storytelling.tension_provider import MissingAnalysisError
        for clip in input_clips:
            try:
                provider.get_tension(clip)
            except MissingAnalysisError:
                missing_clips.append(clip)
            except Exception as exc:
                logger.warning(f"   ⚠️ Story Engine: Failed to read metadata for {os.path.basename(clip)}: {exc}")
                missing_clips.append(clip)
                
        if not missing_clips:
            logger.info("   ✅ All clips have tension analysis.")
            return

        logger.info(f"   ⚠️ Missing analysis for {len(missing_clips)} clips.")
        
        from ..cgpu_utils import is_cgpu_available
        if not is_cgpu_available(require_gpu=False):
            message = "Story Engine requires cgpu for tension analysis but cgpu is not available."
            if self.settings.features.strict_cloud_compute:
                raise RuntimeError(message)
            logger.warning(f"   ⚠️ {message} Proceeding with dummy tension values.")
            return

        logger.info("   ☁️ Offloading tension analysis to cgpu...")
        from ..cgpu_jobs import TensionAnalysisBatchJob
        job = TensionAnalysisBatchJob(missing_clips, output_dir=str(tension_meta_dir))
        result = job.execute()
        if not result.success:
            message = f"Story tension analysis failed: {result.error}"
            if self.settings.features.strict_cloud_compute:
                raise RuntimeError(message)
            logger.warning(f"   ⚠️ {message}")
            return

        analyzed = result.metadata.get("clip_count", len(missing_clips))
        logger.info(f"   ✅ Tension analysis complete for {analyzed} clip(s).")

    def _run_story_assembly(self):
        """
        Executes the montage assembly using the Story Engine.
        """
        logger.info("   📖 Story Engine: Assembling narrative...")
        
        # Ensure audio analysis is done
        if not hasattr(self.ctx, 'audio_result') or self.ctx.audio_result is None:
            self._analyze_music()

        # Initialize duration, transitions, and renderer
        self._init_audio_duration()
        self._init_crossfade_settings()
        self._init_progressive_renderer()

        # Determine style/arc
        style_name = "dynamic"
        story_arc_spec = None
        if self.ctx.editing_instructions:
            style_name = self.ctx.editing_instructions.get('style', {}).get('name', 'dynamic')
            story_arc_spec = self.ctx.editing_instructions.get("story_arc")

        # Determine output filename and logo
        self._setup_output_paths(style_name)

        # Setup tension provider + solver
        tension_meta_dir = self._get_tension_metadata_dir()
        allow_dummy = not self.settings.features.strict_cloud_compute
        provider = TensionProvider(tension_meta_dir, allow_dummy=allow_dummy)
        if story_arc_spec:
            arc = StoryArc.from_spec(story_arc_spec)
        else:
            arc = StoryArc.from_preset(style_name)
        solver = StorySolver(arc, provider)

        # Build beat list (clamped to target duration)
        duration = self.ctx.target_duration or self.ctx.audio_result.duration
        beats = [t for t in self.ctx.audio_result.beat_times.tolist() if 0.0 <= t <= duration]
        if not beats:
            beats = [0.0]

        min_clip_duration = None
        max_clip_duration = None
        if self.ctx.editing_instructions:
            constraints = self.ctx.editing_instructions.get("constraints", {})
            min_clip_duration = coerce_float(constraints.get("min_clip_duration_sec"))
            max_clip_duration = coerce_float(constraints.get("max_clip_duration_sec"))
            if min_clip_duration is not None and max_clip_duration is not None and min_clip_duration > max_clip_duration:
                min_clip_duration, max_clip_duration = max_clip_duration, min_clip_duration

        # Build timeline plan
        input_clips = self.ctx.video_files or self._get_files(self.ctx.input_dir, ('.mp4', '.mov', '.mkv'))
        timeline_events = solver.solve(input_clips, duration, beats)
        if not timeline_events:
            raise RuntimeError("Story Engine produced no timeline events")

        # Map scenes by source path (fallback to full clip duration if scenes missing)
        scenes_by_path: Dict[str, List[Dict[str, Any]]] = {}
        for scene in self.ctx.all_scenes_dicts:
            scenes_by_path.setdefault(scene['path'], []).append(scene)
            scene.setdefault('usage_count', 0)

        if input_clips:
            from ..video_metadata import probe_duration
            for clip_path in input_clips:
                if clip_path not in scenes_by_path:
                    clip_duration = probe_duration(clip_path)
                    if clip_duration > 0:
                        scenes_by_path[clip_path] = [{
                            "path": clip_path,
                            "start": 0.0,
                            "end": clip_duration,
                            "duration": clip_duration,
                            "meta": {},
                            "usage_count": 0,
                        }]

        for scenes in scenes_by_path.values():
            self.rng.shuffle(scenes)

        logger.info(f"   ✅ Generated {len(timeline_events)} cuts based on '{style_name}' arc.")

        self.ctx.cut_number = 0
        self.ctx.current_time = 0.0

        # Assemble timeline using existing clip processing pipeline
        for idx, event in enumerate(timeline_events):
            start_time = float(event.get('time', 0.0))
            if start_time >= duration:
                break

            if idx < len(timeline_events) - 1:
                end_time = float(timeline_events[idx + 1].get('time', duration))
            else:
                end_time = duration

            cut_duration = max(0.5, min(end_time - start_time, duration - start_time))
            if min_clip_duration is not None:
                cut_duration = max(cut_duration, min_clip_duration)
            if max_clip_duration is not None:
                cut_duration = min(cut_duration, max_clip_duration)
            if cut_duration <= 0:
                continue

            clip_path = event.get('clip')
            if not clip_path:
                continue

            candidates = [s for s in scenes_by_path.get(clip_path, []) if s.get('duration', 0.0) >= cut_duration]
            if not candidates:
                candidates = scenes_by_path.get(clip_path, [])
            if not candidates:
                logger.warning(f"   ⚠️ Story Engine: No scenes available for {os.path.basename(clip_path)}")
                continue

            min_usage = min(s.get('usage_count', 0) for s in candidates)
            least_used = [s for s in candidates if s.get('usage_count', 0) == min_usage]
            scene = self.rng.choice(least_used) if least_used else self.rng.choice(candidates)

            scene_start = float(scene.get('start', 0.0))
            scene_end = float(scene.get('end', scene_start + cut_duration))
            scene_duration = float(scene.get('duration', max(scene_end - scene_start, cut_duration)))
            if cut_duration > scene_duration:
                cut_duration = max(0.5, scene_duration)

            max_start = scene_end - cut_duration
            if max_start <= scene_start:
                clip_start = scene_start
            else:
                clip_start = self.rng.uniform(scene_start, max_start)

            self.ctx.current_time = start_time
            self.ctx.beat_idx = idx
            current_energy = float(event.get('clip_tension') or event.get('target_tension') or 0.5)
            selection_score = float(event.get('score', 0.0))

            self._process_and_add_clip(
                scene,
                clip_start,
                cut_duration,
                current_energy,
                selection_score,
            )

            scene['usage_count'] = scene.get('usage_count', 0) + 1
            self.ctx.cut_number += 1
            self.ctx.current_time = start_time + cut_duration

        # Flush remaining futures to finalize segments
        self._flush_pending_futures()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "AudioAnalysisResult",
    "SceneInfo",
    "OutputProfile",
    "ClipMetadata",
    "MontageContext",
    "MontageResult",
    # Main class
    "MontageBuilder",
]
