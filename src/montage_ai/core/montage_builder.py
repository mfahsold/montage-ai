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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np

from ..config import get_settings, Settings
from ..logger import logger
from .analysis_cache import get_analysis_cache


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
        avg = self.avg_energy
        if avg > 0.6:
            return "high"
        elif avg < 0.4:
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
class MontageContext:
    """
    Encapsulates all state for a montage job.

    Replaces the dozens of local variables flowing through create_montage().
    All mutable state is stored here, making the pipeline stages pure functions
    that take context and return results.
    """
    # Job identification
    job_id: str
    variant_id: int

    # Settings (immutable reference)
    settings: Settings

    # Paths
    input_dir: Path
    music_dir: Path
    assets_dir: Path
    output_dir: Path
    temp_dir: Path

    # Feature flags (may be modified by Creative Director)
    stabilize: bool = False
    upscale: bool = False
    enhance: bool = False

    # Creative Director instructions (JSON from LLM)
    editing_instructions: Optional[Dict[str, Any]] = None

    # Semantic query for clip selection (Phase 2: Semantic Storytelling)
    semantic_query: Optional[str] = None

    # Audio analysis results
    audio_result: Optional[AudioAnalysisResult] = None

    # Scene detection results
    all_scenes: List[SceneInfo] = field(default_factory=list)
    all_scenes_dicts: List[Dict[str, Any]] = field(default_factory=list)  # Legacy format
    video_files: List[str] = field(default_factory=list)

    # Output profile
    output_profile: Optional[OutputProfile] = None

    # Timeline state
    target_duration: float = 0.0
    current_time: float = 0.0
    beat_idx: int = 0
    cut_number: int = 0
    estimated_total_cuts: int = 0

    # Clip metadata for timeline export
    clips_metadata: List[ClipMetadata] = field(default_factory=list)

    # Cut pattern state (for dynamic pacing)
    current_pattern: Optional[List[float]] = None
    pattern_idx: int = 0

    # Previous clip state (for continuity rules)
    last_used_path: Optional[str] = None
    last_shot_type: Optional[str] = None
    last_clip_end_time: Optional[float] = None

    # Crossfade settings
    enable_xfade: bool = False
    xfade_duration: float = 0.3

    # Rendering
    output_filename: Optional[str] = None
    render_duration: float = 0.0
    logo_path: Optional[str] = None

    # Timing
    start_time: float = field(default_factory=time.time)

    def reset_timeline_state(self):
        """Reset timeline state for a fresh pass."""
        self.current_time = 0.0
        self.beat_idx = 0
        self.cut_number = 0
        self.clips_metadata = []
        self.current_pattern = None
        self.pattern_idx = 0
        self.last_used_path = None
        self.last_shot_type = None
        self.last_clip_end_time = None

    def get_story_position(self) -> float:
        """Get current position in story arc (0-1)."""
        if self.target_duration > 0:
            return self.current_time / self.target_duration
        return 0.0

    def get_story_phase(self) -> str:
        """Get current story phase based on position."""
        pos = self.get_story_position()
        if pos < 0.15:
            return "intro"
        elif pos < 0.40:
            return "build"
        elif pos < 0.70:
            return "climax"
        elif pos < 0.90:
            return "sustain"
        return "outro"

    def elapsed_time(self) -> float:
        """Time since job started."""
        return time.time() - self.start_time


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
    ):
        """
        Initialize the MontageBuilder.

        Args:
            variant_id: Variant number for this montage (1-based)
            settings: Optional Settings instance (uses global if None)
            editing_instructions: Optional Creative Director instructions
        """
        self.settings = settings or get_settings()
        self.variant_id = variant_id
        self.editing_instructions = editing_instructions

        # Initialize context
        self.ctx = self._create_context()

        # Component references (lazy initialized)
        self._monitor = None
        self._footage_pool = None
        self._progressive_renderer = None
        self._memory_manager = None
        self._clip_enhancer = None
        self._intelligent_selector = None

    def _create_context(self) -> MontageContext:
        """Create a fresh MontageContext from settings."""
        return MontageContext(
            job_id=self.settings.job_id,
            variant_id=self.variant_id,
            settings=self.settings,
            input_dir=self.settings.paths.input_dir,
            music_dir=self.settings.paths.music_dir,
            assets_dir=self.settings.paths.assets_dir,
            output_dir=self.settings.paths.output_dir,
            temp_dir=self.settings.paths.temp_dir,
            stabilize=self.settings.features.stabilize,
            upscale=self.settings.features.upscale,
            enhance=self.settings.features.enhance,
            editing_instructions=self.editing_instructions,
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
            self._setup_workspace()

            # Phase 2: Analyze
            self._analyze_assets()

            # Phase 3: Plan
            self._plan_montage()

            # Phase 4: Enhance (if enabled)
            if self.ctx.stabilize or self.ctx.upscale or self.ctx.enhance:
                self._enhance_assets()

            # Phase 5: Render
            self._render_output()

            # Phase 6: Cleanup
            self._cleanup()

            # Build result
            return MontageResult(
                success=True,
                output_path=self.ctx.output_filename,
                duration=self.ctx.current_time,
                cut_count=self.ctx.cut_number,
                render_time=self.ctx.render_duration,
                file_size_mb=self._get_output_file_size(),
                stats=self._collect_stats(),
            )

        except Exception as e:
            logger.error(f"❌ Montage build failed: {e}")
            self._cleanup()
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

    def _setup_workspace(self):
        """
        Phase 1: Initialize workspace.

        - Apply Creative Director effects overrides
        - Initialize monitoring
        - Initialize intelligent clip selector
        - Ensure directories exist
        """
        logger.info("   📁 Setting up workspace...")

        # Ensure temp directory exists
        os.makedirs(str(self.ctx.temp_dir), exist_ok=True)

        # Apply Creative Director effects (ENV takes precedence)
        self._apply_creative_director_effects()

        # Initialize monitor (if available)
        self._init_monitor()

        # Initialize intelligent clip selector (if available)
        self._init_intelligent_selector()

        logger.info(f"   🎨 Effects: STABILIZE={self.ctx.stabilize}, UPSCALE={self.ctx.upscale}, ENHANCE={self.ctx.enhance}")

    def _analyze_assets(self):
        """
        Phase 2: Analyze audio and video assets.

        - Load and analyze music (beats, tempo, energy)
        - Load videos and detect scenes
        - Run AI content analysis on scenes
        - Determine output profile
        """
        logger.info("\n   🎵 Analyzing assets...")

        # Analyze music
        self._analyze_music()

        # Detect scenes in videos
        self._detect_scenes()

        # Determine output profile
        self._determine_output_profile()

        # Initialize footage pool
        self._init_footage_pool()

    def _plan_montage(self):
        """
        Phase 3: Plan the montage timeline.

        - Calculate target duration
        - Initialize progressive renderer
        - Run clip selection loop (beat-synced, AI-scored)
        """
        logger.info("\n   📋 Planning montage...")

        # Initialize audio clip and calculate duration
        self._init_audio_duration()

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
        self.ctx.output_filename = os.path.join(
            str(self.ctx.output_dir),
            f"gallery_montage_{self.ctx.job_id}_v{self.variant_id}_{style_name}.mp4"
        )

        # Check for logo
        logo_files = self._get_files(self.ctx.assets_dir, ('.png', '.jpg'))
        self.ctx.logo_path = logo_files[0] if logo_files else None

        # Run the main assembly loop
        logger.info("   ✂️ Assembling cuts...")
        self._run_assembly_loop()

    def _enhance_assets(self):
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

    def _render_output(self):
        """
        Phase 5: Render final output.

        - Finalize progressive renderer (or legacy path)
        - Add audio
        - Add logo overlay (if present)
        """
        logger.info("\n   🎬 Rendering output...")

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

    def _cleanup(self):
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

    def _analyze_music(self):
        """Load and analyze music file with caching support."""
        from ..audio_analysis import get_beat_times, analyze_music_energy

        # Get music files
        music_files = self._get_files(self.ctx.music_dir, ('.mp3', '.wav'))
        if not music_files:
            raise ValueError("No music found in music directory")

        # Select music file (rotate through variants)
        music_index = (self.variant_id - 1) % len(music_files)
        music_path = music_files[music_index]

        # Try cache first
        cache = get_analysis_cache()
        cached = cache.load_audio(music_path)

        if cached:
            # Cache hit - use cached analysis
            logger.info("   ⚡ Using cached audio analysis")
            self.ctx.audio_result = AudioAnalysisResult(
                music_path=music_path,
                beat_times=np.array(cached.beat_times),
                tempo=cached.tempo,
                energy_times=np.array(cached.energy_times),
                energy_values=np.array(cached.energy_values),
                duration=cached.duration,
            )
        else:
            # Cache miss - analyze and cache
            beat_info = get_beat_times(music_path, verbose=self.settings.features.verbose)
            energy_profile = analyze_music_energy(music_path, verbose=self.settings.features.verbose)

            # Store in context
            self.ctx.audio_result = AudioAnalysisResult(
                music_path=music_path,
                beat_times=beat_info.beat_times,
                tempo=beat_info.tempo,
                energy_times=energy_profile.times,
                energy_values=energy_profile.rms,
                duration=beat_info.duration,
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
        """Detect scenes in all video files with caching support."""
        from ..scene_analysis import detect_scenes, analyze_scene_content
        from .analysis_cache import get_analysis_cache

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

        # Detect scenes in each video
        for v_path in video_files:
            # Check cache first
            cached = cache.load_scenes(v_path, threshold)
            if cached:
                cache_hits += 1
                scenes = [(s["start"], s["end"]) for s in cached.scenes]
            else:
                # Fresh detection
                scenes = detect_scenes(v_path, threshold=threshold)
                # Save to cache (convert to Scene-like objects)
                cache.save_scenes(v_path, threshold, scenes)

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

        # AI scene analysis (limit to first 20 for speed)
        logger.info("   🤖 AI Director is watching footage...")
        scenes_to_analyze = self.ctx.all_scenes[:20]
        for scene in scenes_to_analyze:
            meta = analyze_scene_content(scene.path, scene.midpoint)
            scene.meta = meta

        # Shuffle for variety
        random.shuffle(self.ctx.all_scenes)

        # Create legacy dict format (needed for footage_manager compatibility)
        self.ctx.all_scenes_dicts = [s.to_dict() for s in self.ctx.all_scenes]

        logger.info(f"   📹 Found {len(self.ctx.all_scenes)} scenes in {len(video_files)} videos")

    def _determine_output_profile(self):
        """Determine output profile from input footage."""
        from ..video_metadata import determine_output_profile as _determine_profile

        profile = _determine_profile(self.ctx.video_files)

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
        cut_patterns = [
            [4, 4, 4, 4],       # Steady pace
            [2, 2, 4, 8],       # Accelerate then hold
            [8, 4, 2, 2],       # Long shot into fast cuts
            [1, 1, 2, 3, 5],    # Fibonacci
            [1.5, 1.5, 5],      # Syncopated
        ]

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

            # Find best start point
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

        pacing_speed = "dynamic"
        if self.ctx.editing_instructions is not None:
            pacing_speed = self.ctx.editing_instructions.get('pacing', {}).get('speed', 'dynamic')

        if pacing_speed == "very_fast":
            return 1
        elif pacing_speed == "fast":
            return 2 if tempo < 130 else 4
        elif pacing_speed == "medium":
            return 4
        elif pacing_speed == "slow":
            return 8
        elif pacing_speed == "very_slow":
            return 16 if tempo < 100 else 8
        else:  # dynamic
            if self.ctx.current_pattern is None or self.ctx.pattern_idx >= len(self.ctx.current_pattern):
                self.ctx.current_pattern = calculate_dynamic_cut_length(
                    current_energy, tempo, self.ctx.current_time,
                    self.ctx.target_duration, cut_patterns
                )
                self.ctx.pattern_idx = 0

            beats = self.ctx.current_pattern[self.ctx.pattern_idx]
            self.ctx.pattern_idx += 1
            return beats

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

    def _select_clip(
        self,
        available_footage,
        current_energy: float,
        unique_videos: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Score and select the best clip for this cut."""
        from ..scene_analysis import calculate_visual_similarity, detect_motion_blur

        # Convert to scene dicts for scoring
        valid_scenes = [
            s for s in self.ctx.all_scenes_dicts
            if id(s) in [c.clip_id for c in available_footage]
        ]

        if not valid_scenes:
            return None, 0

        candidates = valid_scenes[:20]
        best_score = -1000
        selected_scene = candidates[0]

        for scene in candidates:
            score = 0

            # Rule 1: Fresh clips bonus
            scene_id = id(scene)
            footage_clip = next((c for c in available_footage if c.clip_id == scene_id), None)
            if footage_clip:
                if footage_clip.usage_count == 0:
                    score += 50
                elif footage_clip.usage_count == 1:
                    score += 20
                else:
                    score -= footage_clip.usage_count * 10

                # Story arc phase matching
                story_position = self.ctx.get_story_position()
                phase = self.ctx.get_story_phase()
                if phase == "intro" and current_energy < 0.4:
                    score += 15
                elif phase == "build" and 0.4 <= current_energy < 0.7:
                    score += 15
                elif phase == "climax" and current_energy >= 0.7:
                    score += 15
                elif phase == "outro" and current_energy < 0.5:
                    score += 15

            # Rule 2: Avoid jump cuts
            if scene['path'] == self.ctx.last_used_path:
                if unique_videos > 1:
                    score -= 50
                else:
                    score -= 5

            # Rule 3: AI Content Matching
            meta = scene.get('meta', {})
            action = meta.get('action', 'medium')
            shot = meta.get('shot', 'medium')

            if current_energy > 0.6 and action == 'high':
                score += 20
            if current_energy < 0.4 and action == 'low':
                score += 20

            # Rule 4: Shot Variation
            if self.ctx.last_shot_type and shot == self.ctx.last_shot_type:
                score -= 10
            else:
                score += 10

            # Rule 5: Match cut detection
            if self.ctx.last_clip_end_time is not None and self.ctx.last_used_path is not None:
                try:
                    similarity = calculate_visual_similarity(
                        self.ctx.last_used_path, self.ctx.last_clip_end_time,
                        scene['path'], scene['start']
                    )
                    if similarity > 0.7:
                        score += 30
                except Exception:
                    pass

            # Rule 6: Semantic matching (Phase 2: Semantic Storytelling)
            if self.ctx.semantic_query and (meta.get('tags') or meta.get('caption')):
                try:
                    from ..semantic_matcher import get_semantic_matcher
                    matcher = get_semantic_matcher()
                    if matcher.is_available:
                        sem_result = matcher.match_query_to_clip(self.ctx.semantic_query, meta)
                        # Up to +40 points for strong semantic match
                        score += int(sem_result.overall_score * 40)
                except Exception:
                    pass

            # Randomness factor
            score += random.randint(-15, 15)
            scene['_heuristic_score'] = score

        # Probabilistic selection from top candidates
        candidates.sort(key=lambda x: x.get('_heuristic_score', -1000), reverse=True)
        top_n = min(3, len(candidates))
        if top_n > 0:
            top_candidates = candidates[:top_n]
            min_score = min(c.get('_heuristic_score', 0) for c in top_candidates)
            weights = [c.get('_heuristic_score', 0) - min_score + 10 for c in top_candidates]
            selected_scene = random.choices(top_candidates, weights=weights, k=1)[0]
            best_score = selected_scene.get('_heuristic_score', 0)

        return selected_scene, best_score

    def _process_and_add_clip(
        self,
        scene: Dict[str, Any],
        clip_start: float,
        cut_duration: float,
        current_energy: float,
        selection_score: float
    ):
        """Process clip (enhance if needed) and add to timeline."""
        # Track enhancements
        stabilize_applied = False
        upscale_applied = False
        enhance_applied = False

        # Generate temp paths
        temp_clip_name = f"temp_clip_{self.ctx.beat_idx}_{random.randint(0, 9999)}.mp4"
        temp_clip_path = os.path.join(str(self.ctx.temp_dir), temp_clip_name)

        # Extract subclip
        self._extract_subclip(scene['path'], clip_start, cut_duration, temp_clip_path)

        # Apply enhancements
        current_path = temp_clip_path
        temp_files = [temp_clip_path]

        if self.ctx.stabilize and self._clip_enhancer:
            stab_path = os.path.join(str(self.ctx.temp_dir), f"stab_{temp_clip_name}")
            result = self._clip_enhancer.stabilize(current_path, stab_path)
            if result != current_path:
                current_path = result
                temp_files.append(stab_path)
                stabilize_applied = True

        if self.ctx.upscale and self._clip_enhancer:
            upscale_path = os.path.join(str(self.ctx.temp_dir), f"upscale_{temp_clip_name}")
            result = self._clip_enhancer.upscale(current_path, upscale_path)
            if result != current_path:
                current_path = result
                temp_files.append(upscale_path)
                upscale_applied = True

        if self.ctx.enhance and self._clip_enhancer:
            enhance_path = os.path.join(str(self.ctx.temp_dir), f"enhance_{temp_clip_name}")
            result = self._clip_enhancer.enhance(current_path, enhance_path)
            if result != current_path:
                current_path = result
                temp_files.append(enhance_path)
                enhance_applied = True

        # Add to progressive renderer or legacy clips list
        if self._progressive_renderer:
            # Progressive path: write normalized clip and add to renderer
            final_clip_path = os.path.join(
                str(self.ctx.temp_dir),
                f"clip_{self.ctx.job_id}_{self.ctx.cut_number:04d}.mp4"
            )
            self._normalize_clip(current_path, final_clip_path)
            self._progressive_renderer.add_clip_path(final_clip_path)

            # Cleanup intermediate temp files
            for tf in temp_files:
                try:
                    if os.path.exists(tf):
                        os.remove(tf)
                except Exception:
                    pass

        # Store metadata
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
            enhancements={
                'stabilized': stabilize_applied,
                'upscaled': upscale_applied,
                'enhanced': enhance_applied,
            }
        )
        self.ctx.clips_metadata.append(clip_meta)

    def _extract_subclip(
        self, video_path: str, start: float, duration: float, output_path: str
    ):
        """Extract a subclip using FFmpeg."""
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)

    def _normalize_clip(self, input_path: str, output_path: str):
        """Normalize clip for concatenation."""
        import subprocess

        profile = self.ctx.output_profile
        if not profile:
            # Fallback: just copy
            import shutil
            shutil.copy(input_path, output_path)
            return

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={profile.width}:{profile.height}:force_original_aspect_ratio=decrease,pad={profile.width}:{profile.height}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(profile.fps),
            "-c:v", profile.codec,
            "-pix_fmt", profile.pix_fmt,
            "-crf", str(self.settings.encoding.crf),
            "-preset", "fast",
            "-an",  # No audio for individual clips
            output_path
        ]

        if profile.profile:
            cmd.extend(["-profile:v", profile.profile])
        if profile.level:
            cmd.extend(["-level", profile.level])

        subprocess.run(cmd, capture_output=True, timeout=60)


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
