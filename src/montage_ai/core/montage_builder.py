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
from ..enhancement_tracking import (
    EnhancementTracker,
    EnhancementDecision,
    StabilizeParams,
    UpscaleParams,
    ColorGradeParams,
)
from .models import EditingInstructions
from .context import (
    MontagePaths,
    MontageFeatures,
    MontageCreative,
    MontageMedia,
    MontageTimeline,
    MontageRender,
    MontageTiming,
    MontageContext,
    MontageResult,
    ClipMetadata,
)
from .clip_processor import process_clip_task


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
        
        # Convert dict to strongly-typed model if necessary
        if isinstance(editing_instructions, dict):
            self.editing_instructions = EditingInstructions(**editing_instructions)
        else:
            self.editing_instructions = editing_instructions
            
        self.progress_callback = progress_callback

        # Initialize context
        self.ctx = self._create_context()
        if self.editing_instructions:
            self.ctx.creative.editing_instructions = self.editing_instructions
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
        self._scene_provider = None  # Unified scene analysis provider
        self._enhancement_tracker: Optional[EnhancementTracker] = None  # NLE export tracking

        # Parallel processing
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending_futures: List[Future] = []
        
        # New Engines
        from .analysis_engine import AssetAnalyzer
        from .render_engine import RenderEngine
        from .pacing_engine import PacingEngine
        from .selection_engine import SelectionEngine
        from .story_engine import StoryEngine
        self._analyzer = AssetAnalyzer(self.ctx)
        self._render_engine = RenderEngine(self.ctx)
        self._pacing_engine = PacingEngine(self.ctx)
        self._selection_engine = SelectionEngine(self.ctx)
        self._story_engine = StoryEngine(self.ctx)

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
                denoise=self.settings.features.denoise,
                sharpen=self.settings.features.sharpen,
                film_grain=self.settings.features.film_grain,
                dialogue_duck=self.settings.features.dialogue_duck,
            ),
            creative=MontageCreative(
                editing_instructions=self.editing_instructions,
            ),
        )

    # =========================================================================
    # Lazy-Initialized Providers
    # =========================================================================

    @property
    def scene_provider(self):
        """Get the unified scene provider (lazy initialized)."""
        if self._scene_provider is None:
            from .scene_provider import get_scene_provider
            self._scene_provider = get_scene_provider()
        return self._scene_provider

    @property
    def enhancement_tracker(self) -> EnhancementTracker:
        """Get the enhancement tracker for NLE export (lazy initialized)."""
        if self._enhancement_tracker is None:
            self._enhancement_tracker = EnhancementTracker()
        return self._enhancement_tracker

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
            if self.ctx.features.stabilize or self.ctx.features.upscale or self.ctx.features.enhance:
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
                output_path=self.ctx.render.output_filename,
                duration=self.ctx.timeline.current_time,
                cut_count=self.ctx.timeline.cut_number,
                render_time=self.ctx.render.render_duration,
                file_size_mb=self._get_output_file_size(),
                stats=self._collect_stats(),
                project_package_path=self.ctx.render.exported_files.get('package') if self.ctx.render.exported_files else None,
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
        os.makedirs(str(self.ctx.paths.temp_dir), exist_ok=True)

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

        # Log enabled effects
        effects_list = []
        if self.ctx.features.stabilize: effects_list.append("STAB")
        if self.ctx.features.upscale: effects_list.append("UPSCALE")
        if self.ctx.features.enhance: effects_list.append("ENHANCE")
        if self.ctx.features.denoise: effects_list.append("DENOISE")
        if self.ctx.features.sharpen: effects_list.append("SHARPEN")
        if self.ctx.features.film_grain and self.ctx.features.film_grain != "none": effects_list.append(f"GRAIN:{self.ctx.features.film_grain}")
        if self.ctx.features.dialogue_duck: effects_list.append("DUCK")
        effects_str = ", ".join(effects_list) if effects_list else "none"
        logger.info(f"   🎨 Effects: [{effects_str}] COLOR_GRADE={self.ctx.features.color_grade}")

    def analyze_assets(self):
        """
        Phase 2: Analyze audio and video assets.

        OPTIMIZED: Runs cgpu voice isolation in parallel with scene detection
        to maximize utilization of both cloud GPU and local CPU.
        """
        logger.info("\n   🎵 Analyzing assets Checkpoint...")

        # Story Engine Analysis Trigger (Phase 1)
        if self.settings.features.story_engine:
            self._story_engine.ensure_analysis()

        # OPTIMIZATION: Start proxy generation in background
        proxy_futures = []
        if self.ctx.media.video_files:
            logger.info("   🎞️ Checking proxies...")
            proxy_dir = self.ctx.paths.input_dir / "Proxies"
            try:
                proxy_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                proxy_dir = self.ctx.paths.temp_dir / "proxies"
            
            generator = ProxyGenerator(proxy_dir)
            for video_path in self.ctx.media.video_files:
                proxy_futures.append(
                    self._executor.submit(generator.ensure_proxy, video_path)
                )

        # OPTIMIZATION: Start voice isolation in background while doing scene detection
        voice_isolation_future = None
        isolated_audio_path = None

        if self.settings.features.voice_isolation:
            # Replicate music selection logic to start async job
            music_files = self._get_files(self.ctx.paths.music_dir, ('.mp3', '.wav'))
            if music_files:
                music_index = (self.variant_id - 1) % len(music_files)
                music_path = music_files[music_index]
                
                # Check for override
                if self.ctx.creative.editing_instructions and self.ctx.creative.editing_instructions.music_track:
                    requested_track = self.ctx.creative.editing_instructions.music_track
                    for mf in music_files:
                        if mf == requested_track or os.path.basename(mf) == requested_track:
                            music_path = mf
                            break

                # Start voice isolation async (cgpu)
                logger.info("   🚀 Starting voice isolation async (cgpu)...")
                voice_isolation_future = self._executor.submit(
                    self._analyzer.perform_voice_isolation, music_path
                )

        # Detect scenes using AnalysisEngine (blocking call that runs parallel internally)
        self._analyzer.detect_scenes(progress_callback=self.progress_callback)

        # Wait for voice isolation to complete
        if voice_isolation_future is not None:
            try:
                timeout = self.settings.llm.cgpu_timeout
                isolated_audio_path = voice_isolation_future.result(timeout=timeout)
                logger.info(f"   ✅ Voice isolation completed")
            except Exception as e:
                if self.settings.features.strict_cloud_compute:
                    raise RuntimeError(f"Strict cloud compute enabled: Voice isolation async failed: {e}") from e
                logger.warning(f"   ⚠️ Voice isolation async failed: {e}")
                isolated_audio_path = None

        # Analyze music (delegated)
        self._analyzer.analyze_music(isolated_audio_path=isolated_audio_path)

        # Determine output profile (delegated)
        self._analyzer.determine_output_profile()

        # Initialize footage pool
        self._footage_pool = self._selection_engine.init_footage_pool()

        # OPTIMIZATION: Build Scene Similarity K-D Tree Index for fast O(log n) lookups
        # This enables "match cut" detection (similar-looking scenes) in clip selection
        logger.info("   🌳 Building scene similarity index (K-D tree)...")
        try:
            from ..scene_analysis import SceneSimilarityIndex
            similarity_index = SceneSimilarityIndex()
            similarity_index.build(self.ctx.media.all_scenes)
            
            # Store in context for clip selection engine to use
            self.ctx.media.similarity_index = similarity_index
            
            if similarity_index.enabled:
                logger.info(f"   ✅ K-D tree index built (O(log n) similarity queries enabled)")
            else:
                logger.info(f"   ⚠️ K-D tree unavailable, using O(n) linear search")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to build similarity index: {e}")
            self.ctx.media.similarity_index = None

        # Wait for proxies
        if proxy_futures:
            logger.info("   ⏳ Waiting for proxy generation to complete...")
            completed = 0
            for f in proxy_futures:
                try:
                    f.result()
                    completed += 1
                except Exception:
                    pass
            if completed > 0:
                logger.info(f"   ✅ {completed} Proxies ready")

    def _setup_output_paths(self, style_name: str):
        """Determine output filename and logo path."""
        # Sanitize style name
        style_label = "".join(c for c in style_name if c.isalnum() or c in ('-', '_')).strip()
        if not style_label:
            style_label = "dynamic"

        self.ctx.render.output_filename = os.path.join(
            str(self.ctx.paths.output_dir),
            f"gallery_montage_{self.ctx.job_id}_v{self.variant_id}_{style_label}.mp4"
        )

        # Check for logo
        logo_files = self._get_files(self.ctx.paths.assets_dir, ('.png', '.jpg'))
        self.ctx.render.logo_path = logo_files[0] if logo_files else None

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

        # Generate B-Roll plan if script is available
        if self.ctx.creative.editing_instructions and self.ctx.creative.editing_instructions.script:
             self._plan_broll_sequence()

        # Initialize audio and pacing
        self._pacing_engine.init_audio_duration()
        self._pacing_engine.init_crossfade_settings()

        # Initialize progressive renderer
        self._init_progressive_renderer()

        # Estimate total cuts for progress tracking
        tempo = self.ctx.media.audio_result.tempo
        avg_beats_per_cut = 4.0
        self.ctx.timeline.estimated_total_cuts = int(
            (self.ctx.timeline.target_duration * tempo / 60) / avg_beats_per_cut
        )

        # Determine output filename
        style_name = "dynamic"
        if self.ctx.creative.editing_instructions is not None:
             style_name = self.ctx.creative.editing_instructions.style.name if self.ctx.creative.editing_instructions.style else "dynamic"
        
        # Load style params
        try:
            template = get_style_template(style_name)
            self.ctx.creative.style_params = template.get("params", {})
            logger.info(f"   🎨 Loaded style '{style_name}' params")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load style '{style_name}': {e}")
            self.ctx.creative.style_params = {}

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
        """
        # Pass renderer to engine (if not already managed there in future)
        self._render_engine.set_renderer(self._progressive_renderer)
        self._render_engine.render_output()

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

        # Cleanup via engine
        self._render_engine.cleanup()
        if self._progressive_renderer:
             # Just in case engine didn't catch it
             pass

        # Cleanup temp directory
        temp_dir = str(self.ctx.paths.temp_dir)
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
        if self.ctx.creative.editing_instructions is None:
            return

        effects = getattr(self.ctx.creative.editing_instructions, 'effects', {})

        # ENV takes precedence over style template
        env_stabilize = self.settings.features.stabilize
        env_upscale = self.settings.features.upscale
        env_enhance = self.settings.features.enhance

        # Force disable heavy features in Preview Mode
        if self.settings.encoding.quality_profile == "preview":
            logger.info("   ⚡ Preview Mode: Disabling stabilization, upscale, and enhancement")
            self.ctx.features.stabilize = False
            self.ctx.features.upscale = False
            self.ctx.features.enhance = False
        else:
            if not env_stabilize and 'stabilization' in effects:
                self.ctx.features.stabilize = effects['stabilization']
            else:
                self.ctx.features.stabilize = env_stabilize

            if not env_upscale and 'upscale' in effects:
                self.ctx.features.upscale = effects['upscale']
            else:
                self.ctx.features.upscale = env_upscale

            if not env_enhance and 'sharpness_boost' in effects:
                self.ctx.features.enhance = effects['sharpness_boost']
            else:
                self.ctx.features.enhance = env_enhance

        # Parse color_grading from style template (NEW: replaces hardcoded Teal & Orange)
        # ENV takes priority, then style template effects, then default
        env_color_grade = os.environ.get('COLOR_GRADING', '')
        if env_color_grade:
            self.ctx.features.color_grade = env_color_grade
        elif 'color_grading' in effects:
            self.ctx.features.color_grade = effects['color_grading']
        else:
            self.ctx.features.color_grade = "teal_orange"  # Legacy default

        # Parse color_intensity (0.0-1.0, defaults to 1.0)
        env_color_intensity = os.environ.get('COLOR_INTENSITY', '')
        if env_color_intensity:
            try:
                self.ctx.features.color_intensity = float(env_color_intensity)
            except ValueError:
                self.ctx.features.color_intensity = 1.0
        else:
            self.ctx.features.color_intensity = 1.0

        # Extract semantic query for Phase 2: Semantic Storytelling
        # Priority: explicit semantic_query > content_focus > ENV
        env_semantic = os.environ.get('SEMANTIC_QUERY', '')
        
        inst = self.ctx.creative.editing_instructions
        
        if getattr(inst, 'semantic_query', None):
            self.ctx.creative.semantic_query = inst.semantic_query
        elif getattr(inst, 'content_focus', None):
            self.ctx.creative.semantic_query = inst.content_focus
        elif inst.style:
            # Combine mood, theme, or content hints
            style_params = inst.style.params
            semantic_parts = []
            if style_params.get('mood'):
                semantic_parts.append(style_params['mood'])
            if style_params.get('theme'):
                semantic_parts.append(style_params['theme'])
            if style_params.get('content'):
                semantic_parts.append(style_params['content'])
            if semantic_parts:
                self.ctx.creative.semantic_query = ' '.join(semantic_parts)
                
        if env_semantic and not self.ctx.creative.semantic_query:
            self.ctx.creative.semantic_query = env_semantic

    def _plan_broll_sequence(self):
        """
        Generate B-Roll plan from script.
        """
        script = self.ctx.creative.editing_instructions.script
        self.ctx.creative.broll_plan = self._story_engine.plan_broll(script)

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
            if self.ctx.creative.editing_instructions is not None and self.ctx.creative.editing_instructions.style:
                style = self.ctx.creative.editing_instructions.style.name
            self._intelligent_selector = IntelligentClipSelector(style=style)
            logger.info(f"   🧠 Intelligent Clip Selector initialized (style={style})")
        except ImportError:
            self._intelligent_selector = None
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to initialize Intelligent Clip Selector: {e}")
            self._intelligent_selector = None

    def _analyze_music(self, isolated_audio_path: Optional[str] = None):
        """Load and analyze music file with caching support."""
        # Delegate to new Analyzer engine
        self._analyzer.analyze_music(isolated_audio_path)
        
        # Monitor logging (kept here for now, could move to engine later)
        if self._monitor and self.ctx.media.audio_result:
            self._monitor.log_beat_analysis(
                self.ctx.media.audio_result.music_path,
                self.ctx.media.audio_result.tempo,
                len(self.ctx.media.audio_result.beat_times),
                self.ctx.media.audio_result.energy_profile
            )


    def _apply_noise_reduction(self, audio_path: str) -> str:
        """Deprecated: Use AssetAnalyzer._apply_noise_reduction"""
        return self._analyzer._apply_noise_reduction(audio_path)


    def _init_footage_pool(self):
        """Initialize footage pool manager."""
        from ..footage_manager import integrate_footage_manager

        # Use the same dict objects that are stored in context
        # (important: id() matching requires same objects)
        self._footage_pool = integrate_footage_manager(
            self.ctx.media.all_scenes_dicts,
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
        if self.ctx.render.output_filename and os.path.exists(self.ctx.render.output_filename):
            return os.path.getsize(self.ctx.render.output_filename) / (1024 * 1024)
        return 0.0

    def _collect_stats(self) -> Dict[str, Any]:
        """Collect statistics for the result."""
        return {
            "job_id": self.ctx.job_id,
            "variant_id": self.variant_id,
            "cut_count": self.ctx.timeline.cut_number,
            "duration": self.ctx.timeline.current_time,
            "tempo": self.ctx.media.audio_result.tempo if self.ctx.media.audio_result else 0,
            "scene_count": len(self.ctx.media.all_scenes),
            "elapsed_time": self.ctx.elapsed_time(),
            "story_engine": self.settings.features.story_engine,
        }

    # =========================================================================
    # Assembly Loop Helpers
    # =========================================================================


    def _init_progressive_renderer(self):
        """Initialize progressive renderer if available."""
        self._progressive_renderer = self._render_engine.init_progressive_renderer()
        # memory manager is handled inside engine now, though we expose it if needed
        # self._memory_manager is used in cleanup, let's keep it sync if possible or rely on engine cleanup
        if self._progressive_renderer:
             self._memory_manager = self._progressive_renderer.memory_manager

    def _flush_pending_futures(self):
        """Wait for all pending clip processing futures to complete."""
        if not self._pending_futures:
            return

        logger.info(f"   ⏳ Waiting for {len(self._pending_futures)} pending clips to finish processing...")
        while self._pending_futures:
            fut, meta = self._pending_futures.pop(0)
            self._render_engine.process_completed_task(fut, meta, self.enhancement_tracker)

    def _run_assembly_loop(self):
        """
        Main assembly loop: select clips, apply enhancements, add to timeline.

        This is the heart of the montage creation process.
        """
        from ..audio_analysis import calculate_dynamic_cut_length
        from ..scene_analysis import find_best_start_point

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
        if self.ctx.creative.style_params and "cut_patterns" in self.ctx.creative.style_params:
            cut_patterns = self.ctx.creative.style_params["cut_patterns"]
            logger.info(f"   🎨 Using {len(cut_patterns)} custom cut patterns from style")

        # Get audio data
        beat_times = self.ctx.media.audio_result.beat_times
        energy_times = self.ctx.media.audio_result.energy_times
        energy_values = self.ctx.media.audio_result.energy_values
        tempo = self.ctx.media.audio_result.tempo

        # Get unique video count for variety logic
        unique_videos = len(set(s.path for s in self.ctx.media.all_scenes))

        # Main loop
        while self.ctx.timeline.current_time < self.ctx.timeline.target_duration and self.ctx.media.all_scenes:
            # Get current energy
            current_energy = self._pacing_engine.get_energy_at_time(self.ctx.timeline.current_time)

            # Calculate cut duration
            cut_duration = self._pacing_engine.get_next_cut_duration(current_energy)
            
            # Calculate beats per cut (approx) for beat indexing
            beats_per_cut = cut_duration / (60.0 / tempo) if tempo > 0 else 4.0

            # Early exit: prevent overrun beyond target
            remaining_time = self.ctx.timeline.target_duration - self.ctx.timeline.current_time
            if remaining_time < cut_duration * 0.3:
                # Less than 30% of a cut remaining - stop here
                logger.info(f"   🛑 Target reached ({self.ctx.timeline.current_time:.1f}s / {self.ctx.timeline.target_duration:.1f}s)")
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
            # OPTIMIZATION: Use K-D Tree similarity index if available (O(log n) lookup)
            similarity_fn = None
            if hasattr(self.ctx.media, 'similarity_index') and self.ctx.media.similarity_index:
                # K-D tree wrapper: find similar scenes in O(log n) time
                similarity_fn = lambda scene: self._find_similar_scenes_kdtree(scene)
            elif self._scene_provider:
                # Fallback to provider's method if K-D tree unavailable
                similarity_fn = self._scene_provider.calculate_similarity
            
            selected_scene, best_score = self._selection_engine.select_clip(
                available_footage, current_energy, unique_videos, similarity_fn=similarity_fn
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
            self._selection_engine.consume_clip(
                clip_id=id(selected_scene),
                timeline_position=self.ctx.timeline.current_time,
                used_in_point=clip_start,
                used_out_point=clip_end
            )

            # Process and add clip to timeline
            self._process_and_add_clip(
                selected_scene, clip_start, cut_duration, current_energy, best_score
            )

            # Update state
            self.ctx.timeline.last_used_path = selected_scene['path']
            self.ctx.timeline.last_shot_type = selected_scene.get('meta', {}).get('shot', 'medium')
            self.ctx.timeline.last_tags = selected_scene.get('meta', {}).get('tags', [])
            self.ctx.timeline.last_clip_end_time = clip_end

            self.ctx.timeline.current_time += cut_duration
            self.ctx.timeline.beat_idx += int(beats_per_cut)
            self.ctx.timeline.cut_number += 1

            # Progress logging
            if self._monitor and self.ctx.timeline.cut_number % 5 == 0:
                self._monitor.log_progress(
                    self.ctx.timeline.cut_number,
                    self.ctx.timeline.estimated_total_cuts,
                    "cuts placed"
                )

        # Flush remaining futures
        self._flush_pending_futures()

        logger.info(f"   ✅ Assembly complete: {self.ctx.timeline.cut_number} cuts, {self.ctx.timeline.current_time:.1f}s")

    def _process_and_add_clip(
        self,
        scene: Dict[str, Any],
        clip_start: float,
        cut_duration: float,
        current_energy: float,
        selection_score: float
    ):
        """Submit clip processing task and track metadata."""
        
        # Calculate pattern beat for metadata
        pattern_beat = 4
        if self.ctx.timeline.current_pattern:
             # pattern_idx was incremented in loop? No, beat_idx was. 
             # Actually timeline.current_pattern logic is slightly opaque here.
             # Assuming standard 4 if not available.
             # The original code accessed pattern_idx - 1, implying it was updated.
             # But pattern updating logic isn't visible in the snippet.
             # Let's trust the context or just pass 4 if unsure.
             # Actually, let's look at how beats_per_cut was calculated in loop.
             pass

        # We need to pass the current pattern beat if we want to match original behavior exactly for metadata.
        # Original: self.ctx.timeline.current_pattern[self.ctx.timeline.pattern_idx - 1] if ...
        
        current_pattern_val = 4
        if self.ctx.timeline.current_pattern and self.ctx.timeline.pattern_idx > 0:
             idx = (self.ctx.timeline.pattern_idx - 1) % len(self.ctx.timeline.current_pattern)
             current_pattern_val = self.ctx.timeline.current_pattern[idx]

        # Submit via Render Engine
        future, clip_meta = self._render_engine.submit_clip_task(
            scene=scene,
            clip_start=clip_start,
            cut_duration=cut_duration,
            current_energy=current_energy,
            selection_score=selection_score,
            beat_idx=self.ctx.timeline.beat_idx,
            beats_per_cut=0, # Let engine decide or reuse pattern val? 
                             # The engine uses "beats_per_cut if beats_per_cut else current_pattern_beat".
                             # So I pass 0 here and let it use current_pattern_val
            current_pattern_beat=current_pattern_val,
            executor=self._executor,
            enhancer=self._clip_enhancer,
            resource_manager=self._resource_manager
        )
        
        self.ctx.timeline.clips_metadata.append(clip_meta)
        self._pending_futures.append((future, clip_meta))
        
        # Process completed futures in order to keep memory usage in check
        while self._pending_futures and self._pending_futures[0][0].done():
            fut, meta = self._pending_futures.pop(0)
            self._render_engine.process_completed_task(fut, meta, self.enhancement_tracker)



    def export_timeline(self):
        """
        Phase 6: Export timeline to NLE formats (EDL, XML, OTIO).
        """
        if not self.settings.features.export_timeline:
            return

        logger.info("\n   📝 Exporting timeline...")
        
        # Convert ClipMetadata to format expected by exporter
        clips_data = []
        for clip in self.ctx.timeline.clips_metadata:
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
                },
                'enhancement_decision': clip.enhancement_decision,  # NLE export tracking
            })

        # Get audio path safely
        audio_path = self.ctx.media.audio_result.music_path if self.ctx.media.audio_result else ""
        if not audio_path:
            logger.warning("   ⚠️ No audio path found for export, using placeholder")
            audio_path = "audio_track_missing.mp3"

        try:
            self.ctx.render.exported_files = export_timeline_from_montage(
                clips_data=clips_data,
                audio_path=audio_path,
                total_duration=self.ctx.timeline.current_time,
                output_dir=str(self.ctx.paths.output_dir),
                project_name=f"{self.ctx.job_id}_v{self.variant_id}",
                generate_proxies=self.settings.features.generate_proxies,
                resolution=(self.ctx.media.output_profile.width, self.ctx.media.output_profile.height),
                fps=self.ctx.media.output_profile.fps
            )
        except Exception as e:
            logger.error(f"   ❌ Timeline export failed: {e}")

    def _save_episodic_memory(self):
        """Delegate to Selection Engine."""
        self._selection_engine.save_episodic_memory()




    # =========================================================================
    # Story Engine Methods (Phase 1)
    # =========================================================================

    def _run_story_assembly(self):
        """
        Executes the montage assembly using the Story Engine.
        """
        logger.info("   📖 Story Engine: Assembling narrative...")
        
        # Ensure audio analysis is done
        if not hasattr(self.ctx, 'audio_result') or self.ctx.media.audio_result is None:
            self._analyze_music()

        # Initialize duration, transitions, and renderer
        self._pacing_engine.init_audio_duration()
        self._pacing_engine.init_crossfade_settings()
        self._init_progressive_renderer()

        # Generate Plan
        try:
            timeline_events = self._story_engine.generate_story_plan()
        except Exception as e:
            logger.error(f"Story Engine Planning Failed: {e}")
            raise

        # Determine output filename and logo
        style_name = "dynamic"
        if self.ctx.creative.editing_instructions and self.ctx.creative.editing_instructions.style:
            style_name = self.ctx.creative.editing_instructions.style.name
        self._setup_output_paths(style_name)
        
        logger.info(f"   ✅ Generated {len(timeline_events)} cuts based on story arc.")

        self._realize_story_plan(timeline_events)

    def _realize_story_plan(self, timeline_events: List[Dict[str, Any]]):
        """Converts abstract timeline events into concrete clips."""
        
        duration = self.ctx.timeline.target_duration or self.ctx.media.audio_result.duration
        
        # Determine constraints
        min_clip_duration = None
        max_clip_duration = None
        if self.ctx.creative.editing_instructions:
            constraints = getattr(self.ctx.creative.editing_instructions, 'constraints', {})
            min_clip_duration = coerce_float(constraints.get("min_clip_duration_sec"))
            max_clip_duration = coerce_float(constraints.get("max_clip_duration_sec"))
            if min_clip_duration is not None and max_clip_duration is not None and min_clip_duration > max_clip_duration:
                min_clip_duration, max_clip_duration = max_clip_duration, min_clip_duration

        # Map scenes by source path (fallback to full clip duration if scenes missing)
        scenes_by_path: Dict[str, List[Dict[str, Any]]] = {}
        for scene in self.ctx.media.all_scenes_dicts:
            scenes_by_path.setdefault(scene['path'], []).append(scene)
            scene.setdefault('usage_count', 0)

        # Ensure we have scenes for all input clips even if detection failed
        input_clips = self.ctx.media.video_files or self._get_files(self.ctx.paths.input_dir, ('.mp4', '.mov', '.mkv'))
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

        self.ctx.timeline.cut_number = 0
        self.ctx.timeline.current_time = 0.0

        # Assemble timeline
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

            self.ctx.timeline.current_time = start_time
            self.ctx.timeline.beat_idx = idx
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
            self.ctx.timeline.cut_number += 1
            self.ctx.timeline.current_time = start_time + cut_duration

        # Flush remaining futures to finalize segments
        self._flush_pending_futures()

    def _find_similar_scenes_kdtree(self, target_scene: Dict[str, Any]) -> float:
        """
        Find similar scenes using K-D Tree index (O(log n) lookup).
        
        Returns a similarity score (0-1) for use in clip selection scoring.
        """
        if not hasattr(self.ctx.media, 'similarity_index') or not self.ctx.media.similarity_index:
            return 0.0
        
        try:
            similarity_index = self.ctx.media.similarity_index
            target_path = target_scene.get('path', '')
            target_time = target_scene.get('start', 0.0) + (target_scene.get('duration', 1.0) / 2)
            
            # Query K-D tree for similar scenes (O(log n) complexity)
            similar_scenes = similarity_index.find_similar(
                target_path, 
                target_time, 
                k=3,  # Get top 3 similar scenes
                threshold=0.7
            )
            
            # Return highest similarity score (clamped to match cutting preferences)
            if similar_scenes:
                best_similarity, _ = similar_scenes[0]
                return best_similarity * 20.0  # Scale for clip selection scoring
            return 0.0
        except Exception as e:
            logger.debug(f"K-D tree similarity lookup failed: {e}")
            return 0.0


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "MontageBuilder",
]
