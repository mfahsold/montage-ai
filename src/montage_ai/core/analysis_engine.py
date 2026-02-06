"""
Analysis Engine Module

Extracts analysis logic from MontageBuilder to separate concerns.
Handles Audio Analysis (Beats, Energy, Voice Isolation) and Video Scene Detection.
"""

import os
import random
import psutil
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict, Iterable
# OPTIMIZATION Phase 3: ProcessPoolExecutor for CPU-bound tasks (bypasses GIL)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np

from ..logger import logger
from ..config import Settings
from .context import AudioAnalysisResult, SceneInfo, MontageContext
from ..utils import file_exists_and_valid
from ..media_files import list_media_files
from .analysis_cache import get_analysis_cache


# OPTIMIZATION Phase 3: Module-level worker for ProcessPoolExecutor (pickle compatibility)
def _detect_video_scenes_worker(v_path: str, threshold: float) -> Tuple[str, List]:
    """
    Detect scenes in a single video (process-safe).
    Top-level function required for ProcessPoolExecutor pickling.
    """
    from ..scene_analysis import detect_scenes
    try:
        scenes = detect_scenes(v_path, threshold=threshold)
        return v_path, scenes
    except Exception as e:
        # Use print since logger may not work across process boundaries
        print(f"   ⚠️ Scene detection failed for {v_path}: {e}")
        return v_path, []


# =============================================================================
# Resource Monitoring Utility
# =============================================================================

def get_resource_snapshot() -> Dict[str, Any]:
    """
    Get current resource usage as a dict for progress updates.
    Can be spread into progress callback payloads for SSE broadcast.

    Returns:
        Dict with cpu_percent, memory_mb, memory_pressure, gpu_util keys.
        Values may be None if collection fails.
    """
    result: Dict[str, Any] = {
        "cpu_percent": None,
        "memory_mb": None,
        "memory_pressure": None,
        "gpu_util": None,
    }

    try:
        process = psutil.Process(os.getpid())
        result["cpu_percent"] = process.cpu_percent(interval=0.1)
        result["memory_mb"] = process.memory_info().rss / 1024 / 1024
    except Exception:
        pass

    try:
        from ..memory_monitor import get_memory_manager
        mem_manager = get_memory_manager()
        result["memory_pressure"] = mem_manager.get_memory_pressure_level()
    except Exception:
        pass

    try:
        from ..cgpu_utils import get_cgpu_metrics
        result["gpu_util"] = get_cgpu_metrics()
    except Exception:
        pass

    return result


def _log_resource_usage(phase: str, progress_pct: int = 0) -> Dict[str, Any]:
    """
    Log CPU, memory, and disk usage for progress visibility.
    Call periodically during long-running tasks.

    Returns:
        Resource snapshot dict (same as get_resource_snapshot).
    """
    snapshot = get_resource_snapshot()

    try:
        cpu_pct = snapshot.get("cpu_percent") or 0
        mem_mb = snapshot.get("memory_mb") or 0

        # System-wide resource usage for CLI logging
        sys_cpu = psutil.cpu_percent(interval=0.05)
        sys_mem_pct = psutil.virtual_memory().percent

        # Log in compact format
        progress_str = f" [{progress_pct}%]" if progress_pct > 0 else ""
        gpu_str = f" | GPU: {snapshot['gpu_util']}" if snapshot.get("gpu_util") else ""
        logger.info(f"   📊 {phase}{progress_str}: CPU process={cpu_pct:.1f}% sys={sys_cpu:.1f}% | Memory process={mem_mb:.0f}MB sys={sys_mem_pct:.1f}%{gpu_str}")
    except Exception as e:
        logger.debug(f"Resource logging failed: {e}")

    return snapshot


class AssetAnalyzer:
    """
    Handles analysis of audio and video assets.
    """
    def __init__(self, context: MontageContext):
        self.ctx = context
        self.settings = context.settings
        
        # Dependencies that are usually lazy loaded in methods
        self._resource_manager = None 

    def determine_output_profile(self) -> None:
        """Determine output profile from input footage."""
        from ..video_metadata import determine_output_profile as _determine_profile, OutputProfile
        from .. import segment_writer as segment_writer_module

        if not self.ctx.media.video_files:
             logger.warning("   ⚠️ No video files to determine profile from")
             return

        profile = _determine_profile(self.ctx.media.video_files)
        
        # Override if export resolution explicitly requested
        export_cfg = self.settings.export
        # Guard against MagicMock truthiness in tests; rely on env presence
        has_explicit_width = os.environ.get("EXPORT_WIDTH") is not None
        has_explicit_height = os.environ.get("EXPORT_HEIGHT") is not None
        if has_explicit_width or has_explicit_height:
            try:
                # Use env overrides when provided, fallback to config values
                w = int(os.environ.get("EXPORT_WIDTH", getattr(export_cfg, "resolution_width", 0) or 0))
                h = int(os.environ.get("EXPORT_HEIGHT", getattr(export_cfg, "resolution_height", 0) or 0))
                if w > 0 and h > 0:
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
            except (TypeError, ValueError):
                pass

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
        # We need to ensure we are creating the correct object for context
        self.ctx.media.output_profile = OutputProfile(
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
        segment_writer_module.STANDARD_WIDTH = profile.width
        segment_writer_module.STANDARD_HEIGHT = profile.height
        segment_writer_module.STANDARD_FPS = profile.fps
        segment_writer_module.STANDARD_PIX_FMT = profile.pix_fmt
        segment_writer_module.TARGET_CODEC = profile.codec
        segment_writer_module.TARGET_PROFILE = profile.profile
        segment_writer_module.TARGET_LEVEL = profile.level
        
        if self.settings.features.verbose:
            logger.info(f"\n   🧭 Output profile: {self.ctx.media.output_profile.width}x{self.ctx.media.output_profile.height} @ {self.ctx.media.output_profile.fps:.1f}fps")

    def analyze_music(self, isolated_audio_path: Optional[str] = None) -> None:
        """
        Load and analyze music file with caching support.
        Populates self.ctx.media.audio_result.
        """
        from ..audio_analysis import analyze_audio_parallel, detect_music_sections

        # Get music files
        music_files = self._get_files(self.ctx.paths.music_dir, ('.mp3', '.wav'))
        if not music_files:
            raise ValueError("No music found in music directory")

        # Select music file (rotate through variants)
        music_index = (self.ctx.variant_id - 1) % len(music_files)
        music_path = music_files[music_index]

        # Allow specific track override from editing instructions
        if self.ctx.creative.editing_instructions and self.ctx.creative.editing_instructions.music_track:
            requested_track = self.ctx.creative.editing_instructions.music_track
            # Find exact match or filename match
            match = None
            for mf in music_files:
                if mf == requested_track or os.path.basename(mf) == requested_track:
                    match = mf
                    break
            
            if match:
                logger.info(f"   🎵 Using requested track: {os.path.basename(match)}")
                music_path = match
            else:
                logger.warning(f"   ⚠️ Requested track '{requested_track}' not found, falling back to variant default")

        # Use pre-isolated audio if provided (from async processing)
        # Otherwise apply voice isolation synchronously if enabled
        if isolated_audio_path:
            music_path = isolated_audio_path
        elif self.settings.features.voice_isolation:
            music_path = self.perform_voice_isolation(music_path)

        # Apply noise reduction if enabled (can stack with voice_isolation or run standalone)
        if self.settings.features.noise_reduction:
            music_path = self._apply_noise_reduction(music_path)

        # Try cache first
        cache = get_analysis_cache()
        cached = cache.load_audio(music_path)

        if cached:
            # Cache hit - use cached analysis
            logger.info("   ⚡ Using cached audio analysis")
            
            from ..audio_analysis import EnergyProfile
            ep = EnergyProfile(
                times=np.array(cached.energy_times),
                rms=np.array(cached.energy_values),
                sample_rate=22050, 
                hop_length=512 
            )
            sections = detect_music_sections(ep)

            self.ctx.media.audio_result = AudioAnalysisResult(
                music_path=music_path,
                beat_times=np.array(cached.beat_times),
                tempo=cached.tempo,
                energy_times=np.array(cached.energy_times),
                energy_values=np.array(cached.energy_values),
                duration=cached.duration,
                sections=sections
            )
        else:
            # Cache miss - analyze and cache (PARALLEL for ~2x speedup)
            beat_info, energy_profile = analyze_audio_parallel(
                music_path, verbose=self.settings.features.verbose
            )
            sections = detect_music_sections(energy_profile)

            # Store in context
            self.ctx.media.audio_result = AudioAnalysisResult(
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
            
    def perform_voice_isolation(self, audio_path: str) -> str:
        """
        Apply voice isolation to audio before analysis.
        Uses cgpu/demucs to separate vocals from music.
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
                stems = result.metadata.get("stems", {})
                if stems.get("no_vocals"):
                    logger.info(f"   ✅ Voice isolated: Using accompaniment for analysis: {stems['no_vocals']}")
                    return stems["no_vocals"]
                
                if result.output_path:
                    logger.info(f"   ✅ Voice isolated: {result.output_path}")
                    return result.output_path
            else:
                if self.settings.features.strict_cloud_compute:
                    raise RuntimeError(f"Voice isolation failed: {result.error}")
                logger.warning(f"   Voice isolation failed: {result.error}")
                return audio_path

        except Exception as e:
            if self.settings.features.strict_cloud_compute:
                raise
            logger.warning(f"   Voice isolation error: {e}")
            return audio_path

    def detect_scenes(self, progress_callback: Optional[Any] = None) -> None:
        """
        Detect scenes in all video files with caching and parallel processing.
        Populates self.ctx.media.all_scenes and self.ctx.media.video_files.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
        from ..scene_analysis import detect_scenes, analyze_scene_content, clear_histogram_cache
        from .analysis_cache import get_analysis_cache

        # Clear histogram cache from previous runs
        clear_histogram_cache()

        # Get video files (allow pre-selected list from upstream)
        video_files = self.ctx.media.video_files or []
        if not video_files:
            video_files = self._get_files(self.ctx.paths.input_dir, self.settings.file_types.video_extensions)
        if not video_files:
            raise ValueError("No videos found in input directory")

        # EDGE CASE: Massive footage influx (Intelligent Sampling)
        # SOTA 2026: Prevents infrastructure overload by prioritizing key content
        if len(video_files) > 100:
            logger.info(f"   ⚠️ MASSIVE footage detected ({len(video_files)} clips). Switching to Intelligent Sampling...")
            rng = random.Random(42)
            # Sample: First 25, last 25, and scattered 25 in middle
            if len(video_files) > 75:
                sampled = video_files[:25] + video_files[-25:] + rng.sample(video_files[25:-25], 25)
                # Deduplicate while preserving order
                video_files_sampled = []
                seen = set()
                for v in sampled:
                    if v not in seen:
                        video_files_sampled.append(v)
                        seen.add(v)
                video_files = video_files_sampled
                logger.info(f"   🎯 Sampled down to {len(video_files)} high-priority clips for analysis.")

        # Apply preview fast-path filtering regardless of where the file list came from
        video_files = self._apply_preview_input_limits(video_files)

        video_files = self._filter_supported_videos(video_files)
        if not video_files:
            raise ValueError("No supported videos found in input directory")

        self.ctx.media.video_files = video_files
        self.ctx.media.all_scenes = []
        self.ctx.media.all_scenes_dicts = []

        # Scene detection threshold (centralized)
        threshold = self.settings.thresholds.scene_threshold
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

        # Cluster-Mode: Parallelize across nodes via K8s
        detected_scenes = {}  # path -> scenes list

        if uncached_videos and self.settings.features.cluster_mode:
            try:
                detected_scenes = self._detect_scenes_distributed(uncached_videos, threshold, progress_callback)
            except Exception as e:
                logger.warning(f"   ⚠️ Distributed scene detection failed ({e}), falling back to local processing")
                # Fall through to local processing below
        
        if uncached_videos and not detected_scenes:
            # OPTIMIZATION Phase 3: ProcessPoolExecutor for CPU-intensive scene detection
            # Bypasses Python GIL for 2-4x speedup on multi-core systems
            # PHASE 5: Scale with available cores (cluster-optimized)
            cpu_count = os.cpu_count() or 2
            cfg_workers = self.settings.processing.max_scene_workers

            # CRITICAL: Respect LOW_MEMORY_MODE for constrained environments
            if self.settings.features.low_memory_mode:
                # Strict sequential processing - never exceed configured workers
                max_workers = min(len(uncached_videos), cfg_workers)
                logger.info(f"   ⚠️ LOW_MEMORY_MODE: Sequential scene detection ({max_workers} worker)")
            else:
                max_workers = min(
                    len(uncached_videos),
                    max(4, cpu_count // 2),  # Use 50% of cores (reserve for system)
                    cfg_workers
                )
            logger.info(f"   🚀 ProcessPool scene detection ({len(uncached_videos)} videos, {max_workers}/{cpu_count} CPU workers)")
            
            # Progress tracking setup
            total_tasks = len(uncached_videos)
            completed_tasks = 0
            last_log_time = time.time()

            try:
                # ProcessPoolExecutor for true parallelism (bypasses GIL)
                per_file_timeout = int(getattr(self.settings.processing, "scene_detection_per_file_timeout_seconds", 120))
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_detect_video_scenes_worker, v, threshold): v for v in uncached_videos}

                    for future in as_completed(futures):
                        v = futures[future]
                        try:
                            # Bound waiting on individual futures to avoid a single hung task stalling the loop
                            v_path, scenes = future.result(timeout=per_file_timeout)
                        except TimeoutError:
                            # Best-effort cancellation and mark as empty scenes
                            try:
                                future.cancel()
                            except Exception:
                                pass
                            logger.warning("   ⚠️ Scene detection timed out for %s (>%ss); skipping", os.path.basename(v), per_file_timeout)
                            try:
                                from montage_ai import telemetry
                                telemetry.record_event("scene_detection_timeout", {"file": os.path.basename(v), "timeout_s": per_file_timeout})
                            except Exception:
                                pass
                            v_path, scenes = v, []
                        except Exception as exc:
                            # Record error and continue
                            logger.warning("   ⚠️ Scene detection failed for %s: %s", os.path.basename(v), exc)
                            try:
                                from montage_ai import telemetry
                                telemetry.record_event("scene_detection_failed", {"file": os.path.basename(v), "error": str(exc)[:200]})
                            except Exception:
                                pass
                            v_path, scenes = v, []

                        detected_scenes[v_path] = scenes
                        # Save to cache
                        try:
                            cache.save_scenes(v_path, threshold, scenes)
                        except Exception:
                            logger.debug("Failed to save scenes to cache for %s", v_path)
                        
                        # Update progress
                        completed_tasks += 1
                        progress_pct = int((completed_tasks / total_tasks) * 100)
                        basename = os.path.basename(v_path)

                        # Log + collect resource snapshot every 5 seconds (rate-limited)
                        now = time.time()
                        snapshot = {}
                        if now - last_log_time >= 5.0:
                            snapshot = _log_resource_usage("Scene Detection", progress_pct)
                            last_log_time = now

                        if progress_callback:
                            msg = f"Detecting scenes in {basename} ({completed_tasks}/{total_tasks})"
                            # New dict-based callback with resources
                            progress_callback({
                                "percent": progress_pct,
                                "message": msg,
                                "current_item": basename,
                                **snapshot,
                            })
            except Exception as e:
                # Fallback to ThreadPool if ProcessPool fails (e.g., pickle issues)
                logger.warning(f"   ⚠️ ProcessPool failed ({e}), falling back to ThreadPool")
                from ..config_pools import PoolConfig
                thread_workers = PoolConfig.thread_workers()
                with ThreadPoolExecutor(max_workers=thread_workers) as executor:
                    futures = {executor.submit(_detect_video_scenes_worker, v, threshold): v for v in uncached_videos}
                    completed_tasks = 0
                    last_log_time = time.time()
                    for future in as_completed(futures):
                        v_path, scenes = future.result()
                        detected_scenes[v_path] = scenes
                        cache.save_scenes(v_path, threshold, scenes)
                        completed_tasks += 1
                        progress_pct = int((completed_tasks / total_tasks) * 100)
                        basename = os.path.basename(v_path)

                        # Log + collect resource snapshot every 5 seconds
                        now = time.time()
                        snapshot = {}
                        if now - last_log_time >= 5.0:
                            snapshot = _log_resource_usage("Scene Detection (ThreadPool)", progress_pct)
                            last_log_time = now

                        if progress_callback:
                            msg = f"Detecting scenes in {basename} ({completed_tasks}/{total_tasks})"
                            progress_callback({
                                "percent": progress_pct,
                                "message": msg,
                                "current_item": basename,
                                **snapshot,
                            })

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
                    self.ctx.media.all_scenes.append(scene_info)

        if cache_hits > 0:
            logger.info(f"   ⚡ Used cached scene detection for {cache_hits}/{len(video_files)} videos")

        # Parallel AI scene analysis
        # OPTIMIZATION: Configurable limit (was hardcoded to 20)
        # Set MAX_AI_ANALYZE_SCENES=0 for unlimited analysis
        max_analyze = self.settings.processing.max_ai_analyze_scenes
        if max_analyze > 0 and len(self.ctx.media.all_scenes) > max_analyze:
            logger.info(f"   🤖 AI Director is watching footage... (analyzing {max_analyze}/{len(self.ctx.media.all_scenes)} scenes)")
            scenes_to_analyze = self.ctx.media.all_scenes[:max_analyze]
        else:
            logger.info(f"   🤖 AI Director is watching all {len(self.ctx.media.all_scenes)} scenes...")
            scenes_to_analyze = self.ctx.media.all_scenes

        def analyze_scene(scene):
            """Analyze a single scene (thread-safe)."""
            try:
                meta = analyze_scene_content(scene.path, scene.midpoint)
                return scene, meta
            except Exception:
                return scene, {}

        # OPTIMIZATION: Scale workers with scene count (up to CPU cores)
        cpu_count = os.cpu_count() or 4
        if self.settings.features.low_memory_mode:
            # LOW_MEMORY_MODE: Sequential AI analysis to minimize memory spikes
            ai_workers = 1
            logger.debug("   ⚠️ LOW_MEMORY_MODE: Sequential AI scene analysis")
        else:
            ai_workers = min(len(scenes_to_analyze), max(4, cpu_count // 2))

        total_ai = len(scenes_to_analyze)
        if total_ai == 0:
            logger.debug("   🤖 No scenes to run AI analysis on; skipping AI analysis")
        else:
            with ThreadPoolExecutor(max_workers=ai_workers) as executor:
                futures = [executor.submit(analyze_scene, s) for s in scenes_to_analyze]
                completed_ai = 0
                start_time = time.time()

                for future in as_completed(futures):
                    scene, meta = future.result()
                    scene.meta = meta

                    # Update progress
                    completed_ai += 1
                    if self.ctx.job_id:  # Only if we have a job context
                        from ..monitoring import get_monitor
                        monitor = get_monitor()

                        elapsed = time.time() - start_time
                        avg_dur = elapsed / completed_ai

                        # Log to progress tracker if available via MontageBuilder
                        # Since AssetAnalyzer doesn't have direct ref to progress, 
                        # we use the monitor or just standard logging for now.
                        # Actually, we can assume the caller (MontageBuilder) set up self.progress_callback
                        if completed_ai % max(1, total_ai // 10) == 0:
                            logger.info(f"   🤖 AI Analysis: {completed_ai}/{total_ai} ({int(completed_ai/total_ai*100)}%)")

        # Shuffle for variety
        random.shuffle(self.ctx.media.all_scenes)

        # Create legacy dict format (needed for footage_manager compatibility)
        self.ctx.media.all_scenes_dicts = [s.to_dict() for s in self.ctx.media.all_scenes]

        logger.info(f"   📹 Found {len(self.ctx.media.all_scenes)} scenes in {len(video_files)} videos")

    def _apply_preview_input_limits(self, video_files: List[str]) -> List[str]:
        """Return a filtered list according to preview limits (size + count).

        This helper centralizes preview fast-path logic so it can be tested
        independently and reused by other callers.
        """
        try:
            profile = (self.settings.encoding.quality_profile or "").lower()
        except Exception:
            profile = ""

        if profile != "preview":
            return list(video_files)

        try:
            max_size_mb = int(self.settings.processing.preview_max_input_size_mb)
            max_files = int(self.settings.processing.preview_max_files)
        except Exception:
            max_size_mb = int(os.environ.get("PREVIEW_MAX_INPUT_SIZE_MB", "200"))
            max_files = int(os.environ.get("PREVIEW_MAX_FILES", "3"))

        filtered: List[str] = []
        skipped: List[tuple[str, int]] = []
        for p in video_files:
            try:
                size_mb = os.path.getsize(p) / (1024 * 1024)
            except Exception:
                size_mb = 0
            if size_mb > max_size_mb:
                skipped.append((p, int(size_mb)))
                continue
            filtered.append(p)
            if len(filtered) >= max_files:
                break

        if skipped:
            for sfile, sz in skipped:
                logger.info("   ⚠️ Skipping large file for preview: %s (%dMB)", os.path.basename(sfile), sz)
        if len(filtered) < len(video_files):
            logger.info("   ℹ️ Preview-fast-path will analyze %d/%d files", len(filtered), len(video_files))

        return filtered

    def _detect_scenes_distributed(self, video_paths: List[str], threshold: float, progress_callback: Optional[Any] = None) -> Dict[str, List]:
        """
        Distribute scene detection across the cluster using Kubernetes Jobs.
        """
        from ..cluster.job_submitter import JobSubmitter
        from ..cluster.distributed_scene_detection import aggregate_shard_results

        logger.info(f"   🌐 Cluster Mode: Submitting distributed scene detection job for {len(video_paths)} videos...")
        
        submitter = JobSubmitter()
        
        # Parallelism matches Cluster settings
        parallelism = min(len(video_paths), self.settings.features.cluster_parallelism)
        
        try:
            job = submitter.submit_scene_detection(
                video_paths=video_paths,
                parallelism=parallelism,
                threshold=threshold
            )
            
            logger.info(f"   🚀 Job submitted: {job.name} (parallelism={parallelism})")
            
            # Monitor progress
            last_succeeded = -1
            status_errors = 0
            max_errors = max(1, int(self.settings.cluster.status_max_errors))
            poll_interval = max(1, int(self.settings.cluster.status_poll_interval_seconds))
            request_timeout = max(1, int(self.settings.cluster.status_request_timeout_seconds))

            final_status = None
            while True:
                try:
                    status = submitter.get_job_status(job.name, request_timeout=request_timeout)
                except Exception as e:
                    status_errors += 1
                    logger.warning(
                        "   ⚠️ Cluster status check failed (%d/%d): %s",
                        status_errors,
                        max_errors,
                        e
                    )
                    if status_errors >= max_errors:
                        raise RuntimeError(
                            f"Distributed job {job.name} status check failed after {status_errors} attempts"
                        ) from e
                    time.sleep(poll_interval)
                    continue

                if status is None or status.is_not_found:
                    status_errors += 1
                    logger.warning(
                        "   ⚠️ Cluster status unavailable (%d/%d); retrying...",
                        status_errors,
                        max_errors
                    )
                    if status_errors >= max_errors:
                        raise RuntimeError(
                            f"Distributed job {job.name} status unavailable after {status_errors} attempts"
                        )
                    time.sleep(poll_interval)
                    continue

                status_errors = 0

                if status.succeeded > last_succeeded:
                    logger.info(f"   ⏳ Progress: {status.succeeded}/{parallelism} workers completed")
                    last_succeeded = status.succeeded

                    if progress_callback:
                        progress_callback({
                            "percent": int((status.succeeded / parallelism) * 100),
                            "message": f"Cluster processing: {status.succeeded}/{parallelism} nodes complete",
                        })

                if status.is_complete:
                    final_status = status
                    break
                time.sleep(poll_interval)

            if not final_status or not final_status.is_successful:
                total_failed = final_status.failed if final_status else "unknown"
                raise RuntimeError(f"Distributed job {job.name} failed (failed={total_failed})")
                
            logger.info(f"   ✅ Distributed job {job.name} finished successfully")
            
            # Aggregate results
            output_dir = str(self.settings.paths.scene_cache_dir)
            scenes_data = aggregate_shard_results(output_dir, job.name)
            
            # Convert list of dicts to path-mapped dict
            results = {}
            for s in scenes_data:
                path = s["path"]
                if path not in results:
                    results[path] = []
                results[path].append((s["start"], s["end"]))
                
            return results
        except Exception as e:
            logger.error(f"   ❌ Distributed scene detection failed: {e}")
            raise

    def _get_files(self, directory: str, extensions: Iterable[str]) -> List[str]:
        """Get valid files from directory."""
        root = Path(directory)
        if not root.exists():
            return []
        return [str(path) for path in list_media_files(root, extensions, recursive=True)]

    def _filter_supported_videos(self, video_files: List[str]) -> List[str]:
        """Drop missing or unsupported video files before analysis."""
        from ..video_metadata import probe_metadata

        existing: List[str] = []
        missing: List[str] = []
        for path in video_files:
            if file_exists_and_valid(path):
                existing.append(path)
            else:
                missing.append(path)

        if missing:
            logger.warning(
                "Skipping missing inputs: %s",
                ", ".join(os.path.basename(p) for p in missing),
            )

        supported: List[str] = []
        skipped: List[str] = []
        for path in existing:
            meta = probe_metadata(path)
            if meta and meta.width > 0 and meta.height > 0:
                supported.append(path)
            else:
                skipped.append(path)

        if skipped:
            logger.warning(
                "Skipping unsupported inputs: %s",
                ", ".join(os.path.basename(p) for p in skipped),
            )

        return supported
