"""
Analysis Engine Module

Extracts analysis logic from MontageBuilder to separate concerns.
Handles Audio Analysis (Beats, Energy, Voice Isolation) and Video Scene Detection.
"""

import os
import random
import numpy as np
from typing import List, Optional, Tuple, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..logger import logger
from ..config import Settings
from .context import AudioAnalysisResult, SceneInfo, MontageContext
from ..utils import file_exists_and_valid
from .analysis_cache import get_analysis_cache

class AssetAnalyzer:
    """
    Handles analysis of audio and video assets.
    """
    def __init__(self, context: MontageContext):
        self.ctx = context
        self.settings = context.settings
        
        # Dependencies that are usually lazy loaded in methods
        self._resource_manager = None 

    def analyze_music(self, isolated_audio_path: Optional[str] = None) -> None:
        """
        Load and analyze music file with caching support.
        Populates self.ctx.media.audio_result.
        """
        from ..audio_analysis import get_beat_times, analyze_music_energy, detect_music_sections

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
            # Cache miss - analyze and cache
            beat_info = get_beat_times(music_path, verbose=self.settings.features.verbose)
            energy_profile = analyze_music_energy(music_path, verbose=self.settings.features.verbose)
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
            
    def _apply_voice_isolation(self, audio_path: str) -> str:
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
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from ..scene_analysis import detect_scenes, analyze_scene_content, clear_histogram_cache
        from .analysis_cache import get_analysis_cache

        # Clear histogram cache from previous runs
        clear_histogram_cache()

        # Get video files
        video_files = self._get_files(self.ctx.paths.input_dir, ('.mp4', '.mov'))
        if not video_files:
            raise ValueError("No videos found in input directory")

        self.ctx.media.video_files = video_files
        self.ctx.media.all_scenes = []
        self.ctx.media.all_scenes_dicts = []

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
            # Simple heuristic for workers: use max 4 or num_videos
            max_workers = min(4, len(uncached_videos))
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
                    if progress_callback:
                        local_pct = int((completed_tasks / total_tasks) * 100)
                        msg = f"Detecting scenes in {os.path.basename(v_path)} ({completed_tasks}/{total_tasks})"
                        progress_callback(local_pct, msg)

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

        # Parallel AI scene analysis (limit to first 20 for speed)
        logger.info("   🤖 AI Director is watching footage...")
        scenes_to_analyze = self.ctx.media.all_scenes[:20]

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
        random.shuffle(self.ctx.media.all_scenes)

        # Create legacy dict format (needed for footage_manager compatibility)
        self.ctx.media.all_scenes_dicts = [s.to_dict() for s in self.ctx.media.all_scenes]

        logger.info(f"   📹 Found {len(self.ctx.media.all_scenes)} scenes in {len(video_files)} videos")
        """Get valid files from directory."""
        if not os.path.exists(directory):
            return []
        
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(extensions) and not filename.startswith('._'):
                     files.append(os.path.join(root, filename))
        return sorted(files)
