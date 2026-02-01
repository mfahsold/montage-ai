"""
Render Engine Module

Extracts rendering logic from MontageBuilder to separate concerns.
Handles progressive rendering finalization and FFmpeg concatenation.
"""

import os
import time
import random
import shutil
import uuid
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List
from ..logger import logger
from .context import MontageContext, ClipMetadata
from ..segment_writer import SegmentWriter, ProgressiveRenderer, STANDARD_WIDTH, STANDARD_HEIGHT
from .clip_processor import process_clip_task
from ..enhancement_tracking import (
    EnhancementTracker,
    EnhancementDecision,
    StabilizeParams,
    UpscaleParams,
    ColorGradeParams,
)
from ..clip_enhancement import DenoiseConfig, SharpenConfig, FilmGrainConfig

class RenderEngine:
    """
    Handles final rendering of the montage.
    """
    def __init__(self, context: MontageContext):
        self.ctx = context
        self.settings = context.settings
        self._progressive_renderer: Optional[SegmentWriter] = None
        self._progress_callback: Optional[Any] = None

    def init_progressive_renderer(self):
        """Initialize segment writer for progressive output."""
        try:
            from ..segment_writer import ProgressiveRenderer
            from ..memory_monitor import get_memory_manager

            # Phase 4: Resolution-aware adaptive batch sizing
            low_memory = self.settings.features.low_memory_mode
            
            # Get representative clip metadata for resolution detection
            if self.ctx.media.output_profile:
                width = self.ctx.media.output_profile.width
                height = self.ctx.media.output_profile.height
                batch_size = self.settings.high_res.get_adaptive_batch_size_for_resolution(
                    width=width,
                    height=height,
                    low_memory=low_memory
                )
                if width * height >= 15_000_000:  # 6K+
                    logger.info(f"   📐 High-res output: {width}x{height}, batch_size={batch_size}")
            else:
                # Fallback to general adaptive batch size
                batch_size = self.settings.processing.get_adaptive_batch_size(low_memory)

            if low_memory:
                logger.info(f"   ⚠️ LOW_MEMORY_MODE: Batch size reduced to {batch_size}")

            self._progressive_renderer = ProgressiveRenderer(
                batch_size=batch_size,
                output_dir=os.path.join(str(self.ctx.paths.temp_dir), f"segments_{self.ctx.job_id}"),
                memory_manager=get_memory_manager(),
                job_id=self.ctx.job_id,
                enable_xfade=self.ctx.timeline.enable_xfade,
                xfade_duration=self.ctx.timeline.xfade_duration,
                ffmpeg_crf=self.settings.encoding.crf,
                normalize_clips=self.settings.encoding.normalize_clips,
                target_width=width if self.ctx.media.output_profile else STANDARD_WIDTH,
                target_height=height if self.ctx.media.output_profile else STANDARD_HEIGHT,
                target_fps=self.ctx.media.output_profile.fps if self.ctx.media.output_profile else 30.0,
            )
            logger.info(f"   ✅ Progressive Renderer initialized (batch={batch_size})")
            return self._progressive_renderer
        except ImportError:
            self._progressive_renderer = None
            logger.warning("   ⚠️ Progressive Renderer not available")
            return None

    def submit_clip_task(
        self,
        scene: Dict[str, Any],
        clip_start: float,
        cut_duration: float,
        current_energy: float,
        selection_score: float,
        beat_idx: int,
        beats_per_cut: float,
        current_pattern_beat: float,  # Pass the value from pattern if available
        executor: Any,
        enhancer: Any,
        resource_manager: Any
    ) -> Tuple[Any, ClipMetadata]:
        """Submit clip processing task and create initial metadata."""
        
        # Generate temp paths
        temp_clip_name = f"temp_clip_{self.ctx.job_id}_{beat_idx}_{uuid.uuid4().hex}.mp4"
        
        # Submit task
        if executor:
            future = executor.submit(
                process_clip_task,
                scene_path=scene['path'],
                clip_start=clip_start,
                cut_duration=cut_duration,
                temp_dir=str(self.ctx.paths.temp_dir),
                temp_clip_name=temp_clip_name,
                ctx_stabilize=self.ctx.features.stabilize,
                ctx_upscale=self.ctx.features.upscale,
                ctx_enhance=self.ctx.features.enhance,
                ctx_color_grade=self.ctx.features.color_grade,
                ctx_denoise=self.ctx.features.denoise,
                ctx_sharpen=self.ctx.features.sharpen,
                ctx_film_grain=self.ctx.features.film_grain,
                enhancer=enhancer,
                output_profile=self.ctx.media.output_profile,
                settings=self.settings,
                resource_manager=resource_manager
            )
        else:
            raise RuntimeError("Executor not initialized")

        # Create metadata (enhancements will be updated later)
        clip_meta = ClipMetadata(
            source_path=scene['path'],
            start_time=clip_start,
            duration=cut_duration,
            timeline_start=self.ctx.timeline.current_time,
            energy=current_energy,
            action=scene.get('meta', {}).get('action', 'medium'),
            shot=scene.get('meta', {}).get('shot', 'medium'),
            beat_idx=beat_idx,
            beats_per_cut=beats_per_cut if beats_per_cut else current_pattern_beat,
            selection_score=selection_score,
            enhancements={} # Updated on completion
        )
        return future, clip_meta

    def process_completed_task(
        self,
        future: Any,
        meta: ClipMetadata,
        enhancement_tracker: EnhancementTracker
    ):
        """Process result of a clip task and feed progressive renderer."""
        try:
            final_path, enhancements, temp_files = future.result()
            meta.enhancements = enhancements

            # Create EnhancementDecision for NLE export tracking
            decision = enhancement_tracker.create_decision(
                source_path=meta.source_path,
                timeline_in=meta.timeline_start,
                timeline_out=meta.timeline_start + meta.duration,
            )

            # Record applied enhancements with parameters
            if enhancements.get('stabilized'):
                decision.record_stabilize(StabilizeParams(
                    method="vidstab",
                    smoothing=30,
                    crop_mode="black",
                ))
            if enhancements.get('upscaled'):
                decision.record_upscale(UpscaleParams(
                    method="realesrgan" if self.settings.features.upscale else "lanczos",
                    scale_factor=2,
                ))
            if enhancements.get('enhanced'):
                decision.record_color_grade(ColorGradeParams(
                    preset=self.ctx.features.color_grade or "teal_orange",
                    intensity=self.ctx.features.color_intensity,
                ))
            
            if enhancements.get('denoised'):
                decision.record_denoise(DenoiseConfig(spatial_strength=0.3))
            if enhancements.get('sharpened'):
                decision.record_sharpen(SharpenConfig(amount=0.5))
            if enhancements.get('film_grain'):
                decision.record_film_grain(FilmGrainConfig(grain_type=self.ctx.features.film_grain, enabled=True))

            meta.enhancement_decision = decision

            if self._progressive_renderer:
                self._progressive_renderer.add_clip_path(final_path)

                # Cleanup intermediate temp files
                final_path_abs = os.path.abspath(final_path)
                for tf in temp_files:
                    if tf == final_path or not os.path.exists(tf):
                        continue
                    try:
                        if os.path.isdir(tf):
                            tf_abs = os.path.abspath(tf)
                            if os.path.commonpath([final_path_abs, tf_abs]) == tf_abs:
                                continue
                            shutil.rmtree(tf, ignore_errors=True)
                        else:
                            os.remove(tf)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error processing clip {meta.source_path}: {e}")
    
    def set_renderer(self, renderer: SegmentWriter):
        self._progressive_renderer = renderer

    def render_output(self) -> None:
        """
        Phase 5: Render final output.
        """
        logger.info("\n   🎬 Rendering output...")

        if self.settings.processing.should_skip_output(self.ctx.render.output_filename):
            logger.info(f"   ♻️ Output exists, skipping render: {os.path.basename(self.ctx.render.output_filename)}")
            self.ctx.render.render_duration = 0.0
            return

        render_start_time = time.time()

        if self._progressive_renderer:
            # Progressive path: finalize with FFmpeg
            logger.info(f"   🔗 Finalizing with Progressive Renderer ({self._progressive_renderer.get_segment_count()} segments)...")

            output_path = getattr(self.ctx.render, "output_filename", "")
            current_item = os.path.basename(output_path) if output_path else "render"
            self._report_render_progress(10, "Rendering final segments", current_item=current_item)

            audio_duration = self.ctx.timeline.target_duration
            audio_path = self.ctx.media.audio_result.music_path

            # Apply dialogue ducking if enabled
            if self.ctx.features.dialogue_duck and audio_path:
                try:
                    from ..dialogue_ducking import apply_ducking_to_audio
                    # Use voice track if available (from voice isolation), else try original audio
                    voice_path = getattr(self.ctx.media.audio_result, 'voice_path', None) or audio_path
                    ducked_audio_path = os.path.join(str(self.ctx.paths.temp_dir), "ducked_audio.m4a")
                    duck_level = self.settings.features.dialogue_duck_level
                    logger.info(f"   🔇 Applying dialogue ducking ({duck_level}dB)...")
                    result = apply_ducking_to_audio(
                        music_path=audio_path,
                        voice_path=voice_path,
                        output_path=ducked_audio_path,
                        duck_level_db=duck_level
                    )
                    if result and os.path.exists(result):
                        audio_path = result
                        logger.info("   ✅ Dialogue ducking applied")
                except Exception as e:
                    logger.warning(f"   ⚠️ Dialogue ducking failed: {e}")

            self._report_render_progress(60, "Finalizing segments", current_item=current_item)
            success = self._progressive_renderer.finalize(
                output_path=self.ctx.render.output_filename,
                audio_path=audio_path,
                audio_duration=audio_duration,
                logo_path=self.ctx.render.logo_path
            )

            if success:
                method_str = "xfade" if self.ctx.timeline.enable_xfade else "-c copy"
                self.ctx.render.render_duration = time.time() - render_start_time
                logger.info(f"   ✅ Final video rendered via FFmpeg ({method_str}) in {self.ctx.render.render_duration:.1f}s")
                if self.ctx.render.logo_path:
                    logger.info(f"   🏷️ Logo overlay: {os.path.basename(self.ctx.render.logo_path)}")
                self._report_render_progress(100, "Render complete", current_item=current_item)
            else:
                raise RuntimeError("Progressive render failed")
        else:
            # Legacy path
            raise NotImplementedError("Legacy rendering not implemented in RenderEngine")

    def cleanup(self):
        """Cleanup render resources."""
        if self._progressive_renderer:
             try:
                 self._progressive_renderer.cleanup()
             except Exception:
                 pass

    def set_progress_callback(self, callback: Optional[Any]) -> None:
        """Register progress callback for render-phase updates."""
        self._progress_callback = callback

    def render_distributed(self) -> None:
        """
        Distributed Phase 5: Render final output across multiple nodes.
        """
        logger.info("\n   🌐 Entering Distributed Render Mode (Phase 2)...")
        
        if not self.ctx.timeline.clips_metadata:
            logger.warning("   ⚠️ No clips in timeline to render.")
            return

        render_start_time = time.time()
        job_id = self.ctx.job_id
        
        # 1. Prepare shared storage path for clips metadata
        temp_dir = Path(self.ctx.paths.temp_dir)
        clips_json_path = temp_dir / f"clips_{job_id}.json"
        
        # Convert ClipMetadata objects to dicts for JSON
        import dataclasses
        clips_data = [dataclasses.asdict(c) for c in self.ctx.timeline.clips_metadata]
        
        with open(clips_json_path, "w") as f:
            json.dump(clips_data, f)
            
        logger.info(f"   📝 Saved timeline metadata ({len(clips_data)} clips) to {clips_json_path}")

        # 2. Submit K8s Jobs
        try:
            from ..cluster.job_submitter import JobSubmitter
            submitter = JobSubmitter()
            
            parallelism = self.settings.features.cluster_parallelism
            logger.info(f"   🚀 Submitting {parallelism} rendering shards to cluster...")
            
            # Note: We need to implement submit_render_job in JobSubmitter
            # Or use a generic submit_job method
            cluster_tier = self.settings.features.cluster_render_tier
            shard_env = {}
            hwaccel = (self.settings.gpu.ffmpeg_hwaccel or "").strip()
            if hwaccel:
                shard_env["FFMPEG_HWACCEL"] = hwaccel
            output_codec = (self.settings.gpu.output_codec or "").strip()
            if output_codec:
                shard_env["OUTPUT_CODEC"] = output_codec

            job_spec = submitter.submit_generic_job(
                job_id=f"render-{job_id}",
                command=[
                    "python", "-m", "montage_ai.cluster.distributed_rendering",
                    "--clips-json", str(clips_json_path),
                    "--shard-count", str(parallelism),
                    "--output-dir", str(temp_dir / f"segments_{job_id}"),
                    "--job-id", job_id,
                    "--quality", self.settings.encoding.quality_profile
                ],
                parallelism=parallelism,
                component="distributed-rendering",
                env=shard_env or None,
                tier=cluster_tier
            )
            
            # 3. Wait for completion
            self._report_render_progress(20, "Waiting for cluster nodes to render segments...")
            submitter.wait_for_job(job_spec.name)
            
            if not submitter.is_job_successful(job_spec.name):
                raise RuntimeError("Cluster rendering failed. Check K8s logs.")
                
            # 4. Finalize: Concatenate segments from all shards
            self._report_render_progress(80, "Aggregating segments from nodes...")
            
            # Collect all shard reports and segment paths
            all_segments = []
            segments_dir = temp_dir / f"segments_{job_id}"
            
            # Search for shard reports
            import glob
            report_files = glob.glob(str(segments_dir / "shard_*" / "shard_report.json"))
            report_files.sort() # Ensure shards stay in order
            
            for report_path in report_files:
                with open(report_path, "r") as f:
                    report = json.load(f)
                    all_segments.extend(report.get("segments", []))
            
            logger.info(f"   🔗 Aggregated {len(all_segments)} segments from {len(report_files)} shards")

            # Finalize using ProgressiveRenderer logic
            output_path = self.ctx.render.output_filename
            audio_path = self.ctx.media.audio_result.music_path
            audio_duration = self.ctx.timeline.target_duration
            
            # We use ProgressiveRenderer to wrap SegmentWriter
            from ..segment_writer import ProgressiveRenderer
            pr = ProgressiveRenderer(output_dir=str(segments_dir))
            # Manually inject segments into the inner segment_writer
            from ..segment_writer import SegmentInfo
            for seg_path in all_segments:
                pr.segment_writer.segments.append(SegmentInfo(path=seg_path, duration=0, clip_count=0))
                
            success = pr.finalize(
                output_path=output_path,
                audio_path=audio_path,
                audio_duration=audio_duration,
                logo_path=self.ctx.render.logo_path
            )
            
            if success:
                self.ctx.render.render_duration = time.time() - render_start_time
                logger.info(f"   ✅ Distributed render complete in {self.ctx.render.render_duration:.1f}s")
                self._report_render_progress(100, "Render complete")
            else:
                raise RuntimeError("Final concatenation failed")

        except Exception as e:
            logger.error(f"   ❌ Distributed render failed: {e}")
            raise

    def _collect_render_resources(self) -> Dict[str, Any]:
        """Grab CPU/memory/GPU metrics for progress updates."""
        try:
            from .analysis_engine import get_resource_snapshot
            return get_resource_snapshot()
        except Exception as exc:
            logger.debug(f"Render resource snapshot skipped: {exc}")
            return {}

    def _report_render_progress(
        self,
        percent: int,
        message: str,
        current_item: Optional[str] = None,
    ) -> None:
        """Emit a progress callback with optional hardware metrics."""
        if not self._progress_callback:
            return

        update = {
            "percent": max(0, min(100, int(percent))),
            "message": message,
        }
        if current_item:
            update["current_item"] = current_item

        snapshot = self._collect_render_resources()
        for key, value in snapshot.items():
            if value is not None:
                update[key] = value

        try:
            self._progress_callback(update)
        except Exception as exc:
            logger.debug(f"Render progress callback failed: {exc}")

    def export_timeline(self):
        """
        Phase 6: Export timeline to NLE formats (EDL, XML, OTIO).
        """
        from ..timeline_exporter import export_timeline_from_montage

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
                project_name=f"{self.ctx.job_id}_v{self.ctx.variant_id}",
                generate_proxies=self.settings.features.generate_proxies,
                resolution=(self.ctx.media.output_profile.width, self.ctx.media.output_profile.height) if self.ctx.media.output_profile else (1920, 1080),
                fps=self.ctx.media.output_profile.fps if self.ctx.media.output_profile else 30.0
            )
        except Exception as e:
            logger.error(f"   ❌ Timeline export failed: {e}")
