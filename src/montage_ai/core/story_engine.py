"""
Story Engine Module

Integrates the Narrative Storytelling engine into the Montage pipeline.
Handles tension analysis and arc-based timeline generation.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..logger import logger
from ..config import Settings
from .context import MontageContext
from ..utils import coerce_float

# Lazy imports for optional components
try:
    from ..storytelling import StoryArc, TensionProvider, StorySolver
    from ..storytelling.tension_provider import MissingAnalysisError
    from ..cgpu_utils import is_cgpu_available
    from ..cgpu_jobs import TensionAnalysisBatchJob
    from ..video_metadata import probe_duration
except ImportError:
    pass

class StoryEngine:
    """
    Handles interaction with the Storytelling subsystem.
    """
    
    def __init__(self, context: MontageContext):
        self.ctx = context
        self.settings = context.settings

    def _get_tension_metadata_dir(self) -> Path:
        """Resolve the directory for tension metadata."""
        return self.settings.paths.tension_metadata_dir

    def ensure_analysis(self):
        """
        Identifies clips missing tension analysis and triggers cgpu jobs.
        """
        logger.info("   📖 Story Engine: Checking for missing analysis...")
        
        # 1. Identify all input clips
        input_clips = self.ctx.media.video_files
        if not input_clips:
            # Fallback scan if context not populated (rare)
            # Reimplementing basic scan from builder
            directory = self.ctx.paths.input_dir
            if os.path.isdir(str(directory)):
                 input_clips = []
                 for f in os.listdir(str(directory)):
                     if f.lower().endswith(('.mp4', '.mov', '.mkv')):
                         input_clips.append(os.path.join(str(directory), f))
        
        if not input_clips:
            logger.warning("   ⚠️ Story Engine: No input clips found for analysis.")
            return
        
        # 2. Check which ones are missing metadata
        tension_meta_dir = self._get_tension_metadata_dir()
        tension_meta_dir.mkdir(parents=True, exist_ok=True)

        provider = TensionProvider(tension_meta_dir)
        missing_clips = []
        
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
        
        if not is_cgpu_available(require_gpu=False):
            message = "Story Engine requires cgpu for tension analysis but cgpu is not available."
            if self.settings.features.strict_cloud_compute:
                raise RuntimeError(message)
            logger.warning(f"   ⚠️ {message} Proceeding with dummy tension values.")
            return

        logger.info("   ☁️ Offloading tension analysis to cgpu...")
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

    def generate_story_plan(self) -> List[Dict[str, Any]]:
        """
        Generates a list of timeline events based on the story arc.
        Returns a list of dicts suitable for processing into clips.
        """
        logger.info("   📖 Story Engine: Planning narrative...")

        # Determine style/arc
        style_name = "dynamic"
        story_arc_spec = None
        if self.ctx.creative.editing_instructions:
            if self.ctx.creative.editing_instructions.style:
                style_name = self.ctx.creative.editing_instructions.style.name
            story_arc_spec = getattr(self.ctx.creative.editing_instructions, "story_arc", None)

        # Setup tension provider + solver
        tension_meta_dir = self._get_tension_metadata_dir()
        allow_dummy = not self.settings.features.strict_cloud_compute
        provider = TensionProvider(tension_meta_dir, allow_dummy=allow_dummy)
        
        if story_arc_spec:
            arc = StoryArc.from_spec(story_arc_spec)
        else:
            arc = StoryArc.from_preset(style_name)
        
        # Pull parameters from spec if available
        fatigue_sensitivity = 0.4
        momentum_weight = 0.1
        if story_arc_spec:
             fatigue_sensitivity = getattr(story_arc_spec, "fatigue_sensitivity", 0.4)
             momentum_weight = getattr(story_arc_spec, "momentum_weight", 0.1)

        solver = StorySolver(
            arc, 
            provider, 
            fatigue_sensitivity=fatigue_sensitivity,
            momentum_weight=momentum_weight
        )

        # Build beat list
        duration = self.ctx.timeline.target_duration or self.ctx.media.audio_result.duration
        beats = [t for t in self.ctx.media.audio_result.beat_times.tolist() if 0.0 <= t <= duration]
        if not beats:
            beats = [0.0]

        # Resolve input clips
        input_clips = self.ctx.media.video_files

        timeline_events = solver.solve(input_clips, duration, beats)
        if not timeline_events:
            raise RuntimeError("Story Engine produced no timeline events")
            
        return timeline_events

    def plan_broll(self, script: str) -> Optional[List[Dict[str, Any]]]:
        """
        Generate B-Roll plan from script.
        
        Uses BrollPlanner to find relevant clips for script segments.
        Estimates timing based on word count if no timestamps provided.
        """
        try:
            from ..broll_planner import plan_broll
            
            if not script:
                return None
            logger.info(f"   📜 Planning B-Roll for script: {script[:50]}...")
            
            # Plan B-Roll (find suggestions)
            # We use input_dir from context
            plan = plan_broll(
                script, 
                input_dir=str(self.ctx.paths.input_dir),
                top_k=3,
                analyze_first=True
            )
            
            # Estimate timing
            # Simple heuristic: Distribute segments evenly or by word count
            # Use timeline target duration if set, otherwise need another heuristic
            duration = self.ctx.timeline.target_duration
            if duration <= 0:
                 duration = 60.0 # Fallback
            
            total_words = len(script.split())
            duration_per_word = duration / max(1, total_words)
            
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
            
            logger.info(f"   ✅ B-Roll Plan created with {len(plan)} segments")
            return plan
            
        except Exception as e:
            logger.warning(f"   ⚠️ B-Roll planning failed: {e}")
            return None
