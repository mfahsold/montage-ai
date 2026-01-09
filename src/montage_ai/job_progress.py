"""
Progress logging for montage jobs.

Provides structured, phase-based progress updates with timing and metrics.

Usage:
    progress = JobProgress("hitchcock_1234", total_scenes=160)
    progress.scene_detection_start()
    progress.log_phase("Scene Detection", scenes_found=42, duration=2.3)
    progress.scene_detection_complete(scenes=42)
    
    progress.optical_flow_start()
    progress.log_clip_progress(current_clip=5, total_clips=160, duration=12.5)
    progress.optical_flow_complete(avg_time_per_clip=12.5)
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from .logger import logger


class Phase(Enum):
    """Job phases for structured progress tracking."""
    INITIALIZATION = "Initialization"
    SCENE_DETECTION = "Scene Detection"
    PROXY_GENERATION = "Proxy Generation"
    METADATA_EXTRACTION = "Metadata Extraction"
    OPTICAL_FLOW_ANALYSIS = "Optical Flow Analysis"
    CLIP_SELECTION = "Clip Selection"
    ASSEMBLY = "Assembly"
    RENDERING = "Rendering"
    FINALIZATION = "Finalization"


@dataclass
class PhaseMetrics:
    """Metrics for a single job phase."""
    phase: Phase
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    items_processed: int = 0
    total_items: Optional[int] = None
    errors: int = 0
    warnings: int = 0
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def items_per_second(self) -> float:
        """Processing speed."""
        if self.duration == 0:
            return 0
        return self.items_processed / self.duration
    
    @property
    def progress_pct(self) -> float:
        """Progress percentage (0-100)."""
        if self.total_items is None or self.total_items == 0:
            return 0
        return (self.items_processed / self.total_items) * 100


class JobProgress:
    """Track and log job progress with structured metrics."""
    
    def __init__(self, job_id: str, total_scenes: int = 0):
        self.job_id = job_id
        self.total_scenes = total_scenes
        self.start_time = time.time()
        self.phases: Dict[Phase, PhaseMetrics] = {}
        self.current_phase: Optional[Phase] = None
    
    def _start_phase(self, phase: Phase, total_items: Optional[int] = None) -> None:
        """Start tracking a phase."""
        if self.current_phase and self.current_phase in self.phases:
            self._end_current_phase()
        
        self.current_phase = phase
        self.phases[phase] = PhaseMetrics(phase=phase, total_items=total_items)
        
        logger.info(f"🎬 [Job {self.job_id}] Starting: {phase.value}")
    
    def _end_current_phase(self) -> None:
        """End current phase and log summary."""
        if not self.current_phase or self.current_phase not in self.phases:
            return
        
        metrics = self.phases[self.current_phase]
        metrics.end_time = time.time()
        
        # Build summary
        summary = f"✓ {metrics.phase.value} completed in {metrics.duration:.1f}s"
        if metrics.items_processed > 0:
            summary += f" ({metrics.items_processed} items @ {metrics.items_per_second:.1f}/s)"
        
        logger.info(f"🎬 [Job {self.job_id}] {summary}")
    
    # Phase helpers
    def scene_detection_start(self) -> None:
        """Start scene detection phase."""
        self._start_phase(Phase.SCENE_DETECTION)
    
    def scene_detection_complete(self, scenes: int) -> None:
        """Log scene detection completion."""
        if Phase.SCENE_DETECTION in self.phases:
            self.phases[Phase.SCENE_DETECTION].items_processed = scenes
        self._end_current_phase()
    
    def proxy_generation_start(self) -> None:
        """Start proxy generation phase."""
        self._start_phase(Phase.PROXY_GENERATION)
    
    def proxy_generation_complete(self, proxy_height: int, duration: float) -> None:
        """Log proxy generation completion."""
        if Phase.PROXY_GENERATION in self.phases:
            self.phases[Phase.PROXY_GENERATION].items_processed = 1
        self._end_current_phase()
        logger.info(f"   720p proxy generated in {duration:.1f}s")
    
    def metadata_extraction_start(self, total_scenes: int) -> None:
        """Start metadata extraction phase."""
        self._start_phase(Phase.METADATA_EXTRACTION, total_items=total_scenes)
    
    def log_clip_metadata(self, clip_number: int, total_clips: int, duration: float) -> None:
        """Log individual clip metadata extraction."""
        if Phase.METADATA_EXTRACTION in self.phases:
            metrics = self.phases[Phase.METADATA_EXTRACTION]
            metrics.items_processed = clip_number
            progress = (clip_number / total_clips) * 100 if total_clips > 0 else 0
            
            if clip_number % max(1, total_clips // 10) == 0:  # Log every 10%
                logger.info(
                    f"   Metadata: {clip_number}/{total_clips} "
                    f"({progress:.0f}%) - {duration:.1f}s per clip"
                )
    
    def metadata_extraction_complete(self) -> None:
        """Log metadata extraction completion."""
        self._end_current_phase()
    
    def optical_flow_start(self, total_clips: int) -> None:
        """Start optical flow analysis phase."""
        self._start_phase(Phase.OPTICAL_FLOW_ANALYSIS, total_items=total_clips)
    
    def log_optical_flow_progress(self, clip_number: int, total_clips: int, 
                                   duration_per_clip: float, eta_seconds: Optional[float] = None) -> None:
        """Log optical flow progress."""
        if Phase.OPTICAL_FLOW_ANALYSIS in self.phases:
            metrics = self.phases[Phase.OPTICAL_FLOW_ANALYSIS]
            metrics.items_processed = clip_number
            progress = (clip_number / total_clips) * 100 if total_clips > 0 else 0
            
            if clip_number % max(1, total_clips // 10) == 0 or clip_number == 1:
                status = f"   Optical Flow: {clip_number}/{total_clips} ({progress:.0f}%)"
                status += f" - {duration_per_clip:.1f}s per clip"
                if eta_seconds:
                    minutes = eta_seconds / 60
                    status += f" - ETA: {minutes:.1f}m"
                logger.info(status)
    
    def optical_flow_complete(self) -> None:
        """Log optical flow completion."""
        self._end_current_phase()
    
    def assembly_start(self) -> None:
        """Start assembly phase."""
        self._start_phase(Phase.ASSEMBLY)
    
    def assembly_complete(self) -> None:
        """Log assembly completion."""
        self._end_current_phase()
    
    def rendering_start(self) -> None:
        """Start rendering phase."""
        self._start_phase(Phase.RENDERING)
    
    def log_render_progress(self, progress_pct: float, eta_seconds: Optional[float] = None) -> None:
        """Log render progress."""
        status = f"   Rendering: {progress_pct:.0f}%"
        if eta_seconds:
            minutes = eta_seconds / 60
            status += f" - ETA: {minutes:.1f}m"
        logger.info(status)
    
    def rendering_complete(self) -> None:
        """Log rendering completion."""
        self._end_current_phase()
    
    def finalization_start(self) -> None:
        """Start finalization phase."""
        self._start_phase(Phase.FINALIZATION)
    
    def finalization_complete(self, output_path: str) -> None:
        """Log finalization completion."""
        self._end_current_phase()
        
        total_duration = time.time() - self.start_time
        logger.info(f"✨ [Job {self.job_id}] Completed in {total_duration/60:.1f} minutes")
        logger.info(f"   Output: {output_path}")
    
    def log_error(self, phase: Optional[Phase] = None, error_msg: str = "") -> None:
        """Log an error in current or specified phase."""
        target_phase = phase or self.current_phase
        if target_phase and target_phase in self.phases:
            self.phases[target_phase].errors += 1
        logger.error(f"❌ [Job {self.job_id}] {error_msg}")
    
    def log_warning(self, warning_msg: str) -> None:
        """Log a warning in current phase."""
        if self.current_phase and self.current_phase in self.phases:
            self.phases[self.current_phase].warnings += 1
        logger.warning(f"⚠️  [Job {self.job_id}] {warning_msg}")
    
    def summary(self) -> str:
        """Return summary of all phases."""
        lines = [f"📊 Job Summary: {self.job_id}"]
        total_time = time.time() - self.start_time
        lines.append(f"   Total Time: {total_time/60:.1f} minutes")
        
        for phase_name, metrics in self.phases.items():
            pct = (metrics.duration / total_time * 100) if total_time > 0 else 0
            line = f"   {metrics.phase.value}: {metrics.duration:.1f}s ({pct:.0f}%)"
            if metrics.errors > 0:
                line += f" - {metrics.errors} errors"
            lines.append(line)
        
        return "\n".join(lines)
