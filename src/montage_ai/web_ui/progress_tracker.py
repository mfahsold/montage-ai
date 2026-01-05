"""
Enhanced Progress Tracking for Web UI

Provides detailed progress information for jobs including:
- Phase tracking (analyzing, rendering, exporting)
- Progress percentage per phase
- Time estimates
- Cancellation support
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

class JobPhase(str, Enum):
    """Job execution phases"""
    QUEUED = "queued"
    ANALYZING = "analyzing"
    SELECTING = "selecting"
    RENDERING = "rendering"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class ProgressInfo:
    """Detailed progress information for a job"""
    job_id: str
    phase: JobPhase
    progress: float  # 0.0 to 1.0
    message: str
    phase_name: str  # Human-readable phase name
    time_elapsed: float  # seconds
    time_remaining: Optional[float] = None  # seconds, estimated
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    current_step_num: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['phase'] = self.phase.value
        return data
    
    def to_sse_message(self) -> str:
        """Format as SSE message string"""
        import json
        return json.dumps(self.to_dict())

class ProgressTracker:
    """Track progress for a single job"""
    
    # Phase weights for overall progress calculation
    PHASE_WEIGHTS = {
        JobPhase.QUEUED: 0.0,
        JobPhase.ANALYZING: 0.15,
        JobPhase.SELECTING: 0.10,
        JobPhase.RENDERING: 0.65,
        JobPhase.EXPORTING: 0.10,
        JobPhase.COMPLETE: 1.0,
    }
    
    # Human-readable phase names
    PHASE_NAMES = {
        JobPhase.QUEUED: "Queued",
        JobPhase.ANALYZING: "Analyzing Footage",
        JobPhase.SELECTING: "Selecting Clips",
        JobPhase.RENDERING: "Rendering Video",
        JobPhase.EXPORTING: "Exporting Files",
        JobPhase.COMPLETE: "Complete",
        JobPhase.ERROR: "Error",
        JobPhase.CANCELLED: "Cancelled",
    }
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.current_phase = JobPhase.QUEUED
        self.phase_progress = 0.0  # 0.0 to 1.0 within current phase
        self.start_time = time.time()
        self.phase_start_time = self.start_time
        self.cancelled = False
        
    def update_phase(self, phase: JobPhase, message: str = "") -> ProgressInfo:
        """Update to a new phase"""
        self.current_phase = phase
        self.phase_progress = 0.0
        self.phase_start_time = time.time()
        
        return self._create_progress_info(message or self.PHASE_NAMES[phase])
    
    def update_progress(self, progress: float, message: str = "", 
                       current_step: Optional[str] = None,
                       current_step_num: Optional[int] = None,
                       total_steps: Optional[int] = None) -> ProgressInfo:
        """Update progress within current phase"""
        self.phase_progress = max(0.0, min(1.0, progress))
        
        return self._create_progress_info(
            message or self.PHASE_NAMES[self.current_phase],
            current_step=current_step,
            current_step_num=current_step_num,
            total_steps=total_steps
        )
    
    def cancel(self) -> ProgressInfo:
        """Mark job as cancelled"""
        self.cancelled = True
        self.current_phase = JobPhase.CANCELLED
        return self._create_progress_info("Job cancelled by user")
    
    def _calculate_overall_progress(self) -> float:
        """Calculate overall progress across all phases"""
        # Get cumulative weight of completed phases
        completed_weight = 0.0
        for phase in JobPhase:
            if phase == self.current_phase:
                break
            if phase in self.PHASE_WEIGHTS:
                completed_weight += self.PHASE_WEIGHTS[phase]
        
        # Add progress within current phase
        current_phase_weight = self.PHASE_WEIGHTS.get(self.current_phase, 0.0)
        current_progress = completed_weight + (current_phase_weight * self.phase_progress)
        
        return min(1.0, current_progress)
    
    def _estimate_time_remaining(self) -> Optional[float]:
        """Estimate time remaining based on progress"""
        overall_progress = self._calculate_overall_progress()
        
        if overall_progress <= 0.01:
            return None  # Not enough data yet
        
        elapsed = time.time() - self.start_time
        estimated_total = elapsed / overall_progress
        remaining = estimated_total - elapsed
        
        return max(0.0, remaining)
    
    def _create_progress_info(self, message: str,
                             current_step: Optional[str] = None,
                             current_step_num: Optional[int] = None,
                             total_steps: Optional[int] = None) -> ProgressInfo:
        """Create ProgressInfo object"""
        return ProgressInfo(
            job_id=self.job_id,
            phase=self.current_phase,
            progress=self._calculate_overall_progress(),
            message=message,
            phase_name=self.PHASE_NAMES[self.current_phase],
            time_elapsed=time.time() - self.start_time,
            time_remaining=self._estimate_time_remaining(),
            current_step=current_step,
            current_step_num=current_step_num,
            total_steps=total_steps
        )

class ProgressManager:
    """Manage progress trackers for multiple jobs"""
    
    def __init__(self):
        self._trackers: Dict[str, ProgressTracker] = {}
    
    def create_tracker(self, job_id: str) -> ProgressTracker:
        """Create a new progress tracker for a job"""
        tracker = ProgressTracker(job_id)
        self._trackers[job_id] = tracker
        return tracker
    
    def get_tracker(self, job_id: str) -> Optional[ProgressTracker]:
        """Get existing tracker for a job"""
        return self._trackers.get(job_id)
    
    def remove_tracker(self, job_id: str) -> None:
        """Remove tracker when job is complete"""
        self._trackers.pop(job_id, None)
    
    def cancel_job(self, job_id: str) -> Optional[ProgressInfo]:
        """Cancel a running job"""
        tracker = self.get_tracker(job_id)
        if tracker:
            return tracker.cancel()
        return None

# Global progress manager instance
_progress_manager = ProgressManager()

def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance"""
    return _progress_manager
