"""
CGPU Jobs - Unified Cloud GPU Job Architecture

Provides a robust, job-based interface for offloading heavy compute to cgpu.
Each job follows the lifecycle: prepare → upload → run → download → cleanup.

Usage:
    from montage_ai.cgpu_jobs import CGPUJobManager, TranscribeJob, UpscaleJob, StabilizeJob

    manager = CGPUJobManager()

    # Submit jobs
    job = TranscribeJob(audio_path="/data/audio.wav", model="medium")
    manager.submit(job)

    # Process all queued jobs
    results = manager.process_queue()
"""

from .base import CGPUJob, JobStatus, JobResult
from .manager import CGPUJobManager

# Job implementations (lazy imports to avoid circular deps)
__all__ = [
    # Core
    "CGPUJob",
    "JobStatus",
    "JobResult",
    "CGPUJobManager",
    # Jobs (imported on demand)
    "TranscribeJob",
    "UpscaleJob",
    "StabilizeJob",
]


def __getattr__(name):
    """Lazy import job implementations."""
    if name == "TranscribeJob":
        from .transcribe import TranscribeJob
        return TranscribeJob
    elif name == "UpscaleJob":
        from .upscale import UpscaleJob
        return UpscaleJob
    elif name == "StabilizeJob":
        from .stabilize import StabilizeJob
        return StabilizeJob
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
