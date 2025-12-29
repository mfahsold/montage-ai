"""
Data models for the Web UI.

Provides structured types for job tracking, phase management, and API responses.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


# =============================================================================
# Pipeline Phase Definitions
# =============================================================================

PIPELINE_PHASES = [
    ("ingest", "Analyzing footage"),
    ("audio", "Processing audio"),
    ("creative", "Creative direction"),
    ("assembly", "Assembling timeline"),
    ("enhance", "Enhancing clips"),
    ("render", "Final render"),
]

PHASE_NAME_TO_INDEX = {name: idx for idx, (name, _) in enumerate(PIPELINE_PHASES)}


# =============================================================================
# Job Phase Tracking
# =============================================================================

@dataclass
class JobPhase:
    """
    Structured phase information for job progress tracking.

    Attributes:
        name: Phase identifier (e.g., "ingest", "render")
        label: Human-readable phase name (e.g., "Analyzing footage")
        number: Current phase number (1-based)
        total: Total number of phases
        started_at: ISO timestamp when phase started
        progress_percent: Progress within current phase (0-100)
    """
    name: str
    label: str
    number: int
    total: int
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    progress_percent: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_name(cls, name: str, progress: int = 0) -> "JobPhase":
        """Create JobPhase from phase name."""
        idx = PHASE_NAME_TO_INDEX.get(name, 0)
        label = PIPELINE_PHASES[idx][1] if idx < len(PIPELINE_PHASES) else name
        return cls(
            name=name,
            label=label,
            number=idx + 1,
            total=len(PIPELINE_PHASES),
            progress_percent=progress,
        )

    @classmethod
    def initial(cls) -> "JobPhase":
        """Create initial phase (queued state)."""
        return cls(
            name="queued",
            label="Waiting in queue",
            number=0,
            total=len(PIPELINE_PHASES),
            progress_percent=0,
        )


# =============================================================================
# Job Status
# =============================================================================

@dataclass
class JobStatus:
    """
    Complete job status including phase information.

    Attributes:
        id: Unique job identifier
        style: Selected editing style
        status: Job state (queued, running, completed, failed)
        phase: Current phase information
        created_at: Job creation timestamp
        started_at: Processing start timestamp
        completed_at: Completion timestamp
        error: Error message if failed
        output_path: Path to output file if completed
    """
    id: str
    style: str
    status: str  # queued, running, completed, failed
    phase: JobPhase
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Ensure phase is also a dict
        if isinstance(result.get("phase"), dict):
            pass  # Already converted by asdict
        return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PIPELINE_PHASES",
    "PHASE_NAME_TO_INDEX",
    "JobPhase",
    "JobStatus",
]
