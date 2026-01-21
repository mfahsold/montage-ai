"""
Video Workflow Abstract Base Class - Unified Processing Pipeline

Provides shared infrastructure for all video workflows:
- Creator (Montage): Beat-synced editing with scene detection
- Shorts Studio: Vertical video reframing with captions
- Future: Trailer, Podcast, etc.

SOTA Design Pattern: Template Method + Strategy Pattern
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..logger import logger
from ..core.job_store import JobStore


# =============================================================================
# Workflow Phases (Shared State Machine)
# =============================================================================

class WorkflowPhase(Enum):
    """Standard phases across all workflows."""
    QUEUED = "queued"
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    RENDERING = "rendering"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class WorkflowOptions:
    """Base options for all workflows."""
    input_path: str
    output_dir: str
    job_id: str
    quality_profile: str = "standard"  # preview, standard, high
    
    # Shared enhancement options
    stabilize: bool = False
    upscale: bool = False
    enhance: bool = False
    
    # Extensible options dict for workflow-specific settings
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "success": self.success,
            "output_path": self.output_path,
            "error": self.error,
            "metadata": self.metadata,
            "duration_seconds": self.duration_seconds
        }


# =============================================================================
# Abstract Base Class
# =============================================================================

class VideoWorkflow(ABC):
    """
    Abstract base class for video workflows.
    
    Implements Template Method pattern:
    - execute() defines the processing skeleton
    - Subclasses implement workflow-specific steps
    
    Provides:
    - Job state management (via JobStore)
    - Progress tracking with phases
    - Error handling and recovery
    - Shared resource management
    """
    
    def __init__(self, options: WorkflowOptions):
        """
        Initialize workflow.
        
        Args:
            options: Workflow configuration
        """
        self.options = options
        self.job_store = JobStore()
        self.current_phase = WorkflowPhase.QUEUED
        self._start_time: Optional[datetime] = None
        self._result: Optional[WorkflowResult] = None
    
    # =========================================================================
    # Template Method (Defines Processing Skeleton)
    # =========================================================================
    
    def execute(self) -> WorkflowResult:
        """
        Execute workflow (Template Method).
        
        Defines the standard processing pipeline:
        1. Initialize
        2. Validate
        3. Analyze
        4. Process
        5. Render
        6. Export
        7. Cleanup
        
        Returns:
            WorkflowResult with success status and output path
        """
        self._start_time = datetime.now()
        
        try:
            # Phase 1: Initialize
            self._update_phase(WorkflowPhase.INITIALIZING)
            logger.info(f"[{self.workflow_name}] Initializing job {self.options.job_id}")
            self.initialize()
            
            # Phase 2: Validate
            logger.info(f"[{self.workflow_name}] Validating inputs...")
            self.validate()
            
            # Phase 3: Analyze
            self._update_phase(WorkflowPhase.ANALYZING)
            logger.info(f"[{self.workflow_name}] Analyzing video...")
            analysis_result = self.analyze()
            
            # Phase 4: Process
            self._update_phase(WorkflowPhase.PROCESSING)
            logger.info(f"[{self.workflow_name}] Processing...")
            processing_result = self.process(analysis_result)
            
            # Phase 5: Render
            self._update_phase(WorkflowPhase.RENDERING)
            logger.info(f"[{self.workflow_name}] Rendering...")
            render_result = self.render(processing_result)
            
            # Phase 6: Export
            self._update_phase(WorkflowPhase.EXPORTING)
            logger.info(f"[{self.workflow_name}] Exporting...")
            output_path = self.export(render_result)
            
            # Phase 7: Cleanup
            logger.info(f"[{self.workflow_name}] Cleaning up...")
            self.cleanup()
            
            # Success
            self._update_phase(WorkflowPhase.COMPLETED)
            duration = (datetime.now() - self._start_time).total_seconds()
            
            self._result = WorkflowResult(
                success=True,
                output_path=output_path,
                metadata=self.get_metadata(),
                duration_seconds=duration
            )
            
            # Persist result to JobStore
            # Persist result to JobStore (retry to avoid transient visibility races)
            try:
                self.job_store.update_job_with_retry(
                    self.options.job_id,
                    {
                        "result": self._result.to_dict(),
                        "output_file": os.path.basename(output_path),
                        "metadata": self.get_metadata(),
                        "status": "finished",
                    },
                    retries=4,
                )
            except Exception:
                logger.exception("Failed to persist workflow result for %s", self.options.job_id)

            logger.info(f"[{self.workflow_name}] Completed in {duration:.1f}s: {output_path}")
            return self._result
            
        except Exception as e:
            # Failure
            self._update_phase(WorkflowPhase.FAILED)
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[{self.workflow_name}] Failed: {error_msg}")
            # Log full traceback for debugging
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            self._result = WorkflowResult(
                success=False,
                error=error_msg
            )
            
            return self._result
    
    # =========================================================================
    # Abstract Methods (Workflow-Specific Implementation)
    # =========================================================================
    
    @property
    @abstractmethod
    def workflow_name(self) -> str:
        """Human-readable workflow name (e.g., "Montage Creator", "Shorts Studio")."""
        pass
    
    @property
    @abstractmethod
    def workflow_type(self) -> str:
        """Machine-readable workflow type (e.g., "montage", "shorts")."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize workflow-specific resources.
        
        Example:
        - Load models
        - Setup temporary directories
        - Initialize analyzers
        """
        pass
    
    @abstractmethod
    def validate(self) -> None:
        """
        Validate inputs and prerequisites.
        
        Should raise exception if validation fails.
        """
        pass
    
    @abstractmethod
    def analyze(self) -> Any:
        """
        Analyze input video.
        
        Returns:
            Analysis results (workflow-specific type)
        """
        pass
    
    @abstractmethod
    def process(self, analysis_result: Any) -> Any:
        """
        Process video based on analysis.
        
        Args:
            analysis_result: Output from analyze()
            
        Returns:
            Processing results (workflow-specific type)
        """
        pass
    
    @abstractmethod
    def render(self, processing_result: Any) -> Any:
        """
        Render final video.
        
        Args:
            processing_result: Output from process()
            
        Returns:
            Render results (workflow-specific type)
        """
        pass
    
    @abstractmethod
    def export(self, render_result: Any) -> str:
        """
        Export final output.
        
        Args:
            render_result: Output from render()
            
        Returns:
            Path to final output file
        """
        pass
    
    def cleanup(self) -> None:
        """
        Cleanup temporary resources.
        
        Default implementation does nothing.
        Override if needed.
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get workflow metadata for result.
        
        Default implementation returns basic info.
        Override to add workflow-specific metadata.
        """
        return {
            "workflow_type": self.workflow_type,
            "job_id": self.options.job_id,
            "quality_profile": self.options.quality_profile
        }
    
    # =========================================================================
    # Shared Utilities
    # =========================================================================
    
    def _update_phase(
        self,
        phase: WorkflowPhase,
        progress_percent: Optional[int] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Update job phase in JobStore.
        
        Args:
            phase: New phase
            progress_percent: Optional progress percentage (0-100)
            message: Optional status message
        """
        self.current_phase = phase
        
        updates = {
            "phase": {
                "name": phase.value,
                "label": phase.value.replace("_", " ").title()
            }
        }
        
        if progress_percent is not None:
            updates["progress_percent"] = progress_percent
        
        if message:
            updates["message"] = message
        
        if phase == WorkflowPhase.COMPLETED:
            updates["status"] = "completed"
            updates["completed_at"] = datetime.now().isoformat()
        elif phase == WorkflowPhase.FAILED:
            updates["status"] = "failed"
        elif phase != WorkflowPhase.QUEUED:
            updates["status"] = "running"
        
        try:
            self.job_store.update_job_with_retry(self.options.job_id, updates, retries=3)
        except Exception:
            logger.exception("Failed to persist workflow phase update for %s", self.options.job_id)
    
    def _update_progress(
        self,
        percent: int,
        message: Optional[str] = None,
        current_item: Optional[str] = None,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
        gpu_util: Optional[str] = None,
        memory_pressure: Optional[str] = None,
    ) -> None:
        """
        Update progress percentage with optional resource metrics.

        Args:
            percent: Progress percentage (0-100)
            message: Optional status message
            current_item: Current file/clip being processed
            cpu_percent: Process CPU usage %
            memory_mb: Process memory (RSS) in MB
            gpu_util: GPU utilization string
            memory_pressure: Memory pressure level
        """
        updates: Dict[str, Any] = {"progress_percent": percent}
        if message:
            updates["message"] = message
        if current_item:
            updates["current_item"] = current_item
        if cpu_percent is not None:
            updates["cpu_percent"] = cpu_percent
        if memory_mb is not None:
            updates["memory_mb"] = memory_mb
        if gpu_util:
            updates["gpu_util"] = gpu_util
        if memory_pressure:
            updates["memory_pressure"] = memory_pressure

        # Use retry for high-value progress updates (near completion) to reduce visibility races.
        try:
            if percent >= 90:
                # Persist high-value updates with a small retry budget to avoid API staleness.
                self.job_store.update_job_with_retry(self.options.job_id, updates, retries=2)
            else:
                # Best-effort for frequent, low-value progress updates to avoid added latency.
                self.job_store.update_job(self.options.job_id, updates)
        except Exception:
            logger.exception("Failed to persist progress update for %s", self.options.job_id)


# =============================================================================
# Factory
# =============================================================================

def create_workflow(workflow_type: str, options: WorkflowOptions) -> VideoWorkflow:
    """
    Factory function to create workflow instances.
    
    Args:
        workflow_type: Type of workflow ("montage", "shorts", etc.)
        options: Workflow configuration
        
    Returns:
        VideoWorkflow instance
        
    Raises:
        ValueError: If workflow_type is unknown
    """
    # Import workflow implementations
    from .montage_workflow import MontageWorkflow
    from .shorts_workflow import ShortsWorkflow
    
    workflows = {
        "montage": MontageWorkflow,
        "shorts": ShortsWorkflow,
    }
    
    workflow_class = workflows.get(workflow_type)
    if not workflow_class:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    return workflow_class(options)
