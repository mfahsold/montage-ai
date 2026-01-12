"""
Cluster Module - Distributed Processing for Montage AI

Provides abstractions for:
- Node capability detection
- Task routing based on hardware
- Parallel job distribution
- Result aggregation
- Distributed scene detection
"""

from .node_capabilities import (
    NodeCapability,
    ClusterManager,
    ClusterMode,
    GPUType,
    TaskType,
    get_cluster_manager,
    reset_cluster_manager,
    detect_local_hardware,
)
from .task_router import TaskRouter, DistributedJob
from .job_submitter import JobSubmitter, JobSpec, JobStatus

__all__ = [
    # Core classes
    "NodeCapability",
    "ClusterManager",
    "ClusterMode",
    "GPUType",
    "TaskType",
    "TaskRouter",
    "DistributedJob",
    # Job submission
    "JobSubmitter",
    "JobSpec",
    "JobStatus",
    # Functions
    "get_cluster_manager",
    "reset_cluster_manager",
    "detect_local_hardware",
]
