"""
Processing Configuration Module

Video/audio processing settings and thresholds.
"""

import os
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

from . import BaseConfig, _env_override


def _get_effective_cpu_count() -> int:
    """Get effective CPU count respecting container limits."""
    # Try cgroup v2
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.exists():
            parts = cpu_max.read_text().strip().split()
            if len(parts) >= 2 and parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                return max(1, quota // period)
    except (OSError, ValueError):
        pass

    # Try cgroup v1
    try:
        cfs_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        cfs_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if cfs_quota.exists() and cfs_period.exists():
            quota = int(cfs_quota.read_text().strip())
            period = int(cfs_period.read_text().strip())
            if quota > 0:
                return max(1, quota // period)
    except (OSError, ValueError):
        pass

    return multiprocessing.cpu_count()


@dataclass
class StabilizationConfig(BaseConfig):
    """Video stabilization settings."""

    enabled: bool = True
    smoothing: int = field(
        default_factory=lambda: _env_override("stabilize_smoothing", 10)
    )
    max_shift: float = field(
        default_factory=lambda: _env_override("stabilize_max_shift", 0.05)
    )
    crop_ratio: float = field(
        default_factory=lambda: _env_override("stabilize_crop_ratio", 0.95)
    )


@dataclass
class ProcessingConfig(BaseConfig):
    """Processing pipeline configuration."""

    # Parallelism
    max_workers: int = field(
        default_factory=lambda: _env_override(
            "max_workers", min(8, _get_effective_cpu_count())
        )
    )
    scene_detection_workers: int = field(
        default_factory=lambda: _env_override(
            "scene_workers", min(4, _get_effective_cpu_count() // 2)
        )
    )

    # Memory
    batch_size: int = field(default_factory=lambda: _env_override("batch_size", 25))
    max_memory_gb: float = field(
        default_factory=lambda: _env_override("max_memory_gb", 8.0)
    )

    # Quality profiles
    quality_profile: str = field(
        default_factory=lambda: _env_override("quality_profile", "standard")
    )  # preview, standard, high, master

    # Stabilization
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)

    # Timeouts (seconds)
    scene_detection_timeout: int = field(
        default_factory=lambda: _env_override("scene_timeout", 300)
    )
    upscale_timeout: int = field(
        default_factory=lambda: _env_override("upscale_timeout", 600)
    )
    render_timeout: int = field(
        default_factory=lambda: _env_override("render_timeout", 1800)
    )

    def validate(self) -> list:
        """Validate processing configuration."""
        errors = []

        valid_profiles = ["preview", "standard", "high", "master"]
        if self.quality_profile not in valid_profiles:
            errors.append(
                f"Invalid quality_profile: {self.quality_profile}. Use: {valid_profiles}"
            )

        if self.max_workers < 1:
            errors.append("max_workers must be >= 1")

        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")

        return errors
