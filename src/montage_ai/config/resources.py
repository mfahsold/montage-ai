"""
Resources Configuration Module

System resource limits and monitoring.
"""

from dataclasses import dataclass, field

from . import BaseConfig, _env_override


@dataclass
class ResourcesConfig(BaseConfig):
    """System resource configuration."""

    # Memory
    memory_safety_margin_mb: int = field(
        default_factory=lambda: _env_override("memory_safety_margin_mb", 500)
    )
    max_memory_percent: float = field(
        default_factory=lambda: _env_override("max_memory_percent", 85.0)
    )

    # Disk
    min_disk_free_gb: float = field(
        default_factory=lambda: _env_override("min_disk_free_gb", 5.0)
    )

    # Monitoring
    mem_interval_sec: float = field(
        default_factory=lambda: _env_override("monitoring_mem_interval", 5.0)
    )
