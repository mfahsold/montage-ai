"""
Cloud Configuration Module

Cloud GPU offloading and cluster settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from . import BaseConfig, _env_override


@dataclass
class CloudConfig(BaseConfig):
    """Cloud GPU and offloading configuration."""

    # CGPU settings
    cgpu_enabled: bool = field(
        default_factory=lambda: _env_override("cgpu_enabled", False)
    )
    cgpu_project: Optional[str] = field(
        default_factory=lambda: _env_override("cgpu_project", None)
    )
    cgpu_region: str = field(
        default_factory=lambda: _env_override("cgpu_region", "us-central1")
    )

    # Offloading behavior
    offload_upscale: bool = field(
        default_factory=lambda: _env_override("offload_upscale", True)
    )
    offload_llm: bool = field(
        default_factory=lambda: _env_override("offload_llm", True)
    )

    # Limits
    max_cloud_jobs: int = field(
        default_factory=lambda: _env_override("max_cloud_jobs", 5)
    )
    cloud_timeout: int = field(
        default_factory=lambda: _env_override("cloud_timeout", 1800)
    )


@dataclass
class ClusterConfig(BaseConfig):
    """Cluster deployment configuration."""

    # Cluster mode
    enabled: bool = field(default_factory=lambda: _env_override("cluster_mode", False))

    # Service discovery
    namespace: str = field(
        default_factory=lambda: _env_override("cluster_namespace", "default")
    )
    domain: str = field(
        default_factory=lambda: _env_override("cluster_domain", "cluster.local")
    )

    # Sharding
    shard_index: int = field(default_factory=lambda: _env_override("shard_index", 0))
    shard_count: int = field(default_factory=lambda: _env_override("shard_count", 1))

    # Services
    ollama_service: str = field(
        default_factory=lambda: _env_override("ollama_service", "ollama")
    )
    redis_service: str = field(
        default_factory=lambda: _env_override("redis_service", "redis")
    )
