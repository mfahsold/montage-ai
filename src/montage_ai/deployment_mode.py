"""
Unified Deployment Mode Detection for Montage AI

This module consolidates the previously fragmented mode detection:
- CLUSTER_MODE (boolean) - temp directory, cluster offloading
- MONTAGE_CLUSTER_MODE (enum) - node/hardware discovery

Into a single DEPLOYMENT_MODE with two clear states:
- local: Development mode, direct execution, local storage
- cluster: Kubernetes deployment, queue-based, shared storage

Usage:
    from montage_ai.deployment_mode import get_deployment_mode, is_cluster_mode

    mode = get_deployment_mode()
    if is_cluster_mode():
        # Use shared storage, queue-based jobs, K8s node discovery
        ...
"""

import os
from enum import Enum
from typing import Optional


class DeploymentMode(Enum):
    """Canonical deployment modes for Montage AI."""
    LOCAL = "local"      # Development: direct execution, local paths
    CLUSTER = "cluster"  # Production: K8s, queues, shared storage


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse boolean from string with fallback."""
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_deployment_mode() -> DeploymentMode:
    """
    Detect the current deployment mode.

    Priority:
    1. DEPLOYMENT_MODE env var (explicit override)
    2. Derive from CLUSTER_MODE + MONTAGE_CLUSTER_MODE (backwards compat)
    3. Auto-detect from environment (K8s service account present?)

    Returns:
        DeploymentMode.CLUSTER if running in Kubernetes cluster
        DeploymentMode.LOCAL otherwise
    """
    # 1. Explicit override
    explicit = os.environ.get("DEPLOYMENT_MODE", "").lower()
    if explicit in ("local", "cluster"):
        return DeploymentMode(explicit)

    # 2. Backwards compatibility: derive from existing env vars
    cluster_mode = _parse_bool(os.environ.get("CLUSTER_MODE"), False)
    montage_cluster_mode = os.environ.get("MONTAGE_CLUSTER_MODE", "auto").lower()

    if cluster_mode or montage_cluster_mode == "k8s":
        return DeploymentMode.CLUSTER

    if montage_cluster_mode == "local":
        return DeploymentMode.LOCAL

    # 3. Auto-detect: check for K8s service account
    if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
        return DeploymentMode.CLUSTER

    # 4. Check for common K8s env vars
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return DeploymentMode.CLUSTER

    return DeploymentMode.LOCAL


def is_cluster_mode() -> bool:
    """Check if running in cluster deployment mode."""
    return get_deployment_mode() == DeploymentMode.CLUSTER


def is_local_mode() -> bool:
    """Check if running in local development mode."""
    return get_deployment_mode() == DeploymentMode.LOCAL


def get_legacy_cluster_mode() -> bool:
    """
    Get CLUSTER_MODE boolean for backwards compatibility.

    New code should use is_cluster_mode() instead.
    """
    mode = get_deployment_mode()
    return mode == DeploymentMode.CLUSTER


def get_legacy_montage_cluster_mode() -> str:
    """
    Get MONTAGE_CLUSTER_MODE string for backwards compatibility.

    Returns: "local", "k8s", "config", or "auto"
    """
    # If explicitly set, return as-is
    explicit = os.environ.get("MONTAGE_CLUSTER_MODE")
    if explicit:
        return explicit.lower()

    # Derive from deployment mode
    mode = get_deployment_mode()
    return "k8s" if mode == DeploymentMode.CLUSTER else "local"


# Module-level cache for performance
_cached_mode: Optional[DeploymentMode] = None


def get_cached_deployment_mode() -> DeploymentMode:
    """
    Get cached deployment mode (for hot paths).

    Mode is determined once at first call and cached.
    Use reload_deployment_mode() to refresh.
    """
    global _cached_mode
    if _cached_mode is None:
        _cached_mode = get_deployment_mode()
    return _cached_mode


def reload_deployment_mode() -> DeploymentMode:
    """Force reload of cached deployment mode."""
    global _cached_mode
    _cached_mode = get_deployment_mode()
    return _cached_mode
