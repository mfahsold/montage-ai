"""
Cluster Configuration - Defines encoding nodes for distributed processing.

Configure your cluster nodes via environment variable or config file.

Environment Variable (JSON):
    export MONTAGE_CLUSTER_NODES='[
        {"name": "gpu-node", "hostname": "10.0.0.10", "encoder_type": "nvenc", "priority": 100},
        {"name": "cpu-node", "hostname": "10.0.0.11", "encoder_type": "cpu", "priority": 10}
    ]'

Or create a config file at ~/.config/montage-ai/cluster.json

Example node configuration:
    {
        "name": "my-gpu",           # Unique node name
        "hostname": "10.0.0.10",    # IP or hostname
        "encoder_type": "nvenc",    # nvenc, nvmpi, vaapi, rocm, qsv, cpu
        "max_concurrent": 2,        # Max parallel encode jobs
        "priority": 100,            # Higher = preferred (0-100)
        "ssh_user": "user",         # SSH username for remote nodes
        "vram_mb": 8000,            # GPU VRAM in MB (0 for CPU)
        "capabilities": ["h264_nvenc", "hevc_nvenc"]
    }

Usage:
    from montage_ai.cluster_config import get_cluster_nodes, create_cluster_router

    router = create_cluster_router(enable_cgpu=True)
    result = await encode_segment(router, input, output)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from .encoder_router import EncoderNode, EncoderRouter
from .logger import logger


# =============================================================================
# Cluster Node Loading
# =============================================================================

def _load_nodes_from_env() -> List[EncoderNode]:
    """Load cluster nodes from MONTAGE_CLUSTER_NODES environment variable."""
    nodes_json = os.environ.get("MONTAGE_CLUSTER_NODES", "")
    if not nodes_json:
        return []

    try:
        nodes_data = json.loads(nodes_json)
        return [EncoderNode(**node) for node in nodes_data]
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse MONTAGE_CLUSTER_NODES: {e}")
        return []


def _load_nodes_from_config() -> List[EncoderNode]:
    """Load cluster nodes from config file."""
    config_paths = [
        Path.home() / ".config" / "montage-ai" / "cluster.json",
        Path("/etc/montage-ai/cluster.json"),
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    nodes_data = json.load(f)
                logger.info(f"Loaded cluster config from {config_path}")
                return [EncoderNode(**node) for node in nodes_data]
            except (json.JSONDecodeError, TypeError, OSError) as e:
                logger.warning(f"Failed to load {config_path}: {e}")

    return []


def _get_default_nodes() -> List[EncoderNode]:
    """Return empty list - users must configure their own cluster."""
    return []


# Load nodes: env var takes precedence, then config file, then empty default
CLUSTER_NODES: List[EncoderNode] = (
    _load_nodes_from_env() or
    _load_nodes_from_config() or
    _get_default_nodes()
)

if CLUSTER_NODES:
    logger.info(f"Loaded {len(CLUSTER_NODES)} cluster nodes")
else:
    logger.debug("No cluster nodes configured (local-only mode)")


# =============================================================================
# Cluster Router Factory
# =============================================================================

def get_cluster_nodes(
    include_cpu: bool = True,
    min_priority: int = 0,
) -> List[EncoderNode]:
    """
    Get list of cluster nodes filtered by criteria.

    Args:
        include_cpu: Include CPU-only nodes
        min_priority: Minimum priority threshold

    Returns:
        Filtered list of EncoderNode objects.
    """
    nodes = []
    for node in CLUSTER_NODES:
        if not include_cpu and node.encoder_type == "cpu":
            continue
        if node.priority < min_priority:
            continue
        nodes.append(node)
    return nodes


def create_cluster_router(
    enable_cgpu: bool = True,
    include_remote: bool = True,
    gpu_only: bool = False,
) -> EncoderRouter:
    """
    Create an EncoderRouter configured for the cluster.

    Args:
        enable_cgpu: Enable Cloud GPU for heavy tasks (upscale, long duration)
        include_remote: Include remote SSH nodes
        gpu_only: Only include GPU-capable nodes

    Returns:
        Configured EncoderRouter instance.
    """
    router = EncoderRouter(enable_cgpu=enable_cgpu)

    if include_remote:
        for node in get_cluster_nodes(include_cpu=not gpu_only):
            # Skip local node (already added by router)
            if node.hostname in ("localhost", "127.0.0.1"):
                continue
            router.add_node(node)

    logger.info(f"Cluster router initialized with {len(router.nodes)} nodes")
    for node in router.nodes:
        logger.debug(f"  - {node.name}: {node.encoder_type} @ {node.hostname} (prio={node.priority})")

    return router


def get_best_encoder_for_task(
    task_type: str,
    duration_seconds: float = 60,
    input_resolution: tuple = (1920, 1080),
    output_resolution: tuple = (1920, 1080),
) -> str:
    """
    Determine the best encoder node for a given task.

    Energy-efficient routing:
    - Jetson/T14s for standard encoding (low power)
    - Fluxibri only for 4K/heavy VRAM tasks
    - CGPU for upscaling and long transcription

    Args:
        task_type: Type of task (encode, upscale, stabilize, transcribe)
        duration_seconds: Video duration
        input_resolution: Input (width, height)
        output_resolution: Output (width, height)

    Returns:
        Node name or "cgpu" for cloud processing.
    """
    out_w, out_h = output_resolution
    in_w, in_h = input_resolution

    # Heavy tasks -> CGPU (offload to cloud)
    is_4k_output = out_w >= 3840 or out_h >= 2160
    is_4k_input = in_w >= 3840 or in_h >= 2160
    is_long = duration_seconds > 300  # 5 minutes

    if task_type == "upscale" and (is_4k_output or is_long):
        return "cgpu"

    if task_type == "transcribe" and is_long:
        return "cgpu"

    # 4K tasks need Fluxibri's VRAM (21GB)
    if is_4k_input or is_4k_output:
        return "fluxibri"

    # Standard encoding -> Jetson (energy efficient, ~15W)
    if task_type in ("encode", "color_grade"):
        return "jetson"

    # Stabilization -> Jetson (GPU accelerated, low power)
    if task_type == "stabilize":
        return "jetson"

    # Default -> Jetson (most energy efficient GPU)
    return "jetson"


# =============================================================================
# Environment-based Configuration
# =============================================================================

def get_encoder_from_env() -> Optional[str]:
    """
    Get encoder preference from environment variable.

    Supports:
    - MONTAGE_ENCODER=fluxibri  (use specific node)
    - MONTAGE_ENCODER=cgpu      (force cloud GPU)
    - MONTAGE_ENCODER=local     (local only)
    - MONTAGE_ENCODER=auto      (automatic selection)
    """
    encoder = os.environ.get("MONTAGE_ENCODER", "auto").lower()

    if encoder == "auto":
        return None

    if encoder in ("cgpu", "cloud"):
        return "cgpu"

    if encoder in ("local", "localhost"):
        return "local"

    # Check if it's a known node name
    for node in CLUSTER_NODES:
        if node.name.lower() == encoder:
            return node.name

    return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CLUSTER_NODES",
    "get_cluster_nodes",
    "create_cluster_router",
    "get_best_encoder_for_task",
    "get_encoder_from_env",
]
