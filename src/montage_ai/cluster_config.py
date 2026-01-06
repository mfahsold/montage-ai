"""
Cluster Configuration - Defines encoding nodes for distributed processing.

This file contains the actual cluster hardware configuration based on
the K3s cluster at Fluxibri.

Nodes:
- fluxibri: AMD RX 7900 XT/XTX (21GB VRAM) - Primary heavy encode
- jetson: NVIDIA Jetson (8GB) - Edge video processing
- t14s: Qualcomm Snapdragon X Elite (Adreno) - ARM64 encode
- esprimo: Intel CPU - Light tasks
- workers: Raspberry Pi / AMD64 - CPU fallback

Usage:
    from montage_ai.cluster_config import get_cluster_nodes, create_cluster_router

    router = create_cluster_router(enable_cgpu=True)
    result = await encode_segment(router, input, output)
"""

from __future__ import annotations

import os
from typing import List, Optional

from .encoder_router import EncoderNode, EncoderRouter
from .logger import logger


# =============================================================================
# Cluster Node Definitions
# =============================================================================

CLUSTER_NODES = [
    # === ENERGY-EFFICIENT GPU NODES (Priority) ===

    # NVIDIA Jetson - Edge processing, low power (~15W)
    EncoderNode(
        name="jetson",
        hostname="192.168.1.15",
        encoder_type="nvmpi",
        max_concurrent=2,
        priority=100,  # Highest - energy efficient dedicated GPU
        ssh_user="codeai",
        vram_mb=8000,  # ~8GB shared memory
        capabilities=["h264_nvmpi", "hevc_nvmpi"],
    ),
    # Qualcomm Snapdragon X Elite - ARM64, ultra low power (~8W)
    EncoderNode(
        name="t14s",
        hostname="192.168.1.237",
        encoder_type="adreno",
        max_concurrent=2,
        priority=90,  # High - most energy efficient
        ssh_user="codeai",
        vram_mb=0,  # Shared memory
        capabilities=["h264_v4l2m2m"],
    ),

    # === POWER-HUNGRY GPU (Last Resort for Local) ===

    # AMD RX 7900 XT/XTX - High power (~300W), but massive VRAM
    # Use for: 4K, long renders, batch jobs when efficiency doesn't matter
    EncoderNode(
        name="fluxibri",
        hostname="192.168.1.16",
        encoder_type="rocm",
        max_concurrent=4,  # Large VRAM allows multiple streams
        priority=50,  # Lower priority - only when needed
        ssh_user="codeai",
        vram_mb=21458,  # 21GB VRAM
        capabilities=["h264_vaapi", "hevc_vaapi", "av1_vaapi"],
    ),

    # === CPU FALLBACK NODES ===

    # Intel Esprimo - CPU encoding
    EncoderNode(
        name="esprimo",
        hostname="192.168.1.157",
        encoder_type="cpu",
        max_concurrent=4,
        priority=20,  # Low priority - CPU only
        ssh_user="codeai",
        capabilities=["libx264", "libx265"],
    ),
    # AMD64 Worker - CPU encoding
    EncoderNode(
        name="worker-amd64",
        hostname="192.168.1.37",
        encoder_type="cpu",
        max_concurrent=8,
        priority=15,
        ssh_user="codeai",
        capabilities=["libx264", "libx265"],
    ),
]


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
