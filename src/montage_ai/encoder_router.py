"""
Multi-GPU Encoder Router - Distributed Encoding Across Heterogeneous Hardware.

Supports:
- AMD ROCm (RX 7900 XTX with 20GB VRAM)
- NVIDIA Jetson (NVMPI)
- Qualcomm Adreno (V4L2)
- Intel QSV
- VAAPI (AMD/Intel Linux)
- Cloud GPU (cgpu) for heavy tasks

Usage:
    from montage_ai.encoder_router import EncoderRouter, encode_segment

    router = EncoderRouter(enable_cgpu=True)
    router.add_node(EncoderNode(name="gpu-node", hostname="10.0.0.10", ...))

    # Single segment
    result = await encode_segment(router, "/in/raw.mp4", "/out/encoded.mp4")

    # Parallel encoding
    results = await encode_segments_parallel(router, segments)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .core.hardware import get_best_hwaccel, HWConfig
from .logger import logger

# Optional cgpu import
try:
    from .cgpu_utils import is_cgpu_available
    CGPU_AVAILABLE = True
except ImportError:
    CGPU_AVAILABLE = False

    def is_cgpu_available() -> bool:
        return False


class TaskType(Enum):
    """Types of encoding tasks."""
    ENCODE = "encode"
    UPSCALE = "upscale"
    STABILIZE = "stabilize"
    TRANSCRIBE = "transcribe"
    COLOR_GRADE = "color_grade"


@dataclass
class EncoderNode:
    """Configuration for a single encoding node."""
    name: str
    hostname: str
    encoder_type: str  # vaapi, nvmpi, nvenc, rocm, adreno, qsv, cpu
    max_concurrent: int = 2
    priority: int = 5  # Higher = preferred
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None
    ssh_port: int = 22
    vram_mb: int = 0  # 0 = unknown
    capabilities: List[str] = field(default_factory=list)  # ["h264", "hevc", "av1"]

    @property
    def is_local(self) -> bool:
        """Check if this node is localhost."""
        return self.hostname in ("localhost", "127.0.0.1", os.uname().nodename)

    @property
    def is_gpu(self) -> bool:
        """Check if this is a GPU encoder."""
        return self.encoder_type not in ("cpu", "none")


@dataclass
class EncodeResult:
    """Result of an encoding operation."""
    success: bool
    output_path: str
    node_name: str
    duration_ms: int = 0
    error: Optional[str] = None


class EncoderRouter:
    """
    Routes encoding tasks to the best available hardware.

    Priority order:
    1. CGPU (for heavy tasks: 4K upscale, long duration)
    2. Local GPU with highest priority
    3. Remote GPU nodes via SSH
    4. CPU fallback
    """

    def __init__(self, enable_cgpu: bool = False):
        """
        Initialize router with auto-detected local hardware.

        Args:
            enable_cgpu: Enable Cloud GPU for heavy tasks.
        """
        self.nodes: List[EncoderNode] = []
        self._load: Dict[str, int] = {}  # node_name -> current jobs
        self._enable_cgpu = enable_cgpu

        # Auto-detect local hardware
        hw_config = get_best_hwaccel()
        self.local_node = self._create_local_node(hw_config)
        if self.local_node:
            self.add_node(self.local_node)


# Backwards-compatible alias to clarify this is the cluster/distributed router.
ClusterEncoderRouter = EncoderRouter

    def _create_local_node(self, hw_config: HWConfig) -> EncoderNode:
        """Create local node from hardware config."""
        return EncoderNode(
            name="local",
            hostname="localhost",
            encoder_type=hw_config.type,
            max_concurrent=2 if hw_config.is_gpu else os.cpu_count() or 4,
            priority=10 if hw_config.is_gpu else 1,
            capabilities=[hw_config.encoder],
        )

    @property
    def cgpu_available(self) -> bool:
        """Check if CGPU is available and enabled."""
        return self._enable_cgpu and CGPU_AVAILABLE and is_cgpu_available()

    def add_node(self, node: EncoderNode) -> None:
        """Add an encoding node to the router."""
        self.nodes.append(node)
        self._load[node.name] = 0
        logger.info(f"Added encoder node: {node.name} ({node.encoder_type}) @ {node.hostname}")

    def remove_node(self, name: str) -> bool:
        """Remove a node by name."""
        for i, node in enumerate(self.nodes):
            if node.name == name:
                del self.nodes[i]
                del self._load[name]
                return True
        return False

    def get_node_load(self, name: str) -> int:
        """Get current job count for a node."""
        return self._load.get(name, 0)

    def increment_load(self, name: str) -> None:
        """Increment job count for a node."""
        self._load[name] = self._load.get(name, 0) + 1

    def decrement_load(self, name: str) -> None:
        """Decrement job count for a node."""
        current = self._load.get(name, 0)
        self._load[name] = max(0, current - 1)

    def select_best_node(
        self,
        encoder_type: Optional[str] = None,
        require_gpu: bool = False,
        min_vram_mb: int = 0,
    ) -> Optional[EncoderNode]:
        """
        Select the best available node for encoding.

        Args:
            encoder_type: Require specific encoder type (vaapi, nvmpi, etc.)
            require_gpu: Only consider GPU nodes
            min_vram_mb: Minimum VRAM required

        Returns:
            Best available node or None if no suitable node found.
        """
        candidates = []

        for node in self.nodes:
            # Filter by encoder type
            if encoder_type and node.encoder_type != encoder_type:
                continue

            # Filter by GPU requirement
            if require_gpu and not node.is_gpu:
                continue

            # Filter by VRAM
            if min_vram_mb > 0 and node.vram_mb < min_vram_mb:
                continue

            # Check capacity
            current_load = self._load.get(node.name, 0)
            if current_load >= node.max_concurrent:
                continue

            candidates.append(node)

        if not candidates:
            return None

        # Sort by priority (descending), then by current load (ascending)
        candidates.sort(key=lambda n: (-n.priority, self._load.get(n.name, 0)))

        return candidates[0]

    def get_node_by_name(self, name: str) -> Optional[EncoderNode]:
        """Get a specific node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None


# =============================================================================
# Task Routing
# =============================================================================

def route_task(
    router: EncoderRouter,
    task_type: str,
    input_resolution: Tuple[int, int] = (1920, 1080),
    output_resolution: Tuple[int, int] = (1920, 1080),
    duration_seconds: float = 60,
) -> str:
    """
    Determine where to route a task: "local", "remote", or "cgpu".

    Heavy tasks (4K upscale, long duration) route to CGPU if available.
    Light tasks stay local.
    """
    # Check for heavy tasks that should go to CGPU
    if router.cgpu_available:
        out_width, out_height = output_resolution
        is_4k_upscale = out_width >= 3840 or out_height >= 2160
        is_long_duration = duration_seconds > 180  # 3 minutes
        is_upscale_task = task_type == "upscale"

        if is_upscale_task and (is_4k_upscale or is_long_duration):
            logger.info(f"Routing {task_type} to CGPU (4K: {is_4k_upscale}, long: {is_long_duration})")
            return "cgpu"

    # Check for local GPU
    best_node = router.select_best_node(require_gpu=True)
    if best_node and best_node.is_local:
        return "local"

    # Check for remote nodes
    if best_node and not best_node.is_local:
        return "remote"

    # Fallback to local CPU
    return "local"


# =============================================================================
# FFmpeg Execution
# =============================================================================

async def _run_ffmpeg_local(
    input_path: str,
    output_path: str,
    hw_config: HWConfig,
    extra_args: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Run FFmpeg locally with hardware acceleration."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]

    # Add decoder args
    cmd.extend(hw_config.decoder_args)

    # Input
    cmd.extend(["-i", input_path])

    # Filters (hwupload if needed)
    if hw_config.hwupload_filter:
        cmd.extend(["-vf", hw_config.hwupload_filter])

    # Extra args (e.g., scale, filters)
    if extra_args:
        cmd.extend(extra_args)

    # Encoder args
    cmd.extend(hw_config.encoder_args)

    # Quality
    if hw_config.type in ("nvenc", "nvmpi"):
        cmd.extend(["-cq", "23", "-b:v", "0"])
    elif hw_config.type == "vaapi":
        cmd.extend(["-qp", "23"])
    elif hw_config.type == "qsv":
        cmd.extend(["-global_quality", "23"])
    elif hw_config.type == "cpu":
        cmd.extend(["-crf", "23", "-preset", "medium"])

    # Audio
    cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    # Output
    cmd.append(output_path)

    logger.debug(f"FFmpeg local: {' '.join(cmd)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return True, output_path
        else:
            error = stderr.decode()[:500]
            logger.error(f"FFmpeg failed: {error}")
            return False, error

    except Exception as e:
        logger.exception(f"FFmpeg execution error: {e}")
        return False, str(e)


async def _run_ffmpeg_ssh(
    node: EncoderNode,
    input_path: str,
    output_path: str,
    extra_args: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Run FFmpeg on remote node via SSH."""
    # Build SSH command
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]

    if node.ssh_key_path:
        ssh_cmd.extend(["-i", node.ssh_key_path])

    ssh_cmd.extend(["-p", str(node.ssh_port)])

    user_host = f"{node.ssh_user}@{node.hostname}" if node.ssh_user else node.hostname
    ssh_cmd.append(user_host)

    # Build FFmpeg command for remote execution
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", input_path,
    ]

    if extra_args:
        ffmpeg_cmd.extend(extra_args)

    # Encoder-specific args
    if node.encoder_type == "nvmpi":
        ffmpeg_cmd.extend(["-c:v", "h264_nvmpi", "-qp", "23"])
    elif node.encoder_type == "vaapi":
        ffmpeg_cmd.extend([
            "-vaapi_device", "/dev/dri/renderD128",
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi", "-qp", "23",
        ])
    elif node.encoder_type == "rocm":
        ffmpeg_cmd.extend([
            "-vaapi_device", "/dev/dri/renderD128",
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi", "-qp", "23",
        ])
    else:
        ffmpeg_cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium"])

    ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k", output_path])

    # Escape the command for SSH
    remote_cmd = " ".join(f'"{arg}"' if " " in arg else arg for arg in ffmpeg_cmd)
    ssh_cmd.append(remote_cmd)

    logger.debug(f"FFmpeg SSH: {' '.join(ssh_cmd[:5])} ...")

    try:
        process = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return True, output_path
        else:
            error = stderr.decode()[:500]
            logger.error(f"SSH FFmpeg failed: {error}")
            return False, error

    except Exception as e:
        logger.exception(f"SSH execution error: {e}")
        return False, str(e)


# =============================================================================
# High-Level API
# =============================================================================

async def encode_segment(
    router: EncoderRouter,
    input_path: str,
    output_path: str,
    prefer_gpu: bool = True,
    extra_args: Optional[List[str]] = None,
) -> EncodeResult:
    """
    Encode a single video segment using the best available hardware.

    Args:
        router: EncoderRouter instance
        input_path: Input video path
        output_path: Output video path
        prefer_gpu: Prefer GPU encoding if available
        extra_args: Additional FFmpeg arguments

    Returns:
        EncodeResult with success status and details.
    """
    import time
    start_time = time.monotonic()

    # Select best node
    node = router.select_best_node(require_gpu=prefer_gpu)
    if not node:
        node = router.select_best_node()  # Fallback to any node

    if not node:
        return EncodeResult(
            success=False,
            output_path=output_path,
            node_name="none",
            error="No encoding nodes available",
        )

    # Track load
    router.increment_load(node.name)

    try:
        if node.is_local:
            hw_config = get_best_hwaccel()
            success, result = await _run_ffmpeg_local(
                input_path, output_path, hw_config, extra_args
            )
        else:
            success, result = await _run_ffmpeg_ssh(
                node, input_path, output_path, extra_args
            )

        duration_ms = int((time.monotonic() - start_time) * 1000)

        return EncodeResult(
            success=success,
            output_path=output_path if success else "",
            node_name=node.name,
            duration_ms=duration_ms,
            error=result if not success else None,
        )

    finally:
        router.decrement_load(node.name)


async def encode_segments_parallel(
    router: EncoderRouter,
    segments: List[Dict[str, str]],
    prefer_gpu: bool = True,
) -> List[EncodeResult]:
    """
    Encode multiple segments in parallel across available nodes.

    Args:
        router: EncoderRouter instance
        segments: List of {"input": path, "output": path} dicts
        prefer_gpu: Prefer GPU encoding

    Returns:
        List of EncodeResult for each segment.
    """
    tasks = [
        encode_segment(
            router,
            seg["input"],
            seg["output"],
            prefer_gpu=prefer_gpu,
        )
        for seg in segments
    ]

    return await asyncio.gather(*tasks)


# =============================================================================
# Cluster Discovery
# =============================================================================

def discover_cluster_nodes() -> List[EncoderNode]:
    """
    Discover encoding nodes from K8s cluster.

    Uses kubectl to list nodes and infer GPU capabilities from labels.
    """
    nodes: List[EncoderNode] = []

    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "wide", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning("kubectl not available or cluster not accessible")
            return nodes

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            name = parts[0]
            status = parts[1]
            internal_ip = parts[5] if len(parts) > 5 else ""

            if status != "Ready":
                continue

            # Infer encoder type from node name
            encoder_type = "cpu"
            priority = 1

            if "jetson" in name.lower():
                encoder_type = "nvmpi"
                priority = 8
            elif "fluxibri" in name.lower():
                encoder_type = "rocm"  # AMD with ROCm
                priority = 10
            elif "t14" in name.lower() or "adreno" in name.lower():
                encoder_type = "adreno"
                priority = 6

            node = EncoderNode(
                name=name,
                hostname=internal_ip or name,
                encoder_type=encoder_type,
                priority=priority,
                ssh_user="codeai",  # Default user
            )
            nodes.append(node)

    except subprocess.TimeoutExpired:
        logger.warning("kubectl timed out")
    except FileNotFoundError:
        logger.debug("kubectl not found")

    return nodes


async def probe_node_capabilities(hostname: str, ssh_user: str) -> Dict[str, Any]:
    """
    Probe a remote node for GPU capabilities via SSH.

    Returns dict with encoders, vram, etc.
    """
    caps: Dict[str, Any] = {
        "encoders": [],
        "vram_mb": 0,
        "hw_type": "cpu",
    }

    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
        f"{ssh_user}@{hostname}",
        "ffmpeg -encoders 2>/dev/null | grep -E 'nvenc|nvmpi|vaapi|qsv|amf|v4l2'"
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            output = stdout.decode()
            if "nvmpi" in output:
                caps["encoders"].append("nvmpi")
                caps["hw_type"] = "nvmpi"
            if "nvenc" in output:
                caps["encoders"].append("nvenc")
                caps["hw_type"] = "nvenc"
            if "vaapi" in output:
                caps["encoders"].append("vaapi")
                if caps["hw_type"] == "cpu":
                    caps["hw_type"] = "vaapi"
            if "amf" in output:
                caps["encoders"].append("amf")
                caps["hw_type"] = "rocm"
            if "v4l2" in output:
                caps["encoders"].append("v4l2")
                if caps["hw_type"] == "cpu":
                    caps["hw_type"] = "adreno"

    except Exception as e:
        logger.debug(f"Failed to probe {hostname}: {e}")

    return caps


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "EncoderNode",
    "EncoderRouter",
    "ClusterEncoderRouter",
    "EncodeResult",
    "TaskType",
    "encode_segment",
    "encode_segments_parallel",
    "route_task",
    "discover_cluster_nodes",
    "probe_node_capabilities",
]
