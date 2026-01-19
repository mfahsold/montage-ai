"""
Node Capabilities - Hardware-Aware Task Routing

Provides automatic hardware detection and intelligent task routing for
heterogeneous clusters. Works in multiple modes:

1. **Local Mode**: Auto-detect hardware on current machine (default)
2. **Kubernetes Mode**: Discover nodes via K8s API
3. **Config Mode**: Load cluster definition from YAML file
4. **Hybrid Mode**: Combine K8s discovery with local detection

Usage:
    from montage_ai.cluster import get_cluster_manager, TaskType

    # Auto-detect local hardware (no K8s required)
    cluster = get_cluster_manager()

    # Use K8s cluster discovery
    cluster = get_cluster_manager(mode="k8s")

    # Load from config file
    cluster = get_cluster_manager(config_path="/path/to/cluster.yaml")

    # Get best node for task
    nodes = cluster.get_nodes_for_task(TaskType.GPU_ENCODING, resolution=(3840, 2160))

Environment Variables:
    MONTAGE_CLUSTER_MODE: "local", "k8s", "config", "auto" (default: "auto")
    MONTAGE_CLUSTER_CONFIG: Path to cluster config YAML
    MONTAGE_FORCE_CPU: Set to "1" to disable GPU detection
    MONTAGE_BARE_METAL_ONLY: Set to "1" to prefer bare-metal nodes for local tasks
"""

import json
import os
import platform
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from ..cgpu_utils import is_cgpu_available
except ImportError:
    def is_cgpu_available(*args, **kwargs): return False

from ..logger import logger


# =============================================================================
# Enums
# =============================================================================

class TaskType(Enum):
    """Task types that can be distributed across the cluster."""
    GPU_ENCODING = auto()         # Video encoding with GPU acceleration
    CPU_SCENE_DETECTION = auto()  # CPU-intensive scene analysis
    MEMORY_INTENSIVE = auto()     # 4K/8K processing, large batches
    PROXY_GENERATION = auto()     # Lightweight proxy creation
    AUDIO_ANALYSIS = auto()       # Beat detection, audio processing
    THUMBNAIL_EXTRACTION = auto() # Quick thumbnail generation
    FINAL_RENDER = auto()         # Final concatenation with effects
    ML_INFERENCE = auto()         # LLM/VLM-based clip selection
    CLOUD_UPSCALE = auto()        # Upscaling via cgpu/cloud
    CLOUD_TRANSCRIPTION = auto()  # Transcription via cgpu/cloud


class GPUType(Enum):
    """GPU types with associated encoder info."""
    NONE = ("none", None, None)
    AMD_ROCM = ("amd-rocm", "h264_amf", "amf")
    AMD_VAAPI = ("amd-vaapi", "h264_vaapi", "vaapi")
    NVIDIA_NVENC = ("nvidia-nvenc", "h264_nvenc", "cuda")
    NVIDIA_TEGRA = ("nvidia-tegra", "h264_nvenc", "cuda")
    INTEL_QSV = ("intel-qsv", "h264_qsv", "qsv")
    INTEL_VAAPI = ("intel-vaapi", "h264_vaapi", "vaapi")
    APPLE_VIDEOTOOLBOX = ("apple-vt", "h264_videotoolbox", "videotoolbox")
    QUALCOMM_ADRENO = ("qualcomm-adreno", None, None)
    CGPU = ("cgpu", "h264_nvenc", "cuda")

    def __init__(self, id_: str, encoder: Optional[str], hwaccel: Optional[str]):
        self.id = id_
        self.encoder = encoder
        self.hwaccel = hwaccel

    @property
    def value(self) -> str:
        return self.id


class ClusterMode(Enum):
    """Cluster discovery modes."""
    LOCAL = "local"       # Single machine, auto-detect hardware
    K8S = "k8s"          # Kubernetes cluster discovery
    CONFIG = "config"     # Load from YAML config file
    AUTO = "auto"        # Auto-detect best mode


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NodeCapability:
    """Hardware capabilities of a single node (local machine or cluster node)."""
    name: str
    cpu_cores: int
    memory_gb: float
    architecture: str  # x86_64, aarch64, arm64
    gpu_type: GPUType = GPUType.NONE
    gpu_vram_gb: float = 0.0
    gpu_name: str = ""
    encoder: Optional[str] = None  # FFmpeg encoder name
    hwaccel: Optional[str] = None  # FFmpeg hwaccel value
    is_control_plane: bool = False
    is_local: bool = False  # True if this is the current machine
    benchmark_score: float = 1.0  # Performance multiplier from node benchmark
    labels: Dict[str, str] = field(default_factory=dict)
    address: str = ""  # IP or hostname for remote nodes

    def __post_init__(self):
        # Set encoder/hwaccel from GPU type if not specified
        if self.encoder is None and self.gpu_type != GPUType.NONE:
            self.encoder = self.gpu_type.encoder
        if self.hwaccel is None and self.gpu_type != GPUType.NONE:
            self.hwaccel = self.gpu_type.hwaccel
        try:
            self.benchmark_score = float(self.benchmark_score)
        except (TypeError, ValueError):
            self.benchmark_score = 1.0
        self.benchmark_score = max(0.1, min(self.benchmark_score, 10.0))

    @property
    def is_gpu_node(self) -> bool:
        return self.gpu_type != GPUType.NONE

    @property
    def is_arm(self) -> bool:
        return self.architecture in ("arm64", "aarch64", "armv8")

    @property
    def is_high_cpu(self) -> bool:
        return self.cpu_cores >= 8

    @property
    def is_high_memory(self) -> bool:
        return self.memory_gb >= 16

    @property
    def is_bare_metal(self) -> bool:
        """Heuristic for bare-metal nodes (prefers explicit labels)."""
        label_value = (
            self.labels.get("montage-ai/bare-metal")
            or self.labels.get("bare-metal")
            or self.labels.get("baremetal")
        )
        if isinstance(label_value, str) and label_value.lower() in ("true", "1", "yes"):
            return True

        instance_type = (
            self.labels.get("node.kubernetes.io/instance-type")
            or self.labels.get("beta.kubernetes.io/instance-type")
            or ""
        ).lower()
        return "bare" in instance_type or "metal" in instance_type

    def can_handle_task(self, task: TaskType, resolution: Tuple[int, int] = (1920, 1080)) -> bool:
        """Check if this node can handle a specific task type."""
        pixels = resolution[0] * resolution[1]
        is_4k_plus = pixels >= 8_294_400  # 4K = 3840x2160

        if task == TaskType.GPU_ENCODING:
            return self.is_gpu_node and self.encoder is not None

        elif task == TaskType.CPU_SCENE_DETECTION:
            if is_4k_plus:
                return self.cpu_cores >= 4 and self.memory_gb >= 8
            return self.cpu_cores >= 2 and self.memory_gb >= 4

        elif task == TaskType.MEMORY_INTENSIVE:
            if is_4k_plus:
                return self.memory_gb >= 16
            return self.memory_gb >= 8

        elif task == TaskType.FINAL_RENDER:
            if is_4k_plus and self.is_gpu_node:
                return self.gpu_vram_gb >= 4
            return self.cpu_cores >= 4 or self.is_gpu_node

        elif task == TaskType.PROXY_GENERATION:
            return not self.is_control_plane and self.cpu_cores >= 2

        elif task == TaskType.AUDIO_ANALYSIS:
            return self.cpu_cores >= 2 and self.memory_gb >= 2

        elif task == TaskType.THUMBNAIL_EXTRACTION:
            return not self.is_control_plane

        elif task == TaskType.ML_INFERENCE:
            return self.memory_gb >= 4

        elif task == TaskType.CLOUD_UPSCALE:
            return self.gpu_type == GPUType.CGPU

        elif task == TaskType.CLOUD_TRANSCRIPTION:
            return self.gpu_type == GPUType.CGPU

        return True

    def get_priority_for_task(self, task: TaskType, resolution: Tuple[int, int] = (1920, 1080)) -> int:
        """Get priority score for this task (higher = better)."""
        score = 0

        if task == TaskType.GPU_ENCODING:
            if self.gpu_type in (GPUType.AMD_ROCM, GPUType.NVIDIA_NVENC):
                score = 100 + int(self.gpu_vram_gb * 4)
            elif self.gpu_type == GPUType.NVIDIA_TEGRA:
                score = 80 + int(self.gpu_vram_gb * 4)
            elif self.gpu_type == GPUType.APPLE_VIDEOTOOLBOX:
                score = 90  # Apple Silicon is very efficient
            elif self.gpu_type in (GPUType.INTEL_QSV, GPUType.INTEL_VAAPI, GPUType.AMD_VAAPI):
                score = 50 + int(self.gpu_vram_gb * 2)
            elif self.gpu_type == GPUType.CGPU:
                score = 120  # Cloud GPUs are powerful but have latency

        elif task == TaskType.CPU_SCENE_DETECTION:
            score = self.cpu_cores * 8 + int(self.memory_gb)

        elif task == TaskType.MEMORY_INTENSIVE:
            score = int(self.memory_gb * 3) + self.cpu_cores

        elif task == TaskType.FINAL_RENDER:
            if self.is_gpu_node:
                score = 100 + int(self.gpu_vram_gb * 2)
                if self.gpu_type == GPUType.CGPU:
                    score += 20
            else:
                score = self.cpu_cores * 5

        elif task == TaskType.PROXY_GENERATION:
            score = 50

        elif task == TaskType.AUDIO_ANALYSIS:
            score = self.cpu_cores * 6 + int(self.memory_gb // 2)

        elif task == TaskType.ML_INFERENCE:
            if self.gpu_type == GPUType.CGPU:
                score = 200  # Preferred for ML tasks
            elif self.is_gpu_node:
                score = 100 + int(self.gpu_vram_gb * 3)
            elif self.is_arm:
                score = 60 + int(self.memory_gb)
            else:
                score = 40 + int(self.memory_gb // 2)

        elif task == TaskType.CLOUD_UPSCALE:
            if self.gpu_type == GPUType.CGPU:
                score = 100

        elif task == TaskType.CLOUD_TRANSCRIPTION:
            if self.gpu_type == GPUType.CGPU:
                score = 100

        # Prefer local node for single-machine setups
        if self.is_local:
            score += 5

        # Prefer bare-metal resources for consistent throughput
        if task not in (TaskType.CLOUD_UPSCALE, TaskType.CLOUD_TRANSCRIPTION) and self.is_bare_metal:
            score += 8

        # Apply benchmark multiplier (higher = faster)
        if self.benchmark_score != 1.0:
            score = int(round(score * self.benchmark_score))

        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "architecture": self.architecture,
            "gpu_type": self.gpu_type.value,
            "gpu_vram_gb": self.gpu_vram_gb,
            "gpu_name": self.gpu_name,
            "encoder": self.encoder,
            "hwaccel": self.hwaccel,
            "is_control_plane": self.is_control_plane,
            "benchmark_score": self.benchmark_score,
            "labels": self.labels,
            "address": self.address,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeCapability":
        """Create from dictionary."""
        gpu_type_str = data.get("gpu_type", "none")
        gpu_type = GPUType.NONE
        for gt in GPUType:
            if gt.value == gpu_type_str:
                gpu_type = gt
                break

        return cls(
            name=data.get("name", "unknown"),
            cpu_cores=data.get("cpu_cores", 1),
            memory_gb=data.get("memory_gb", 1.0),
            architecture=data.get("architecture", platform.machine()),
            gpu_type=gpu_type,
            gpu_vram_gb=data.get("gpu_vram_gb", 0.0),
            gpu_name=data.get("gpu_name", ""),
            encoder=data.get("encoder"),
            hwaccel=data.get("hwaccel"),
            is_control_plane=data.get("is_control_plane", False),
            benchmark_score=data.get("benchmark_score", 1.0),
            labels=data.get("labels", {}),
            address=data.get("address", ""),
        )


# =============================================================================
# Hardware Detection (Local Machine)
# =============================================================================

def _get_cpu_count() -> int:
    """Get number of CPU cores."""
    try:
        # Prefer physical cores over logical
        import multiprocessing
        return multiprocessing.cpu_count()
    except Exception:
        return os.cpu_count() or 1


def _get_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        # Linux
        if Path("/proc/meminfo").exists():
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / 1024 / 1024

        # macOS
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / 1024 / 1024 / 1024

        # Fallback
        import psutil
        return psutil.virtual_memory().total / 1024 / 1024 / 1024
    except Exception:
        return 8.0  # Conservative default


def _get_architecture() -> str:
    """Get CPU architecture."""
    arch = platform.machine().lower()
    # Normalize architecture names
    if arch in ("x86_64", "amd64"):
        return "x86_64"
    elif arch in ("arm64", "aarch64"):
        return "arm64"
    elif arch.startswith("arm"):
        return "arm"
    return arch


def _detect_nvidia_gpu() -> Optional[Tuple[GPUType, float, str]]:
    """Detect NVIDIA GPU using nvidia-smi."""
    if os.environ.get("MONTAGE_FORCE_CPU") == "1":
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            name = parts[0].strip()
            vram_mb = float(parts[1].strip()) if len(parts) > 1 else 0
            vram_gb = vram_mb / 1024

            # Detect Tegra/Jetson
            if "tegra" in name.lower() or "jetson" in name.lower():
                return (GPUType.NVIDIA_TEGRA, vram_gb, name)
            return (GPUType.NVIDIA_NVENC, vram_gb, name)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_amd_gpu() -> Optional[Tuple[GPUType, float, str]]:
    """Detect AMD GPU using rocm-smi or vainfo."""
    if os.environ.get("MONTAGE_FORCE_CPU") == "1":
        return None

    # Try ROCm first
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if "card0" in data:
                vram_bytes = data["card0"].get("VRAM Total Memory (B)", 0)
                vram_gb = vram_bytes / 1024 / 1024 / 1024

                # Get GPU name
                name_result = subprocess.run(
                    ["rocm-smi", "--showproductname", "--json"],
                    capture_output=True, text=True, timeout=5
                )
                name = "AMD GPU"
                if name_result.returncode == 0:
                    name_data = json.loads(name_result.stdout)
                    name = name_data.get("card0", {}).get("Card SKU", "AMD GPU")

                return (GPUType.AMD_ROCM, vram_gb, name)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # Fallback to VAAPI detection for AMD
    try:
        result = subprocess.run(
            ["vainfo"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "AMD" in result.stdout.upper():
            return (GPUType.AMD_VAAPI, 0.0, "AMD GPU (VAAPI)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _detect_intel_gpu() -> Optional[Tuple[GPUType, float, str]]:
    """Detect Intel GPU using vainfo or QSV."""
    if os.environ.get("MONTAGE_FORCE_CPU") == "1":
        return None

    # Check for Intel QSV
    try:
        result = subprocess.run(
            ["vainfo"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.upper()
            if "INTEL" in output or "I965" in output or "IRIX" in output:
                # Try to detect QSV support
                ffmpeg_result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=5
                )
                if ffmpeg_result.returncode == 0 and "h264_qsv" in ffmpeg_result.stdout:
                    return (GPUType.INTEL_QSV, 0.0, "Intel QSV")
                return (GPUType.INTEL_VAAPI, 0.0, "Intel VAAPI")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _detect_apple_gpu() -> Optional[Tuple[GPUType, float, str]]:
    """Detect Apple Silicon GPU."""
    if os.environ.get("MONTAGE_FORCE_CPU") == "1":
        return None

    if platform.system() != "Darwin":
        return None

    try:
        # Check for VideoToolbox encoder support
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "h264_videotoolbox" in result.stdout:
            # Get chip name
            chip_result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            chip_name = chip_result.stdout.strip() if chip_result.returncode == 0 else "Apple Silicon"
            return (GPUType.APPLE_VIDEOTOOLBOX, 0.0, chip_name)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _detect_cgpu() -> Optional[Tuple[GPUType, float, str]]:
    """Detect if cgpu cloud resource is available."""
    if os.environ.get("MONTAGE_FORCE_CPU") == "1":
        return None

    # We use a try/except because cgpu_utils might not be fully configured
    try:
        if is_cgpu_available(require_gpu=True):
            # cgpu usually provides T4 or A100 (16GB+)
            return (GPUType.CGPU, 16.0, "Google Colab T4/A100")
    except Exception:
        pass
    return None


def _detect_gpu() -> Tuple[GPUType, float, str]:
    """Detect best available GPU."""
    # Try each GPU type in priority order
    for detector in [_detect_nvidia_gpu, _detect_amd_gpu, _detect_intel_gpu, _detect_apple_gpu, _detect_cgpu]:
        result = detector()
        if result:
            return result
    return (GPUType.NONE, 0.0, "")


@lru_cache(maxsize=1)
def detect_local_hardware() -> NodeCapability:
    """
    Detect hardware capabilities of the local machine.

    Results are cached for performance.
    """
    hostname = platform.node() or "localhost"
    cpu_cores = _get_cpu_count()
    memory_gb = _get_memory_gb()
    architecture = _get_architecture()
    gpu_type, gpu_vram, gpu_name = _detect_gpu()

    logger.debug(
        f"Detected local hardware: {cpu_cores} cores, {memory_gb:.1f}GB RAM, "
        f"{architecture}, GPU: {gpu_type.value}"
    )

    return NodeCapability(
        name=hostname,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        architecture=architecture,
        gpu_type=gpu_type,
        gpu_vram_gb=gpu_vram,
        gpu_name=gpu_name,
        is_local=True,
    )


# =============================================================================
# Kubernetes Discovery
# =============================================================================

def _is_k8s_available() -> bool:
    """Check if kubectl is available and can connect to a cluster."""
    if shutil.which("kubectl") is None:
        return False

    try:
        result = subprocess.run(
            ["kubectl", "cluster-info", "--request-timeout=3s"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _discover_k8s_nodes() -> List[NodeCapability]:
    """Discover nodes from Kubernetes cluster."""
    nodes = []

    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            logger.warning("Failed to get K8s nodes")
            return nodes

        data = json.loads(result.stdout)
        for item in data.get("items", []):
            node = _parse_k8s_node(item)
            if node:
                nodes.append(node)

    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning(f"K8s node discovery failed: {e}")

    return nodes


def _parse_k8s_node(item: Dict[str, Any]) -> Optional[NodeCapability]:
    """Parse a Kubernetes node object into NodeCapability."""
    try:
        metadata = item.get("metadata", {})
        status = item.get("status", {})
        capacity = status.get("capacity", {})
        node_info = status.get("nodeInfo", {})
        labels = metadata.get("labels", {})

        name = metadata.get("name", "unknown")

        # Parse CPU (can be "4" or "4000m")
        cpu_str = capacity.get("cpu", "1")
        if cpu_str.endswith("m"):
            cpu_cores = int(cpu_str[:-1]) // 1000
        else:
            cpu_cores = int(cpu_str)

        # Parse memory (can be "16Gi", "16000Mi", "16000000Ki")
        mem_str = capacity.get("memory", "1Gi")
        memory_gb = _parse_k8s_memory(mem_str)

        # Detect architecture
        arch = node_info.get("architecture", "amd64")
        if arch == "amd64":
            arch = "x86_64"

        # Detect GPU from labels and resources
        gpu_type, gpu_vram = _detect_k8s_gpu(labels, capacity)
        benchmark_score = _parse_k8s_benchmark_score(labels)

        # Check if control plane
        is_control_plane = any(
            key in labels for key in [
                "node-role.kubernetes.io/control-plane",
                "node-role.kubernetes.io/master"
            ]
        )

        # Get address
        addresses = status.get("addresses", [])
        address = ""
        # Priority 1: InternalIP, Priority 2: ExternalIP, Priority 3: Hostname
        for addr_type in ["InternalIP", "ExternalIP", "Hostname"]:
            for addr in addresses:
                if addr.get("type") == addr_type:
                    address = addr.get("address", "")
                    break
            if address:
                break

        return NodeCapability(
            name=name,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            architecture=arch,
            gpu_type=gpu_type,
            gpu_vram_gb=gpu_vram,
            is_control_plane=is_control_plane,
            benchmark_score=benchmark_score,
            labels=labels,
            address=address,
        )

    except Exception as e:
        logger.warning(f"Failed to parse K8s node: {e}")
        return None


def _parse_k8s_memory(mem_str: str) -> float:
    """Parse Kubernetes memory string to GB."""
    mem_str = mem_str.strip()

    # Match patterns like "16Gi", "16000Mi", "16000000Ki", "16000000000"
    match = re.match(r"(\d+)([KMGTP]i?)?", mem_str, re.IGNORECASE)
    if not match:
        return 1.0

    value = float(match.group(1))
    unit = (match.group(2) or "").upper()

    multipliers = {
        "": 1 / (1024 ** 3),  # bytes to GB
        "K": 1 / (1024 ** 2),
        "KI": 1 / (1024 ** 2),
        "M": 1 / 1024,
        "MI": 1 / 1024,
        "G": 1,
        "GI": 1,
        "T": 1024,
        "TI": 1024,
    }

    return value * multipliers.get(unit, 1 / (1024 ** 3))


def _parse_k8s_benchmark_score(labels: Dict[str, str]) -> float:
    """Parse benchmark score label from a node."""
    for key in (
        "montage-ai/bench-score",
        "montage-ai/benchmark-score",
        "bench-score",
        "benchmark-score",
    ):
        value = labels.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _detect_k8s_gpu(labels: Dict[str, str], capacity: Dict[str, str]) -> Tuple[GPUType, float]:
    """Detect GPU type from K8s node labels and capacity."""
    # Check for NVIDIA GPU
    nvidia_gpu = capacity.get("nvidia.com/gpu", "0")
    if nvidia_gpu != "0":
        # Check for Tegra/Jetson
        if labels.get("accelerator") == "nvidia-tegra" or "jetson" in labels.get("node.kubernetes.io/instance-type", "").lower():
            return (GPUType.NVIDIA_TEGRA, 8.0)  # Assume 8GB shared
        return (GPUType.NVIDIA_NVENC, 0.0)

    # Check for AMD GPU
    amd_gpu = capacity.get("amd.com/gpu", "0")
    if amd_gpu != "0":
        if labels.get("amd.com/gpu") == "present":
            return (GPUType.AMD_ROCM, 0.0)
        return (GPUType.AMD_VAAPI, 0.0)

    # Check for Intel GPU via labels
    if labels.get("intel.feature.node.kubernetes.io/gpu") == "true":
        return (GPUType.INTEL_QSV, 0.0)

    # Check for cgpu cloud resource via label
    if labels.get("montage-ai/cgpu") == "true" or labels.get("cgpu") == "true":
        return (GPUType.CGPU, 16.0)

    return (GPUType.NONE, 0.0)


# =============================================================================
# Configuration File Loading
# =============================================================================

def _load_config_file(config_path: str) -> List[NodeCapability]:
    """Load cluster configuration from YAML file."""
    if not YAML_AVAILABLE:
        logger.warning("PyYAML not installed, cannot load config file")
        return []

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return []

    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        nodes = []
        for node_data in data.get("nodes", []):
            node = NodeCapability.from_dict(node_data)
            nodes.append(node)

        logger.info(f"Loaded {len(nodes)} nodes from config: {config_path}")
        return nodes

    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return []


# =============================================================================
# Cluster Manager
# =============================================================================

class ClusterManager:
    """
    Manages cluster node capabilities and task routing.

    Supports multiple discovery modes:
    - LOCAL: Single machine with auto-detected hardware
    - K8S: Kubernetes cluster discovery
    - CONFIG: Load from YAML config file
    - AUTO: Try K8s first, fallback to local
    """

    def __init__(
        self,
        mode: Optional[ClusterMode] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize ClusterManager.

        Args:
            mode: Discovery mode (defaults to AUTO)
            config_path: Path to cluster config YAML (for CONFIG mode)
        """
        self._lock = threading.Lock()
        # Read from environment if not specified
        if mode is None:
            mode_str = os.environ.get("MONTAGE_CLUSTER_MODE", "auto").lower()
            mode = ClusterMode(mode_str) if mode_str in [m.value for m in ClusterMode] else ClusterMode.AUTO

        if config_path is None:
            config_path = os.environ.get("MONTAGE_CLUSTER_CONFIG")

        self.mode = mode
        self.config_path = config_path
        self._nodes: Dict[str, NodeCapability] = {}
        self._local_node: Optional[NodeCapability] = None
        self._refresh_nodes()

    def _refresh_nodes(self):
        """Refresh node list based on mode."""
        with self._lock:
            self._nodes.clear()

            if self.mode == ClusterMode.CONFIG and self.config_path:
                nodes = _load_config_file(self.config_path)
                for node in nodes:
                    self._nodes[node.name] = node

            elif self.mode == ClusterMode.K8S:
                if _is_k8s_available():
                    nodes = _discover_k8s_nodes()
                    for node in nodes:
                        self._nodes[node.name] = node
                else:
                    logger.warning("K8s not available, falling back to local mode")
                    self._add_local_node()

            elif self.mode == ClusterMode.LOCAL:
                self._add_local_node()

            elif self.mode == ClusterMode.AUTO:
                # Try K8s first
                if _is_k8s_available():
                    nodes = _discover_k8s_nodes()
                    for node in nodes:
                        self._nodes[node.name] = node
                    logger.info(f"Auto-detected K8s cluster with {len(nodes)} nodes")
                else:
                    self._add_local_node()
                    logger.info("Running in local mode (no K8s cluster detected)")

        if not self._nodes:
            logger.warning("No nodes discovered, adding local node as fallback")
            self._add_local_node()

        # Always try to add CGPU if available
        self._add_cgpu_node()

    def _add_local_node(self):
        """Add the local machine as a node."""
        self._local_node = detect_local_hardware()
        self._nodes[self._local_node.name] = self._local_node

    def _add_cgpu_node(self):
        """Add a virtual cloud node if cgpu is available."""
        # Only add if not already present (could be in config or k8s)
        if "cloud-cgpu" in self._nodes:
            return

        try:
            if is_cgpu_available(require_gpu=True):
                node = NodeCapability(
                    name="cloud-cgpu",
                    cpu_cores=8,
                    memory_gb=32.0,
                    architecture="x86_64",
                    gpu_type=GPUType.CGPU,
                    gpu_vram_gb=16.0,
                    gpu_name="Google Colab T4/A100",
                    labels={"tier": "cloud", "purpose": "acceleration"},
                    address="colab.google.com"
                )
                self._nodes[node.name] = node
                logger.info("Added virtual cloud-cgpu node to cluster")
        except Exception:
            pass

    @property
    def nodes(self) -> List[NodeCapability]:
        """Get all worker nodes (excludes control plane)."""
        return [n for n in self._nodes.values() if not n.is_control_plane]

    @property
    def gpu_nodes(self) -> List[NodeCapability]:
        """Get nodes with GPU capabilities."""
        return [n for n in self.nodes if n.is_gpu_node]

    @property
    def arm_nodes(self) -> List[NodeCapability]:
        """Get ARM architecture nodes."""
        return [n for n in self.nodes if n.is_arm]

    @property
    def high_memory_nodes(self) -> List[NodeCapability]:
        """Get nodes with 16GB+ RAM."""
        return [n for n in self.nodes if n.is_high_memory]

    @property
    def total_cpu_cores(self) -> int:
        """Total CPU cores across all worker nodes."""
        return sum(n.cpu_cores for n in self.nodes)

    @property
    def total_memory_gb(self) -> float:
        """Total memory across all worker nodes."""
        return sum(n.memory_gb for n in self.nodes)

    @property
    def local_node(self) -> Optional[NodeCapability]:
        """Get the local node (if in local mode)."""
        return self._local_node

    def get_node(self, name: str) -> Optional[NodeCapability]:
        """Get a specific node by name."""
        return self._nodes.get(name)

    def get_nodes_for_task(
        self,
        task: TaskType,
        resolution: Tuple[int, int] = (1920, 1080),
        max_nodes: Optional[int] = None
    ) -> List[NodeCapability]:
        """
        Get nodes capable of handling a task, sorted by priority.

        Args:
            task: Type of task to run
            resolution: Video resolution for capability checking
            max_nodes: Maximum number of nodes to return

        Returns:
            List of capable nodes, highest priority first
        """
        capable = [
            n for n in self.nodes
            if n.can_handle_task(task, resolution)
        ]

        if self._is_bare_metal_only_mode(task):
            bare_nodes = [n for n in capable if n.is_bare_metal]
            if bare_nodes:
                capable = bare_nodes
            else:
                logger.warning(
                    "Bare-metal-only routing enabled for %s, but no bare-metal nodes are available; falling back.",
                    task.name,
                )

        # Sort by priority (descending)
        capable.sort(
            key=lambda n: n.get_priority_for_task(task, resolution),
            reverse=True
        )

        if max_nodes:
            return capable[:max_nodes]
        return capable

    def _is_bare_metal_only_mode(self, task: TaskType) -> bool:
        """Return True when bare-metal-only routing is enabled for this task."""
        if task in (TaskType.CLOUD_UPSCALE, TaskType.CLOUD_TRANSCRIPTION):
            return False
        value = os.environ.get("MONTAGE_BARE_METAL_ONLY", "").strip().lower()
        return value in ("1", "true", "yes", "on")

    def get_best_node_for_task(
        self,
        task: TaskType,
        resolution: Tuple[int, int] = (1920, 1080)
    ) -> Optional[NodeCapability]:
        """Get the single best node for a task."""
        nodes = self.get_nodes_for_task(task, resolution, max_nodes=1)
        return nodes[0] if nodes else None

    def get_parallel_distribution(
        self,
        task: TaskType,
        num_items: int,
        resolution: Tuple[int, int] = (1920, 1080)
    ) -> Dict[str, List[int]]:
        """
        Distribute items across capable nodes for parallel processing.

        Args:
            task: Type of task
            num_items: Number of items to distribute
            resolution: Video resolution

        Returns:
            Dict mapping node name to list of item indices
        """
        nodes = self.get_nodes_for_task(task, resolution)
        if not nodes:
            return {}

        distribution: Dict[str, List[int]] = {n.name: [] for n in nodes}

        # Weighted distribution based on priority scores
        total_score = sum(n.get_priority_for_task(task, resolution) for n in nodes)
        if total_score == 0:
            # Equal distribution
            for i in range(num_items):
                node = nodes[i % len(nodes)]
                distribution[node.name].append(i)
        else:
            # Weighted distribution
            assigned = 0
            for node in nodes:
                score = node.get_priority_for_task(task, resolution)
                share = max(1, int((score / total_score) * num_items))
                for _ in range(share):
                    if assigned < num_items:
                        distribution[node.name].append(assigned)
                        assigned += 1

            # Assign remaining items round-robin
            node_idx = 0
            while assigned < num_items:
                distribution[nodes[node_idx].name].append(assigned)
                assigned += 1
                node_idx = (node_idx + 1) % len(nodes)

        return {k: v for k, v in distribution.items() if v}

    def print_cluster_summary(self):
        """Print a summary of cluster capabilities."""
        mode_name = self.mode.value.upper()
        title = f"CLUSTER SUMMARY ({mode_name} MODE)"

        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

        print(f"\nTotal Resources:")
        print(f"  Worker Nodes: {len(self.nodes)}")
        print(f"  Total CPU:    {self.total_cpu_cores} cores")
        print(f"  Total RAM:    {self.total_memory_gb:.1f} GB")
        print(f"  GPU Nodes:    {len(self.gpu_nodes)}")

        if self.nodes:
            print(f"\nNodes:")
            for node in sorted(self.nodes, key=lambda n: n.name):
                gpu_info = f" | GPU: {node.gpu_type.value}" if node.is_gpu_node else ""
                local_tag = " [LOCAL]" if node.is_local else ""
                print(f"  {node.name}: {node.cpu_cores} CPU, {node.memory_gb:.1f}GB{gpu_info}{local_tag}")

        print(f"\nTask Routing Preview:")
        for task in [TaskType.GPU_ENCODING, TaskType.CPU_SCENE_DETECTION, TaskType.MEMORY_INTENSIVE]:
            best = self.get_best_node_for_task(task, resolution=(1920, 1080))
            if best:
                print(f"  {task.name}: {best.name}")
            else:
                print(f"  {task.name}: No capable node")

        print("=" * 70 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Export cluster configuration as dictionary."""
        return {
            "mode": self.mode.value,
            "nodes": [n.to_dict() for n in self._nodes.values()]
        }

    def save_config(self, path: str):
        """Save cluster configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise RuntimeError("PyYAML not installed")

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

        logger.info(f"Saved cluster config to {path}")


# =============================================================================
# Module-Level Functions
# =============================================================================

# Singleton instance
_cluster_manager: Optional[ClusterManager] = None


def get_cluster_manager(
    mode: Optional[str] = None,
    config_path: Optional[str] = None,
    force_refresh: bool = False
) -> ClusterManager:
    """
    Get or create the global ClusterManager instance.

    Args:
        mode: Cluster mode ("local", "k8s", "config", "auto")
        config_path: Path to config file (for "config" mode)
        force_refresh: Force recreation of the manager

    Returns:
        ClusterManager instance
    """
    global _cluster_manager

    if _cluster_manager is None or force_refresh:
        cluster_mode = None
        if mode:
            try:
                cluster_mode = ClusterMode(mode.lower())
            except ValueError:
                logger.warning(f"Unknown cluster mode '{mode}', using AUTO")

        _cluster_manager = ClusterManager(mode=cluster_mode, config_path=config_path)

    return _cluster_manager


def reset_cluster_manager():
    """Reset the singleton ClusterManager (useful for testing)."""
    global _cluster_manager
    _cluster_manager = None
    detect_local_hardware.cache_clear()


# =============================================================================
# Example Configuration
# =============================================================================

EXAMPLE_CONFIG = """
# Example cluster configuration for montage-ai
# Save as ~/.config/montage-ai/cluster.yaml
# Set MONTAGE_CLUSTER_CONFIG=/path/to/cluster.yaml to use

nodes:
  # GPU rendering node
  - name: gpu-server
    cpu_cores: 8
    memory_gb: 64.0
    architecture: x86_64
    gpu_type: nvidia-nvenc
    gpu_vram_gb: 24.0
    gpu_name: "RTX 4090"
    address: "192.168.1.100"

  # High-CPU node for scene detection
  - name: compute-node
    cpu_cores: 32
    memory_gb: 128.0
    architecture: x86_64
    address: "192.168.1.101"

  # ARM node for ML inference
  - name: arm-inference
    cpu_cores: 8
    memory_gb: 16.0
    architecture: arm64
    address: "192.168.1.102"
"""


if __name__ == "__main__":
    # Test local hardware detection
    print("Testing hardware detection...")
    local = detect_local_hardware()
    print(f"Local: {local.name}, {local.cpu_cores} cores, {local.memory_gb:.1f}GB")
    print(f"GPU: {local.gpu_type.value}, VRAM: {local.gpu_vram_gb:.1f}GB")

    print("\n" + "-" * 40 + "\n")

    # Test cluster manager
    cluster = get_cluster_manager()
    cluster.print_cluster_summary()

    # Test task distribution
    print("Distribution for 10 videos (Scene Detection):")
    dist = cluster.get_parallel_distribution(
        TaskType.CPU_SCENE_DETECTION,
        num_items=10,
        resolution=(1920, 1080)
    )
    for node, items in dist.items():
        print(f"  {node}: {len(items)} items")
