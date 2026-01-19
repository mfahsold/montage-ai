"""
Resource Manager - Unified Compute Resource Detection and Allocation

Consolidates all compute resource detection into a single module with priority:
    1. Cloud GPU (cgpu) - highest priority for heavy compute
    2. Local GPU (NVENC/VAAPI/QSV) - second priority for encoding/rendering
    3. Local CPU - fallback, can be distributed via K3s

Usage:
    from montage_ai.resource_manager import get_resource_manager

    rm = get_resource_manager()

    # Get best available encoder
    encoder = rm.get_encoder()  # Returns FFmpegConfig with GPU if available

    # Check resource status
    rm.print_status()

    # Get compute backend for heavy tasks (upscaling, transcription)
    backend = rm.get_compute_backend()
    # Returns: "cgpu", "local_gpu", "cpu"

DRY Principle: All resource detection logic consolidated here.
"""

import os
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

from .logger import logger
from .config import get_effective_cpu_count, get_settings

# Re-use ClusterManager for advanced resource detection
try:
    from .cluster import get_cluster_manager, ClusterMode
    CLUSTER_AVAILABLE = True
except ImportError:
    CLUSTER_AVAILABLE = False

if TYPE_CHECKING:
    from .ffmpeg_config import FFmpegConfig



# =============================================================================
# Enums and Types
# =============================================================================

class ComputeBackend(Enum):
    """Available compute backends in priority order."""
    CGPU = "cgpu"           # Cloud GPU via cgpu (Colab)
    LOCAL_GPU = "local_gpu" # Local GPU (NVENC, VAAPI, etc.)
    CPU = "cpu"             # Local CPU


class TaskType(Enum):
    """Types of compute tasks with different resource requirements."""
    ENCODING = "encoding"           # Video encoding (benefits from GPU)
    UPSCALING = "upscaling"         # AI upscaling (requires GPU)
    TRANSCRIPTION = "transcription" # Whisper (benefits from GPU)
    ANALYSIS = "analysis"           # Scene/beat analysis (CPU OK)
    VOICE_ISOLATION = "voice_iso"   # Demucs (benefits from GPU)


# =============================================================================
# Resource Status
# =============================================================================

@dataclass
class ResourceStatus:
    """Current status of available compute resources."""
    # Cloud GPU (cgpu)
    cgpu_available: bool = False
    cgpu_gpu_info: Optional[str] = None
    cgpu_serve_available: bool = False  # For LLM tasks

    # Local GPU
    local_gpu_available: bool = False
    local_gpu_type: Optional[str] = None  # nvenc, nvmpi, vaapi, qsv, videotoolbox
    local_gpu_info: Optional[str] = None

    # CPU info
    cpu_cores: int = 0

    # K3s cluster info
    k3s_available: bool = False
    k3s_nodes: List[str] = field(default_factory=list)
    k3s_ready_nodes: int = 0

    @property
    def best_backend(self) -> ComputeBackend:
        """Get best available backend for heavy compute."""
        if self.cgpu_available:
            return ComputeBackend.CGPU
        if self.local_gpu_available:
            return ComputeBackend.LOCAL_GPU
        return ComputeBackend.CPU

    @property
    def has_any_gpu(self) -> bool:
        """Check if any GPU is available."""
        return self.cgpu_available or self.local_gpu_available


# =============================================================================
# Resource Manager
# =============================================================================

class ResourceManager:
    """
    Unified resource detection and allocation manager.

    Automatically detects available compute resources and provides
    optimal configuration for different task types.
    """

    def __init__(self, refresh: bool = True):
        """
        Initialize resource manager.

        Args:
            refresh: If True, immediately detect available resources
        """
        self._status: Optional[ResourceStatus] = None
        self._ffmpeg_config_cache: Dict[str, "FFmpegConfig"] = {}

        if refresh:
            self.refresh()

    def refresh(self) -> ResourceStatus:
        """
        Refresh resource detection.

        Probes all available compute resources and updates status.

        Returns:
            Current ResourceStatus
        """
        status = ResourceStatus()

        # 1. Use ClusterManager for unified detection if available
        if CLUSTER_AVAILABLE:
            try:
                cluster = get_cluster_manager()
                
                # Check for cgpu in cluster
                cgpu_node = cluster.get_node("cloud-cgpu")
                if cgpu_node:
                    status.cgpu_available = True
                    status.cgpu_gpu_info = cgpu_node.gpu_name
                else:
                    # Fallback to direct detection
                    status.cgpu_available, status.cgpu_gpu_info = self._detect_cgpu()
                
                # Detect cluster properties via kubectl (preserves legacy behavior)
                status.k3s_available, status.k3s_nodes, status.k3s_ready_nodes = \
                    self._detect_k3s()
                
                # Detect local GPU from cluster manager's local node
                local = cluster.local_node
                if local and local.is_gpu_node:
                    status.local_gpu_available = True
                    status.local_gpu_type = local.gpu_type.value
                    status.local_gpu_info = local.gpu_name
                else:
                    # Fallback
                    status.local_gpu_available, status.local_gpu_type, status.local_gpu_info = \
                        self._detect_local_gpu()
            except Exception as e:
                logger.debug(f"ClusterManager-based detection failed: {e}")
                # Fallback to manual detection
                status.cgpu_available, status.cgpu_gpu_info = self._detect_cgpu()
                status.local_gpu_available, status.local_gpu_type, status.local_gpu_info = \
                    self._detect_local_gpu()
                status.k3s_available, status.k3s_nodes, status.k3s_ready_nodes = \
                    self._detect_k3s()
        else:
            # Traditional detection
            status.cgpu_available, status.cgpu_gpu_info = self._detect_cgpu()
            status.local_gpu_available, status.local_gpu_type, status.local_gpu_info = \
                self._detect_local_gpu()
            status.k3s_available, status.k3s_nodes, status.k3s_ready_nodes = \
                self._detect_k3s()

        status.cgpu_serve_available = self._detect_cgpu_serve()
        status.cpu_cores = get_effective_cpu_count()

        self._status = status
        return status

    @property
    def status(self) -> ResourceStatus:
        """Get current resource status (cached)."""
        if self._status is None:
            self.refresh()
        return self._status

    # =========================================================================
    # Detection Methods
    # =========================================================================

    def _detect_cgpu(self) -> Tuple[bool, Optional[str]]:
        """
        Detect cgpu availability.

        Returns:
            (available, gpu_info) tuple
        """
        try:
            from .cgpu_utils import is_cgpu_available, check_cgpu_gpu
        except ImportError:
            return False, None
        except Exception as e:
            logger.debug(f"cgpu detection error: {e}")
            return False, None

        if not is_cgpu_available():
            return False, None

        gpu_ok, gpu_info = check_cgpu_gpu()
        if gpu_ok:
            return True, gpu_info
        return True, "GPU available (details unknown)"

    def _detect_cgpu_serve(self) -> bool:
        """Check if cgpu serve (LLM endpoint) is available."""
        llm = get_settings().llm
        if not llm.cgpu_enabled:
            return False

        host = llm.cgpu_host
        port = llm.cgpu_port

        try:
            import requests
            response = requests.get(
                f"http://{host}:{port}/v1/models",
                timeout=5
            )
            return response.status_code == 200
        except ImportError:
            logger.debug("requests library not available")
            return False
        except requests.RequestException as e:
            logger.debug(f"cgpu serve connection failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected cgpu serve error: {e}")
            return False

    def _detect_local_gpu(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect local GPU encoders.

        Returns:
            (available, type, info) tuple
        """
        from .core import hardware
        from .config_timeouts import TimeoutConfig
        
        hw_config = hardware.get_best_hwaccel()

        if not hw_config.is_gpu:
            return False, None, None

        info = f"{hw_config.type.upper()} ({hw_config.encoder})"

        # For NVIDIA, try to get device name
        if hw_config.type in ("nvenc", "nvmpi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=TimeoutConfig.probe_quick()
                )
                if result.returncode == 0 and result.stdout.strip():
                    info = f"NVENC: {result.stdout.strip()}"
            except FileNotFoundError:
                logger.debug("nvidia-smi not found")
            except subprocess.TimeoutExpired:
                logger.debug("nvidia-smi timeout")
            except Exception as e:
                logger.debug(f"nvidia-smi error: {e}")

            if hw_config.type == "nvmpi":
                try:
                    tegra_release = "/etc/nv_tegra_release"
                    if os.path.exists(tegra_release):
                        release_info = Path(tegra_release).read_text(encoding="utf-8").strip()
                        info = f"Jetson: {release_info}"
                except (OSError, IOError) as e:
                    logger.debug(f"Jetson release file read error: {e}")

        # For VAAPI, try to get device info
        elif hw_config.type == "vaapi":
            try:
                result = subprocess.run(
                    ["vainfo"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Driver version' in line:
                            info = f"VAAPI: {line.split(':')[-1].strip()}"
                            break
            except FileNotFoundError:
                logger.debug("vainfo not found")
            except subprocess.TimeoutExpired:
                logger.debug("vainfo timeout")
            except Exception as e:
                logger.debug(f"vainfo error: {e}")

        return True, hw_config.type, info

    def _detect_k3s(self) -> Tuple[bool, List[str], int]:
        """
        Detect K3s cluster availability.

        Returns:
            (available, node_names, ready_count) tuple
        """
        from .config_timeouts import TimeoutConfig
        
        if not shutil.which("kubectl"):
            return False, [], 0

        try:
            result = subprocess.run(
                ["kubectl", "get", "nodes", "-o",
                 "custom-columns=NAME:.metadata.name,STATUS:.status.conditions[-1].type"],
                capture_output=True,
                text=True,
                timeout=TimeoutConfig.probe_kubernetes()
            )

            if result.returncode != 0:
                return False, [], 0

            nodes = []
            ready_count = 0

            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 2:
                    node_name = parts[0]
                    status = parts[1]
                    nodes.append(node_name)
                    if status == "Ready":
                        ready_count += 1

            return len(nodes) > 0, nodes, ready_count

        except (subprocess.TimeoutExpired, Exception):
            return False, [], 0

    # =========================================================================
    # Resource Allocation
    # =========================================================================

    def get_compute_backend(self, task: TaskType = TaskType.ENCODING) -> ComputeBackend:
        """
        Get best compute backend for a task type.

        Args:
            task: Type of task to run

        Returns:
            Best available ComputeBackend
        """
        status = self.status

        # GPU-required tasks
        if task in (TaskType.UPSCALING, TaskType.VOICE_ISOLATION):
            if status.cgpu_available:
                return ComputeBackend.CGPU
            # These tasks REQUIRE GPU, but we return local_gpu or cpu
            # and let the caller handle the fallback/warning
            if status.local_gpu_available:
                return ComputeBackend.LOCAL_GPU
            return ComputeBackend.CPU

        # GPU-beneficial tasks
        if task == TaskType.ENCODING:
            if status.local_gpu_available:
                return ComputeBackend.LOCAL_GPU
            if status.cgpu_available:
                # cgpu is overkill for encoding, but usable
                return ComputeBackend.CGPU
            return ComputeBackend.CPU

        if task == TaskType.TRANSCRIPTION:
            if status.cgpu_available:
                return ComputeBackend.CGPU
            if status.local_gpu_available:
                return ComputeBackend.LOCAL_GPU
            return ComputeBackend.CPU

        # CPU-OK tasks
        return ComputeBackend.CPU

    def get_encoder(self,
                    prefer_gpu: bool = True,
                    cache_key: str = "default") -> "FFmpegConfig":
        """
        Get FFmpeg encoder configuration with optimal settings.

        Uses caching for efficiency. GPU encoder is selected based on
        current resource status.

        Args:
            prefer_gpu: Whether to prefer GPU encoding
            cache_key: Key for caching (use different keys for different profiles)

        Returns:
            FFmpegConfig with optimal settings
        """
        if cache_key in self._ffmpeg_config_cache:
            return self._ffmpeg_config_cache[cache_key]

        status = self.status

        # Respect FFMPEG_HWACCEL env var if explicitly set to "none"
        env_hwaccel = os.environ.get("FFMPEG_HWACCEL", "").lower()
        if env_hwaccel in ("none", "cpu"):
            hwaccel = "none"
        elif prefer_gpu and status.local_gpu_available:
            hwaccel = status.local_gpu_type
        elif prefer_gpu:
            hwaccel = "auto"  # Will auto-detect
        else:
            hwaccel = "none"

        from .ffmpeg_config import FFmpegConfig
        config = FFmpegConfig(hwaccel=hwaccel)
        self._ffmpeg_config_cache[cache_key] = config

        return config

    def get_optimal_threads(self) -> int:
        """
        Get optimal thread count for CPU-bound tasks.

        Reserves some cores for system responsiveness.

        Returns:
            Optimal thread count
        """
        cores = self.status.cpu_cores

        # Reserve 1-2 cores for system
        if cores <= 2:
            return cores
        elif cores <= 4:
            return cores - 1
        else:
            return cores - 2

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def print_status(self):
        """Print detailed resource status."""
        status = self.status

        logger.info("\n" + "=" * 60)
        logger.info("  RESOURCE MANAGER - Compute Resource Status")
        logger.info("=" * 60)

        # Cloud GPU
        logger.info("\n  Cloud GPU (cgpu):")
        if status.cgpu_available:
            logger.info(f"    Status: AVAILABLE")
            if status.cgpu_gpu_info:
                logger.info(f"    GPU: {status.cgpu_gpu_info}")
        else:
            logger.info(f"    Status: Not available")
            if os.environ.get("CGPU_GPU_ENABLED", "false").lower() != "true":
                logger.info(f"    Hint: Set CGPU_GPU_ENABLED=true to enable")

        if status.cgpu_serve_available:
            logger.info(f"    LLM Endpoint: Running")

        # Local GPU
        logger.info("\n  Local GPU:")
        if status.local_gpu_available:
            logger.info(f"    Status: AVAILABLE")
            logger.info(f"    Type: {status.local_gpu_type.upper() if status.local_gpu_type else 'Unknown'}")
            if status.local_gpu_info:
                logger.info(f"    Info: {status.local_gpu_info}")
        else:
            logger.info(f"    Status: Not available")
            logger.info(f"    Hint: Set FFMPEG_HWACCEL=auto to auto-detect")

        # CPU
        logger.info("\n  CPU:")
        logger.info(f"    Cores: {status.cpu_cores}")
        logger.info(f"    Optimal Threads: {self.get_optimal_threads()}")

        # K3s Cluster
        logger.info("\n  K3s Cluster:")
        if status.k3s_available:
            logger.info(f"    Status: AVAILABLE")
            logger.info(f"    Nodes: {len(status.k3s_nodes)} total, {status.k3s_ready_nodes} ready")
            for node in status.k3s_nodes:
                logger.info(f"      - {node}")
        else:
            logger.info(f"    Status: Not available")

        # Summary
        logger.info("\n  Summary:")
        logger.info(f"    Best Backend: {status.best_backend.value.upper()}")
        logger.info(f"    GPU Available: {'Yes' if status.has_any_gpu else 'No'}")
        logger.info(f"    Cluster Ready: {'Yes' if status.k3s_ready_nodes > 0 else 'No'}")

        logger.info("\n" + "=" * 60)

    def get_status_line(self) -> str:
        """
        Get single-line status summary for logging.

        Returns:
            Status line like "cgpu:T4 | gpu:nvenc | k3s:5/6"
        """
        status = self.status
        parts = []

        if status.cgpu_available:
            gpu_short = status.cgpu_gpu_info.split(',')[0] if status.cgpu_gpu_info else "OK"
            parts.append(f"cgpu:{gpu_short}")

        if status.local_gpu_available:
            parts.append(f"gpu:{status.local_gpu_type}")

        if status.k3s_available:
            parts.append(f"k3s:{status.k3s_ready_nodes}/{len(status.k3s_nodes)}")

        parts.append(f"cpu:{status.cpu_cores}")

        return " | ".join(parts)


# =============================================================================
# Singleton Access
# =============================================================================

_resource_manager: Optional[ResourceManager] = None


def get_resource_manager(refresh: bool = False) -> ResourceManager:
    """
    Get singleton ResourceManager instance.

    Args:
        refresh: If True, refresh resource detection

    Returns:
        ResourceManager instance
    """
    global _resource_manager

    if _resource_manager is None:
        _resource_manager = ResourceManager(refresh=True)
    elif refresh:
        _resource_manager.refresh()

    return _resource_manager


def print_resource_status():
    """Print resource status (convenience function)."""
    get_resource_manager().print_status()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ComputeBackend",
    "TaskType",
    # Data classes
    "ResourceStatus",
    # Main class
    "ResourceManager",
    # Functions
    "get_resource_manager",
    "print_resource_status",
]


# =============================================================================
# CLI Self-Test
# =============================================================================

if __name__ == "__main__":
    print_resource_status()
