"""
Centralized Configuration for Montage AI

Single Source of Truth for all paths, feature flags, and settings.
Uses Pydantic for validation and environment variable loading.

Usage:
    from montage_ai.config import settings

    input_dir = settings.paths.input_dir
    if settings.features.upscale:
        ...
"""

import os
import multiprocessing
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Tuple, List
from dataclasses import dataclass, field

from montage_ai.config_parser import ConfigParser


# =============================================================================
# Path Configuration
# =============================================================================
@dataclass
class PathConfig:
    """All filesystem paths used by Montage AI."""

    input_dir: Path = field(default_factory=lambda: Path(os.environ.get("INPUT_DIR", "/data/input")))
    music_dir: Path = field(default_factory=lambda: Path(os.environ.get("MUSIC_DIR", "/data/music")))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", "/data/output")))
    assets_dir: Path = field(default_factory=lambda: Path(os.environ.get("ASSETS_DIR", "/data/assets")))
    # OPTIMIZATION: Use RAM disk on Linux for 30-40% faster temp file I/O
    # /dev/shm is tmpfs mounted in RAM, falls back to /tmp if not available
    temp_dir: Path = field(default_factory=lambda: Path(os.environ.get("TEMP_DIR") or ("/dev/shm" if Path("/dev/shm").exists() and Path("/dev/shm").is_dir() else "/tmp")))
    lut_dir: Path = field(default_factory=lambda: Path(os.environ.get("LUT_DIR", "/data/luts")))
    session_dir: Path = field(default_factory=lambda: Path(os.environ.get("SESSION_DIR", "/tmp/montage_sessions")))
    transcript_dir: Path = field(default_factory=lambda: Path(os.environ.get("TRANSCRIPT_DIR", "/tmp/montage_transcript")))
    shorts_dir: Path = field(default_factory=lambda: Path(os.environ.get("SHORTS_DIR", "/tmp/montage_shorts")))
    metadata_cache_dir: Path = field(
        default_factory=lambda: (
            Path(os.environ.get("METADATA_CACHE_DIR"))
            if os.environ.get("METADATA_CACHE_DIR")
            else (
                # Prefer XDG cache (user-writable); fallback to ~/.cache
                Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                / "montage_ai"
                / "metadata"
            )
        )
    )
    tension_metadata_dir: Path = field(
        default_factory=lambda: (
            Path(os.environ.get("TENSION_METADATA_DIR"))
            if os.environ.get("TENSION_METADATA_DIR")
            else (
                (
                    Path(os.environ.get("METADATA_CACHE_DIR")) / "tension"
                    if os.environ.get("METADATA_CACHE_DIR")
                    else (
                        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                        / "montage_ai"
                        / "tension"
                    )
                )
            )
        )
    )
    style_preset_path: Optional[Path] = field(default_factory=lambda: Path(os.environ.get("STYLE_PRESET_PATH")) if os.environ.get("STYLE_PRESET_PATH") else None)
    style_preset_dir: Optional[Path] = field(default_factory=lambda: Path(os.environ.get("STYLE_PRESET_DIR") or os.environ.get("STYLE_TEMPLATES_DIR")) if (os.environ.get("STYLE_PRESET_DIR") or os.environ.get("STYLE_TEMPLATES_DIR")) else None)

    def ensure_directories(self) -> None:
        """Create all directories if they don't exist."""
        for path in [self.input_dir, self.music_dir, self.output_dir, self.assets_dir, self.metadata_cache_dir, self.tension_metadata_dir, self.session_dir, self.transcript_dir, self.shorts_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_log_path(self, job_id: str) -> Path:
        """Get log file path for a job."""
        return self.output_dir / f"render_{job_id}.log"


def _read_cgroup_cpu_limit() -> Optional[int]:
    """Best-effort CPU limit from cgroups (returns None when unlimited/unknown)."""
    # cgroup v2
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.exists():
            parts = cpu_max.read_text().strip().split()
            if len(parts) >= 2 and parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                if quota > 0 and period > 0:
                    return max(1, int(quota / period))
    except (OSError, ValueError):
        pass

    # cgroup v1
    try:
        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_path.exists() and period_path.exists():
            quota = int(quota_path.read_text().strip())
            period = int(period_path.read_text().strip())
            if quota > 0 and period > 0:
                return max(1, int(quota / period))
    except (OSError, ValueError):
        pass

    return None


def _parse_int(raw: Optional[str], default: int) -> int:
    """Parse an int from env strings with fallback."""
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def get_cluster_shard_index() -> int:
    """Cluster shard index with Kubernetes Indexed Job fallback."""
    raw = os.environ.get("CLUSTER_SHARD_INDEX")
    if raw is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX", "0")
    return _parse_int(raw, 0)


def get_cluster_shard_count() -> int:
    """Total shard count for cluster sharding."""
    return _parse_int(os.environ.get("CLUSTER_SHARD_COUNT", "1"), 1)


def get_effective_cpu_count() -> int:
    """Best-effort CPU count (respects cgroup limits + affinity when available)."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = multiprocessing.cpu_count() or 1

    cgroup_limit = _read_cgroup_cpu_limit()
    if cgroup_limit is None:
        return max(1, affinity)
    return max(1, min(affinity, cgroup_limit))


# =============================================================================
# Feature Flags
# =============================================================================
@dataclass
class FeatureConfig:
    """Feature toggles for enhancement pipeline."""

    stabilize: bool = field(default_factory=ConfigParser.make_bool_parser("STABILIZE", False))
    upscale: bool = field(default_factory=ConfigParser.make_bool_parser("UPSCALE", False))
    enhance: bool = field(default_factory=ConfigParser.make_bool_parser("ENHANCE", True))
    preserve_aspect: bool = field(default_factory=ConfigParser.make_bool_parser("PRESERVE_ASPECT", False))
    export_timeline: bool = field(default_factory=ConfigParser.make_bool_parser("EXPORT_TIMELINE", False))
    generate_proxies: bool = field(default_factory=ConfigParser.make_bool_parser("GENERATE_PROXIES", False))
    llm_clip_selection: bool = field(default_factory=ConfigParser.make_bool_parser("LLM_CLIP_SELECTION", True))
    deep_analysis: bool = field(default_factory=ConfigParser.make_bool_parser("DEEP_ANALYSIS", False))
    verbose: bool = field(default_factory=ConfigParser.make_bool_parser("VERBOSE", True))
    enable_ai_filter: bool = field(default_factory=ConfigParser.make_bool_parser("ENABLE_AI_FILTER", False))
    # Phase 4: Agentic Creative Loop - LLM evaluates and refines cuts iteratively
    creative_loop: bool = field(default_factory=ConfigParser.make_bool_parser("CREATIVE_LOOP", False))
    creative_loop_max_iterations: int = field(default_factory=ConfigParser.make_int_parser("CREATIVE_LOOP_MAX_ITERATIONS", 3))

    # Episodic memory for analysis caching (experimental)
    episodic_memory: bool = field(default_factory=ConfigParser.make_bool_parser("EPISODIC_MEMORY", False))

    # Storytelling Engine (Phase 1 scaffolding)
    story_engine: bool = field(default_factory=ConfigParser.make_bool_parser("ENABLE_STORY_ENGINE", False))
    strict_cloud_compute: bool = field(default_factory=ConfigParser.make_bool_parser("STRICT_CLOUD_COMPUTE", False))

    # Shorts Workflow (Vertical Video + Smart Reframing)
    shorts_mode: bool = field(default_factory=ConfigParser.make_bool_parser("SHORTS_MODE", False))
    reframe_mode: str = field(default_factory=ConfigParser.make_str_parser("REFRAME_MODE", "auto"))  # auto, speaker, center, custom

    # 2025 P0/P1: Burn-in captions and voice isolation
    captions: bool = field(default_factory=ConfigParser.make_bool_parser("CAPTIONS", False))
    captions_style: str = field(default_factory=ConfigParser.make_str_parser("CAPTIONS_STYLE", "tiktok"))  # tiktok, minimal, bold, karaoke
    transcription_model: str = field(default_factory=ConfigParser.make_str_parser("TRANSCRIPTION_MODEL", "medium"))
    
    # Audio Polish: Clean Audio = Voice Isolation + Denoise (single toggle)
    # CLEAN_AUDIO is the new consolidated toggle, VOICE_ISOLATION is legacy
    voice_isolation: bool = field(default_factory=lambda: (
        ConfigParser.parse_bool("CLEAN_AUDIO", False) or ConfigParser.parse_bool("VOICE_ISOLATION", False)
    ))
    voice_isolation_model: str = field(default_factory=ConfigParser.make_str_parser("VOICE_ISOLATION_MODEL", "htdemucs"))

    # Noise Reduction: DeepFilterNet for lightweight noise removal (faster than voice_isolation)
    # Use for podcasts, interviews, vlogs with background noise
    noise_reduction: bool = field(default_factory=ConfigParser.make_bool_parser("NOISE_REDUCTION", False))
    noise_reduction_strength: int = field(default_factory=ConfigParser.make_int_parser("NOISE_REDUCTION_STRENGTH", 100))

    # Video Enhancement (per-clip)
    denoise: bool = field(default_factory=ConfigParser.make_bool_parser("DENOISE", False))
    sharpen: bool = field(default_factory=ConfigParser.make_bool_parser("SHARPEN", False))

    # Film Grain Simulation: none, 35mm, 16mm, 8mm, digital
    film_grain: str = field(default_factory=ConfigParser.make_str_parser("FILM_GRAIN", "none"))

    # Dialogue Ducking: Auto-duck music during speech
    dialogue_duck: bool = field(default_factory=ConfigParser.make_bool_parser("DIALOGUE_DUCK", False))
    dialogue_duck_level: float = field(default_factory=ConfigParser.make_float_parser("DIALOGUE_DUCK_LEVEL", -12.0))

    # Performance: Low-resource hardware mode (longer timeouts, smaller batches, sequential processing)
    low_memory_mode: bool = field(default_factory=ConfigParser.make_bool_parser("LOW_MEMORY_MODE", False))

    # Cluster Mode: Distribute heavy processing across multiple Kubernetes nodes
    cluster_mode: bool = field(default_factory=ConfigParser.make_bool_parser("CLUSTER_MODE", False))
    cluster_parallelism: int = field(default_factory=ConfigParser.make_int_parser("CLUSTER_PARALLELISM", 4))

    # Color/levels normalization controls (can be disabled for clean footage)
    colorlevels: bool = field(default_factory=ConfigParser.make_bool_parser("COLORLEVELS", True))
    luma_normalize: bool = field(default_factory=ConfigParser.make_bool_parser("LUMA_NORMALIZE", True))


# =============================================================================
# Processing Configuration (High-Res Support)
# =============================================================================
@dataclass
class ProcessingSettings:
    """Processing settings with adaptive batch sizing for high-resolution media."""
    
    batch_size: int = field(default_factory=lambda: int(os.environ.get("BATCH_SIZE", "5")))
    max_input_resolution: int = field(default_factory=lambda: int(os.environ.get("MAX_INPUT_RESOLUTION", "8294400")))  # 4K default
    warn_threshold_resolution: int = 33177600  # 6K (6144x3160)
    
    def get_adaptive_batch_size_for_resolution(
        self, width: int, height: int, low_memory: bool = False, memory_gb: Optional[float] = None
    ) -> int:
        """Adaptive batch sizing based on input resolution and available memory.

        Resolution Brackets (base values):
        - 1080p (2MP): batch_size = 25
        - 4K (8MP): batch_size = 8
        - 6K (19MP): batch_size = 2
        - 8K+ (33MP): batch_size = 1 (warning issued)

        Memory Adjustments:
        - <4GB: Quarter batch size
        - <8GB: Half batch size
        - 8GB+: Full batch size

        Args:
            width: Video width in pixels
            height: Video height in pixels
            low_memory: Deprecated - use memory_gb instead
            memory_gb: Available memory in GB (auto-detected if None)
        """
        pixels = width * height

        # Determine base batch size from resolution
        # Mapping chosen to favor conservative batch sizes for high-res inputs
        # - 8K+ (>33MP): batch_size = 1 (warn)
        # - 6K (>=15MP, <33MP): batch_size = 1 (reduced)
        # - 4K (>=8MP, <15MP): batch_size = 2
        # - else: use configured base (default BATCH_SIZE env, typically 5)
        if pixels > 33_177_600:  # 8K+ (33MP+)
            from .logger import logger
            logger.warning(
                f"⚠️ Very high resolution: {width}x{height} ({pixels/1e6:.1f}MP). "
                "Using batch_size=1. Consider proxy workflow for better performance."
            )
            base = 1
        elif pixels >= 15_000_000:  # 6K (15-33MP)
            from .logger import logger
            logger.warning(
                f"⚠️ High resolution detected: {width}x{height} ({pixels/1e6:.1f}MP). "
                "Using reduced batch size. Consider proxy workflow for better performance."
            )
            base = 1
        elif pixels >= 8_000_000:  # 4K (8-15MP)
            base = 2
        else:
            # Use configured batch_size (env BATCH_SIZE) for typical HD/SD inputs
            base = self.batch_size if self.batch_size > 0 else 5

        # Default memory to 8GB for deterministic behavior unless explicitly provided
        if memory_gb is None:
            memory_gb = 8.0  # Assume 8GB by default for consistent batch sizing in tests/environments

        # Apply memory-based adjustment
        # - low_memory flag halves the batch size (but doesn't reduce 4K/6K below intended bracket)
        # - only apply aggressive reductions for very low memory (<4GB)
        if low_memory:
            # Reduce only when base > 2 to avoid dropping 4K/6K below safe minimum
            return max(1, base // 2) if base > 2 else base
        if memory_gb < 4:
            # Very constrained environment: reduce significantly
            return max(1, base // 4)

        return base
    
    def validate_resolution(self, width: int, height: int) -> bool:
        """Validate resolution is within safe limits."""
        from .logger import logger
        pixels = width * height
        
        if pixels > self.warn_threshold_resolution:
            logger.warning(
                f"⚠️ High resolution detected: {width}x{height} ({pixels/1e6:.1f}MP). "
                "Consider using proxy workflow for better performance."
            )
        
        if pixels > self.max_input_resolution * 4:  # 8K
            logger.error(
                f"❌ Resolution {width}x{height} exceeds safe limits. "
                "Use proxy workflow or reduce resolution."
            )
            return False
        
        return True


# =============================================================================
# GPU Configuration
# =============================================================================
@dataclass
class GPUConfig:
    """GPU and hardware acceleration settings."""

    use_gpu: str = field(default_factory=lambda: os.environ.get("USE_GPU", "auto").lower())
    output_codec: str = field(default_factory=lambda: os.environ.get("OUTPUT_CODEC", "h264"))
    ffmpeg_hwaccel: str = field(default_factory=lambda: os.environ.get("FFMPEG_HWACCEL", "auto"))
    use_ffmpeg_mcp: bool = field(default_factory=lambda: os.environ.get("USE_FFMPEG_MCP", "false").lower() == "true")
    ffmpeg_mcp_endpoint: str = field(default_factory=lambda: os.environ.get("FFMPEG_MCP_ENDPOINT", "http://ffmpeg-mcp.montage-ai.svc.cluster.local:8080"))
    # Force routing encoding to CGPU service when beneficial or explicitly requested
    force_cgpu_encoding: bool = field(default_factory=lambda: os.environ.get("FORCE_CGPU_ENCODING", "false").lower() == "true")


# =============================================================================
# LLM Configuration
# =============================================================================
@dataclass
class LLMConfig:
    """LLM backend configuration."""

    # OpenAI-compatible API (KubeAI, vLLM, LocalAI)
    openai_api_base: str = field(default_factory=lambda: os.environ.get("OPENAI_API_BASE", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "not-needed"))
    openai_model: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", ""))
    openai_vision_model: str = field(default_factory=lambda: os.environ.get("OPENAI_VISION_MODEL", ""))

    # Ollama (local fallback)
    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llava"))
    director_model: str = field(default_factory=lambda: os.environ.get("DIRECTOR_MODEL", "llama3.1:70b"))

    # Google AI
    google_api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    google_ai_model: str = field(default_factory=lambda: os.environ.get("GOOGLE_AI_MODEL", "gemini-2.0-flash"))
    google_ai_endpoint: str = field(default_factory=lambda: os.environ.get("GOOGLE_AI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models"))

    # cgpu (Colab GPU)
    cgpu_enabled: bool = field(default_factory=lambda: os.environ.get("CGPU_ENABLED", "false").lower() == "true")
    cgpu_gpu_enabled: bool = field(default_factory=lambda: os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true")
    cgpu_host: str = field(default_factory=lambda: os.environ.get("CGPU_HOST", "127.0.0.1"))
    cgpu_port: int = field(default_factory=lambda: int(os.environ.get("CGPU_PORT", "5021")))
    cgpu_model: str = field(default_factory=lambda: os.environ.get("CGPU_MODEL", "gemini-2.0-flash"))
    cgpu_output_dir: str = field(default_factory=lambda: os.environ.get("CGPU_OUTPUT_DIR", ""))
    cgpu_timeout: int = field(default_factory=lambda: int(os.environ.get("CGPU_TIMEOUT", "1200")))
    cgpu_max_concurrency: int = field(default_factory=lambda: int(os.environ.get("CGPU_MAX_CONCURRENCY", "1")))
    
    # General LLM settings
    timeout: int = field(default_factory=lambda: int(os.environ.get("LLM_TIMEOUT", "60")))

    @property
    def has_openai_backend(self) -> bool:
        """Check if OpenAI-compatible backend is configured."""
        return bool(self.openai_api_base and self.openai_model)

    @property
    def has_google_backend(self) -> bool:
        """Check if Google AI backend is configured."""
        return bool(self.google_api_key)


# =============================================================================
# Cloud Configuration (Pro / Monetization)
# =============================================================================
@dataclass
class CloudConfig:
    """
    Montage Cloud (Pro) configuration.
    
    Handles authentication and endpoints for the paid 'Pro' tier services
    (Cloud GPU offloading, hosted Web UI, etc.).
    """
    api_key: str = field(default_factory=lambda: os.environ.get("MONTAGE_CLOUD_API_KEY", ""))
    endpoint: str = field(default_factory=lambda: os.environ.get("MONTAGE_CLOUD_ENDPOINT", "https://api.montage.ai/v1"))
    enabled: bool = field(default_factory=lambda: os.environ.get("MONTAGE_CLOUD_ENABLED", "false").lower() == "true")
    
    # Billing / Usage tracking
    track_usage: bool = field(default_factory=lambda: os.environ.get("TRACK_USAGE", "true").lower() == "true")


@dataclass
class ClusterConfig:
    """
    Kubernetes cluster and registry configuration.
    
    Syncs with deploy/config.env values for distributed processing.
    """
    registry_host: str = field(default_factory=lambda: os.environ.get("REGISTRY_HOST", "192.168.1.12"))
    registry_port: str = field(default_factory=lambda: os.environ.get("REGISTRY_PORT", "30500"))
    image_name: str = field(default_factory=lambda: os.environ.get("IMAGE_NAME", "montage-ai"))
    image_tag: str = field(default_factory=lambda: os.environ.get("IMAGE_TAG", "latest-amd64"))
    namespace: str = field(default_factory=lambda: os.environ.get("CLUSTER_NAMESPACE", "montage-ai"))
    image_pull_secret: Optional[str] = field(default_factory=lambda: os.environ.get("IMAGE_PULL_SECRET"))
    pvc_name: str = field(default_factory=lambda: os.environ.get("PVC_NAME", "montage-ai-data"))

    # Resource Tiers (Canonical Fluxibri Tiers)
    tiers: dict = field(default_factory=lambda: {
        "minimal": {"requests": {"cpu": "10m", "memory": "32Mi"}, "limits": {"cpu": "100m", "memory": "128Mi"}},
        "small": {"requests": {"cpu": "100m", "memory": "256Mi"}, "limits": {"cpu": "500m", "memory": "1Gi"}},
        "medium": {"requests": {"cpu": "500m", "memory": "1Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}},
        "large": {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"cpu": "8", "memory": "16Gi"}},
        "gpu": {"requests": {"cpu": "1", "memory": "4Gi"}, "limits": {"cpu": "16", "memory": "32Gi"}}
    })

    @property
    def image_full(self) -> str:
        """Full remote image path for Kubernetes."""
        host = self.registry_host
        port = self.registry_port
        name = self.image_name
        tag = self.image_tag
        
        # Build image path
        if port:
            return f"{host}:{port}/{name}:{tag}"
        return f"{host}/{name}:{tag}"


# =============================================================================
# Motion Analysis & Performance Tuning
# =============================================================================
@dataclass
class MotionAnalysisConfig:
    """Optical flow and motion detection parameters for performance tuning."""
    
    # Optical flow (Farneback algorithm) parameters
    optical_flow_pyr_scale: float = field(default_factory=lambda: float(os.environ.get("OPTICAL_FLOW_PYR_SCALE", "0.5")))
    optical_flow_levels: int = field(default_factory=lambda: int(os.environ.get("OPTICAL_FLOW_LEVELS", "3")))
    optical_flow_winsize: int = field(default_factory=lambda: int(os.environ.get("OPTICAL_FLOW_WINSIZE", "15")))
    optical_flow_iterations: int = field(default_factory=lambda: int(os.environ.get("OPTICAL_FLOW_ITERATIONS", "3")))
    optical_flow_poly_n: int = field(default_factory=lambda: int(os.environ.get("OPTICAL_FLOW_POLY_N", "5")))
    optical_flow_poly_sigma: float = field(default_factory=lambda: float(os.environ.get("OPTICAL_FLOW_POLY_SIGMA", "1.2")))
    
    # Motion magnitude normalization (typical range 0-10)
    motion_magnitude_scale: float = field(default_factory=lambda: float(os.environ.get("MOTION_MAGNITUDE_SCALE", "10.0")))
    
    # Optical flow direction threshold (determines if motion is static/left/right)
    motion_direction_threshold: float = field(default_factory=lambda: float(os.environ.get("MOTION_DIRECTION_THRESHOLD", "0.5")))
    
    # Blur/focus detection: Laplacian variance threshold
    # Higher = stricter quality threshold; typical range sharp ~1000, blurry ~100
    blur_detection_variance_threshold: float = field(default_factory=lambda: float(os.environ.get("BLUR_DETECTION_VARIANCE_THRESHOLD", "1000.0")))
    
    # Histogram bins for color matching (higher = more precise but slower)
    histogram_bins: int = field(default_factory=lambda: int(os.environ.get("HISTOGRAM_BINS", "64")))
    
    # Motion sampling strategy: "full" (all frames) or "adaptive" (skip low-motion scenes)
    motion_sampling_mode: str = field(default_factory=lambda: os.environ.get("MOTION_SAMPLING_MODE", "adaptive"))


# =============================================================================
# Analysis & Scene Detection Constants
# =============================================================================
@dataclass
class AnalysisConstants:
    """Centralized constants for scene analysis and feature extraction."""
    
    # Scene Detection (PySceneDetect)
    scene_min_length_frames: int = field(default_factory=ConfigParser.make_int_parser("SCENE_MIN_LENGTH_FRAMES", 15))  # ~0.5s @ 30fps
    
    # Optical Flow (Farneback algorithm)
    optical_flow_pyr_scale: float = field(default_factory=ConfigParser.make_float_parser("OF_PYR_SCALE", 0.5))
    optical_flow_levels: int = field(default_factory=ConfigParser.make_int_parser("OF_LEVELS", 3))
    optical_flow_winsize: int = field(default_factory=ConfigParser.make_int_parser("OF_WINSIZE", 15))
    
    # Downsampling & Proxy
    default_downsampling_height: int = field(default_factory=ConfigParser.make_int_parser("DEFAULT_DOWNSAMPLE_HEIGHT", 1080))
    proxy_generation_timeout_seconds: int = field(default_factory=ConfigParser.make_int_parser("PROXY_GEN_TIMEOUT", 3600))  # 1 hour
    
    # Preview Resolution (for web UI thumbnails)
    preview_height: int = field(default_factory=ConfigParser.make_int_parser("PREVIEW_HEIGHT", 360))


# =============================================================================
# Proxy & Analysis Configuration
# =============================================================================
@dataclass
class ProxyConfig:
    """Proxy video generation for analysis acceleration.
    
    For large/long videos, proxy mode generates a lightweight version
    for feature extraction and analysis, then applies results to full-res render.
    """
    
    # Enable proxy mode for analysis (auto-enabled for large videos)
    enable_proxy_analysis: bool = field(default_factory=lambda: os.environ.get("ENABLE_PROXY_ANALYSIS", "true").lower() == "true")
    
    # Proxy resolution (height, maintains aspect ratio)
    # 720p = Good balance; maintains features while 2-3x speedup
    # Options: 360, 540, 720, 1080
    proxy_height: int = field(default_factory=lambda: int(os.environ.get("PROXY_HEIGHT", "720")))
    
    # Auto-enable threshold: if video is larger than this duration, use proxy
    # (in seconds: 600s = 10 minutes)
    auto_proxy_duration_threshold: float = field(default_factory=lambda: float(os.environ.get("AUTO_PROXY_DURATION_THRESHOLD", "600.0")))
    
    # Auto-enable threshold: if video is larger than this pixel count, use proxy
    # (pixels: 1920*1080 = 2.07M; YouTube raw ~1920*1440 = 2.76M)
    auto_proxy_resolution_threshold: int = field(default_factory=lambda: int(os.environ.get("AUTO_PROXY_RESOLUTION_THRESHOLD", "2000000")))
    
    # Reuse proxy for multiple analysis passes (e.g. scene detection + metadata)
    cache_proxies: bool = field(default_factory=lambda: os.environ.get("CACHE_PROXIES", "true").lower() == "true")
    
    def should_use_proxy(self, duration_seconds: float, width: int, height: int) -> bool:
        """Determine if proxy should be used based on input characteristics."""
        if not self.enable_proxy_analysis:
            return False
        
        # Check duration threshold
        if duration_seconds > self.auto_proxy_duration_threshold:
            return True
        
        # Check resolution threshold
        pixel_count = width * height
        if pixel_count > self.auto_proxy_resolution_threshold:
            return True
        
        return False


# =============================================================================
# Encoding Configuration
# =============================================================================
@dataclass
class EncodingConfig:
    """FFmpeg and video encoding settings."""

    quality_profile: str = field(default_factory=lambda: os.environ.get("QUALITY_PROFILE", "standard").lower())
    codec: str = field(default_factory=lambda: os.environ.get("OUTPUT_CODEC", "libx264"))
    crf: int = field(default_factory=lambda: int(os.environ.get("FINAL_CRF", "18")))
    preset: str = field(default_factory=lambda: os.environ.get("FFMPEG_PRESET", "medium"))
    hwaccel: str = field(default_factory=lambda: os.environ.get("FFMPEG_HWACCEL", "auto"))
    pix_fmt: str = field(default_factory=lambda: os.environ.get("OUTPUT_PIX_FMT", "yuv420p"))
    profile: str = field(default_factory=lambda: os.environ.get("OUTPUT_PROFILE", "high"))
    level: str = field(default_factory=lambda: os.environ.get("OUTPUT_LEVEL", "4.1"))
    threads: int = field(default_factory=lambda: int(os.environ.get("FFMPEG_THREADS", "0")))
    audio_codec: str = field(default_factory=lambda: os.environ.get("OUTPUT_AUDIO_CODEC", "aac"))
    audio_bitrate: str = field(default_factory=lambda: os.environ.get("OUTPUT_AUDIO_BITRATE", "192k"))
    normalize_clips: bool = field(default_factory=lambda: os.environ.get("NORMALIZE_CLIPS", "true").lower() == "true")
    extract_reencode: bool = field(default_factory=lambda: os.environ.get("EXTRACT_REENCODE", "false").lower() == "true")

    def __post_init__(self) -> None:
        """Apply quality profile defaults when explicit env overrides are absent."""
        profile = (self.quality_profile or "standard").lower()
        preset_map = {
            "preview": "ultrafast",
            "standard": "medium",
            "high": "slow",
            "master": "slow",
        }
        crf_map = {
            "preview": 28,
            "standard": 18,
            "high": 17,
            "master": 16,
        }

        if "FFMPEG_PRESET" not in os.environ and profile in preset_map:
            self.preset = preset_map[profile]
        if "FINAL_CRF" not in os.environ and profile in crf_map:
            self.crf = crf_map[profile]

        if profile == "master":
            if "OUTPUT_CODEC" not in os.environ:
                self.codec = "libx265"
            if "OUTPUT_PIX_FMT" not in os.environ:
                self.pix_fmt = "yuv420p10le"
            if "OUTPUT_PROFILE" not in os.environ:
                self.profile = "main10"


# =============================================================================
# Export Configuration
# =============================================================================
@dataclass
class ExportConfig:
    """Timeline export and NLE integration settings."""
    
    resolution_width: int = field(default_factory=lambda: int(os.environ.get("EXPORT_WIDTH", "1080")))
    resolution_height: int = field(default_factory=lambda: int(os.environ.get("EXPORT_HEIGHT", "1920")))
    fps: float = field(default_factory=lambda: float(os.environ.get("EXPORT_FPS", "30.0")))
    project_name_template: str = field(default_factory=lambda: os.environ.get("EXPORT_PROJECT_NAME", "fluxibri_montage"))
    explicit_resolution_override: bool = field(init=False)

    def __post_init__(self) -> None:
        self.explicit_resolution_override = (
            "EXPORT_WIDTH" in os.environ or "EXPORT_HEIGHT" in os.environ
        )


# =============================================================================
# Upscaling Configuration
# =============================================================================
@dataclass
class UpscaleConfig:
    """Upscaling configuration for local and cloud pipelines."""

    model: str = field(default_factory=lambda: os.environ.get("UPSCALE_MODEL", "realesrgan-x4plus"))
    scale: int = field(default_factory=lambda: int(os.environ.get("UPSCALE_SCALE", "2")))
    frame_format: str = field(default_factory=lambda: os.environ.get("UPSCALE_FRAME_FORMAT", "jpg"))
    tile_size: int = field(default_factory=lambda: int(os.environ.get("UPSCALE_TILE_SIZE", "512")))
    crf: int = field(default_factory=lambda: int(os.environ.get("UPSCALE_CRF", os.environ.get("FINAL_CRF", "18"))))


# =============================================================================
# Processing Configuration
# =============================================================================
@dataclass
class ProcessingConfig:
    """Batch processing and performance settings."""

    batch_size: int = field(default_factory=ConfigParser.make_int_parser("BATCH_SIZE", 25))
    max_scene_workers: int = field(default_factory=lambda: ConfigParser.parse_int("MAX_SCENE_WORKERS", os.cpu_count() or 2))
    force_gc: bool = field(default_factory=ConfigParser.make_bool_parser("FORCE_GC", True))
    parallel_enhance: bool = field(default_factory=ConfigParser.make_bool_parser("PARALLEL_ENHANCE", True))
    skip_existing_outputs: bool = field(default_factory=ConfigParser.make_bool_parser("SKIP_EXISTING_OUTPUTS", True))
    force_reprocess: bool = field(default_factory=ConfigParser.make_bool_parser("FORCE_REPROCESS", False))
    min_output_bytes: int = field(default_factory=ConfigParser.make_int_parser("MIN_OUTPUT_BYTES", 1024))
    job_id_strategy: str = field(default_factory=lambda: ConfigParser.parse_str("JOB_ID_STRATEGY", "timestamp").lower())
    max_parallel_jobs: int = field(default_factory=lambda: ConfigParser.parse_int("MAX_PARALLEL_JOBS", max(get_effective_cpu_count() - 1, get_effective_cpu_count())))
    max_concurrent_jobs: int = field(default_factory=ConfigParser.make_int_parser("MAX_CONCURRENT_JOBS", max(4, get_effective_cpu_count() - 1)))
    max_scene_reuse: int = field(default_factory=ConfigParser.make_int_parser("MAX_SCENE_REUSE", 3))
    cluster_shard_index: int = field(default_factory=get_cluster_shard_index)
    cluster_shard_count: int = field(default_factory=get_cluster_shard_count)

    # OPTIMIZATION: Configurable AI scene analysis limit (0 = unlimited)
    # Was hardcoded to 20 scenes - now can be scaled up for better quality
    max_ai_analyze_scenes: int = field(default_factory=ConfigParser.make_int_parser("MAX_AI_ANALYZE_SCENES", 50))

    # Adaptive settings for low-resource hardware
    clip_prefetch_count: int = field(default_factory=ConfigParser.make_int_parser("CLIP_PREFETCH_COUNT", 3))
    analysis_timeout: int = field(default_factory=ConfigParser.make_int_parser("ANALYSIS_TIMEOUT", 120))  # seconds
    ffprobe_timeout: int = field(default_factory=ConfigParser.make_int_parser("FFPROBE_TIMEOUT", 10))
    ffmpeg_short_timeout: int = field(default_factory=ConfigParser.make_int_parser("FFMPEG_SHORT_TIMEOUT", 30))
    ffmpeg_timeout: int = field(default_factory=ConfigParser.make_int_parser("FFMPEG_TIMEOUT", 120))
    ffmpeg_long_timeout: int = field(default_factory=ConfigParser.make_int_parser("FFMPEG_LONG_TIMEOUT", 600))
    render_timeout: int = field(default_factory=ConfigParser.make_int_parser("RENDER_TIMEOUT", 3600))  # 1 hour default

    def get_adaptive_batch_size(self, low_memory: bool = False, memory_gb: Optional[float] = None) -> int:
        """Get batch size adjusted for memory constraints.

        Args:
            low_memory: Force low memory mode (deprecated - use memory_gb)
            memory_gb: Available memory in GB (auto-detected if None)

        Returns:
            Adjusted batch size based on available memory
        """
        # Check LOW_MEMORY_MODE environment variable
        if low_memory or ConfigParser.parse_bool("LOW_MEMORY_MODE", False):
            return max(1, self.batch_size // 4)

        # Auto-detect memory if not provided
        if memory_gb is None:
            try:
                import psutil
                memory_gb = psutil.virtual_memory().available / (1024 ** 3)
            except ImportError:
                return self.batch_size  # Can't detect, use default

        # Memory-based adjustment
        if memory_gb < 4:
            return max(1, self.batch_size // 4)
        elif memory_gb < 8:
            return max(1, self.batch_size // 2)

        return self.batch_size

    def get_adaptive_parallel_jobs(self, low_memory: bool = False) -> int:
        """Get parallel job count adjusted for memory constraints."""
        if low_memory or ConfigParser.parse_bool("LOW_MEMORY_MODE", False):
            return 1  # Sequential processing in low memory mode
        return self.max_parallel_jobs

    def get_cluster_shard(self) -> Tuple[int, int]:
        """Normalize cluster shard index/count with safe defaults."""
        count = max(1, int(self.cluster_shard_count))
        index = max(0, int(self.cluster_shard_index))
        if index >= count:
            index = index % count
        return index, count

    def get_sharded_variants(self, total_variants: int) -> List[int]:
        """
        Compute variant ids assigned to this shard.

        Uses round-robin sharding: variant (1-indexed) is assigned when
        (variant - 1) % shard_count == shard_index.
        """
        if total_variants <= 0:
            return []
        shard_index, shard_count = self.get_cluster_shard()
        if shard_count <= 1:
            return list(range(1, total_variants + 1))
        return [
            variant_id
            for variant_id in range(1, total_variants + 1)
            if (variant_id - 1) % shard_count == shard_index
        ]

    def should_skip_output(self, output_path: Optional[Path]) -> bool:
        """
        Check whether an output can be reused for idempotent runs.

        Returns True when SKIP_EXISTING_OUTPUTS=true, FORCE_REPROCESS=false,
        and the output file exists with at least MIN_OUTPUT_BYTES.
        """
        if self.force_reprocess or not self.skip_existing_outputs:
            return False
        if not output_path:
            return False
        path = output_path if isinstance(output_path, Path) else Path(output_path)
        if not path.exists() or not path.is_file():
            return False
        try:
            return path.stat().st_size >= self.min_output_bytes
        except OSError:
            return False


# =============================================================================
# Creative Configuration
# =============================================================================
@dataclass
class CreativeConfig:
    """Creative and editing parameters."""

    cut_style: str = field(default_factory=lambda: os.environ.get("CUT_STYLE", "dynamic").lower())
    creative_prompt: str = field(default_factory=lambda: os.environ.get("CREATIVE_PROMPT", "").strip())
    target_duration: float = field(default_factory=lambda: float(os.environ.get("TARGET_DURATION", "0") or "0"))
    music_start: float = field(default_factory=lambda: float(os.environ.get("MUSIC_START", "0") or "0"))
    music_end: Optional[float] = field(default_factory=lambda: float(os.environ.get("MUSIC_END", "0") or "0") or None)
    num_variants: int = field(default_factory=lambda: int(os.environ.get("NUM_VARIANTS", "1")))
    enable_xfade: str = field(default_factory=lambda: os.environ.get("ENABLE_XFADE", ""))  # "", "true", "false"
    xfade_duration: float = field(default_factory=lambda: float(os.environ.get("XFADE_DURATION", "0.3")))


# Audio Configuration
# =============================================================================
@dataclass
class AudioConfig:
    """Audio analysis and processing settings."""
    
    silence_threshold: str = field(default_factory=lambda: os.environ.get("SILENCE_THRESHOLD", "-35dB"))
    min_silence_duration: float = field(default_factory=lambda: float(os.environ.get("MIN_SILENCE_DURATION", "0.05")))
    energy_high_threshold: float = field(default_factory=lambda: float(os.environ.get("ENERGY_HIGH_THRESHOLD", "0.6")))
    energy_low_threshold: float = field(default_factory=lambda: float(os.environ.get("ENERGY_LOW_THRESHOLD", "0.4")))


# =============================================================================
# Thresholds Configuration (centralized)
# =============================================================================
@dataclass
class ThresholdsConfig:
    """Detection thresholds and sensitivity parameters (single source of truth).

    Defaults preserve current behavior; env vars allow tuning per deployment.
    """

    # Scene detection (PySceneDetect ContentDetector)
    scene_threshold: float = field(default_factory=ConfigParser.make_float_parser("SCENE_THRESHOLD", 30.0))

    # Speech/VAD (silero/pyannote style)
    speech_threshold: float = field(default_factory=ConfigParser.make_float_parser("SPEECH_THRESHOLD", 0.5))
    speech_min_duration_ms: int = field(default_factory=ConfigParser.make_int_parser("SPEECH_MIN_DURATION", 250))
    speech_min_silence_ms: int = field(default_factory=ConfigParser.make_int_parser("SPEECH_MIN_SILENCE", 100))

    # Silence detection for FFmpeg silencedetect
    # Stored without unit; append 'dB' at call sites when needed
    silence_level_db: str = field(default_factory=ConfigParser.make_str_parser("SILENCE_THRESHOLD", "-35"))
    silence_duration_s: float = field(default_factory=ConfigParser.make_float_parser("SILENCE_DURATION", 0.5))

    # Face detection (MediaPipe)
    face_confidence: float = field(default_factory=ConfigParser.make_float_parser("FACE_CONFIDENCE", 0.6))

    # Audio ducking thresholds (sidechaincompress)
    ducking_core_threshold: float = field(default_factory=ConfigParser.make_float_parser("DUCKING_CORE_THRESHOLD", 0.1))
    ducking_soft_threshold: float = field(default_factory=ConfigParser.make_float_parser("DUCKING_SOFT_THRESHOLD", 0.03))

    # Music analysis
    music_min_duration_s: float = field(default_factory=ConfigParser.make_float_parser("MUSIC_MIN_DURATION", 5.0))


# =============================================================================
# Session / Queue Configuration (Redis)
# =============================================================================
@dataclass
class SessionConfig:
    """Session/backing store configuration (Redis optional)."""

    redis_host: Optional[str] = field(default_factory=ConfigParser.make_str_parser("REDIS_HOST", ""))
    # Prefer REDIS_SERVICE_PORT when provided by Kubernetes service env
    _redis_port_raw: str = field(default_factory=lambda: ConfigParser.parse_str("REDIS_SERVICE_PORT", ConfigParser.parse_str("REDIS_PORT", "6379")))
    queue_fast_name: str = field(default_factory=ConfigParser.make_str_parser("QUEUE_FAST_NAME", "default"))
    queue_heavy_name: str = field(default_factory=ConfigParser.make_str_parser("QUEUE_HEAVY_NAME", "heavy"))
    
    @property
    def redis_port(self) -> int:
        raw = self._redis_port_raw
        try:
            return int(raw) if raw and raw.isdigit() else 6379
        except (TypeError, ValueError):
            return 6379


# =============================================================================
# Cache Configuration
# =============================================================================
@dataclass
class CacheConfig:
    """Caching configuration (TTL and enable/disable flags).

    Defaults preserve current behavior and respect existing env vars.
    """

    # Unify default TTL via legacy env var; allow independent overrides later if needed
    analysis_ttl_hours: int = field(default_factory=ConfigParser.make_int_parser("CACHE_INVALIDATION_HOURS", 24))
    metadata_ttl_hours: int = field(default_factory=ConfigParser.make_int_parser("CACHE_INVALIDATION_HOURS", 24))

    # Legacy toggle only existed for analysis cache; keep semantics
    analysis_enabled: bool = field(default_factory=lambda: not ConfigParser.parse_bool("DISABLE_ANALYSIS_CACHE", False))

    # Unified cache version across subsystems
    version: str = field(default_factory=ConfigParser.make_str_parser("CACHE_VERSION", "1.0"))


# =============================================================================
# Resource / Memory Thresholds Configuration
# =============================================================================
@dataclass
class ResourceThresholdsConfig:
    """Resource limits and memory pressure thresholds."""

    memory_limit_gb: float = field(default_factory=lambda: float(os.environ.get("MEMORY_LIMIT_GB", "8")))
    memory_warning_threshold: float = field(default_factory=lambda: float(os.environ.get("MEMORY_WARNING_THRESHOLD", "0.75")))
    memory_critical_threshold: float = field(default_factory=lambda: float(os.environ.get("MEMORY_CRITICAL_THRESHOLD", "0.90")))
    # Optional safety margin applied when estimating safe batch sizes
    memory_safety_margin_mb: int = field(default_factory=lambda: int(os.environ.get("MEMORY_SAFETY_MARGIN_MB", "500")))


# =============================================================================
# Monitoring Configuration
# =============================================================================
@dataclass
class MonitoringConfig:
    """Monitoring settings for live telemetry and tee logging."""

    mem_interval_sec: float = field(default_factory=lambda: float(os.environ.get("MONITOR_MEM_INTERVAL", "30.0")))
    log_file: Optional[str] = field(default_factory=lambda: os.environ.get("LOG_FILE") or None)
    # KPI target for Time-to-First-Preview in seconds (default: 180s)
    preview_target_seconds: int = field(default_factory=lambda: int(os.environ.get("PREVIEW_TIME_TARGET", "180")))


# =============================================================================
# Preview Configuration
# =============================================================================
@dataclass
class PreviewConfig:
    """Preview profile settings (resolution and speed/quality defaults).

    These values are used by preview generators and analysis/proxy steps and
    can be overridden via environment variables. They intentionally default to
    a fast, low-latency profile to optimize iteration speed.
    """

    width: int = field(default_factory=lambda: int(os.environ.get("PREVIEW_WIDTH", "640")))
    height: int = field(default_factory=lambda: int(os.environ.get("PREVIEW_HEIGHT", "360")))
    crf: int = field(default_factory=lambda: int(os.environ.get("PREVIEW_CRF", "28")))
    preset: str = field(default_factory=lambda: os.environ.get("PREVIEW_PRESET", "ultrafast"))
    # Max duration for generated previews (seconds)
    max_duration: float = field(default_factory=lambda: float(os.environ.get("PREVIEW_MAX_DURATION", "30.0")))


# =============================================================================
# =============================================================================
# File Type Configuration
# =============================================================================
@dataclass
class FileTypeConfig:
    """Allowed file extensions for uploads and processing."""

    video_extensions: Set[str] = field(default_factory=lambda: {'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', 'mxf', 'mts', 'm2ts', 'ts'})
    audio_extensions: Set[str] = field(default_factory=lambda: {'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg'})
    image_extensions: Set[str] = field(default_factory=lambda: {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'})

    def is_video(self, filename: str) -> bool:
        """Check if filename has a video extension."""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in self.video_extensions

    def is_audio(self, filename: str) -> bool:
        """Check if filename has an audio extension."""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in self.audio_extensions

    def is_image(self, filename: str) -> bool:
        """Check if filename has an image extension."""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in self.image_extensions

    def allowed_file(self, filename: str, allowed_extensions: Set[str]) -> bool:
        """Check if file extension is in allowed set."""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in allowed_extensions


# =============================================================================
# Main Settings Class
# =============================================================================
@dataclass
class Settings:
    """
    Main configuration container.

    Usage:
        from montage_ai.config import settings

        # Access paths
        input_dir = settings.paths.input_dir

        # Check features
        if settings.features.upscale:
            ...

        # Get LLM config
        if settings.llm.has_openai_backend:
            ...
    """

    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    high_res: ProcessingSettings = field(default_factory=ProcessingSettings)  # Phase 4: High-res support
    creative: CreativeConfig = field(default_factory=CreativeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    resources: ResourceThresholdsConfig = field(default_factory=ResourceThresholdsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    file_types: FileTypeConfig = field(default_factory=FileTypeConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    analysis: AnalysisConstants = field(default_factory=AnalysisConstants)

    # Job ID (generated per run)
    job_id: str = field(default_factory=lambda: os.environ.get("JOB_ID", ""))

    def __post_init__(self):
        """Validate settings after initialization."""
        # Ensure paths are Path objects
        if isinstance(self.paths.input_dir, str):
            self.paths.input_dir = Path(self.paths.input_dir)
        if isinstance(self.paths.output_dir, str):
            self.paths.output_dir = Path(self.paths.output_dir)
        if not self.job_id:
            self.job_id = self._resolve_job_id()

    def _resolve_job_id(self) -> str:
        """Resolve job id based on configured strategy."""
        strategy = (self.processing.job_id_strategy or "timestamp").lower()
        if strategy in ("hash", "stable", "deterministic"):
            return self._compute_job_id_hash()
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _compute_job_id_hash(self) -> str:
        """
        Compute a deterministic job id from inputs and key settings.

        Uses file metadata (name/size/mtime) and selected config fields.
        """
        hasher = hashlib.sha1()
        hasher.update(b"montage-ai")

        settings_fingerprint = {
            "features": {
                "stabilize": self.features.stabilize,
                "upscale": self.features.upscale,
                "enhance": self.features.enhance,
                "llm_clip_selection": self.features.llm_clip_selection,
                "deep_analysis": self.features.deep_analysis,
                "captions": self.features.captions,
                "voice_isolation": self.features.voice_isolation,
                "colorlevels": self.features.colorlevels,
                "luma_normalize": self.features.luma_normalize,
                "story_engine": self.features.story_engine,
                "strict_cloud_compute": self.features.strict_cloud_compute,
            },
            "creative": {
                "cut_style": self.creative.cut_style,
                "creative_prompt": self.creative.creative_prompt,
                "target_duration": self.creative.target_duration,
                "music_start": self.creative.music_start,
                "music_end": self.creative.music_end,
                "num_variants": self.creative.num_variants,
                "enable_xfade": self.creative.enable_xfade,
                "xfade_duration": self.creative.xfade_duration,
            },
            "encoding": {
                "quality_profile": self.encoding.quality_profile,
                "codec": self.encoding.codec,
                "crf": self.encoding.crf,
                "preset": self.encoding.preset,
                "pix_fmt": self.encoding.pix_fmt,
                "profile": self.encoding.profile,
                "level": self.encoding.level,
                "normalize_clips": self.encoding.normalize_clips,
            },
            "upscale": {
                "model": self.upscale.model,
                "scale": self.upscale.scale,
                "frame_format": self.upscale.frame_format,
                "tile_size": self.upscale.tile_size,
                "crf": self.upscale.crf,
            },
            "audio": {
                "silence_threshold": self.audio.silence_threshold,
                "min_silence_duration": self.audio.min_silence_duration,
                "energy_high_threshold": self.audio.energy_high_threshold,
                "energy_low_threshold": self.audio.energy_low_threshold,
            },
        }
        hasher.update(json.dumps(settings_fingerprint, sort_keys=True).encode("utf-8"))

        for root in (self.paths.input_dir, self.paths.music_dir, self.paths.assets_dir):
            for path in self._iter_hash_files(root):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                try:
                    rel_path = path.relative_to(root)
                except ValueError:
                    rel_path = path.name
                hasher.update(str(rel_path).encode("utf-8"))
                hasher.update(str(stat.st_size).encode("utf-8"))
                hasher.update(str(int(stat.st_mtime)).encode("utf-8"))

        return hasher.hexdigest()[:12]

    def _iter_hash_files(self, root: Path) -> list[Path]:
        """List files for job id hashing (stable order, best-effort)."""
        if not root.exists():
            return []
        if root.is_file():
            return [root]
        try:
            return sorted([p for p in root.rglob("*") if p.is_file()])
        except OSError:
            return []

    def reload(self) -> "Settings":
        """Reload settings from environment (useful after env changes)."""
        return Settings()

    def to_env_dict(self) -> dict:
        """Convert settings to environment variable dict for subprocess."""
        return {
            "INPUT_DIR": str(self.paths.input_dir),
            "MUSIC_DIR": str(self.paths.music_dir),
            "OUTPUT_DIR": str(self.paths.output_dir),
            "ASSETS_DIR": str(self.paths.assets_dir),
            "TEMP_DIR": str(self.paths.temp_dir),
            "METADATA_CACHE_DIR": str(self.paths.metadata_cache_dir),
            "TENSION_METADATA_DIR": str(self.paths.tension_metadata_dir),
            "SESSION_DIR": str(self.paths.session_dir),
            "TRANSCRIPT_DIR": str(self.paths.transcript_dir),
            "SHORTS_DIR": str(self.paths.shorts_dir),
            "STABILIZE": str(self.features.stabilize).lower(),
            "UPSCALE": str(self.features.upscale).lower(),
            "ENHANCE": str(self.features.enhance).lower(),
            "PRESERVE_ASPECT": str(self.features.preserve_aspect).lower(),
            "EXPORT_TIMELINE": str(self.features.export_timeline).lower(),
            "LLM_CLIP_SELECTION": str(self.features.llm_clip_selection).lower(),
            "VERBOSE": str(self.features.verbose).lower(),
            "ENABLE_STORY_ENGINE": str(self.features.story_engine).lower(),
            "STRICT_CLOUD_COMPUTE": str(self.features.strict_cloud_compute).lower(),
            "CGPU_ENABLED": str(self.llm.cgpu_enabled).lower(),
            "MONTAGE_CLOUD_ENABLED": str(self.cloud.enabled).lower(),
            "MONTAGE_CLOUD_ENDPOINT": self.cloud.endpoint,
            "CGPU_GPU_ENABLED": str(self.llm.cgpu_gpu_enabled).lower(),
            "CGPU_MAX_CONCURRENCY": str(max(1, self.llm.cgpu_max_concurrency)),
            "QUALITY_PROFILE": self.encoding.quality_profile,
            "COLORLEVELS": str(self.features.colorlevels).lower(),
            "LUMA_NORMALIZE": str(self.features.luma_normalize).lower(),
            "CUT_STYLE": self.creative.cut_style,
            "CREATIVE_PROMPT": self.creative.creative_prompt,
            "TARGET_DURATION": str(self.creative.target_duration),
            "MUSIC_START": str(self.creative.music_start),
            "MUSIC_END": str(self.creative.music_end or ""),
            "CLUSTER_SHARD_INDEX": str(self.processing.get_cluster_shard()[0]),
            "CLUSTER_SHARD_COUNT": str(self.processing.get_cluster_shard()[1]),
            "SKIP_EXISTING_OUTPUTS": str(self.processing.skip_existing_outputs).lower(),
            "FORCE_REPROCESS": str(self.processing.force_reprocess).lower(),
            "MIN_OUTPUT_BYTES": str(self.processing.min_output_bytes),
            "JOB_ID_STRATEGY": self.processing.job_id_strategy,
            "FFPROBE_TIMEOUT": str(self.processing.ffprobe_timeout),
            "FFMPEG_SHORT_TIMEOUT": str(self.processing.ffmpeg_short_timeout),
            "FFMPEG_TIMEOUT": str(self.processing.ffmpeg_timeout),
            "FFMPEG_LONG_TIMEOUT": str(self.processing.ffmpeg_long_timeout),
            "RENDER_TIMEOUT": str(self.processing.render_timeout),
            "FINAL_CRF": str(self.encoding.crf),
            "EXTRACT_REENCODE": str(self.encoding.extract_reencode).lower(),
            "FORCE_CGPU_ENCODING": str(self.gpu.force_cgpu_encoding).lower(),
            "EXPORT_WIDTH": str(self.export.resolution_width),
            "EXPORT_HEIGHT": str(self.export.resolution_height),
            "EXPORT_FPS": str(self.export.fps),
            "EXPORT_PROJECT_NAME": self.export.project_name_template,
            "UPSCALE_MODEL": self.upscale.model,
            "UPSCALE_SCALE": str(self.upscale.scale),
            "UPSCALE_FRAME_FORMAT": self.upscale.frame_format,
            "UPSCALE_TILE_SIZE": str(self.upscale.tile_size),
            "UPSCALE_CRF": str(self.upscale.crf),
            "JOB_ID": self.job_id,
            # Threshold exports for compatibility with subprocesses/tools
            "SCENE_THRESHOLD": str(self.thresholds.scene_threshold),
            "SPEECH_THRESHOLD": str(self.thresholds.speech_threshold),
            "SILENCE_THRESHOLD": str(self.thresholds.silence_level_db),
            "SILENCE_DURATION": str(self.thresholds.silence_duration_s),
            "FACE_CONFIDENCE": str(self.thresholds.face_confidence),
            "DUCKING_CORE_THRESHOLD": str(self.thresholds.ducking_core_threshold),
            "DUCKING_SOFT_THRESHOLD": str(self.thresholds.ducking_soft_threshold),
            "MUSIC_MIN_DURATION": str(self.thresholds.music_min_duration_s),
            "SPEECH_MIN_DURATION": str(self.thresholds.speech_min_duration_ms),
            "SPEECH_MIN_SILENCE": str(self.thresholds.speech_min_silence_ms),
            # Session/Redis
            "REDIS_HOST": str(self.session.redis_host or ""),
            "REDIS_PORT": str(self.session.redis_port),
            # Cache/TTL (unified legacy keys for compatibility)
            "CACHE_INVALIDATION_HOURS": str(self.cache.analysis_ttl_hours),
            "DISABLE_ANALYSIS_CACHE": str(not self.cache.analysis_enabled).lower(),
            "CACHE_VERSION": self.cache.version,
            # Memory/Resource thresholds
            "MEMORY_LIMIT_GB": str(self.resources.memory_limit_gb),
            "MEMORY_WARNING_THRESHOLD": str(self.resources.memory_warning_threshold),
            "MEMORY_CRITICAL_THRESHOLD": str(self.resources.memory_critical_threshold),
            # Monitoring
            "MONITOR_MEM_INTERVAL": str(self.monitoring.mem_interval_sec),
            "LOG_FILE": self.monitoring.log_file or "",
            # Preview profile exports (for subprocesses/tools)
            "PREVIEW_WIDTH": str(self.preview.width),
            "PREVIEW_HEIGHT": str(self.preview.height),
            "PREVIEW_CRF": str(self.preview.crf),
            "PREVIEW_PRESET": self.preview.preset,
            "PREVIEW_MAX_DURATION": str(self.preview.max_duration),
        }


# =============================================================================
# Global Settings Instance (Singleton)
# =============================================================================
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (lazy initialization)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload of settings from environment."""
    global _settings
    _settings = Settings()
    return _settings


# Note: Do NOT create module-level instance here!
# This causes cached dataclass definition when module is imported.
# Always use get_settings() to ensure fresh Settings with updated dataclass.

# =============================================================================
# Style and Effect Configurations
# =============================================================================

class QualityProfile:
    """Video quality profile enumeration."""
    PREVIEW = "preview"  # 360p, fast
    STANDARD = "standard"  # 1080p
    HIGH = "high"  # 4K
    CUSTOM = "custom"


@dataclass
class StyleConfig:
    """Configuration for a specific montage style."""
    
    name: str
    """Style name (e.g., 'dynamic', 'hitchcock')."""
    
    weights: dict = field(default_factory=dict)
    """Scoring weights for selection."""
    
    preferred_shots: List[str] = field(default_factory=list)
    """Preferred shot types (e.g., 'close', 'wide')."""
    
    transition_style: str = "cut"
    """Transition type: 'cut', 'fade', 'dissolve'."""
    
    cut_frequency: float = 1.0
    """Relative cut frequency (1.0 = normal, 2.0 = fast)."""
    
    color_grading: Optional[str] = None
    """Color grading preset name."""
    
    description: str = ""
    """Style description."""
    
    def get_weight(self, key: str, default: float = 1.0) -> float:
        """Get weight for scoring factor with fallback."""
        return self.weights.get(key, default)
    
    def has_shot_preference(self) -> bool:
        """Check if style has shot type preferences."""
        return len(self.preferred_shots) > 0
    
    def is_high_energy(self) -> bool:
        """Check if style favors high-energy content."""
        return self.cut_frequency > 1.0


@dataclass
class EffectConfig:
    """Video effects configuration."""
    
    stabilize: bool = False
    """Enable video stabilization."""
    
    upscale: bool = False
    """Enable AI upscaling."""
    
    denoise: bool = True
    """Enable denoising."""
    
    color_grade: bool = True
    """Enable color grading."""
    
    audio_enhancement: bool = True
    """Enable audio enhancement."""
    
    motion_blur: float = 0.0
    """Motion blur amount [0.0, 1.0]."""
    
    sharpen: float = 0.0
    """Sharpening amount [0.0, 1.0]."""
    
    def count_enabled_effects(self) -> int:
        """Count enabled boolean effects."""
        return sum([
            self.stabilize,
            self.upscale,
            self.denoise,
            self.color_grade,
            self.audio_enhancement,
        ])


@dataclass
class VideoConfigSpec:
    """Video format specification (distinct from processing settings)."""
    
    width: int = 1920
    """Output video width in pixels."""
    
    height: int = 1080
    """Output video height in pixels."""
    
    fps: float = 24.0
    """Frames per second."""
    
    duration: float = 60.0
    """Target duration in seconds."""
    
    bitrate: str = "5000k"
    """Output bitrate."""
    
    codec: str = "libx264"
    """Video codec."""
    
    preset: str = "medium"
    """Encoding preset (ultrafast, fast, medium, slow, etc)."""
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 16/9
    
    @property
    def is_portrait(self) -> bool:
        """Check if output is portrait orientation."""
        return self.height > self.width
    
    @property
    def is_landscape(self) -> bool:
        """Check if output is landscape orientation."""
        return self.width > self.height


@dataclass
class AudioConfigSpec:
    """Audio format specification."""
    
    codec: str = "aac"
    """Audio codec."""
    
    bitrate: str = "128k"
    """Audio bitrate."""
    
    sample_rate: int = 48000
    """Sample rate in Hz."""
    
    channels: int = 2
    """Number of channels (1 = mono, 2 = stereo)."""
    
    normalize: bool = True
    """Normalize loudness."""
    
    target_loudness: float = -20.0
    """Target loudness in dB (LUFS)."""
    
    ducking_strength: float = 0.5
    """Music ducking when speech present [0.0, 1.0]."""


@dataclass
class MontageSettingsSpec:
    """Complete montage settings - integrates style, effects, video, audio."""
    
    style: StyleConfig
    """Selected montage style."""

    paths: PathConfig = field(default_factory=PathConfig)
    """Filesystem paths used across the pipeline."""

    features: FeatureConfig = field(default_factory=FeatureConfig)
    """Feature toggles and runtime flags."""

    job_id: str = field(default_factory=lambda: os.environ.get("JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S")))
    """Optional job identifier for builder compatibility."""
    
    quality: str = QualityProfile.STANDARD
    """Output quality profile."""
    
    video: VideoConfigSpec = field(default_factory=VideoConfigSpec)
    """Video format settings."""
    
    audio: AudioConfigSpec = field(default_factory=AudioConfigSpec)
    """Audio settings."""
    
    effects: EffectConfig = field(default_factory=EffectConfig)
    """Effects configuration."""
    
    max_transitions: int = 20
    """Maximum number of transitions."""
    
    cache_enabled: bool = True
    """Enable analysis caching."""
    
    @staticmethod
    def create_default(style_name: str = "dynamic") -> "MontageSettingsSpec":
        """Create settings with defaults."""
        style = StyleConfig(name=style_name, weights={"energy": 1.0})
        return MontageSettingsSpec(style=style)
    
    @staticmethod
    def create_preview() -> "MontageSettingsSpec":
        """Create fast preview settings (360p, ultrafast)."""
        style = StyleConfig(name="dynamic", weights={"energy": 1.0})
        video = VideoConfigSpec(width=640, height=360, fps=24.0, preset="ultrafast")
        return MontageSettingsSpec(style=style, quality=QualityProfile.PREVIEW, video=video)
    
    @staticmethod
    def create_hires() -> "MontageSettingsSpec":
        """Create high-resolution 4K settings."""
        style = StyleConfig(name="dynamic", weights={"energy": 1.0})
        video = VideoConfigSpec(width=3840, height=2160, bitrate="15000k", preset="slow")
        return MontageSettingsSpec(style=style, quality=QualityProfile.HIGH, video=video)
    
    def validate(self) -> List[str]:
        """Validate settings, return list of errors."""
        errors = []
        
        if self.video.width <= 0 or self.video.height <= 0:
            errors.append("Invalid video dimensions")
        
        if self.video.fps <= 0:
            errors.append("Invalid FPS")
        
        if self.audio.channels not in [1, 2]:
            errors.append("Invalid audio channels")
        
        if self.max_transitions < 0:
            errors.append("Invalid max transitions")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if settings are valid."""
        return len(self.validate()) == 0