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
    temp_dir: Path = field(default_factory=lambda: Path(os.environ.get("TEMP_DIR", "/tmp")))
    lut_dir: Path = field(default_factory=lambda: Path(os.environ.get("LUT_DIR", "/data/luts")))
    style_preset_path: Optional[Path] = field(default_factory=lambda: Path(os.environ.get("STYLE_PRESET_PATH")) if os.environ.get("STYLE_PRESET_PATH") else None)
    style_preset_dir: Optional[Path] = field(default_factory=lambda: Path(os.environ.get("STYLE_PRESET_DIR") or os.environ.get("STYLE_TEMPLATES_DIR")) if (os.environ.get("STYLE_PRESET_DIR") or os.environ.get("STYLE_TEMPLATES_DIR")) else None)

    def ensure_directories(self) -> None:
        """Create all directories if they don't exist."""
        for path in [self.input_dir, self.music_dir, self.output_dir, self.assets_dir]:
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

    stabilize: bool = field(default_factory=lambda: os.environ.get("STABILIZE", "false").lower() == "true")
    upscale: bool = field(default_factory=lambda: os.environ.get("UPSCALE", "false").lower() == "true")
    enhance: bool = field(default_factory=lambda: os.environ.get("ENHANCE", "true").lower() == "true")
    preserve_aspect: bool = field(default_factory=lambda: os.environ.get("PRESERVE_ASPECT", "false").lower() == "true")
    export_timeline: bool = field(default_factory=lambda: os.environ.get("EXPORT_TIMELINE", "false").lower() == "true")
    generate_proxies: bool = field(default_factory=lambda: os.environ.get("GENERATE_PROXIES", "false").lower() == "true")
    llm_clip_selection: bool = field(default_factory=lambda: os.environ.get("LLM_CLIP_SELECTION", "true").lower() == "true")
    deep_analysis: bool = field(default_factory=lambda: os.environ.get("DEEP_ANALYSIS", "false").lower() == "true")
    verbose: bool = field(default_factory=lambda: os.environ.get("VERBOSE", "true").lower() == "true")
    enable_ai_filter: bool = field(default_factory=lambda: os.environ.get("ENABLE_AI_FILTER", "false").lower() == "true")
    # Phase 4: Agentic Creative Loop - LLM evaluates and refines cuts iteratively
    creative_loop: bool = field(default_factory=lambda: os.environ.get("CREATIVE_LOOP", "false").lower() == "true")
    creative_loop_max_iterations: int = field(default_factory=lambda: int(os.environ.get("CREATIVE_LOOP_MAX_ITERATIONS", "3")))

    # Episodic memory for analysis caching (experimental)
    episodic_memory: bool = field(default_factory=lambda: os.environ.get("EPISODIC_MEMORY", "false").lower() == "true")

    # Storytelling Engine (Phase 1 scaffolding)
    story_engine: bool = field(default_factory=lambda: os.environ.get("ENABLE_STORY_ENGINE", "false").lower() == "true")
    strict_cloud_compute: bool = field(default_factory=lambda: os.environ.get("STRICT_CLOUD_COMPUTE", "false").lower() == "true")

    # Shorts Workflow (Vertical Video + Smart Reframing)
    shorts_mode: bool = field(default_factory=lambda: os.environ.get("SHORTS_MODE", "false").lower() == "true")
    reframe_mode: str = field(default_factory=lambda: os.environ.get("REFRAME_MODE", "auto"))  # auto, speaker, center, custom

    # 2025 P0/P1: Burn-in captions and voice isolation
    captions: bool = field(default_factory=lambda: os.environ.get("CAPTIONS", "false").lower() == "true")
    captions_style: str = field(default_factory=lambda: os.environ.get("CAPTIONS_STYLE", "tiktok"))  # tiktok, minimal, bold, karaoke
    transcription_model: str = field(default_factory=lambda: os.environ.get("TRANSCRIPTION_MODEL", "medium"))
    
    # Audio Polish: Clean Audio = Voice Isolation + Denoise (single toggle)
    # CLEAN_AUDIO is the new consolidated toggle, VOICE_ISOLATION is legacy
    voice_isolation: bool = field(default_factory=lambda: (
        os.environ.get("CLEAN_AUDIO", "false").lower() == "true" or
        os.environ.get("VOICE_ISOLATION", "false").lower() == "true"
    ))
    voice_isolation_model: str = field(default_factory=lambda: os.environ.get("VOICE_ISOLATION_MODEL", "htdemucs"))

    # Noise Reduction: DeepFilterNet for lightweight noise removal (faster than voice_isolation)
    # Use for podcasts, interviews, vlogs with background noise
    noise_reduction: bool = field(default_factory=lambda: os.environ.get("NOISE_REDUCTION", "false").lower() == "true")
    noise_reduction_strength: int = field(default_factory=lambda: int(os.environ.get("NOISE_REDUCTION_STRENGTH", "100")))

    # Performance: Low-resource hardware mode (longer timeouts, smaller batches, sequential processing)
    low_memory_mode: bool = field(default_factory=lambda: os.environ.get("LOW_MEMORY_MODE", "false").lower() == "true")

    # Color/levels normalization controls (can be disabled for clean footage)
    colorlevels: bool = field(default_factory=lambda: os.environ.get("COLORLEVELS", "true").lower() == "true")
    luma_normalize: bool = field(default_factory=lambda: os.environ.get("LUMA_NORMALIZE", "true").lower() == "true")


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

    batch_size: int = field(default_factory=lambda: int(os.environ.get("BATCH_SIZE", "25")))
    force_gc: bool = field(default_factory=lambda: os.environ.get("FORCE_GC", "true").lower() == "true")
    parallel_enhance: bool = field(default_factory=lambda: os.environ.get("PARALLEL_ENHANCE", "true").lower() == "true")
    skip_existing_outputs: bool = field(default_factory=lambda: os.environ.get("SKIP_EXISTING_OUTPUTS", "true").lower() == "true")
    force_reprocess: bool = field(default_factory=lambda: os.environ.get("FORCE_REPROCESS", "false").lower() == "true")
    min_output_bytes: int = field(default_factory=lambda: int(os.environ.get("MIN_OUTPUT_BYTES", "1024")))
    job_id_strategy: str = field(default_factory=lambda: os.environ.get("JOB_ID_STRATEGY", "timestamp").lower())
    max_parallel_jobs: int = field(default_factory=lambda: int(os.environ.get("MAX_PARALLEL_JOBS", str(max(1, get_effective_cpu_count() - 2)))))
    max_concurrent_jobs: int = field(default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_JOBS", "2")))
    max_scene_reuse: int = field(default_factory=lambda: int(os.environ.get("MAX_SCENE_REUSE", "3")))
    cluster_shard_index: int = field(default_factory=get_cluster_shard_index)
    cluster_shard_count: int = field(default_factory=get_cluster_shard_count)

    # Adaptive settings for low-resource hardware
    clip_prefetch_count: int = field(default_factory=lambda: int(os.environ.get("CLIP_PREFETCH_COUNT", "3")))
    analysis_timeout: int = field(default_factory=lambda: int(os.environ.get("ANALYSIS_TIMEOUT", "120")))  # seconds
    ffprobe_timeout: int = field(default_factory=lambda: int(os.environ.get("FFPROBE_TIMEOUT", "10")))
    ffmpeg_short_timeout: int = field(default_factory=lambda: int(os.environ.get("FFMPEG_SHORT_TIMEOUT", "30")))
    ffmpeg_timeout: int = field(default_factory=lambda: int(os.environ.get("FFMPEG_TIMEOUT", "120")))
    ffmpeg_long_timeout: int = field(default_factory=lambda: int(os.environ.get("FFMPEG_LONG_TIMEOUT", "600")))
    render_timeout: int = field(default_factory=lambda: int(os.environ.get("RENDER_TIMEOUT", "3600")))  # 1 hour default

    def get_adaptive_batch_size(self, low_memory: bool = False) -> int:
        """Get batch size adjusted for memory constraints."""
        if low_memory or os.environ.get("LOW_MEMORY_MODE", "false").lower() == "true":
            return max(1, self.batch_size // 4)  # Quarter batch size in low memory mode
        return self.batch_size

    def get_adaptive_parallel_jobs(self, low_memory: bool = False) -> int:
        """Get parallel job count adjusted for memory constraints."""
        if low_memory or os.environ.get("LOW_MEMORY_MODE", "false").lower() == "true":
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
# =============================================================================
# File Type Configuration
# =============================================================================
@dataclass
class FileTypeConfig:
    """Allowed file extensions for uploads and processing."""

    video_extensions: Set[str] = field(default_factory=lambda: {'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'})
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
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    creative: CreativeConfig = field(default_factory=CreativeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    file_types: FileTypeConfig = field(default_factory=FileTypeConfig)

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


# Convenience alias
settings = get_settings()
