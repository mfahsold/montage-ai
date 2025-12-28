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
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
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

    def ensure_directories(self) -> None:
        """Create all directories if they don't exist."""
        for path in [self.input_dir, self.music_dir, self.output_dir, self.assets_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_log_path(self, job_id: str) -> Path:
        """Get log file path for a job."""
        return self.output_dir / f"render_{job_id}.log"


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


# =============================================================================
# GPU Configuration
# =============================================================================
@dataclass
class GPUConfig:
    """GPU and hardware acceleration settings."""

    use_gpu: str = field(default_factory=lambda: os.environ.get("USE_GPU", "auto").lower())
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

    # cgpu (Colab GPU)
    cgpu_enabled: bool = field(default_factory=lambda: os.environ.get("CGPU_ENABLED", "false").lower() == "true")
    cgpu_gpu_enabled: bool = field(default_factory=lambda: os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true")
    cgpu_host: str = field(default_factory=lambda: os.environ.get("CGPU_HOST", "127.0.0.1"))
    cgpu_port: int = field(default_factory=lambda: int(os.environ.get("CGPU_PORT", "5021")))
    cgpu_model: str = field(default_factory=lambda: os.environ.get("CGPU_MODEL", "gemini-2.5-flash"))
    cgpu_timeout: int = field(default_factory=lambda: int(os.environ.get("CGPU_TIMEOUT", "1200")))

    @property
    def has_openai_backend(self) -> bool:
        """Check if OpenAI-compatible backend is configured."""
        return bool(self.openai_api_base and self.openai_model)

    @property
    def has_google_backend(self) -> bool:
        """Check if Google AI backend is configured."""
        return bool(self.google_api_key)


# =============================================================================
# Encoding Configuration
# =============================================================================
@dataclass
class EncodingConfig:
    """FFmpeg and video encoding settings."""

    codec: str = field(default_factory=lambda: os.environ.get("OUTPUT_CODEC", "libx264"))
    crf: int = field(default_factory=lambda: int(os.environ.get("FINAL_CRF", "18")))
    preset: str = field(default_factory=lambda: os.environ.get("FFMPEG_PRESET", "medium"))
    hwaccel: str = field(default_factory=lambda: os.environ.get("FFMPEG_HWACCEL", "auto"))
    pix_fmt: str = field(default_factory=lambda: os.environ.get("OUTPUT_PIX_FMT", "yuv420p"))
    profile: str = field(default_factory=lambda: os.environ.get("OUTPUT_PROFILE", "high"))
    level: str = field(default_factory=lambda: os.environ.get("OUTPUT_LEVEL", "4.1"))
    threads: int = field(default_factory=lambda: int(os.environ.get("FFMPEG_THREADS", "0")))
    normalize_clips: bool = field(default_factory=lambda: os.environ.get("NORMALIZE_CLIPS", "true").lower() == "true")


# =============================================================================
# Processing Configuration
# =============================================================================
@dataclass
class ProcessingConfig:
    """Batch processing and performance settings."""

    batch_size: int = field(default_factory=lambda: int(os.environ.get("BATCH_SIZE", "25")))
    force_gc: bool = field(default_factory=lambda: os.environ.get("FORCE_GC", "true").lower() == "true")
    parallel_enhance: bool = field(default_factory=lambda: os.environ.get("PARALLEL_ENHANCE", "true").lower() == "true")
    max_parallel_jobs: int = field(default_factory=lambda: int(os.environ.get("MAX_PARALLEL_JOBS", str(max(1, multiprocessing.cpu_count() - 2)))))
    max_concurrent_jobs: int = field(default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_JOBS", "2")))
    max_scene_reuse: int = field(default_factory=lambda: int(os.environ.get("MAX_SCENE_REUSE", "3")))


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
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    creative: CreativeConfig = field(default_factory=CreativeConfig)
    file_types: FileTypeConfig = field(default_factory=FileTypeConfig)

    # Job ID (generated per run)
    job_id: str = field(default_factory=lambda: os.environ.get("JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S")))

    def __post_init__(self):
        """Validate settings after initialization."""
        # Ensure paths are Path objects
        if isinstance(self.paths.input_dir, str):
            self.paths.input_dir = Path(self.paths.input_dir)
        if isinstance(self.paths.output_dir, str):
            self.paths.output_dir = Path(self.paths.output_dir)

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
            "CGPU_ENABLED": str(self.llm.cgpu_enabled).lower(),
            "CGPU_GPU_ENABLED": str(self.llm.cgpu_gpu_enabled).lower(),
            "CUT_STYLE": self.creative.cut_style,
            "CREATIVE_PROMPT": self.creative.creative_prompt,
            "TARGET_DURATION": str(self.creative.target_duration),
            "MUSIC_START": str(self.creative.music_start),
            "MUSIC_END": str(self.creative.music_end or ""),
            "FINAL_CRF": str(self.encoding.crf),
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
