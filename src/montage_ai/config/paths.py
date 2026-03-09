"""
Path Configuration Module

All filesystem paths used by Montage AI.
Paths can be configured via YAML or environment variables with MONTAGE_ prefix.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from . import BaseConfig, _env_override


def _select_temp_dir() -> Path:
    """Select optimal temp directory (RAM disk preferred)."""
    env_override = os.environ.get("TEMP_DIR")
    if env_override:
        return Path(env_override)

    # Try /dev/shm (RAM disk) first
    shm_path = Path("/dev/shm")
    if shm_path.exists() and shm_path.is_dir():
        try:
            min_free_mb = int(os.environ.get("TEMP_DIR_MIN_FREE_MB", "512"))
            usage = shutil.disk_usage(shm_path)
            if usage.free >= min_free_mb * 1024 * 1024:
                return shm_path
        except (OSError, ValueError):
            pass

    return Path("/tmp")


@dataclass
class PathConfig(BaseConfig):
    """Filesystem paths configuration."""

    input_dir: Path = field(
        default_factory=lambda: _env_override("input_dir", Path("/data/input"))
    )
    music_dir: Path = field(
        default_factory=lambda: _env_override("music_dir", Path("/data/music"))
    )
    output_dir: Path = field(
        default_factory=lambda: _env_override("output_dir", Path("/data/output"))
    )
    assets_dir: Path = field(
        default_factory=lambda: _env_override("assets_dir", Path("/data/assets"))
    )
    lut_dir: Path = field(
        default_factory=lambda: _env_override("lut_dir", Path("/data/luts"))
    )
    temp_dir: Path = field(default_factory=_select_temp_dir)

    # Cache directories
    session_dir: Path = field(
        default_factory=lambda: _env_override(
            "session_dir", Path("/tmp/montage_sessions")
        )
    )
    transcript_dir: Path = field(
        default_factory=lambda: _env_override(
            "transcript_dir", Path("/tmp/montage_transcript")
        )
    )
    shorts_dir: Path = field(
        default_factory=lambda: _env_override("shorts_dir", Path("/tmp/montage_shorts"))
    )
    scene_cache_dir: Path = field(
        default_factory=lambda: _env_override(
            "scene_cache_dir",
            Path(os.environ.get("OUTPUT_DIR", "/data/output")) / "scene_cache",
        )
    )
    metadata_cache_dir: Path = field(
        default_factory=lambda: _env_override(
            "metadata_cache_dir",
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "montage_ai"
            / "metadata",
        )
    )

    # Style templates
    style_preset_path: Optional[Path] = field(
        default_factory=lambda: _env_override("style_preset_path", None)
    )
    style_preset_dir: Optional[Path] = field(
        default_factory=lambda: _env_override("style_preset_dir", None)
    )

    def ensure_directories(self) -> None:
        """Create all directories if they don't exist."""
        for path in [
            self.input_dir,
            self.music_dir,
            self.output_dir,
            self.assets_dir,
            self.metadata_cache_dir,
            self.session_dir,
            self.transcript_dir,
            self.shorts_dir,
            self.scene_cache_dir,
        ]:
            if path:
                path.mkdir(parents=True, exist_ok=True)

    def get_log_path(self, job_id: str) -> Path:
        """Get log file path for a job."""
        return self.output_dir / f"render_{job_id}.log"

    def validate(self) -> list:
        """Validate path configuration."""
        errors = []

        # Check required directories exist or can be created
        for name, path in [
            ("input_dir", self.input_dir),
            ("music_dir", self.music_dir),
            ("output_dir", self.output_dir),
        ]:
            if path and not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    errors.append(f"Cannot create {name}: {e}")

        return errors
