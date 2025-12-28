"""
Analysis Cache for Montage AI

Nightfly Studio - Persistent caching for expensive analysis operations.

Caches:
- Audio analysis (beat detection, energy profile) from librosa
- Scene boundaries from scenedetect

Cache files are stored as JSON sidecars next to source files:
- music/track.mp3.analysis.json
- input/video.mp4.scenes.json

Cache invalidation:
- Hash-based: file size + mtime (fast, no full-file hash)
- TTL-based: configurable via CACHE_INVALIDATION_HOURS (default: 24h)
- Version-based: CACHE_VERSION bump invalidates all caches

Usage:
    from montage_ai.core.analysis_cache import get_analysis_cache

    cache = get_analysis_cache()

    # Audio analysis
    if cached := cache.load_audio(music_path):
        # Use cached data
        tempo, beats, energy = cached.tempo, cached.beat_times, cached.energy_values
    else:
        # Compute and cache
        beat_info = get_beat_times(music_path)
        energy_profile = analyze_music_energy(music_path)
        cache.save_audio(music_path, beat_info, energy_profile)

    # Scene analysis
    if cached := cache.load_scenes(video_path, threshold):
        scenes = cached.scenes
    else:
        scenes = detect_scenes(video_path, threshold)
        cache.save_scenes(video_path, threshold, scenes)
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import hashlib
import os
from datetime import datetime

from ..logger import logger


# =============================================================================
# Constants
# =============================================================================

CACHE_VERSION = "1.0"
DEFAULT_TTL_HOURS = 24


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CacheEntry:
    """Base class for cached analysis results."""
    version: str
    file_hash: str
    computed_at: str

    @classmethod
    def compute_file_hash(cls, path: str) -> str:
        """
        Fast hash based on file size + mtime.

        This is much faster than computing a full file hash,
        and sufficient for detecting file changes in most cases.
        """
        try:
            stat = os.stat(path)
            hash_input = f"{stat.st_size}:{stat.st_mtime}".encode()
            return hashlib.md5(hash_input).hexdigest()[:16]
        except OSError:
            return ""


@dataclass
class AudioAnalysisEntry(CacheEntry):
    """Cached beat detection and energy analysis."""
    tempo: float
    beat_times: List[float]
    energy_times: List[float]
    energy_values: List[float]
    duration: float
    sample_rate: int


@dataclass
class SceneAnalysisEntry(CacheEntry):
    """Cached scene boundaries."""
    threshold: float
    scenes: List[Dict[str, float]]  # [{"start": 0.0, "end": 5.2}, ...]
    total_scenes: int


# =============================================================================
# Analysis Cache
# =============================================================================

class AnalysisCache:
    """
    Persistent cache for expensive analysis operations.

    Stores analysis results as JSON sidecar files next to source files.
    Automatically invalidates cache when source file changes or TTL expires.
    """

    def __init__(self, ttl_hours: int = DEFAULT_TTL_HOURS, enabled: bool = True):
        """
        Initialize the analysis cache.

        Args:
            ttl_hours: Time-to-live in hours before cache expires
            enabled: Whether caching is enabled (can be disabled for testing)
        """
        self.ttl_hours = ttl_hours
        self.enabled = enabled

    def _cache_path(self, source_path: str, suffix: str) -> Path:
        """Get sidecar cache file path."""
        return Path(f"{source_path}.{suffix}.json")

    def _is_valid(self, cache_path: Path, source_path: str) -> bool:
        """
        Check if cache is valid.

        Validation checks:
        1. Cache file exists
        2. Cache version matches current version
        3. Source file hash matches (file hasn't changed)
        4. Cache hasn't expired (within TTL)
        """
        if not self.enabled:
            return False

        if not cache_path.exists():
            return False

        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)

            # Version check
            if data.get("version") != CACHE_VERSION:
                logger.debug(f"Cache version mismatch: {data.get('version')} != {CACHE_VERSION}")
                return False

            # Hash check (source file changed?)
            current_hash = CacheEntry.compute_file_hash(source_path)
            if data.get("file_hash") != current_hash:
                logger.debug(f"Source file changed: {os.path.basename(source_path)}")
                return False

            # TTL check
            computed_at = datetime.fromisoformat(data["computed_at"])
            age_hours = (datetime.now() - computed_at).total_seconds() / 3600
            if age_hours > self.ttl_hours:
                logger.debug(f"Cache expired: {age_hours:.1f}h > {self.ttl_hours}h TTL")
                return False

            return True

        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            logger.debug(f"Cache validation failed: {e}")
            return False

    def _write_cache(self, cache_path: Path, data: dict) -> bool:
        """Write cache data to file with error handling."""
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except OSError as e:
            logger.debug(f"Failed to write cache: {e}")
            return False

    # -------------------------------------------------------------------------
    # Audio Analysis Cache
    # -------------------------------------------------------------------------

    def load_audio(self, audio_path: str) -> Optional[AudioAnalysisEntry]:
        """
        Load cached audio analysis if valid.

        Args:
            audio_path: Path to the audio file

        Returns:
            AudioAnalysisEntry if cache hit, None if cache miss
        """
        cache_path = self._cache_path(audio_path, "analysis")

        if not self._is_valid(cache_path, audio_path):
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)

            logger.debug(f"Cache hit: audio analysis for {os.path.basename(audio_path)}")
            return AudioAnalysisEntry(
                version=data["version"],
                file_hash=data["file_hash"],
                computed_at=data["computed_at"],
                tempo=data["tempo"],
                beat_times=data["beat_times"],
                energy_times=data["energy_times"],
                energy_values=data["energy_values"],
                duration=data["duration"],
                sample_rate=data["sample_rate"],
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.debug(f"Failed to load audio cache: {e}")
            return None

    def save_audio(self, audio_path: str, beat_info, energy_profile) -> bool:
        """
        Save audio analysis to cache.

        Args:
            audio_path: Path to the audio file
            beat_info: BeatInfo dataclass from audio_analysis
            energy_profile: EnergyProfile dataclass from audio_analysis

        Returns:
            True if cache was saved successfully
        """
        if not self.enabled:
            return False

        entry = AudioAnalysisEntry(
            version=CACHE_VERSION,
            file_hash=CacheEntry.compute_file_hash(audio_path),
            computed_at=datetime.now().isoformat(),
            tempo=float(beat_info.tempo),
            beat_times=beat_info.beat_times.tolist(),
            energy_times=energy_profile.times.tolist(),
            energy_values=energy_profile.rms.tolist(),
            duration=float(beat_info.duration),
            sample_rate=int(beat_info.sample_rate),
        )

        cache_path = self._cache_path(audio_path, "analysis")
        success = self._write_cache(cache_path, asdict(entry))

        if success:
            logger.debug(f"Cached audio analysis: {os.path.basename(audio_path)}")

        return success

    # -------------------------------------------------------------------------
    # Scene Analysis Cache
    # -------------------------------------------------------------------------

    def load_scenes(self, video_path: str, threshold: float) -> Optional[SceneAnalysisEntry]:
        """
        Load cached scene analysis if valid and threshold matches.

        Args:
            video_path: Path to the video file
            threshold: Scene detection threshold (must match cached value)

        Returns:
            SceneAnalysisEntry if cache hit, None if cache miss
        """
        cache_path = self._cache_path(video_path, "scenes")

        if not self._is_valid(cache_path, video_path):
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)

            # Threshold must match (within tolerance)
            cached_threshold = data.get("threshold", 0)
            if abs(cached_threshold - threshold) > 0.1:
                logger.debug(f"Threshold mismatch: {cached_threshold} != {threshold}")
                return None

            logger.debug(f"Cache hit: scene analysis for {os.path.basename(video_path)}")
            return SceneAnalysisEntry(
                version=data["version"],
                file_hash=data["file_hash"],
                computed_at=data["computed_at"],
                threshold=data["threshold"],
                scenes=data["scenes"],
                total_scenes=data["total_scenes"],
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.debug(f"Failed to load scene cache: {e}")
            return None

    def save_scenes(self, video_path: str, threshold: float, scenes: list) -> bool:
        """
        Save scene analysis to cache.

        Args:
            video_path: Path to the video file
            threshold: Scene detection threshold used
            scenes: List of Scene objects with start/end attributes

        Returns:
            True if cache was saved successfully
        """
        if not self.enabled:
            return False

        # Convert Scene objects to dicts
        scene_dicts = []
        for scene in scenes:
            if hasattr(scene, "start") and hasattr(scene, "end"):
                scene_dicts.append({"start": float(scene.start), "end": float(scene.end)})
            elif isinstance(scene, dict):
                scene_dicts.append(scene)
            elif isinstance(scene, (tuple, list)) and len(scene) >= 2:
                scene_dicts.append({"start": float(scene[0]), "end": float(scene[1])})

        entry = SceneAnalysisEntry(
            version=CACHE_VERSION,
            file_hash=CacheEntry.compute_file_hash(video_path),
            computed_at=datetime.now().isoformat(),
            threshold=float(threshold),
            scenes=scene_dicts,
            total_scenes=len(scene_dicts),
        )

        cache_path = self._cache_path(video_path, "scenes")
        success = self._write_cache(cache_path, asdict(entry))

        if success:
            logger.debug(f"Cached scene analysis: {os.path.basename(video_path)} ({len(scene_dicts)} scenes)")

        return success

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def clear_audio_cache(self, audio_path: str) -> bool:
        """Remove cached audio analysis for a specific file."""
        cache_path = self._cache_path(audio_path, "analysis")
        try:
            if cache_path.exists():
                cache_path.unlink()
                return True
        except OSError:
            pass
        return False

    def clear_scene_cache(self, video_path: str) -> bool:
        """Remove cached scene analysis for a specific file."""
        cache_path = self._cache_path(video_path, "scenes")
        try:
            if cache_path.exists():
                cache_path.unlink()
                return True
        except OSError:
            pass
        return False


# =============================================================================
# Global Singleton
# =============================================================================

_cache: Optional[AnalysisCache] = None


def get_analysis_cache() -> AnalysisCache:
    """
    Get the global analysis cache instance.

    Configuration via environment variables:
    - CACHE_INVALIDATION_HOURS: TTL in hours (default: 24)
    - DISABLE_ANALYSIS_CACHE: Set to "true" to disable caching
    """
    global _cache
    if _cache is None:
        ttl = int(os.environ.get("CACHE_INVALIDATION_HOURS", str(DEFAULT_TTL_HOURS)))
        enabled = os.environ.get("DISABLE_ANALYSIS_CACHE", "false").lower() != "true"
        _cache = AnalysisCache(ttl_hours=ttl, enabled=enabled)
    return _cache


def reset_cache() -> None:
    """Reset the global cache instance (useful for testing)."""
    global _cache
    _cache = None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "AnalysisCache",
    "AudioAnalysisEntry",
    "SceneAnalysisEntry",
    "CacheEntry",
    "get_analysis_cache",
    "reset_cache",
    "CACHE_VERSION",
    "DEFAULT_TTL_HOURS",
]
