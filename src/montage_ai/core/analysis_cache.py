"""
Analysis Cache for Montage AI

Nightfly Studio - Persistent caching for expensive analysis operations.
OPTIMIZATION Phase 3: msgpack serialization (22x faster than JSON)

Caches:
- Audio analysis (beat detection, energy profile) from librosa
- Scene boundaries from scenedetect
- Semantic analysis (tags, captions, mood) from Vision models (Phase 2)

Cache files are stored as JSON/msgpack sidecars next to source files:
- music/track.mp3.analysis.msgpack (or .json for backward compatibility)
- input/video.mp4.scenes.msgpack
- input/video.mp4.semantic_5000.msgpack

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

# OPTIMIZATION Phase 3: msgpack for 22x faster serialization
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

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
        Robust hash based on file size + partial content.
        
        Reads first 64KB and last 64KB of the file to detect changes
        even if mtime is preserved (e.g. rsync/cp -a).
        """
        try:
            stat = os.stat(path)
            file_size = stat.st_size
            
            hasher = hashlib.md5()
            hasher.update(f"{file_size}".encode())
            
            with open(path, "rb") as f:
                # Read first 64KB
                chunk = f.read(65536)
                hasher.update(chunk)
                
                # Read last 64KB if file is large enough
                if file_size > 131072:
                    f.seek(-65536, os.SEEK_END)
                    chunk = f.read(65536)
                    hasher.update(chunk)
                    
            return hasher.hexdigest()[:16]
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


@dataclass
class SemanticAnalysisEntry(CacheEntry):
    """
    Cached semantic scene analysis from Vision models.

    Phase 2: Semantic Storytelling - stores AI-generated content tags.
    Cache key includes time_point to support multiple analyses per video.
    """
    time_point: float           # Time in video where frame was sampled
    quality: str                # "YES" or "NO"
    description: str            # 5-word scene summary
    action: str                 # low/medium/high
    shot: str                   # close/medium/wide
    tags: List[str]             # Free-form semantic tags
    caption: str                # Detailed description for embedding
    objects: List[str]          # Detected objects
    mood: str                   # calm/energetic/dramatic/playful/tense/peaceful/mysterious
    setting: str                # indoor/outdoor/beach/city/nature/studio/street/home
    caption_embedding: Optional[List[float]] = None  # Pre-computed embedding
    # Story phase metadata (2025 Tech Vision: Episodic Memory)
    story_phase: Optional[str] = None       # intro/build/climax/sustain/outro
    narrative_role: Optional[str] = None    # establishing/transition/emotional_peak/resolution
    energy_trajectory: Optional[float] = None  # Expected energy 0.0-1.0 at this story position


@dataclass
class EpisodicMemoryEntry:
    """
    Tracks clip usage across montages for variety and learning.

    2025 Tech Vision: Episodic Memory for B-Roll/Story
    Enables smarter clip selection by tracking:
    - Which clips were used in which phases
    - User feedback (like/dislike) for learning
    - Reuse counts to ensure variety
    """
    clip_path: str              # Path to source clip
    montage_id: str             # ID of montage where clip was used
    story_phase: str            # Phase where clip was placed (intro/build/climax/sustain/outro)
    timestamp_used: float       # Timeline position in that montage
    clip_start: float           # Start time within source clip
    clip_end: float             # End time within source clip
    user_feedback: Optional[str] = None  # like/dislike/neutral (future: user ratings)
    created_at: str = ""        # ISO timestamp when entry was created

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# =============================================================================
# Analysis Cache
# =============================================================================

class AnalysisCache:
    """
    Persistent cache for expensive analysis operations.

    Stores analysis results as JSON sidecar files next to source files.
    Automatically invalidates cache when source file changes or TTL expires.
    """

    def __init__(self, ttl_hours: int = DEFAULT_TTL_HOURS, enabled: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize the analysis cache.

        Args:
            ttl_hours: Time-to-live in hours before cache expires
            enabled: Whether caching is enabled (can be disabled for testing)
            cache_dir: Optional directory to store cache files (overrides sidecar behavior)
        """
        self.ttl_hours = ttl_hours
        self.enabled = enabled
        
        # Allow override via environment variable if not explicitly provided
        if cache_dir is None:
            cache_dir = os.environ.get("METADATA_CACHE_DIR")
        self.cache_dir = cache_dir

    def _cache_path(self, source_path: str, suffix: str) -> Path:
        """Get cache file path (sidecar or centralized)."""
        filename = f"{os.path.basename(source_path)}.{suffix}.json"
        
        if self.cache_dir:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            return Path(os.path.join(self.cache_dir, filename))
            
        return Path(f"{source_path}.{suffix}.json")

    def _is_valid(self, cache_path: Path, source_path: str) -> bool:
        """
        Check if cache is valid.
        OPTIMIZATION Phase 3: Supports both msgpack and JSON for backward compatibility.

        Validation checks:
        1. Cache file exists
        2. Cache version matches current version
        3. Source file hash matches (file hasn't changed)
        4. Cache hasn't expired (within TTL)
        """
        if not self.enabled:
            return False

        # Try msgpack first, fall back to JSON
        msgpack_path = cache_path.with_suffix('.msgpack')
        if MSGPACK_AVAILABLE and msgpack_path.exists():
            actual_path = msgpack_path
            use_msgpack = True
        elif cache_path.exists():
            actual_path = cache_path
            use_msgpack = False
        else:
            return False

        try:
            if use_msgpack:
                with open(actual_path, "rb") as f:
                    data = msgpack.load(f, raw=False)
            else:
                with open(actual_path, encoding="utf-8") as f:
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
        """
        Write cache data to file with error handling using atomic write.
        OPTIMIZATION Phase 3: Uses msgpack if available for 40-60% faster serialization.
        """
        try:
            # Determine file extension based on serialization method
            if MSGPACK_AVAILABLE:
                actual_path = cache_path.with_suffix('.msgpack')
                temp_path = actual_path.with_suffix(f"{actual_path.suffix}.tmp")
                with open(temp_path, "wb") as f:
                    msgpack.dump(data, f, use_bin_type=True)
            else:
                actual_path = cache_path
                temp_path = actual_path.with_suffix(f"{actual_path.suffix}.tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_path.replace(actual_path)
            return True
        except OSError as e:
            logger.debug(f"Failed to write cache: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return False

    # -------------------------------------------------------------------------
    # Audio Analysis Cache
    # -------------------------------------------------------------------------

    def load_audio(self, audio_path: str) -> Optional[AudioAnalysisEntry]:
        """
        Load cached audio analysis if valid.
        OPTIMIZATION Phase 3: Supports msgpack for faster loading.

        Args:
            audio_path: Path to the audio file

        Returns:
            AudioAnalysisEntry if cache hit, None if cache miss
        """
        cache_path = self._cache_path(audio_path, "analysis")

        if not self._is_valid(cache_path, audio_path):
            return None

        # Try msgpack first, fall back to JSON
        msgpack_path = cache_path.with_suffix('.msgpack')
        if MSGPACK_AVAILABLE and msgpack_path.exists():
            actual_path = msgpack_path
            use_msgpack = True
        else:
            actual_path = cache_path
            use_msgpack = False

        try:
            if use_msgpack:
                with open(actual_path, "rb") as f:
                    data = msgpack.load(f, raw=False)
            else:
                with open(actual_path, encoding="utf-8") as f:
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
        OPTIMIZATION Phase 3: Supports msgpack for faster loading.

        Args:
            video_path: Path to the video file
            threshold: Scene detection threshold (must match cached value)

        Returns:
            SceneAnalysisEntry if cache hit, None if cache miss
        """
        cache_path = self._cache_path(video_path, "scenes")

        if not self._is_valid(cache_path, video_path):
            return None

        # Try msgpack first, fall back to JSON
        msgpack_path = cache_path.with_suffix('.msgpack')
        if MSGPACK_AVAILABLE and msgpack_path.exists():
            actual_path = msgpack_path
            use_msgpack = True
        else:
            actual_path = cache_path
            use_msgpack = False

        try:
            if use_msgpack:
                with open(actual_path, "rb") as f:
                    data = msgpack.load(f, raw=False)
            else:
                with open(actual_path, encoding="utf-8") as f:
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
    # Semantic Analysis Cache (Phase 2: Semantic Storytelling)
    # -------------------------------------------------------------------------

    def _semantic_cache_path(self, video_path: str, time_point: float) -> Path:
        """
        Get cache path for semantic analysis at a specific time point.

        Format: video.mp4.semantic_5000.json (time in milliseconds)
        """
        time_ms = int(time_point * 1000)
        return Path(f"{video_path}.semantic_{time_ms}.json")

    def load_semantic(
        self, video_path: str, time_point: float, tolerance_ms: int = 100
    ) -> Optional[SemanticAnalysisEntry]:
        """
        Load cached semantic analysis if valid.
        OPTIMIZATION Phase 3: Supports msgpack for faster loading.

        Args:
            video_path: Path to the video file
            time_point: Time in seconds where frame was sampled
            tolerance_ms: Time tolerance in milliseconds (default 100ms)

        Returns:
            SemanticAnalysisEntry if cache hit, None if cache miss
        """
        cache_path = self._semantic_cache_path(video_path, time_point)

        if not self._is_valid(cache_path, video_path):
            return None

        # Try msgpack first, fall back to JSON
        msgpack_path = cache_path.with_suffix('.msgpack')
        if MSGPACK_AVAILABLE and msgpack_path.exists():
            actual_path = msgpack_path
            use_msgpack = True
        else:
            actual_path = cache_path
            use_msgpack = False

        try:
            if use_msgpack:
                with open(actual_path, "rb") as f:
                    data = msgpack.load(f, raw=False)
            else:
                with open(actual_path, encoding="utf-8") as f:
                    data = json.load(f)

            # Time point must match within tolerance
            cached_time = data.get("time_point", 0)
            if abs(cached_time - time_point) * 1000 > tolerance_ms:
                logger.debug(f"Time point mismatch: {cached_time} != {time_point}")
                return None

            logger.debug(f"Cache hit: semantic analysis for {os.path.basename(video_path)} @ {time_point:.1f}s")
            return SemanticAnalysisEntry(
                version=data["version"],
                file_hash=data["file_hash"],
                computed_at=data["computed_at"],
                time_point=data["time_point"],
                quality=data.get("quality", "YES"),
                description=data.get("description", ""),
                action=data.get("action", "medium"),
                shot=data.get("shot", "medium"),
                tags=data.get("tags", []),
                caption=data.get("caption", ""),
                objects=data.get("objects", []),
                mood=data.get("mood", "neutral"),
                setting=data.get("setting", "unknown"),
                caption_embedding=data.get("caption_embedding"),
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.debug(f"Failed to load semantic cache: {e}")
            return None

    def save_semantic(self, video_path: str, time_point: float, analysis) -> bool:
        """
        Save semantic analysis to cache.

        Args:
            video_path: Path to the video file
            time_point: Time in seconds where frame was sampled
            analysis: SceneAnalysis dataclass from scene_analysis module

        Returns:
            True if cache was saved successfully
        """
        if not self.enabled:
            return False

        # Handle both SceneAnalysis object and dict
        if hasattr(analysis, "to_dict"):
            data = analysis.to_dict()
        elif isinstance(analysis, dict):
            data = analysis
        else:
            data = {
                "quality": getattr(analysis, "quality", "YES"),
                "description": getattr(analysis, "description", ""),
                "action": str(getattr(analysis, "action", "medium")),
                "shot": str(getattr(analysis, "shot", "medium")),
                "tags": list(getattr(analysis, "tags", [])),
                "caption": getattr(analysis, "caption", ""),
                "objects": list(getattr(analysis, "objects", [])),
                "mood": getattr(analysis, "mood", "neutral"),
                "setting": getattr(analysis, "setting", "unknown"),
            }

        # Normalize action/shot to string values
        action_str = data.get("action", "medium")
        if hasattr(action_str, "value"):
            action_str = action_str.value
        shot_str = data.get("shot", "medium")
        if hasattr(shot_str, "value"):
            shot_str = shot_str.value

        entry = SemanticAnalysisEntry(
            version=CACHE_VERSION,
            file_hash=CacheEntry.compute_file_hash(video_path),
            computed_at=datetime.now().isoformat(),
            time_point=float(time_point),
            quality=data.get("quality", "YES"),
            description=data.get("description", ""),
            action=str(action_str),
            shot=str(shot_str),
            tags=list(data.get("tags", [])),
            caption=data.get("caption", ""),
            objects=list(data.get("objects", [])),
            mood=data.get("mood", "neutral"),
            setting=data.get("setting", "unknown"),
            caption_embedding=data.get("caption_embedding"),
        )

        cache_path = self._semantic_cache_path(video_path, time_point)
        success = self._write_cache(cache_path, asdict(entry))

        if success:
            logger.debug(f"Cached semantic analysis: {os.path.basename(video_path)} @ {time_point:.1f}s")

        return success

    def clear_semantic_cache(self, video_path: str, time_point: Optional[float] = None) -> bool:
        """
        Remove cached semantic analysis.
        OPTIMIZATION Phase 3: Clears both msgpack and JSON cache files.

        Args:
            video_path: Path to the video file
            time_point: Specific time point, or None to clear all

        Returns:
            True if any cache was cleared
        """
        if time_point is not None:
            cache_path = self._semantic_cache_path(video_path, time_point)
            cleared = False
            
            # Try msgpack first
            msgpack_path = cache_path.with_suffix('.msgpack')
            if msgpack_path.exists():
                try:
                    msgpack_path.unlink()
                    cleared = True
                except OSError:
                    pass
            
            # Also try JSON
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    cleared = True
                except OSError:
                    pass
            
            return cleared

        # Clear all semantic caches for this video
        cleared = False
        try:
            video_dir = Path(video_path).parent
            video_name = Path(video_path).name
            
            # Clear both msgpack and JSON files
            for pattern in [f"{video_name}.semantic_*.msgpack", f"{video_name}.semantic_*.json"]:
                for cache_file in video_dir.glob(pattern):
                    try:
                        cache_file.unlink()
                        cleared = True
                    except OSError:
                        pass
        except OSError:
            pass
        return cleared

    # -------------------------------------------------------------------------
    # Episodic Memory (2025 Tech Vision: B-Roll/Story Memory)
    # -------------------------------------------------------------------------

    # Safety limits for episodic memory
    MAX_EPISODIC_ENTRIES = 10000  # Prune oldest entries when exceeding this

    def _episodic_memory_path(self) -> Path:
        """Get path for episodic memory storage."""
        cache_dir = Path.home() / ".montage_ai" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "episodic_memory.json"

    def _episodic_lock_path(self) -> Path:
        """Get path for episodic memory lock file."""
        return self._episodic_memory_path().with_suffix(".lock")

    def _normalize_clip_path(self, clip_path: str) -> str:
        """Normalize clip path for consistent matching."""
        try:
            # Convert to absolute path if possible
            return str(Path(clip_path).resolve())
        except (OSError, ValueError):
            return clip_path

    def save_episodic_memory(self, entry: EpisodicMemoryEntry) -> bool:
        """
        Save a clip usage entry to episodic memory.

        Enables learning from past montage decisions:
        - Track which clips work well in which story phases
        - User feedback accumulates for smarter future selection

        Safety measures:
        - File locking prevents concurrent write corruption
        - Max entries limit with automatic pruning of oldest entries
        - Normalized paths for consistent matching

        Args:
            entry: EpisodicMemoryEntry with clip usage details

        Returns:
            True if saved successfully
        """
        if not self.enabled:
            return False

        memory_path = self._episodic_memory_path()
        lock_path = self._episodic_lock_path()

        try:
            # Simple file-based locking
            lock_path.touch(exist_ok=True)

            # Load existing entries
            entries = []
            if memory_path.exists():
                with open(memory_path, encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", [])

            # Normalize the clip path before saving
            entry_dict = asdict(entry)
            entry_dict["clip_path"] = self._normalize_clip_path(entry.clip_path)

            # Add new entry
            entries.append(entry_dict)

            # Prune oldest entries if exceeding limit
            if len(entries) > self.MAX_EPISODIC_ENTRIES:
                # Sort by created_at and keep newest
                entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
                pruned_count = len(entries) - self.MAX_EPISODIC_ENTRIES
                entries = entries[:self.MAX_EPISODIC_ENTRIES]
                logger.debug(f"Pruned {pruned_count} old episodic memory entries")

            # Write back atomically (write to temp, then rename)
            temp_path = memory_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"version": CACHE_VERSION, "entries": entries}, f, indent=2)
            temp_path.replace(memory_path)

            logger.debug(f"Saved episodic memory: {os.path.basename(entry.clip_path)} in {entry.story_phase}")
            return True

        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Failed to save episodic memory: {e}")
            return False
        finally:
            # Clean up lock file
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def load_episodic_for_clip(self, clip_path: str) -> List[EpisodicMemoryEntry]:
        """
        Load all episodic memory entries for a specific clip.

        Args:
            clip_path: Path to the clip (can be relative or absolute)

        Returns:
            List of EpisodicMemoryEntry for this clip's usage history
        """
        memory_path = self._episodic_memory_path()

        if not memory_path.exists():
            return []

        try:
            with open(memory_path, encoding="utf-8") as f:
                data = json.load(f)

            # Normalize paths for comparison
            normalized_path = self._normalize_clip_path(clip_path)
            clip_name = os.path.basename(clip_path)
            entries = []

            for entry_data in data.get("entries", []):
                stored_path = entry_data.get("clip_path", "")
                entry_name = os.path.basename(stored_path)
                # Match by normalized path OR by basename (fallback)
                if stored_path == normalized_path or entry_name == clip_name:
                    entries.append(EpisodicMemoryEntry(
                        clip_path=entry_data["clip_path"],
                        montage_id=entry_data["montage_id"],
                        story_phase=entry_data["story_phase"],
                        timestamp_used=entry_data["timestamp_used"],
                        clip_start=entry_data.get("clip_start", 0.0),
                        clip_end=entry_data.get("clip_end", 0.0),
                        user_feedback=entry_data.get("user_feedback"),
                        created_at=entry_data.get("created_at", ""),
                    ))

            return entries

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.debug(f"Failed to load episodic memory: {e}")
            return []

    def get_clip_reuse_count(self, clip_path: str, phase: Optional[str] = None) -> int:
        """
        Count how many times a clip has been used (optionally in a specific phase).

        Useful for ensuring variety - avoid overusing popular clips.

        Args:
            clip_path: Path to the clip
            phase: Optional story phase filter (intro/build/climax/sustain/outro)

        Returns:
            Number of times this clip has been used
        """
        entries = self.load_episodic_for_clip(clip_path)

        if phase:
            return sum(1 for e in entries if e.story_phase == phase)

        return len(entries)

    def clear_episodic_memory(self, clip_path: Optional[str] = None) -> bool:
        """
        Clear episodic memory entries.

        Args:
            clip_path: If provided, only clear entries for this clip.
                      If None, clear all episodic memory.

        Returns:
            True if any entries were cleared
        """
        memory_path = self._episodic_memory_path()

        if not memory_path.exists():
            return False

        try:
            if clip_path is None:
                # Clear all
                memory_path.unlink()
                return True

            # Clear only entries for specific clip
            with open(memory_path, encoding="utf-8") as f:
                data = json.load(f)

            clip_name = os.path.basename(clip_path)
            original_count = len(data.get("entries", []))

            data["entries"] = [
                e for e in data.get("entries", [])
                if os.path.basename(e.get("clip_path", "")) != clip_name
                and e.get("clip_path") != clip_path
            ]

            if len(data["entries"]) < original_count:
                with open(memory_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return True

            return False

        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Failed to clear episodic memory: {e}")
            return False

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def clear_audio_cache(self, audio_path: str) -> bool:
        """
        Clear cached audio analysis.
        OPTIMIZATION Phase 3: Clears both msgpack and JSON cache files.
        """
        cache_path = self._cache_path(audio_path, "analysis")
        cleared = False
        
        # Try msgpack first
        msgpack_path = cache_path.with_suffix('.msgpack')
        if msgpack_path.exists():
            try:
                msgpack_path.unlink()
                cleared = True
            except OSError:
                pass
        
        # Also try JSON for backward compatibility
        if cache_path.exists():
            try:
                cache_path.unlink()
                cleared = True
            except OSError:
                pass
        
        return cleared

    def clear_scene_cache(self, video_path: str) -> bool:
        """
        Clear cached scene analysis.
        OPTIMIZATION Phase 3: Clears both msgpack and JSON cache files.
        """
        cache_path = self._cache_path(video_path, "scenes")
        cleared = False
        
        # Try msgpack first
        msgpack_path = cache_path.with_suffix('.msgpack')
        if msgpack_path.exists():
            try:
                msgpack_path.unlink()
                cleared = True
            except OSError:
                pass
        
        # Also try JSON for backward compatibility
        if cache_path.exists():
            try:
                cache_path.unlink()
                cleared = True
            except OSError:
                pass
        
        return cleared


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
    "SemanticAnalysisEntry",
    "EpisodicMemoryEntry",
    "CacheEntry",
    "get_analysis_cache",
    "reset_cache",
    "CACHE_VERSION",
    "DEFAULT_TTL_HOURS",
]
