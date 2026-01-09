"""
Tests for analysis cache module.

Tests caching of audio analysis and scene detection results.
Nightfly Studio - Montage AI
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import numpy as np

from src.montage_ai.audio_analysis_objects import BeatInfo, EnergyProfile

from src.montage_ai.core.analysis_cache import (
    AnalysisCache,
    AudioAnalysisEntry,
    SceneAnalysisEntry,
    SemanticAnalysisEntry,
    CacheEntry,
    get_analysis_cache,
    reset_cache,
    CACHE_VERSION,
    DEFAULT_TTL_HOURS,
)


class TestCacheEntry:
    """Tests for CacheEntry base class."""

    def test_compute_file_hash_returns_string(self, tmp_path):
        """compute_file_hash returns 16-char hex string."""
        test_file = tmp_path / "test.mp3"
        test_file.write_text("test content")

        hash_val = CacheEntry.compute_file_hash(str(test_file))

        assert isinstance(hash_val, str)
        assert len(hash_val) == 16
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_compute_file_hash_changes_with_content(self, tmp_path):
        """Hash changes when file content changes."""
        test_file = tmp_path / "test.mp3"

        test_file.write_text("content1")
        hash1 = CacheEntry.compute_file_hash(str(test_file))

        test_file.write_text("content2")
        hash2 = CacheEntry.compute_file_hash(str(test_file))

        assert hash1 != hash2

    def test_compute_file_hash_nonexistent_file(self):
        """Hash returns empty string for nonexistent file."""
        hash_val = CacheEntry.compute_file_hash("/nonexistent/file.mp3")
        assert hash_val == ""


class TestAudioAnalysisEntry:
    """Tests for AudioAnalysisEntry dataclass."""

    def test_create_entry(self):
        """AudioAnalysisEntry can be created with all fields."""
        entry = AudioAnalysisEntry(
            version="1.0",
            file_hash="abc123",
            computed_at="2024-01-01T12:00:00",
            tempo=120.0,
            beat_times=[0.5, 1.0, 1.5],
            energy_times=[0.0, 0.5, 1.0],
            energy_values=[0.2, 0.8, 0.5],
            duration=60.0,
            sample_rate=22050,
        )

        assert entry.tempo == 120.0
        assert entry.beat_times == [0.5, 1.0, 1.5]
        assert entry.duration == 60.0


class TestSceneAnalysisEntry:
    """Tests for SceneAnalysisEntry dataclass."""

    def test_create_entry(self):
        """SceneAnalysisEntry can be created with all fields."""
        entry = SceneAnalysisEntry(
            version="1.0",
            file_hash="def456",
            computed_at="2024-01-01T12:00:00",
            threshold=30.0,
            scenes=[{"start": 0.0, "end": 5.0}, {"start": 5.0, "end": 10.0}],
            total_scenes=2,
        )

        assert entry.threshold == 30.0
        assert len(entry.scenes) == 2
        assert entry.total_scenes == 2


class TestAnalysisCache:
    """Tests for AnalysisCache class."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache instance."""
        return AnalysisCache(ttl_hours=24, enabled=True)

    @pytest.fixture
    def disabled_cache(self):
        """Create a disabled cache instance."""
        return AnalysisCache(ttl_hours=24, enabled=False)

    @pytest.fixture
    def audio_file(self, tmp_path):
        """Create a test audio file."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio content")
        return str(audio_path)

    @pytest.fixture
    def video_file(self, tmp_path):
        """Create a test video file."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        return str(video_path)

    # -------------------------------------------------------------------------
    # Audio Cache Tests
    # -------------------------------------------------------------------------

    def test_save_and_load_audio(self, cache, audio_file):
        """Audio analysis can be saved and loaded."""
        # Create real beat_info and energy_profile
        beat_info = BeatInfo(
            tempo=120.0,
            beat_times=np.array([0.5, 1.0, 1.5]),
            duration=60.0,
            sample_rate=22050,
            confidence=0.95
        )

        energy_profile = EnergyProfile(
            times=np.array([0.0, 0.5, 1.0]),
            rms=np.array([0.2, 0.8, 0.5]),
            duration=60.0,
            peaks=np.array([1.0, 1.5])
        )

        # Save
        success = cache.save_audio(audio_file, beat_info, energy_profile)
        assert success is True

        # Load
        loaded = cache.load_audio(audio_file)
        assert loaded is not None
        assert loaded.tempo == 120.0
        assert loaded.beat_times == [0.5, 1.0, 1.5]
        assert loaded.duration == 60.0

    def test_load_audio_cache_miss(self, cache, audio_file):
        """load_audio returns None on cache miss."""
        result = cache.load_audio(audio_file)
        assert result is None

    def test_load_audio_disabled_cache(self, disabled_cache, audio_file):
        """Disabled cache always returns None."""
        result = disabled_cache.load_audio(audio_file)
        assert result is None

    def test_audio_cache_invalidation_on_file_change(self, cache, audio_file):
        """Cache is invalidated when source file changes."""
        # Create and save cache
        beat_info = BeatInfo(
            tempo=120.0,
            beat_times=np.array([0.5]),
            duration=30.0,
            sample_rate=22050,
            confidence=0.95
        )

        energy_profile = EnergyProfile(
            times=np.array([0.0]),
            rms=np.array([0.5]),
            duration=30.0,
            peaks=np.array([])
        )

        assert cache.save_audio(audio_file, beat_info, energy_profile) is True

        # Verify cache hit
        assert cache.load_audio(audio_file) is not None

        # Modify source file
        with open(audio_file, "wb") as f:
            f.write(b"modified audio content")

        # Cache should be invalidated
        assert cache.load_audio(audio_file) is None

    def test_audio_cache_invalidation_on_ttl_expire(self, tmp_path):
        """Cache is invalidated after TTL expires."""
        cache = AnalysisCache(ttl_hours=1, enabled=True)
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"audio")

        # Create cache file with old timestamp
        cache_path = Path(f"{audio_file}.analysis.json")
        old_time = datetime.now() - timedelta(hours=2)

        cache_data = {
            "version": CACHE_VERSION,
            "file_hash": CacheEntry.compute_file_hash(str(audio_file)),
            "computed_at": old_time.isoformat(),
            "tempo": 100.0,
            "beat_times": [],
            "energy_times": [],
            "energy_values": [],
            "duration": 30.0,
            "sample_rate": 22050,
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        # Cache should be expired
        assert cache.load_audio(str(audio_file)) is None

    def test_audio_cache_invalidation_on_version_mismatch(self, cache, audio_file):
        """Cache is invalidated on version mismatch."""
        cache_path = Path(f"{audio_file}.analysis.json")

        cache_data = {
            "version": "0.9",  # Old version
            "file_hash": CacheEntry.compute_file_hash(audio_file),
            "computed_at": datetime.now().isoformat(),
            "tempo": 100.0,
            "beat_times": [],
            "energy_times": [],
            "energy_values": [],
            "duration": 30.0,
            "sample_rate": 22050,
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        # Cache should be invalid due to version mismatch
        assert cache.load_audio(audio_file) is None

    # -------------------------------------------------------------------------
    # Scene Cache Tests
    # -------------------------------------------------------------------------

    def test_save_and_load_scenes(self, cache, video_file):
        """Scene analysis can be saved and loaded."""
        # Create scenes as tuples (like detect_scenes returns)
        scenes = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]

        # Save
        success = cache.save_scenes(video_file, threshold=30.0, scenes=scenes)
        assert success is True

        # Load
        loaded = cache.load_scenes(video_file, threshold=30.0)
        assert loaded is not None
        assert loaded.total_scenes == 3
        assert loaded.scenes[0] == {"start": 0.0, "end": 5.0}

    def test_save_scenes_with_scene_objects(self, cache, video_file):
        """save_scenes works with Scene objects (with start/end attrs)."""
        scene1 = MagicMock()
        scene1.start = 0.0
        scene1.end = 5.0

        scene2 = MagicMock()
        scene2.start = 5.0
        scene2.end = 10.0

        scenes = [scene1, scene2]

        success = cache.save_scenes(video_file, threshold=30.0, scenes=scenes)
        assert success is True

        loaded = cache.load_scenes(video_file, threshold=30.0)
        assert loaded.total_scenes == 2

    def test_load_scenes_cache_miss(self, cache, video_file):
        """load_scenes returns None on cache miss."""
        result = cache.load_scenes(video_file, threshold=30.0)
        assert result is None

    def test_scene_cache_threshold_mismatch(self, cache, video_file):
        """Cache returns None if threshold doesn't match."""
        scenes = [(0.0, 5.0)]
        cache.save_scenes(video_file, threshold=30.0, scenes=scenes)

        # Different threshold should miss
        result = cache.load_scenes(video_file, threshold=50.0)
        assert result is None

    def test_scene_cache_threshold_tolerance(self, cache, video_file):
        """Cache allows small threshold differences (within 0.1)."""
        scenes = [(0.0, 5.0)]
        cache.save_scenes(video_file, threshold=30.0, scenes=scenes)

        # Small difference should still hit
        result = cache.load_scenes(video_file, threshold=30.05)
        assert result is not None

    def test_scene_cache_invalidation_on_file_change(self, cache, video_file):
        """Scene cache is invalidated when video file changes."""
        scenes = [(0.0, 5.0)]
        cache.save_scenes(video_file, threshold=30.0, scenes=scenes)

        # Verify cache hit
        assert cache.load_scenes(video_file, threshold=30.0) is not None

        # Modify source file
        with open(video_file, "wb") as f:
            f.write(b"modified video content")

        # Cache should be invalidated
        assert cache.load_scenes(video_file, threshold=30.0) is None

    # -------------------------------------------------------------------------
    # Cache Management Tests
    # -------------------------------------------------------------------------

    def test_clear_audio_cache(self, cache, audio_file):
        """clear_audio_cache removes cache file."""
        beat_info = BeatInfo(
            tempo=120.0,
            beat_times=np.array([0.5]),
            duration=30.0,
            sample_rate=22050,
            confidence=0.95
        )

        energy_profile = EnergyProfile(
            times=np.array([0.0]),
            rms=np.array([0.5]),
            duration=30.0,
            peaks=np.array([])
        )

        cache.save_audio(audio_file, beat_info, energy_profile)
        assert cache.load_audio(audio_file) is not None

        # Clear cache
        result = cache.clear_audio_cache(audio_file)
        assert result is True
        assert cache.load_audio(audio_file) is None

    def test_clear_scene_cache(self, cache, video_file):
        """clear_scene_cache removes cache file."""
        scenes = [(0.0, 5.0)]
        cache.save_scenes(video_file, threshold=30.0, scenes=scenes)
        assert cache.load_scenes(video_file, threshold=30.0) is not None

        # Clear cache
        result = cache.clear_scene_cache(video_file)
        assert result is True
        assert cache.load_scenes(video_file, threshold=30.0) is None

    def test_clear_nonexistent_cache(self, cache, audio_file):
        """Clearing nonexistent cache returns False."""
        result = cache.clear_audio_cache(audio_file)
        assert result is False

    # -------------------------------------------------------------------------
    # Disabled Cache Tests
    # -------------------------------------------------------------------------

    def test_disabled_cache_never_saves(self, disabled_cache, audio_file):
        """Disabled cache never saves data."""
        beat_info = BeatInfo(
            tempo=120.0,
            beat_times=np.array([]),
            duration=30.0,
            sample_rate=22050,
            confidence=0.95
        )

        energy_profile = EnergyProfile(
            times=np.array([]),
            rms=np.array([]),
            duration=30.0,
            peaks=np.array([])
        )

        result = disabled_cache.save_audio(audio_file, beat_info, energy_profile)
        assert result is False

        # Cache file should not exist
        cache_path = Path(f"{audio_file}.analysis.json")
        assert not cache_path.exists()


class TestGlobalCache:
    """Tests for global cache singleton."""

    def test_get_analysis_cache_returns_singleton(self):
        """get_analysis_cache returns same instance."""
        reset_cache()

        cache1 = get_analysis_cache()
        cache2 = get_analysis_cache()

        assert cache1 is cache2

    def test_reset_cache_clears_singleton(self):
        """reset_cache clears the global instance."""
        cache1 = get_analysis_cache()
        reset_cache()
        cache2 = get_analysis_cache()

        assert cache1 is not cache2

    def test_cache_respects_env_ttl(self):
        """Cache uses CACHE_INVALIDATION_HOURS env var."""
        reset_cache()

        with patch.dict(os.environ, {"CACHE_INVALIDATION_HOURS": "48"}):
            cache = get_analysis_cache()
            assert cache.ttl_hours == 48

        reset_cache()

    def test_cache_respects_disable_env(self):
        """Cache is disabled via DISABLE_ANALYSIS_CACHE env var."""
        reset_cache()

        with patch.dict(os.environ, {"DISABLE_ANALYSIS_CACHE": "true"}):
            cache = get_analysis_cache()
            assert cache.enabled is False

        reset_cache()


class TestCacheRobustness:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def cache(self):
        return AnalysisCache(ttl_hours=24, enabled=True)

    def test_corrupted_cache_file(self, cache, tmp_path):
        """Corrupted cache file is handled gracefully."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"audio")

        cache_path = Path(f"{audio_file}.analysis.json")
        cache_path.write_text("not valid json {{{")

        # Should return None instead of crashing
        result = cache.load_audio(str(audio_file))
        assert result is None

    def test_missing_fields_in_cache(self, cache, tmp_path):
        """Cache with missing fields is handled gracefully."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"audio")

        cache_path = Path(f"{audio_file}.analysis.json")
        cache_data = {
            "version": CACHE_VERSION,
            # Missing other required fields
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        # Should return None instead of crashing
        result = cache.load_audio(str(audio_file))
        assert result is None

    def test_write_permission_error(self, cache, tmp_path):
        """Write permission error is handled gracefully."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"audio")

        beat_info = BeatInfo(
            tempo=120.0,
            beat_times=np.array([]),
            duration=30.0,
            sample_rate=22050,
            confidence=0.95
        )

        energy_profile = EnergyProfile(
            times=np.array([]),
            rms=np.array([]),
            duration=30.0,
            peaks=np.array([])
        )

        # Mock write to fail
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = cache.save_audio(str(audio_file), beat_info, energy_profile)
            assert result is False


class TestSemanticAnalysisEntry:
    """Tests for SemanticAnalysisEntry dataclass (Phase 2)."""

    def test_create_entry(self):
        """SemanticAnalysisEntry can be created with all fields."""
        entry = SemanticAnalysisEntry(
            version="1.0",
            file_hash="abc123",
            computed_at="2024-01-01T12:00:00",
            time_point=5.0,
            quality="YES",
            description="Beach surfing scene",
            action="high",
            shot="wide",
            tags=["beach", "surfing", "ocean"],
            caption="Person surfing on ocean waves",
            objects=["person", "surfboard", "wave"],
            mood="energetic",
            setting="beach",
            caption_embedding=None,
        )

        assert entry.time_point == 5.0
        assert entry.quality == "YES"
        assert entry.tags == ["beach", "surfing", "ocean"]
        assert entry.mood == "energetic"
        assert entry.setting == "beach"


class TestSemanticCache:
    """Tests for semantic analysis caching (Phase 2)."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache instance."""
        return AnalysisCache(ttl_hours=24, enabled=True)

    @pytest.fixture
    def video_file(self, tmp_path):
        """Create a test video file."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        return str(video_path)

    def test_save_and_load_semantic(self, cache, video_file):
        """Semantic analysis can be saved and loaded."""
        # Create mock analysis (similar to SceneAnalysis)
        analysis = MagicMock()
        analysis.quality = "YES"
        analysis.description = "Beach scene"
        analysis.action = "high"
        analysis.shot = "wide"
        analysis.tags = ["beach", "ocean"]
        analysis.caption = "Person on beach"
        analysis.objects = ["person", "sand"]
        analysis.mood = "calm"
        analysis.setting = "beach"
        analysis.to_dict = MagicMock(return_value={
            "quality": "YES",
            "description": "Beach scene",
            "action": "high",
            "shot": "wide",
            "tags": ["beach", "ocean"],
            "caption": "Person on beach",
            "objects": ["person", "sand"],
            "mood": "calm",
            "setting": "beach",
        })

        # Save
        success = cache.save_semantic(video_file, time_point=5.0, analysis=analysis)
        assert success is True

        # Load
        loaded = cache.load_semantic(video_file, time_point=5.0)
        assert loaded is not None
        assert loaded.time_point == 5.0
        assert loaded.quality == "YES"
        assert loaded.tags == ["beach", "ocean"]
        assert loaded.mood == "calm"

    def test_save_semantic_with_dict(self, cache, video_file):
        """save_semantic works with dict analysis."""
        analysis = {
            "quality": "YES",
            "description": "City street",
            "action": "medium",
            "shot": "medium",
            "tags": ["city", "street"],
            "caption": "Urban scene",
            "objects": ["building"],
            "mood": "neutral",
            "setting": "city",
        }

        success = cache.save_semantic(video_file, time_point=10.0, analysis=analysis)
        assert success is True

        loaded = cache.load_semantic(video_file, time_point=10.0)
        assert loaded is not None
        assert loaded.setting == "city"

    def test_load_semantic_cache_miss(self, cache, video_file):
        """load_semantic returns None on cache miss."""
        result = cache.load_semantic(video_file, time_point=5.0)
        assert result is None

    def test_semantic_cache_time_tolerance(self, cache, video_file):
        """Cache validates time_point matches within tolerance."""
        analysis = {"tags": ["test"], "mood": "neutral"}
        cache.save_semantic(video_file, time_point=5.0, analysis=analysis)

        # Exact match works
        result = cache.load_semantic(video_file, time_point=5.0)
        assert result is not None
        assert result.time_point == 5.0

        # Note: Cache filename includes time in ms, so different time points
        # will look for different files. The tolerance check is for verifying
        # the cached time_point matches the requested one, not file lookup.
        result2 = cache.load_semantic(video_file, time_point=5.5)
        assert result2 is None  # Different file, cache miss

    def test_semantic_cache_invalidation_on_file_change(self, cache, video_file):
        """Semantic cache is invalidated when video file changes."""
        analysis = {"tags": ["test"], "mood": "neutral"}
        cache.save_semantic(video_file, time_point=5.0, analysis=analysis)

        # Verify cache hit
        assert cache.load_semantic(video_file, time_point=5.0) is not None

        # Modify source file
        with open(video_file, "wb") as f:
            f.write(b"modified video content")

        # Cache should be invalidated
        assert cache.load_semantic(video_file, time_point=5.0) is None

    def test_clear_semantic_cache_single(self, cache, video_file):
        """clear_semantic_cache removes specific time point cache."""
        analysis = {"tags": ["test"]}
        cache.save_semantic(video_file, time_point=5.0, analysis=analysis)
        cache.save_semantic(video_file, time_point=10.0, analysis=analysis)

        # Clear only 5.0s
        result = cache.clear_semantic_cache(video_file, time_point=5.0)
        assert result is True

        # 5.0s cleared, 10.0s still exists
        assert cache.load_semantic(video_file, time_point=5.0) is None
        assert cache.load_semantic(video_file, time_point=10.0) is not None

    def test_clear_semantic_cache_all(self, cache, video_file):
        """clear_semantic_cache removes all semantic caches for video."""
        analysis = {"tags": ["test"]}
        cache.save_semantic(video_file, time_point=5.0, analysis=analysis)
        cache.save_semantic(video_file, time_point=10.0, analysis=analysis)
        cache.save_semantic(video_file, time_point=15.0, analysis=analysis)

        # Clear all
        result = cache.clear_semantic_cache(video_file, time_point=None)
        assert result is True

        # All cleared
        assert cache.load_semantic(video_file, time_point=5.0) is None
        assert cache.load_semantic(video_file, time_point=10.0) is None
        assert cache.load_semantic(video_file, time_point=15.0) is None

    def test_semantic_cache_path_format(self, cache, video_file):
        """Semantic cache path uses correct format."""
        path = cache._semantic_cache_path(video_file, time_point=5.5)
        assert str(path).endswith(".semantic_5500.json")

        path2 = cache._semantic_cache_path(video_file, time_point=10.0)
        assert str(path2).endswith(".semantic_10000.json")

    def test_disabled_semantic_cache(self, video_file):
        """Disabled cache never saves semantic data."""
        disabled_cache = AnalysisCache(ttl_hours=24, enabled=False)
        analysis = {"tags": ["test"]}

        result = disabled_cache.save_semantic(video_file, time_point=5.0, analysis=analysis)
        assert result is False

        # Cache file should not exist
        from pathlib import Path
        cache_path = Path(f"{video_file}.semantic_5000.json")
        assert not cache_path.exists()
