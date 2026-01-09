"""
Test Utilities - Centralized Test Fixtures and Helpers

Provides reusable test data generators, assertion helpers, and fixtures
for consistent testing across the Montage AI codebase.

Usage:
    from montage_ai.test_utils import (
        create_test_scene,
        create_test_clip,
        assert_valid_clip_metadata,
    )
    
    scene = create_test_scene(duration=5.0)
    clip = create_test_clip(scene_id="test_1")
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
from dataclasses import dataclass, field


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_valid_clip_metadata(clip: Dict[str, Any]) -> None:
    """Assert that clip dict has valid metadata structure.
    
    Args:
        clip: Clip dict to validate
        
    Raises:
        AssertionError: If clip is invalid
    """
    assert isinstance(clip, dict), f"Clip must be dict, got {type(clip)}"
    assert "id" in clip, "Clip missing 'id'"
    assert "source" in clip or "source_path" in clip, "Clip missing source"
    assert "start_time" in clip, "Clip missing 'start_time'"
    assert "end_time" in clip, "Clip missing 'end_time'"
    assert clip["end_time"] > clip["start_time"], "Invalid clip time range"


def assert_valid_scene(scene: Dict[str, Any]) -> None:
    """Assert that scene dict has valid structure.
    
    Args:
        scene: Scene dict to validate
        
    Raises:
        AssertionError: If scene is invalid
    """
    assert isinstance(scene, dict), f"Scene must be dict, got {type(scene)}"
    assert "start_time" in scene or "start" in scene, "Scene missing start time"
    assert "end_time" in scene or "end" in scene, "Scene missing end time"


def assert_valid_ffmpeg_filters(filters: str) -> None:
    """Assert that FFmpeg filter string is syntactically valid.
    
    Args:
        filters: FFmpeg filter string
        
    Raises:
        AssertionError: If filter string is invalid
    """
    assert isinstance(filters, str), f"Filters must be string, got {type(filters)}"
    assert len(filters) > 0, "Filter string cannot be empty"
    # Basic syntax checks
    assert filters.count("[") == filters.count("]"), "Mismatched brackets"
    assert "=" in filters or ":" in filters, "Filter missing parameters"


def assert_time_range_valid(start: float, end: float, duration: float = None) -> None:
    """Assert that time range is valid.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        duration: Optional total duration to check against
        
    Raises:
        AssertionError: If range is invalid
    """
    assert start >= 0, f"Start time must be >= 0, got {start}"
    assert end > start, f"End time ({end}) must be > start ({start})"
    if duration:
        assert end <= duration, f"End time ({end}) exceeds duration ({duration})"


# ============================================================================
# Test Data Generators
# ============================================================================

def create_test_scene(
    scene_id: str = "test_scene",
    start_time: float = 0.0,
    duration: float = 5.0,
    description: Optional[str] = None,
    visual_features: Optional[Dict[str, Any]] = None,
    audio_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a test scene dict with sensible defaults.
    
    Args:
        scene_id: Scene identifier
        start_time: Scene start in seconds
        duration: Scene duration in seconds
        description: Optional scene description
        visual_features: Optional visual feature dict
        audio_features: Optional audio feature dict
        
    Returns:
        Test scene dict
    """
    if visual_features is None:
        visual_features = {
            "motion": 0.5,
            "brightness": 0.5,
            "contrast": 0.5,
            "saturation": 0.5,
        }
    
    if audio_features is None:
        audio_features = {
            "loudness_db": -20.0,
            "has_speech": False,
            "has_music": False,
        }
    
    return {
        "id": scene_id,
        "start_time": start_time,
        "end_time": start_time + duration,
        "description": description or f"Test scene {scene_id}",
        "visual_features": visual_features,
        "audio_features": audio_features,
        "metadata": {
            "test": True,
            "source": "test_utils",
        },
    }


def create_test_clip(
    clip_id: str = "test_clip",
    source_path: str = "/data/test/clip.mp4",
    start_time: float = 0.0,
    end_time: float = 5.0,
    scene_id: Optional[str] = None,
    score: float = 0.8,
) -> Dict[str, Any]:
    """Create a test clip dict.
    
    Args:
        clip_id: Clip identifier
        source_path: Path to source file
        start_time: Clip start in seconds
        end_time: Clip end in seconds
        scene_id: Associated scene ID
        score: Quality score [0.0, 1.0]
        
    Returns:
        Test clip dict
    """
    return {
        "id": clip_id,
        "source_path": source_path,
        "source": source_path,  # Backward compat
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "scene_id": scene_id or "default_scene",
        "score": score,
        "metadata": {
            "test": True,
            "source": "test_utils",
        },
    }


def create_test_clips(
    count: int = 5,
    duration: float = 5.0,
    scene_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Create multiple test clips.
    
    Args:
        count: Number of clips to create
        duration: Duration of each clip
        scene_id: Optional scene ID for all clips
        
    Returns:
        List of test clip dicts
    """
    clips = []
    for i in range(count):
        clips.append(
            create_test_clip(
                clip_id=f"test_clip_{i}",
                source_path=f"/data/test/clip_{i}.mp4",
                start_time=i * duration,
                end_time=(i + 1) * duration,
                scene_id=scene_id,
                score=0.5 + (i * 0.1),  # Variable scores
            )
        )
    return clips


def create_test_montage_config() -> Dict[str, Any]:
    """Create a minimal test montage configuration.
    
    Returns:
        Test config dict
    """
    return {
        "style": "dynamic",
        "quality": "preview",
        "output_width": 1920,
        "output_height": 1080,
        "fps": 24,
        "duration": 60,  # seconds
        "stabilize": False,
        "upscale": False,
        "max_transitions": 20,
        "color_grading": True,
        "audio_ducking": True,
    }


def create_test_ffmpeg_config() -> Dict[str, Any]:
    """Create a test FFmpeg configuration.
    
    Returns:
        Test FFmpeg config dict
    """
    return {
        "preset": "ultrafast",
        "crf": 28,
        "width": 1920,
        "height": 1080,
        "fps": 24,
        "bitrate": "5000k",
        "codec": "libx264",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
    }


# ============================================================================
# Fixture Classes
# ============================================================================

@dataclass
class TempFileFixture:
    """Manages temporary files for testing."""
    
    _temp_dir: Optional[Path] = field(default=None, init=False)
    _files: List[Path] = field(default_factory=list, init=False)
    
    def create_temp_file(
        self,
        name: str = "test.txt",
        content: str = "",
        suffix: str = "",
    ) -> Path:
        """Create a temporary file.
        
        Args:
            name: File name
            content: File content
            suffix: File suffix
            
        Returns:
            Path to temp file
        """
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="montage_test_"))
        
        filepath = self._temp_dir / f"{name}{suffix}"
        filepath.write_text(content)
        self._files.append(filepath)
        return filepath
    
    def create_temp_dir(self, name: str = "test_dir") -> Path:
        """Create a temporary directory.
        
        Args:
            name: Directory name
            
        Returns:
            Path to temp directory
        """
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="montage_test_"))
        
        dirpath = self._temp_dir / name
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath
    
    def cleanup(self) -> None:
        """Clean up all created files and directories."""
        import shutil
        for filepath in self._files:
            if filepath.exists():
                if filepath.is_dir():
                    shutil.rmtree(filepath)
                else:
                    filepath.unlink()
        
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


@dataclass
class MockClip:
    """Mock clip for testing."""
    
    id: str = "mock_clip"
    source_path: str = "/test/mock.mp4"
    start_time: float = 0.0
    end_time: float = 5.0
    duration: float = 5.0
    score: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "id": self.id,
            "source_path": self.source_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "score": self.score,
        }


@dataclass
class MockScene:
    """Mock scene for testing."""
    
    id: str = "mock_scene"
    start_time: float = 0.0
    end_time: float = 10.0
    description: str = "Mock scene"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "description": self.description,
        }


# ============================================================================
# Comparison Helpers
# ============================================================================

def compare_clips(
    clip1: Dict[str, Any],
    clip2: Dict[str, Any],
    tolerance: float = 0.01,
) -> bool:
    """Compare two clips for equality within tolerance.
    
    Args:
        clip1: First clip
        clip2: Second clip
        tolerance: Tolerance for float comparisons
        
    Returns:
        True if clips are effectively equal
    """
    # Compare IDs
    if clip1.get("id") != clip2.get("id"):
        return False
    
    # Compare time ranges with tolerance
    for time_key in ["start_time", "end_time"]:
        t1 = float(clip1.get(time_key, 0))
        t2 = float(clip2.get(time_key, 0))
        if abs(t1 - t2) > tolerance:
            return False
    
    # Compare scores with tolerance
    s1 = float(clip1.get("score", 0.5))
    s2 = float(clip2.get("score", 0.5))
    if abs(s1 - s2) > tolerance:
        return False
    
    return True


def compare_scenes(
    scene1: Dict[str, Any],
    scene2: Dict[str, Any],
    tolerance: float = 0.01,
) -> bool:
    """Compare two scenes for equality within tolerance.
    
    Args:
        scene1: First scene
        scene2: Second scene
        tolerance: Tolerance for float comparisons
        
    Returns:
        True if scenes are effectively equal
    """
    # Compare IDs
    if scene1.get("id") != scene2.get("id"):
        return False
    
    # Compare time ranges with tolerance
    for time_key in ["start_time", "end_time"]:
        t1 = float(scene1.get(time_key, 0))
        t2 = float(scene2.get(time_key, 0))
        if abs(t1 - t2) > tolerance:
            return False
    
    return True


__all__ = [
    # Assertions
    "assert_valid_clip_metadata",
    "assert_valid_scene",
    "assert_valid_ffmpeg_filters",
    "assert_time_range_valid",
    # Generators
    "create_test_scene",
    "create_test_clip",
    "create_test_clips",
    "create_test_montage_config",
    "create_test_ffmpeg_config",
    # Fixtures
    "TempFileFixture",
    "MockClip",
    "MockScene",
    # Comparisons
    "compare_clips",
    "compare_scenes",
]
