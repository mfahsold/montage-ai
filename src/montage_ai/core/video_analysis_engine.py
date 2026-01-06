"""
Video Analysis Engine - Unified Single-Pass Video Analysis

.. deprecated:: 2026.1
    This module is deprecated. Use `scene_analysis.py` for scene detection
    and `scene_provider.py` for unified analysis access.

    Migration:
        # Old way (deprecated)
        from montage_ai.core.video_analysis_engine import VideoAnalysisEngine
        engine = VideoAnalysisEngine(video_path)
        results = engine.analyze()

        # New way (preferred)
        from montage_ai.core.scene_provider import get_scene_provider
        provider = get_scene_provider()
        scenes = provider.detect_scenes(video_path)

Provides a pluggable architecture for video analysis:
- Single video decode pass (efficient I/O)
- Pluggable analyzers (scene detection, face tracking, motion, etc.)
- Shared frame buffering and caching
- Unified result aggregation
"""

import warnings

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import cv2
import numpy as np

from ..logger import logger
from ..ffmpeg_utils import build_ffprobe_cmd
from ..core.cmd_runner import run_command


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class VideoMetadata:
    """Video file metadata."""
    path: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    codec: str

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def is_vertical(self) -> bool:
        return self.aspect_ratio < 1.0

    @property
    def is_horizontal(self) -> bool:
        return self.aspect_ratio > 1.0


@dataclass
class FrameContext:
    """Context for a single frame during analysis."""
    frame_idx: int
    timestamp: float  # seconds
    frame: np.ndarray  # BGR image
    metadata: VideoMetadata

    @property
    def rgb(self) -> np.ndarray:
        """Get RGB version of frame (cached)."""
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)


@dataclass
class AnalysisResult:
    """Result from a single analyzer."""
    analyzer_id: str
    data: Any
    frame_count: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyzer_id": self.analyzer_id,
            "data": self.data,
            "frame_count": self.frame_count,
            "processing_time_ms": self.processing_time_ms
        }


# =============================================================================
# Analyzer Plugin Interface
# =============================================================================

class AnalyzerPlugin(ABC):
    """
    Base class for video analysis plugins.

    Plugins process frames sequentially and accumulate results.
    The engine handles frame decoding and plugin orchestration.
    """

    @property
    @abstractmethod
    def analyzer_id(self) -> str:
        """Unique identifier for this analyzer."""
        pass

    @property
    def sample_rate(self) -> int:
        """
        Process every Nth frame (1 = every frame, 5 = every 5th frame).

        Override for performance optimization on expensive analyzers.
        """
        return 1

    @property
    def requires_rgb(self) -> bool:
        """Whether this analyzer needs RGB frames (vs BGR)."""
        return False

    def on_start(self, metadata: VideoMetadata) -> None:
        """Called before analysis begins. Setup resources here."""
        pass

    @abstractmethod
    def process_frame(self, ctx: FrameContext) -> Optional[Any]:
        """
        Process a single frame.

        Args:
            ctx: Frame context with frame data and metadata

        Returns:
            Optional result for this frame (None to skip)
        """
        pass

    def on_end(self) -> Any:
        """
        Called after all frames processed.

        Returns:
            Final aggregated result for this analyzer
        """
        pass


# =============================================================================
# Built-in Analyzers
# =============================================================================

class SceneChangeAnalyzer(AnalyzerPlugin):
    """
    Detect scene changes using histogram comparison.

    Fast and lightweight - suitable for all videos.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._prev_hist: Optional[np.ndarray] = None
        self._scene_changes: List[Dict[str, Any]] = []
        self._current_scene_start = 0.0

    @property
    def analyzer_id(self) -> str:
        return "scene_change"

    @property
    def sample_rate(self) -> int:
        return 3  # Process every 3rd frame for speed

    def on_start(self, metadata: VideoMetadata) -> None:
        self._prev_hist = None
        self._scene_changes = []
        self._current_scene_start = 0.0

    def process_frame(self, ctx: FrameContext) -> Optional[float]:
        # Calculate histogram
        hsv = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if self._prev_hist is None:
            self._prev_hist = hist
            return None

        # Compare histograms
        similarity = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
        self._prev_hist = hist

        # Detect scene change
        if similarity < self.threshold:
            self._scene_changes.append({
                "timestamp": ctx.timestamp,
                "frame_idx": ctx.frame_idx,
                "similarity": similarity,
                "scene_duration": ctx.timestamp - self._current_scene_start
            })
            self._current_scene_start = ctx.timestamp
            return similarity

        return None

    def on_end(self) -> List[Dict[str, Any]]:
        return self._scene_changes


class MotionAnalyzer(AnalyzerPlugin):
    """
    Analyze motion intensity using optical flow.

    Provides per-frame motion scores for energy detection.
    """

    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None
        self._motion_scores: List[Dict[str, float]] = []

    @property
    def analyzer_id(self) -> str:
        return "motion"

    @property
    def sample_rate(self) -> int:
        return 2  # Every other frame

    def on_start(self, metadata: VideoMetadata) -> None:
        self._prev_gray = None
        self._motion_scores = []

    def process_frame(self, ctx: FrameContext) -> Optional[float]:
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        # Calculate frame difference
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        # Motion score = mean of difference
        motion_score = float(np.mean(diff))

        self._motion_scores.append({
            "timestamp": ctx.timestamp,
            "motion_score": motion_score
        })

        return motion_score

    def on_end(self) -> Dict[str, Any]:
        if not self._motion_scores:
            return {"scores": [], "avg_motion": 0.0, "max_motion": 0.0}

        scores = [s["motion_score"] for s in self._motion_scores]
        return {
            "scores": self._motion_scores,
            "avg_motion": float(np.mean(scores)),
            "max_motion": float(max(scores)),
            "high_motion_segments": self._find_high_motion_segments()
        }

    def _find_high_motion_segments(self, threshold_percentile: float = 75) -> List[Dict[str, float]]:
        """Find segments with above-average motion."""
        if not self._motion_scores:
            return []

        scores = [s["motion_score"] for s in self._motion_scores]
        threshold = np.percentile(scores, threshold_percentile)

        segments = []
        in_segment = False
        segment_start = 0.0

        for entry in self._motion_scores:
            if entry["motion_score"] >= threshold and not in_segment:
                in_segment = True
                segment_start = entry["timestamp"]
            elif entry["motion_score"] < threshold and in_segment:
                in_segment = False
                segments.append({
                    "start": segment_start,
                    "end": entry["timestamp"],
                    "duration": entry["timestamp"] - segment_start
                })

        return segments


class BrightnessAnalyzer(AnalyzerPlugin):
    """
    Analyze brightness/exposure across video.

    Useful for detecting dark/bright segments.
    """

    def __init__(self):
        self._brightness_values: List[Dict[str, float]] = []

    @property
    def analyzer_id(self) -> str:
        return "brightness"

    @property
    def sample_rate(self) -> int:
        return 5  # Every 5th frame is sufficient

    def on_start(self, metadata: VideoMetadata) -> None:
        self._brightness_values = []

    def process_frame(self, ctx: FrameContext) -> float:
        # Convert to grayscale and get mean brightness
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        self._brightness_values.append({
            "timestamp": ctx.timestamp,
            "brightness": brightness
        })

        return brightness

    def on_end(self) -> Dict[str, Any]:
        if not self._brightness_values:
            return {"values": [], "avg_brightness": 128.0}

        values = [v["brightness"] for v in self._brightness_values]
        return {
            "values": self._brightness_values,
            "avg_brightness": float(np.mean(values)),
            "min_brightness": float(min(values)),
            "max_brightness": float(max(values)),
            "std_brightness": float(np.std(values))
        }


# =============================================================================
# Video Analysis Engine
# =============================================================================

class VideoAnalysisEngine:
    """
    Unified video analysis engine with pluggable analyzers.

    Performs single-pass video decoding with multiple analyzers
    processing each frame in sequence. Results are aggregated
    and returned as a dict keyed by analyzer_id.
    """

    def __init__(self, video_path: str):
        """
        Initialize engine for a video file.

        Args:
            video_path: Path to video file

        .. deprecated:: 2026.1
            Use scene_provider.get_scene_provider() instead.
        """
        warnings.warn(
            "VideoAnalysisEngine is deprecated. Use scene_provider.get_scene_provider() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.video_path = video_path
        self._metadata: Optional[VideoMetadata] = None
        self._analyzers: List[AnalyzerPlugin] = []
        self._progress_callback: Optional[Callable[[int, int], None]] = None

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata (lazy-loaded)."""
        if self._metadata is None:
            self._metadata = self._probe_video()
        return self._metadata

    def add_analyzer(self, analyzer: AnalyzerPlugin) -> "VideoAnalysisEngine":
        """
        Add an analyzer plugin.

        Args:
            analyzer: Analyzer plugin instance

        Returns:
            Self for chaining
        """
        self._analyzers.append(analyzer)
        return self

    def set_progress_callback(self, callback: Callable[[int, int], None]) -> "VideoAnalysisEngine":
        """
        Set progress callback.

        Args:
            callback: Function(current_frame, total_frames)

        Returns:
            Self for chaining
        """
        self._progress_callback = callback
        return self

    def analyze(self) -> Dict[str, Any]:
        """
        Run all analyzers on the video.

        Returns:
            Dict mapping analyzer_id -> result
        """
        import time

        if not self._analyzers:
            logger.warning("No analyzers configured, returning empty results")
            return {}

        # Ensure metadata is loaded
        metadata = self.metadata

        # Initialize analyzers
        for analyzer in self._analyzers:
            analyzer.on_start(metadata)

        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        try:
            frame_idx = 0
            results: Dict[str, List[Any]] = {a.analyzer_id: [] for a in self._analyzers}
            timings: Dict[str, float] = {a.analyzer_id: 0.0 for a in self._analyzers}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / metadata.fps
                ctx = FrameContext(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    frame=frame,
                    metadata=metadata
                )

                # Process with each analyzer
                for analyzer in self._analyzers:
                    if frame_idx % analyzer.sample_rate == 0:
                        start = time.perf_counter()
                        result = analyzer.process_frame(ctx)
                        timings[analyzer.analyzer_id] += (time.perf_counter() - start) * 1000

                        if result is not None:
                            results[analyzer.analyzer_id].append(result)

                # Progress callback
                if self._progress_callback and frame_idx % 30 == 0:
                    self._progress_callback(frame_idx, metadata.frame_count)

                frame_idx += 1

        finally:
            cap.release()

        # Finalize analyzers
        final_results = {}
        for analyzer in self._analyzers:
            final_data = analyzer.on_end()
            final_results[analyzer.analyzer_id] = AnalysisResult(
                analyzer_id=analyzer.analyzer_id,
                data=final_data,
                frame_count=len(results[analyzer.analyzer_id]),
                processing_time_ms=timings[analyzer.analyzer_id]
            )

        logger.info(f"Video analysis complete: {len(final_results)} analyzers, {frame_idx} frames")
        return final_results

    def _probe_video(self) -> VideoMetadata:
        """Probe video for metadata using ffprobe."""
        import json

        cmd = build_ffprobe_cmd([
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            self.video_path
        ])

        result = run_command(cmd, capture_output=True, check=True)
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError(f"No video stream found in {self.video_path}")

        # Parse frame rate (e.g., "30/1" or "30000/1001")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

        duration = float(data.get("format", {}).get("duration", 0))

        return VideoMetadata(
            path=self.video_path,
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=fps,
            duration=duration,
            frame_count=int(video_stream.get("nb_frames", int(duration * fps))),
            codec=video_stream.get("codec_name", "unknown")
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_video_quick(video_path: str) -> Dict[str, Any]:
    """
    Quick analysis with default analyzers (scene change, motion, brightness).

    Args:
        video_path: Path to video file

    Returns:
        Analysis results dict
    """
    engine = VideoAnalysisEngine(video_path)
    engine.add_analyzer(SceneChangeAnalyzer())
    engine.add_analyzer(MotionAnalyzer())
    engine.add_analyzer(BrightnessAnalyzer())

    return engine.analyze()


def get_video_metadata(video_path: str) -> VideoMetadata:
    """
    Get video metadata without full analysis.

    Args:
        video_path: Path to video file

    Returns:
        VideoMetadata object
    """
    engine = VideoAnalysisEngine(video_path)
    return engine.metadata
