"""
Unified Scene Analysis Provider - Single Interface for All Scene Detection

Consolidates 7+ overlapping scene analysis implementations into one provider.
Uses Strategy Pattern to support different backends.

Usage:
    from montage_ai.core.scene_provider import get_scene_provider, SceneProvider

    provider = get_scene_provider()  # Auto-selects best available backend
    scenes = provider.detect_scenes("/path/to/video.mp4")
    analysis = provider.analyze_frame("/path/to/video.mp4", time=5.0)

Backends:
    - pyscenedetect: Best quality, requires scenedetect library
    - histogram: Fast, uses OpenCV histogram comparison
    - threshold: Simple, uses frame differencing
    - cgpu: Cloud GPU offloading (optional)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol, Tuple
import os

from ..logger import logger


# =============================================================================
# Unified Data Classes
# =============================================================================

class ActionLevel(str, Enum):
    """Motion intensity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ShotType(str, Enum):
    """Camera shot classification."""
    CLOSE = "close"
    MEDIUM = "medium"
    WIDE = "wide"


@dataclass
class SceneBoundary:
    """
    Unified scene boundary representation.

    Consolidates: Scene (scene_analysis.py), SceneInfo (montage_builder.py)
    """
    start: float
    end: float
    path: str
    confidence: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "path": self.path,
            "duration": self.duration,
            "confidence": self.confidence,
            "meta": self.meta,
        }

    # Backward compatibility with Scene class
    def to_legacy_scene(self) -> 'Scene':
        """Convert to legacy Scene class for backward compatibility."""
        from ..scene_analysis import Scene
        return Scene(
            start=self.start,
            end=self.end,
            path=self.path,
            meta=self.meta
        )


@dataclass
class FrameAnalysis:
    """
    Unified frame/scene content analysis.

    Consolidates: SceneAnalysis (scene_analysis.py), SceneAnalysis (footage_analyzer.py)
    """
    quality: str = "YES"
    description: str = ""
    action: ActionLevel = ActionLevel.MEDIUM
    shot: ShotType = ShotType.MEDIUM
    confidence: float = 0.5

    # Semantic fields
    tags: List[str] = field(default_factory=list)
    caption: str = ""
    objects: List[str] = field(default_factory=list)
    mood: str = ""
    setting: str = ""
    face_count: int = 0

    # Visual metrics
    brightness: float = 0.5
    motion_score: float = 0.0
    blur_score: float = 0.0

    # Narrative potential (from footage_analyzer.py)
    narrative_potential: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality": self.quality,
            "description": self.description,
            "action": self.action.value,
            "shot": self.shot.value,
            "confidence": self.confidence,
            "tags": self.tags,
            "caption": self.caption,
            "objects": self.objects,
            "mood": self.mood,
            "setting": self.setting,
            "face_count": self.face_count,
            "brightness": self.brightness,
            "motion_score": self.motion_score,
            "blur_score": self.blur_score,
            "narrative_potential": self.narrative_potential,
        }

    def to_legacy_analysis(self) -> 'SceneAnalysis':
        """Convert to legacy SceneAnalysis for backward compatibility."""
        from ..scene_analysis import SceneAnalysis as LegacyAnalysis
        from ..scene_analysis import ActionLevel as LegacyAction
        from ..scene_analysis import ShotType as LegacyShot

        return LegacyAnalysis(
            quality=self.quality,
            description=self.description,
            action=LegacyAction(self.action.value),
            shot=LegacyShot(self.shot.value),
            tags=self.tags,
            caption=self.caption,
            objects=self.objects,
            mood=self.mood,
            setting=self.setting,
            face_count=self.face_count,
        )


# =============================================================================
# Provider Interface
# =============================================================================

class SceneDetectionBackend(Protocol):
    """Protocol for scene detection backends."""

    def detect(self, video_path: str, threshold: float) -> List[SceneBoundary]:
        """Detect scene boundaries in video."""
        ...


class FrameAnalysisBackend(Protocol):
    """Protocol for frame/content analysis backends."""

    def analyze(self, video_path: str, time_point: float, semantic: bool) -> FrameAnalysis:
        """Analyze frame content at specific time."""
        ...


class SceneProvider:
    """
    Unified scene analysis provider.

    Automatically selects best available backends for:
    - Scene detection (pyscenedetect, histogram, threshold)
    - Frame analysis (AI vision, local CV)

    Example:
        provider = SceneProvider()
        scenes = provider.detect_scenes("video.mp4")
        analysis = provider.analyze_frame("video.mp4", time=5.0)
    """

    def __init__(
        self,
        detection_backend: Optional[str] = None,
        analysis_backend: Optional[str] = None,
        threshold: float = 30.0,
        enable_cache: bool = True,
    ):
        """
        Initialize scene provider.

        Args:
            detection_backend: Force specific backend (pyscenedetect, histogram, threshold, cgpu)
            analysis_backend: Force specific backend (ai, local)
            threshold: Scene detection threshold (higher = fewer scenes)
            enable_cache: Enable result caching
        """
        self.threshold = threshold
        self.enable_cache = enable_cache

        # Auto-select best available backends
        self._detection_backend = detection_backend or self._select_detection_backend()
        self._analysis_backend = analysis_backend or self._select_analysis_backend()

        logger.debug(f"SceneProvider initialized: detection={self._detection_backend}, analysis={self._analysis_backend}")

    def _select_detection_backend(self) -> str:
        """Select best available detection backend."""
        # Try pyscenedetect first (best quality)
        try:
            import scenedetect
            return "pyscenedetect"
        except ImportError:
            pass

        # Try cgpu if enabled (central settings)
        from ..config import get_settings
        if get_settings().llm.cgpu_enabled:
            return "cgpu"

        # Fall back to histogram (OpenCV-based)
        try:
            import cv2
            return "histogram"
        except ImportError:
            pass

        # Last resort: threshold-based
        return "threshold"

    def _select_analysis_backend(self) -> str:
        """Select best available analysis backend."""
        # Check for AI backends via centralized settings
        from ..config import get_settings
        llm = get_settings().llm
        if llm.cgpu_enabled:
            return "cgpu"
        if llm.has_openai_backend:
            return "openai"
        if bool(llm.ollama_host):
            return "ollama"

        # Local CV fallback
        return "local"

    def detect_scenes(
        self,
        video_path: str,
        threshold: Optional[float] = None,
        min_scene_length: float = 0.5,
    ) -> List[SceneBoundary]:
        """
        Detect scene boundaries in video.

        Args:
            video_path: Path to video file
            threshold: Override default threshold
            min_scene_length: Minimum scene duration in seconds

        Returns:
            List of SceneBoundary objects
        """
        threshold = threshold or self.threshold

        if self._detection_backend == "pyscenedetect":
            return self._detect_pyscenedetect(video_path, threshold, min_scene_length)
        elif self._detection_backend == "cgpu":
            return self._detect_cgpu(video_path, threshold)
        elif self._detection_backend == "histogram":
            return self._detect_histogram(video_path, threshold, min_scene_length)
        else:
            return self._detect_threshold(video_path, threshold, min_scene_length)

    def analyze_frame(
        self,
        video_path: str,
        time_point: float,
        semantic: bool = False,
    ) -> FrameAnalysis:
        """
        Analyze frame content at specific time.

        Args:
            video_path: Path to video file
            time_point: Time in seconds
            semantic: Whether to extract semantic tags

        Returns:
            FrameAnalysis object
        """
        if self._analysis_backend in ("cgpu", "openai", "ollama"):
            return self._analyze_ai(video_path, time_point, semantic)
        else:
            return self._analyze_local(video_path, time_point)

    def calculate_similarity(
        self,
        video1_path: str,
        time1: float,
        video2_path: str,
        time2: float,
    ) -> float:
        """
        Calculate visual similarity between two frames.

        Args:
            video1_path: Path to first video
            time1: Time in first video
            video2_path: Path to second video
            time2: Time in second video

        Returns:
            Similarity score (0-1, 1 = identical)
        """
        from ..scene_analysis import calculate_visual_similarity
        return calculate_visual_similarity(video1_path, time1, video2_path, time2)

    def find_best_cut_point(
        self,
        video_path: str,
        start: float,
        end: float,
        target_duration: Optional[float] = None,
    ) -> float:
        """
        Find optimal cut point within a scene.

        Uses motion analysis to find the most interesting/actionful moment.

        Args:
            video_path: Path to video
            start: Scene start time
            end: Scene end time
            target_duration: Optional target clip duration

        Returns:
            Best start time for cut
        """
        from ..scene_analysis import find_best_start_point
        return find_best_start_point(video_path, start, end, target_duration or (end - start))

    # =========================================================================
    # Backend Implementations
    # =========================================================================

    def _detect_pyscenedetect(
        self,
        video_path: str,
        threshold: float,
        min_scene_length: float,
    ) -> List[SceneBoundary]:
        """Use PySceneDetect for scene detection."""
        from ..scene_analysis import SceneDetector

        detector = SceneDetector(threshold=threshold)
        legacy_scenes = detector.detect(video_path)

        return [
            SceneBoundary(
                start=s.start,
                end=s.end,
                path=s.path,
                confidence=0.9,  # PySceneDetect is high quality
                meta=s.meta,
            )
            for s in legacy_scenes
            if s.duration >= min_scene_length
        ]

    def _detect_cgpu(self, video_path: str, threshold: float) -> List[SceneBoundary]:
        """Use cgpu for scene detection."""
        try:
            from ..cgpu_jobs.analysis import SceneDetectionJob

            job = SceneDetectionJob(input_path=video_path, threshold=threshold)
            result = job.execute()

            if result.success and result.data:
                return [
                    SceneBoundary(
                        start=s.get("start", 0),
                        end=s.get("end", 0),
                        path=video_path,
                        confidence=0.85,
                    )
                    for s in result.data.get("scenes", [])
                ]
        except Exception as e:
            logger.warning(f"cgpu scene detection failed: {e}, falling back to local")

        # Fallback to local
        return self._detect_histogram(video_path, threshold, 0.5)

    def _detect_histogram(
        self,
        video_path: str,
        threshold: float,
        min_scene_length: float,
    ) -> List[SceneBoundary]:
        """Use histogram comparison for scene detection."""
        try:
            from ..core.video_analysis_engine import VideoAnalysisEngine, SceneChangeAnalyzer

            engine = VideoAnalysisEngine(video_path)
            engine.add_analyzer(SceneChangeAnalyzer(threshold=threshold / 100))
            results = engine.analyze()

            scene_changes = results.get("scene_changes", [])
            if not scene_changes:
                # No scene changes = one big scene
                from ..video_metadata import probe_metadata
                meta = probe_metadata(video_path)
                duration = meta.duration if meta else 60.0
                return [SceneBoundary(start=0, end=duration, path=video_path)]

            # Convert scene changes to boundaries
            boundaries = []
            prev_time = 0.0
            for change in scene_changes:
                if change["timestamp"] - prev_time >= min_scene_length:
                    boundaries.append(SceneBoundary(
                        start=prev_time,
                        end=change["timestamp"],
                        path=video_path,
                        confidence=change.get("similarity", 0.5),
                    ))
                    prev_time = change["timestamp"]

            return boundaries

        except Exception as e:
            logger.warning(f"Histogram detection failed: {e}, falling back to threshold")
            return self._detect_threshold(video_path, threshold, min_scene_length)

    def _detect_threshold(
        self,
        video_path: str,
        threshold: float,
        min_scene_length: float,
    ) -> List[SceneBoundary]:
        """Simple threshold-based scene detection."""
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            boundaries = []
            prev_frame = None
            prev_time = 0.0
            frame_idx = 0
            sample_interval = max(1, int(fps / 2))  # Sample 2x per second

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if prev_frame is not None:
                        diff = cv2.absdiff(prev_frame, gray)
                        score = np.mean(diff)

                        if score > threshold:
                            current_time = frame_idx / fps
                            if current_time - prev_time >= min_scene_length:
                                boundaries.append(SceneBoundary(
                                    start=prev_time,
                                    end=current_time,
                                    path=video_path,
                                    confidence=min(1.0, score / 100),
                                ))
                                prev_time = current_time

                    prev_frame = gray

                frame_idx += 1

            cap.release()

            # Add final scene
            if duration - prev_time >= min_scene_length:
                boundaries.append(SceneBoundary(
                    start=prev_time,
                    end=duration,
                    path=video_path,
                ))

            return boundaries if boundaries else [SceneBoundary(start=0, end=duration, path=video_path)]

        except Exception as e:
            logger.error(f"Threshold detection failed: {e}")
            return []

    def _analyze_ai(
        self,
        video_path: str,
        time_point: float,
        semantic: bool,
    ) -> FrameAnalysis:
        """Use AI backend for frame analysis."""
        from ..scene_analysis import SceneContentAnalyzer

        analyzer = SceneContentAnalyzer()
        legacy = analyzer.analyze(video_path, time_point, semantic=semantic)

        return FrameAnalysis(
            quality=legacy.quality,
            description=legacy.description,
            action=ActionLevel(legacy.action.value),
            shot=ShotType(legacy.shot.value),
            tags=legacy.tags,
            caption=legacy.caption,
            objects=legacy.objects,
            mood=legacy.mood,
            setting=legacy.setting,
            face_count=legacy.face_count,
        )

    def _analyze_local(self, video_path: str, time_point: float) -> FrameAnalysis:
        """Use local CV for frame analysis."""
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_point * fps))

            ret, frame = cap.read()
            cap.release()

            if not ret:
                return FrameAnalysis()

            # Basic visual metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0

            # Blur detection (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = 1.0 - min(1.0, laplacian.var() / 500)

            # Face detection
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            face_count = len(faces)

            # Estimate shot type from face size
            if face_count > 0:
                max_face_area = max(w * h for (x, y, w, h) in faces)
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = max_face_area / frame_area
                if face_ratio > 0.1:
                    shot = ShotType.CLOSE
                elif face_ratio > 0.02:
                    shot = ShotType.MEDIUM
                else:
                    shot = ShotType.WIDE
            else:
                shot = ShotType.WIDE

            return FrameAnalysis(
                quality="YES" if brightness > 0.1 and blur_score < 0.8 else "NO",
                brightness=brightness,
                blur_score=blur_score,
                face_count=face_count,
                shot=shot,
            )

        except Exception as e:
            logger.warning(f"Local frame analysis failed: {e}")
            return FrameAnalysis()


# =============================================================================
# Factory Function
# =============================================================================

_default_provider: Optional[SceneProvider] = None


def get_scene_provider(**kwargs) -> SceneProvider:
    """
    Get the default scene provider instance.

    Uses singleton pattern for efficiency.
    Pass kwargs to override defaults.

    Example:
        provider = get_scene_provider()
        provider = get_scene_provider(threshold=40.0)
    """
    global _default_provider

    if kwargs or _default_provider is None:
        _default_provider = SceneProvider(**kwargs)

    return _default_provider


def clear_scene_provider() -> None:
    """Clear the default provider instance (for testing)."""
    global _default_provider
    _default_provider = None


__all__ = [
    # Data classes
    'SceneBoundary',
    'FrameAnalysis',
    'ActionLevel',
    'ShotType',
    # Provider
    'SceneProvider',
    'get_scene_provider',
    'clear_scene_provider',
]
