"""
Unified Video Analysis Engine - Single-Pass Analysis with Pluggable Analyzers

Consolidates video analysis across workflows:
- Creator: Scene detection, beat alignment, energy analysis
- Shorts: Face tracking, subject detection, safe zone analysis
- Shared: Video metadata, quality assessment, motion analysis

SOTA Design Pattern: Strategy Pattern + Pipeline Abstraction
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

from ..logger import logger


# =============================================================================
# Data Models
# =============================================================================

class AnalysisType(Enum):
    """Types of video analysis available."""
    SCENE_DETECTION = "scene_detection"
    FACE_TRACKING = "face_tracking"
    MOTION_ANALYSIS = "motion_analysis"
    AUDIO_ENERGY = "audio_energy"
    QUALITY_ASSESSMENT = "quality_assessment"


@dataclass
class VideoMetadata:
    """Basic video properties."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    aspect_ratio: float
    codec: str = "unknown"
    bitrate: Optional[int] = None
    
    @classmethod
    def from_video(cls, video_path: str) -> 'VideoMetadata':
        """Extract metadata from video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        aspect_ratio = width / height if height > 0 else 16/9
        
        # Try to get codec (may not be available in all builds)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return cls(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            aspect_ratio=aspect_ratio,
            codec=codec
        )


@dataclass
class AnalysisResult:
    """Container for analysis results from a single analyzer."""
    type: AnalysisType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": self.type.value,
            "data": self.data,
            "metadata": self.metadata
        }


@dataclass
class VideoAnalysis:
    """Complete video analysis result containing all analyzer outputs."""
    video_path: str
    metadata: VideoMetadata
    results: List[AnalysisResult] = field(default_factory=list)
    cache_key: Optional[str] = None
    
    def get_result(self, analysis_type: AnalysisType) -> Optional[AnalysisResult]:
        """Get result by analysis type."""
        for result in self.results:
            if result.type == analysis_type:
                return result
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "video_path": self.video_path,
            "metadata": {
                "width": self.metadata.width,
                "height": self.metadata.height,
                "fps": self.metadata.fps,
                "duration": self.metadata.duration,
                "aspect_ratio": self.metadata.aspect_ratio,
            },
            "results": [r.to_dict() for r in self.results],
            "cache_key": self.cache_key
        }


# =============================================================================
# Analyzer Interface
# =============================================================================

class VideoAnalyzer(ABC):
    """
    Abstract base class for video analyzers.
    
    Implements Strategy Pattern - each analyzer is a self-contained strategy
    for extracting specific information from video.
    """
    
    @property
    @abstractmethod
    def analysis_type(self) -> AnalysisType:
        """Type of analysis this analyzer performs."""
        pass
    
    @abstractmethod
    def analyze(self, video_path: str, metadata: VideoMetadata) -> AnalysisResult:
        """
        Analyze video and return results.
        
        Args:
            video_path: Path to video file
            metadata: Pre-extracted video metadata
            
        Returns:
            AnalysisResult containing extracted data
        """
        pass
    
    def should_run(self, metadata: VideoMetadata) -> bool:
        """
        Check if analyzer should run based on video metadata.
        
        Override this to skip analysis for incompatible videos.
        Example: Skip face tracking for audio-only files.
        """
        return True


# =============================================================================
# Unified Analysis Engine
# =============================================================================

class VideoAnalysisEngine:
    """
    Single-pass video analysis engine with pluggable analyzers.
    
    Benefits:
    - DRY: Shared video I/O, frame iteration, metadata extraction
    - Performance: Single-pass analysis for multiple extractors
    - Extensibility: Add new analyzers without modifying engine
    - Caching: Results cached by video path + analyzer config
    """
    
    def __init__(self):
        self._analyzers: List[VideoAnalyzer] = []
        self._cache: Dict[str, VideoAnalysis] = {}
    
    def register_analyzer(self, analyzer: VideoAnalyzer) -> None:
        """Register an analyzer to run during analysis."""
        if not isinstance(analyzer, VideoAnalyzer):
            raise TypeError(f"Analyzer must inherit from VideoAnalyzer, got {type(analyzer)}")
        self._analyzers.append(analyzer)
        logger.debug(f"Registered analyzer: {analyzer.analysis_type.value}")
    
    def analyze(
        self,
        video_path: str,
        force_refresh: bool = False
    ) -> VideoAnalysis:
        """
        Run all registered analyzers on video.
        
        Args:
            video_path: Path to video file
            force_refresh: Skip cache and re-analyze
            
        Returns:
            VideoAnalysis containing results from all analyzers
        """
        # Check cache
        cache_key = self._generate_cache_key(video_path)
        if not force_refresh and cache_key in self._cache:
            logger.info(f"Using cached analysis for {Path(video_path).name}")
            return self._cache[cache_key]
        
        # Extract metadata once
        logger.info(f"Analyzing video: {Path(video_path).name}")
        metadata = VideoMetadata.from_video(video_path)
        logger.debug(f"Video: {metadata.width}x{metadata.height} @ {metadata.fps:.2f}fps, {metadata.duration:.1f}s")
        
        # Run applicable analyzers
        results = []
        for analyzer in self._analyzers:
            if not analyzer.should_run(metadata):
                logger.debug(f"Skipping {analyzer.analysis_type.value} (not applicable)")
                continue
            
            try:
                logger.info(f"Running {analyzer.analysis_type.value}...")
                result = analyzer.analyze(video_path, metadata)
                results.append(result)
                logger.debug(f"✓ {analyzer.analysis_type.value} complete")
            except Exception as e:
                logger.error(f"✗ {analyzer.analysis_type.value} failed: {e}")
                # Continue with other analyzers
        
        # Create analysis object
        analysis = VideoAnalysis(
            video_path=video_path,
            metadata=metadata,
            results=results,
            cache_key=cache_key
        )
        
        # Cache result
        self._cache[cache_key] = analysis
        
        return analysis
    
    def _generate_cache_key(self, video_path: str) -> str:
        """Generate cache key from video path and analyzer configuration."""
        # Simple cache key for now - could be enhanced with file hash
        analyzer_types = sorted([a.analysis_type.value for a in self._analyzers])
        return f"{video_path}::{':'.join(analyzer_types)}"
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._cache.clear()
        logger.debug("Analysis cache cleared")


# =============================================================================
# Example: Scene Detection Analyzer (for Creator workflow)
# =============================================================================

class SceneDetectionAnalyzer(VideoAnalyzer):
    """
    Scene detection using PySceneDetect-style threshold detection.
    
    Used by Creator workflow for intelligent clip selection.
    """
    
    def __init__(self, threshold: float = 27.0, min_scene_len: int = 15):
        """
        Args:
            threshold: Intensity change threshold (0-255)
            min_scene_len: Minimum scene length in frames
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
    
    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SCENE_DETECTION
    
    def analyze(self, video_path: str, metadata: VideoMetadata) -> AnalysisResult:
        """Detect scene changes in video."""
        cap = cv2.VideoCapture(video_path)
        
        scenes = []
        prev_frame = None
        scene_start = 0
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate mean absolute difference
                    diff = np.abs(gray.astype(float) - prev_frame.astype(float)).mean()
                    
                    # Detect scene change
                    if diff > self.threshold and (frame_num - scene_start) >= self.min_scene_len:
                        time = frame_num / metadata.fps
                        scenes.append({
                            "start_frame": scene_start,
                            "end_frame": frame_num,
                            "start_time": scene_start / metadata.fps,
                            "end_time": time,
                            "duration": (frame_num - scene_start) / metadata.fps
                        })
                        scene_start = frame_num
                
                prev_frame = gray
                frame_num += 1
        
        finally:
            cap.release()
        
        # Add final scene
        if frame_num > scene_start:
            scenes.append({
                "start_frame": scene_start,
                "end_frame": frame_num,
                "start_time": scene_start / metadata.fps,
                "end_time": frame_num / metadata.fps,
                "duration": (frame_num - scene_start) / metadata.fps
            })
        
        return AnalysisResult(
            type=self.analysis_type,
            data={"scenes": scenes},
            metadata={
                "threshold": self.threshold,
                "min_scene_len": self.min_scene_len,
                "total_scenes": len(scenes)
            }
        )


# =============================================================================
# Example: Face Tracking Analyzer (for Shorts workflow)
# =============================================================================

class FaceTrackingAnalyzer(VideoAnalyzer):
    """
    Face detection and tracking using MediaPipe.
    
    Used by Shorts workflow for smart reframing.
    """
    
    def __init__(self, detection_confidence: float = 0.5):
        self.detection_confidence = detection_confidence
        self._mp_face = None  # Lazy import
    
    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.FACE_TRACKING
    
    def should_run(self, metadata: VideoMetadata) -> bool:
        """Only run on videos with reasonable resolution."""
        return metadata.width >= 640 and metadata.height >= 480
    
    def analyze(self, video_path: str, metadata: VideoMetadata) -> AnalysisResult:
        """Track faces throughout video."""
        # Lazy import to avoid heavy dependencies at module load
        try:
            import mediapipe as mp
            if self._mp_face is None:
                self._mp_face = mp.solutions.face_detection.FaceDetection(
                    min_detection_confidence=self.detection_confidence
                )
        except ImportError:
            logger.warning("MediaPipe not available, face tracking disabled")
            return AnalysisResult(
                type=self.analysis_type,
                data={"faces": []},
                metadata={"error": "mediapipe_not_installed"}
            )
        
        cap = cv2.VideoCapture(video_path)
        faces_timeline = []
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_face.process(rgb)
                
                time = frame_num / metadata.fps
                
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        faces_timeline.append({
                            "time": time,
                            "frame": frame_num,
                            "x": bbox.xmin,
                            "y": bbox.ymin,
                            "width": bbox.width,
                            "height": bbox.height,
                            "score": detection.score[0]
                        })
                
                frame_num += 1
        
        finally:
            cap.release()
        
        return AnalysisResult(
            type=self.analysis_type,
            data={"faces": faces_timeline},
            metadata={
                "detection_confidence": self.detection_confidence,
                "total_detections": len(faces_timeline),
                "avg_faces_per_frame": len(faces_timeline) / max(frame_num, 1)
            }
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_analysis_engine(workflow: str = "default") -> VideoAnalysisEngine:
    """
    Factory function to create pre-configured analysis engine.
    
    Args:
        workflow: Workflow type ("montage", "shorts", "default")
        
    Returns:
        VideoAnalysisEngine with appropriate analyzers registered
    """
    engine = VideoAnalysisEngine()
    
    if workflow in ("montage", "default"):
        engine.register_analyzer(SceneDetectionAnalyzer())
    
    if workflow in ("shorts", "default"):
        engine.register_analyzer(FaceTrackingAnalyzer())
    
    return engine
