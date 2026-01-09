"""
Scene Analysis Helpers - Unified Scene Processing Utilities

Consolidates common scene analysis functions scattered across:
  - scene_analysis.py
  - core/analysis_engine.py  
  - core/video_analysis_engine.py

Provides reusable utilities for scene processing, normalization,
and feature extraction.

Usage:
    from montage_ai.scene_helpers import SceneProcessor
    
    processor = SceneProcessor()
    normalized = processor.normalize_scene(scene)
    duration = processor.calculate_duration(scene)
    features = processor.extract_features(scene)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Scene:
    """Normalized scene representation."""
    id: str
    start_time: float  # seconds
    end_time: float
    description: Optional[str] = None
    visual_features: Dict[str, Any] = None
    audio_features: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    @property
    def duration(self) -> float:
        """Computed duration in seconds."""
        return self.end_time - self.start_time
    
    def __post_init__(self):
        if self.visual_features is None:
            self.visual_features = {}
        if self.audio_features is None:
            self.audio_features = {}
        if self.metadata is None:
            self.metadata = {}


class SceneProcessor:
    """Unified scene processing utilities."""
    
    # =========================================================================
    # Scene Normalization
    # =========================================================================
    
    @staticmethod
    def normalize_scene(
        scene: Dict[str, Any],
        strict: bool = False,
    ) -> Scene:
        """Normalize raw scene dict to Scene object.
        
        Handles various input formats and fills missing fields.
        
        Args:
            scene: Raw scene dict (from scenedetect, custom analysis, etc.)
            strict: If True, raise on missing required fields
            
        Returns:
            Normalized Scene object
            
        Raises:
            ValueError: If strict=True and required fields missing
        """
        # Extract time bounds
        start = scene.get("start_time") or scene.get("start") or 0.0
        end = scene.get("end_time") or scene.get("end") or scene.get("duration", 0.0)
        
        if not start and not end:
            if strict:
                raise ValueError("Scene missing start/end time")
            return Scene(
                id=scene.get("id", "unknown"),
                start_time=0.0,
                end_time=0.0,
            )
        
        scene_id = scene.get("id") or scene.get("scene_id") or f"scene_{int(start*1000)}"
        
        return Scene(
            id=scene_id,
            start_time=float(start),
            end_time=float(end),
            description=scene.get("description"),
            visual_features=scene.get("visual_features", {}),
            audio_features=scene.get("audio_features", {}),
            metadata=scene.get("metadata", {}),
        )
    
    @staticmethod
    def calculate_duration(scene: Dict[str, Any]) -> float:
        """Calculate scene duration in seconds.
        
        Args:
            scene: Scene dict
            
        Returns:
            Duration in seconds
        """
        start = scene.get("start_time") or scene.get("start") or 0.0
        end = scene.get("end_time") or scene.get("end") or 0.0
        return float(end) - float(start)
    
    @staticmethod
    def is_valid_scene(scene: Dict[str, Any]) -> bool:
        """Check if scene has minimum valid structure.
        
        Args:
            scene: Scene dict to validate
            
        Returns:
            True if scene is structurally valid
        """
        if not isinstance(scene, dict):
            return False
        
        # Must have either time bounds or duration
        has_times = "start_time" in scene or "start" in scene
        has_end = "end_time" in scene or "end" in scene
        has_duration = "duration" in scene
        
        return (has_times and has_end) or has_duration
    
    # =========================================================================
    # Feature Extraction
    # =========================================================================
    
    @staticmethod
    def extract_visual_features(
        scene: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract all visual features from scene.
        
        Args:
            scene: Scene dict with visual metadata
            
        Returns:
            Dict of visual features
        """
        visual = scene.get("visual_features", {})
        if not visual:
            visual = {}
        
        # Ensure standard keys exist
        defaults = {
            "brightness": 0.5,
            "contrast": 0.5,
            "saturation": 0.5,
            "sharpness": 0.5,
            "motion": 0.5,
            "noise_level": 0.0,
            "color_dominant": None,
        }
        
        for key, default in defaults.items():
            if key not in visual:
                visual[key] = default
        
        return visual
    
    @staticmethod
    def extract_audio_features(scene: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all audio features from scene.
        
        Args:
            scene: Scene dict with audio metadata
            
        Returns:
            Dict of audio features
        """
        audio = scene.get("audio_features", {})
        if not audio:
            audio = {}
        
        # Ensure standard keys
        defaults = {
            "loudness_db": -20.0,
            "has_speech": False,
            "has_music": False,
            "has_silence": False,
            "speech_confidence": 0.0,
            "music_energy": 0.0,
        }
        
        for key, default in defaults.items():
            if key not in audio:
                audio[key] = default
        
        return audio
    
    # =========================================================================
    # Scene Comparison
    # =========================================================================
    
    @staticmethod
    def calculate_scene_similarity(
        scene1: Dict[str, Any],
        scene2: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate similarity between two scenes [0.0, 1.0].
        
        Args:
            scene1: First scene
            scene2: Second scene
            weights: Feature weights for similarity calculation
            
        Returns:
            Similarity score [0.0 = different, 1.0 = identical]
        """
        if weights is None:
            weights = {
                "visual": 0.4,
                "audio": 0.3,
                "duration": 0.3,
            }
        
        score = 0.0
        total_weight = 0.0
        
        # Visual similarity
        if weights.get("visual", 0) > 0:
            vis1 = SceneProcessor.extract_visual_features(scene1)
            vis2 = SceneProcessor.extract_visual_features(scene2)
            vis_sim = SceneProcessor._feature_similarity(vis1, vis2)
            score += vis_sim * weights["visual"]
            total_weight += weights["visual"]
        
        # Audio similarity
        if weights.get("audio", 0) > 0:
            aud1 = SceneProcessor.extract_audio_features(scene1)
            aud2 = SceneProcessor.extract_audio_features(scene2)
            aud_sim = SceneProcessor._feature_similarity(aud1, aud2)
            score += aud_sim * weights["audio"]
            total_weight += weights["audio"]
        
        # Duration similarity
        if weights.get("duration", 0) > 0:
            dur1 = SceneProcessor.calculate_duration(scene1)
            dur2 = SceneProcessor.calculate_duration(scene2)
            dur_sim = 1.0 - min(1.0, abs(dur1 - dur2) / (max(dur1, dur2) + 0.1))
            score += dur_sim * weights["duration"]
            total_weight += weights["duration"]
        
        return score / total_weight if total_weight > 0 else 0.5
    
    @staticmethod
    def _feature_similarity(
        features1: Dict[str, Any],
        features2: Dict[str, Any],
    ) -> float:
        """Calculate feature dict similarity.
        
        Args:
            features1: First feature dict
            features2: Second feature dict
            
        Returns:
            Similarity score [0.0, 1.0]
        """
        if not features1 or not features2:
            return 0.5
        
        matches = 0
        total = 0
        
        for key in set(list(features1.keys()) + list(features2.keys())):
            val1 = features1.get(key, 0.5)
            val2 = features2.get(key, 0.5)
            
            # Handle numeric comparison
            try:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalized distance
                    dist = abs(val1 - val2) / max(1.0, abs(val1) + abs(val2))
                    matches += 1.0 - dist
                elif val1 == val2:
                    matches += 1.0
                total += 1
            except (TypeError, ValueError):
                total += 1
        
        return matches / total if total > 0 else 0.5
    
    # =========================================================================
    # Scene Filtering
    # =========================================================================
    
    @staticmethod
    def filter_scenes_by_duration(
        scenes: List[Dict[str, Any]],
        min_duration: float = 0.0,
        max_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filter scenes by duration bounds.
        
        Args:
            scenes: List of scenes
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds (None = unlimited)
            
        Returns:
            Filtered scenes
        """
        result = []
        for scene in scenes:
            duration = SceneProcessor.calculate_duration(scene)
            if duration < min_duration:
                continue
            if max_duration and duration > max_duration:
                continue
            result.append(scene)
        return result
    
    @staticmethod
    def filter_scenes_by_feature(
        scenes: List[Dict[str, Any]],
        feature_key: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Filter scenes by feature value range.
        
        Args:
            scenes: List of scenes
            feature_key: Feature to filter on (e.g., "motion", "loudness_db")
            min_value: Minimum feature value
            max_value: Maximum feature value
            
        Returns:
            Filtered scenes
        """
        result = []
        for scene in scenes:
            visual = SceneProcessor.extract_visual_features(scene)
            audio = SceneProcessor.extract_audio_features(scene)
            
            value = visual.get(feature_key) or audio.get(feature_key)
            if value is None:
                continue
            
            try:
                value = float(value)
                if min_value <= value <= max_value:
                    result.append(scene)
            except (ValueError, TypeError):
                pass
        
        return result
    
    # =========================================================================
    # Scene Merging
    # =========================================================================
    
    @staticmethod
    def merge_adjacent_scenes(
        scenes: List[Dict[str, Any]],
        max_gap: float = 0.1,  # seconds
    ) -> List[Dict[str, Any]]:
        """Merge adjacent scenes separated by small gaps.
        
        Useful for consolidating fragmented scene detections.
        
        Args:
            scenes: Sorted scenes by start time
            max_gap: Maximum gap (seconds) before merging
            
        Returns:
            Merged scenes
        """
        if not scenes:
            return []
        
        merged = []
        current = dict(scenes[0])
        
        for scene in scenes[1:]:
            gap = scene.get("start_time", 0) - current.get("end_time", 0)
            
            if gap <= max_gap:
                # Merge: extend current scene's end time
                current["end_time"] = scene.get("end_time", current.get("end_time"))
                # Merge metadata (audio/visual features, etc.)
                for key in ["visual_features", "audio_features", "metadata"]:
                    if key in scene and scene[key]:
                        if key not in current:
                            current[key] = {}
                        current[key].update(scene.get(key, {}))
            else:
                # Gap too large - save current and start new
                merged.append(current)
                current = dict(scene)
        
        # Add final scene
        merged.append(current)
        return merged
    
    # =========================================================================
    # Start Point Fallback
    # =========================================================================
    
    @staticmethod
    def calculate_fallback_start(
        scene_start: float,
        scene_end: float,
        target_duration: float,
    ) -> float:
        """Calculate fallback start point when clip doesn't fill target duration.
        
        Used when a scene is shorter than needed for the timeline position.
        
        Args:
            scene_start: Scene start time
            scene_end: Scene end time
            target_duration: Required clip duration
            
        Returns:
            Recommended start point (may be before scene_start)
        """
        available_duration = scene_end - scene_start
        
        # If scene is long enough, start from beginning
        if available_duration >= target_duration:
            return scene_start
        
        # If scene is too short, start earlier to get full duration
        deficit = target_duration - available_duration
        fallback_start = scene_start - deficit
        
        # Don't go negative
        return max(0.0, fallback_start)


__all__ = [
    "Scene",
    "SceneProcessor",
]
