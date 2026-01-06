
"""
Highlight Detector - AI-Powered Highlight Detection

Identifies the "best" moments in raw footage by combining multiple signals:
1. Audio Energy (Loud, exciting moments)
2. Visual Action (High motion)
3. Face Presence (Human interest)
4. Semantic Relevance (Matches prompt keywords)

Usage:
    detector = HighlightDetector(scenes, energy_profile)
    highlights = detector.detect(top_k=5)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from ..scene_analysis import Scene, SceneAnalysis
from ..audio_analysis import EnergyProfile
from ..logger import logger

@dataclass
class HighlightMoment:
    """A detected highlight moment."""
    start: float
    end: float
    score: float
    signals: Dict[str, float]  # Breakdown of score components (audio, visual, face)
    description: str = ""

class HighlightDetector:
    def __init__(self, scenes: List[Scene], energy_profile: Optional[EnergyProfile] = None):
        self.scenes = scenes
        self.energy_profile = energy_profile
        
    def detect(self, top_k: int = 5, min_duration: float = 3.0, max_duration: float = 60.0) -> List[HighlightMoment]:
        """
        Detect top highlights based on multi-modal scoring.
        """
        candidates = []
        
        # Pre-calculate audio energy stats if available
        audio_threshold = 0.0
        if self.energy_profile and len(self.energy_profile.rms) > 0:
             audio_threshold = float(np.percentile(self.energy_profile.rms, 70))

        for scene in self.scenes:
            # Skip invalid duration
            start = getattr(scene, 'start_time', getattr(scene, 'start', 0.0))
            end = getattr(scene, 'end_time', getattr(scene, 'end', 0.0))
            duration = end - start
            if duration < min_duration or duration > max_duration:
                continue
                
            # 1. Base Score: Action Level
            # High action = 1.0, Medium = 0.5, Low = 0.2
            action_score = 0.5
            analysis = scene.meta.get('analysis') if hasattr(scene, 'meta') else None
            
            if analysis:
                if str(analysis.action).lower() == "high":
                    action_score = 1.0
                elif str(analysis.action).lower() == "low":
                    action_score = 0.2
            
            # 2. Face Bonus
            face_score = 0.0
            if analysis and analysis.face_count > 0:
                face_score = min(1.0, analysis.face_count * 0.2) # Up to 1.0 for 5 faces
                
            # 3. Audio Energy Score
            audio_score = 0.0
            if self.energy_profile and len(self.energy_profile.times) > 0:
                # Get energy for this segment
                # Find indices in energy profile
                start_val = getattr(scene, 'start_time', getattr(scene, 'start', 0))
                end_val = getattr(scene, 'end_time', getattr(scene, 'end', 0))
                
                start_idx = np.searchsorted(self.energy_profile.times, start_val)
                end_idx = np.searchsorted(self.energy_profile.times, end_val)
                
                if start_idx < end_idx and start_idx < len(self.energy_profile.rms):
                    segment_rms = self.energy_profile.rms[start_idx:end_idx]
                    if len(segment_rms) > 0:
                        peak = float(np.max(segment_rms))
                        # Normalize energy: 0.0 if below threshold, linear up to max
                        if peak > audio_threshold:
                            audio_score = min(1.0, peak * 2.5) 
            
            # 4. Visual Quality
            quality_penalty = 1.0
            if analysis and str(analysis.quality).upper() == "NO":
                quality_penalty = 0.1 # Heavily penalize bad shots
                
            # Composite Score
            # Heavy weight on Audio and Action for highlights
            final_score = (
                (0.35 * action_score) +
                (0.35 * audio_score) + 
                (0.30 * face_score)
            ) * quality_penalty
            
            candidates.append(HighlightMoment(
                start=getattr(scene, 'start_time', getattr(scene, 'start', 0)),
                end=getattr(scene, 'end_time', getattr(scene, 'end', 0)),
                score=final_score,
                signals={
                    "action": action_score,
                    "audio": audio_score,
                    "face": face_score,
                    "quality": quality_penalty
                },
                description=analysis.description if analysis else ""
            ))
            
        # Sort and return top K
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

