"""
Real Audio Analysis Objects - Replace Mocks

Provides non-mock implementations of audio analysis data structures
for use in tests and production code.

Replaces:
  - MagicMock() for beat_info
  - MagicMock() for energy_profile

Usage:
    from montage_ai.audio_analysis_objects import BeatInfo, EnergyProfile
    
    beat_info = BeatInfo(
        tempo=120.0,
        beat_times=[0.5, 1.0, 1.5],
        duration=60.0,
        sample_rate=22050
    )
    
    energy = EnergyProfile(
        times=[0.0, 0.5, 1.0],
        rms=[0.2, 0.8, 0.5],
        duration=60.0
    )
"""

from typing import List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BeatInfo:
    """Real beat detection result object."""
    
    tempo: float
    """Beats per minute (BPM)."""
    
    beat_times: List[float]
    """Beat positions in seconds."""
    
    duration: float
    """Audio duration in seconds."""
    
    sample_rate: int = 22050
    """Sample rate in Hz."""
    
    confidence: float = 1.0
    """Confidence score [0.0, 1.0]."""
    
    def __post_init__(self):
        """Validate and normalize beat times."""
        if isinstance(self.beat_times, np.ndarray):
            self.beat_times = self.beat_times.tolist()
        
        # Ensure all beat times are within duration
        self.beat_times = [t for t in self.beat_times if t <= self.duration]
    
    def to_numpy(self) -> np.ndarray:
        """Convert beat times to numpy array."""
        return np.array(self.beat_times)
    
    def get_beat_intervals(self) -> List[float]:
        """Get intervals between consecutive beats in seconds."""
        if len(self.beat_times) < 2:
            return []
        return [self.beat_times[i+1] - self.beat_times[i] 
                for i in range(len(self.beat_times) - 1)]
    
    def get_average_interval(self) -> float:
        """Get average interval between beats."""
        intervals = self.get_beat_intervals()
        return np.mean(intervals) if intervals else 0.0
    
    def beat_grid(self, resolution: int = 4) -> List[float]:
        """Generate beat grid at specified resolution (beats per measure)."""
        if not self.beat_times:
            return []
        interval = self.get_average_interval()
        if interval == 0:
            return self.beat_times
        return np.arange(self.beat_times[0], self.duration, interval / resolution).tolist()


@dataclass
class EnergyProfile:
    """Real energy/loudness analysis result."""
    
    times: List[float]
    """Time samples in seconds."""
    
    rms: List[float]
    """RMS energy values [0.0, 1.0]."""
    
    duration: float
    """Audio duration in seconds."""
    
    peaks: List[float] = field(default_factory=list)
    """Peak positions (if detected)."""
    
    def __post_init__(self):
        """Validate and normalize energy data."""
        if isinstance(self.times, np.ndarray):
            self.times = self.times.tolist()
        if isinstance(self.rms, np.ndarray):
            self.rms = self.rms.tolist()
        
        # Ensure arrays are same length
        min_len = min(len(self.times), len(self.rms))
        self.times = self.times[:min_len]
        self.rms = self.rms[:min_len]
        
        # Ensure all values are within duration
        valid_indices = [i for i, t in enumerate(self.times) if t <= self.duration]
        self.times = [self.times[i] for i in valid_indices]
        self.rms = [self.rms[i] for i in valid_indices]
    
    def to_numpy(self) -> tuple:
        """Convert to numpy arrays (times, rms)."""
        return np.array(self.times), np.array(self.rms)
    
    def get_mean_energy(self) -> float:
        """Get mean energy across duration."""
        return float(np.mean(self.rms)) if self.rms else 0.0
    
    def get_max_energy(self) -> float:
        """Get maximum energy."""
        return float(np.max(self.rms)) if self.rms else 0.0
    
    def get_min_energy(self) -> float:
        """Get minimum energy."""
        return float(np.min(self.rms)) if self.rms else 0.0
    
    def get_energy_at_time(self, time: float) -> float:
        """Get energy value at specific time (linear interpolation)."""
        if not self.times or not self.rms:
            return 0.0
        
        # Find closest time
        times_arr = np.array(self.times)
        idx = np.searchsorted(times_arr, time)
        
        if idx == 0:
            return self.rms[0]
        if idx == len(self.times):
            return self.rms[-1]
        
        # Linear interpolation
        t0, t1 = self.times[idx-1], self.times[idx]
        e0, e1 = self.rms[idx-1], self.rms[idx]
        
        if t1 == t0:
            return e0
        
        alpha = (time - t0) / (t1 - t0)
        return e0 + alpha * (e1 - e0)
    
    def get_high_energy_regions(self, threshold: float = 0.7) -> List[tuple]:
        """Get regions with energy above threshold.
        
        Args:
            threshold: Energy threshold [0.0, 1.0]
            
        Returns:
            List of (start, end) tuples for high-energy regions
        """
        regions = []
        in_region = False
        region_start = 0.0
        
        for time, energy in zip(self.times, self.rms):
            if energy >= threshold and not in_region:
                region_start = time
                in_region = True
            elif energy < threshold and in_region:
                regions.append((region_start, time))
                in_region = False
        
        # Close final region if needed
        if in_region:
            regions.append((region_start, self.duration))
        
        return regions


@dataclass
class SceneDetectionResult:
    """Real scene detection result."""
    
    threshold: float
    """Detection threshold used."""
    
    scenes: List[dict]
    """Scene dicts with 'start' and 'end' keys."""
    
    total_scenes: int
    """Total number of scenes detected."""
    
    def __post_init__(self):
        """Normalize scene data."""
        self.total_scenes = len(self.scenes)
    
    def get_durations(self) -> List[float]:
        """Get duration of each scene."""
        return [s.get('end') - s.get('start', 0) for s in self.scenes]
    
    def get_average_scene_duration(self) -> float:
        """Get average scene duration."""
        durations = self.get_durations()
        return float(np.mean(durations)) if durations else 0.0
    
    def get_scene_at_time(self, time: float) -> Optional[dict]:
        """Get scene that contains given time."""
        for scene in self.scenes:
            if scene['start'] <= time <= scene['end']:
                return scene
        return None


@dataclass
class ColorAnalysisResult:
    """Real color analysis result."""
    
    dominant_color: tuple  # (R, G, B)
    """Dominant color as RGB tuple [0-255]."""
    
    color_histogram: dict = field(default_factory=dict)
    """Color distribution."""
    
    saturation_mean: float = 0.5
    """Average saturation [0.0, 1.0]."""
    
    brightness_mean: float = 0.5
    """Average brightness [0.0, 1.0]."""
    
    contrast: float = 0.5
    """Image contrast [0.0, 1.0]."""
    
    def get_dominant_hex(self) -> str:
        """Get dominant color as hex string."""
        r, g, b = int(self.dominant_color[0]), int(self.dominant_color[1]), int(self.dominant_color[2])
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def is_high_saturation(self, threshold: float = 0.7) -> bool:
        """Check if image is highly saturated."""
        return self.saturation_mean >= threshold
    
    def is_bright(self, threshold: float = 0.6) -> bool:
        """Check if image is bright."""
        return self.brightness_mean >= threshold
    
    def is_high_contrast(self, threshold: float = 0.6) -> bool:
        """Check if image has high contrast."""
        return self.contrast >= threshold


__all__ = [
    "BeatInfo",
    "EnergyProfile",
    "SceneDetectionResult",
    "ColorAnalysisResult",
]
