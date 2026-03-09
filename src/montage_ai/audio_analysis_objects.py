"""Compatibility audio analysis objects.

Deprecated compatibility layer for code paths that still import
`audio_analysis_objects`. New code should use objects from `audio_analysis.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import warnings

import numpy as np

warnings.warn(
    "audio_analysis_objects.py is deprecated. Use data classes from audio_analysis.py instead.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class BeatInfo:
    """Backward-compatible beat detection result object."""

    tempo: float
    beat_times: List[float]
    duration: float
    sample_rate: int = 22050
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.beat_times, np.ndarray):
            self.beat_times = self.beat_times.tolist()
        self.beat_times = [float(t) for t in self.beat_times if float(t) <= self.duration]


@dataclass
class EnergyProfile:
    """Backward-compatible energy profile object."""

    times: List[float]
    rms: List[float]
    duration: float
    peaks: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.times, np.ndarray):
            self.times = self.times.tolist()
        if isinstance(self.rms, np.ndarray):
            self.rms = self.rms.tolist()
        if isinstance(self.peaks, np.ndarray):
            self.peaks = self.peaks.tolist()

        count = min(len(self.times), len(self.rms))
        self.times = [float(v) for v in self.times[:count]]
        self.rms = [float(v) for v in self.rms[:count]]
        self.peaks = [float(v) for v in self.peaks]

        valid_indices = [i for i, t in enumerate(self.times) if t <= self.duration]
        self.times = [self.times[i] for i in valid_indices]
        self.rms = [self.rms[i] for i in valid_indices]


__all__ = ["BeatInfo", "EnergyProfile"]
