"""
Montage AI Core Module

Contains the core pipeline components:
- MontageBuilder: Main orchestration class
- MontageContext: Job state container
"""

from .montage_builder import (
    MontageBuilder,
    MontageContext,
    MontageResult,
    AudioAnalysisResult,
    SceneInfo,
    OutputProfile,
    ClipMetadata,
)

__all__ = [
    "MontageBuilder",
    "MontageContext",
    "MontageResult",
    "AudioAnalysisResult",
    "SceneInfo",
    "OutputProfile",
    "ClipMetadata",
]
