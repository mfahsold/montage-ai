from .base import CGPUJob, JobStatus, JobResult
from .manager import CGPUJobManager

# Job implementations (lazy imports to avoid heavy deps at import time)
__all__ = [
    # Core
    "CGPUJob",
    "JobStatus",
    "JobResult",
    "CGPUJobManager",
    # Jobs (imported on demand)
    "TranscribeJob",
    "UpscaleJob",
    "StabilizeJob",
    "SceneDetectionJob",
    "BeatAnalysisJob",
    "FFmpegRenderJob",
    "VoiceIsolationJob",
    "NoiseReductionJob",
    # Future jobs (skeleton implementations)
    "InterpolationJob",
    "LUTGeneratorJob",
]


def __getattr__(name):
    """Lazy import job implementations."""
    if name == "TranscribeJob":
        from .transcribe import TranscribeJob
        return TranscribeJob
    elif name == "UpscaleJob":
        from .upscale import UpscaleJob
        return UpscaleJob
    elif name == "StabilizeJob":
        from .stabilize import StabilizeJob
        return StabilizeJob
    elif name == "SceneDetectionJob":
        from .analysis import SceneDetectionJob
        return SceneDetectionJob
    elif name == "BeatAnalysisJob":
        from .analysis import BeatAnalysisJob
        return BeatAnalysisJob
    elif name == "FFmpegRenderJob":
        from .render import FFmpegRenderJob
        return FFmpegRenderJob
    # Voice/Audio jobs
    elif name == "VoiceIsolationJob":
        from .voice_isolation import VoiceIsolationJob
        return VoiceIsolationJob
    elif name == "NoiseReductionJob":
        from .voice_isolation import NoiseReductionJob
        return NoiseReductionJob
    # Future jobs (skeleton implementations)
    elif name == "InterpolationJob":
        from .interpolation import InterpolationJob
        return InterpolationJob
    elif name == "LUTGeneratorJob":
        from .lut_generator import LUTGeneratorJob
        return LUTGeneratorJob
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
