"""
Montage AI Exception Hierarchy

Structured exception types for better error handling and debugging.
All exceptions inherit from MontageError for easy catching.

Usage:
    from montage_ai.exceptions import VideoAnalysisError, RenderError

    try:
        analyze_video(path)
    except VideoAnalysisError as e:
        logger.error(f"Analysis failed: {e}")
"""


class MontageError(Exception):
    """Base exception for all Montage AI errors."""
    pass


# =============================================================================
# Video Analysis Errors
# =============================================================================

class VideoAnalysisError(MontageError):
    """Error during video analysis (scene detection, beat detection, etc.)."""
    pass


class SceneDetectionError(VideoAnalysisError):
    """Error during scene detection."""
    pass


class BeatDetectionError(VideoAnalysisError):
    """Error during audio beat detection."""
    pass


class MetadataExtractionError(VideoAnalysisError):
    """Error extracting video/audio metadata via ffprobe."""
    pass


# =============================================================================
# Rendering Errors
# =============================================================================

class RenderError(MontageError):
    """Error during video rendering."""
    pass


class FFmpegError(RenderError):
    """FFmpeg command failed."""

    def __init__(self, message: str, command: str = None, stderr: str = None):
        super().__init__(message)
        self.command = command
        self.stderr = stderr


class MemoryError(RenderError):
    """Out of memory during rendering."""
    pass


class EncoderError(RenderError):
    """Hardware encoder not available or failed."""
    pass


# =============================================================================
# LLM/AI Errors
# =============================================================================

class LLMError(MontageError):
    """Error communicating with LLM backend."""
    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""
    pass


class LLMParseError(LLMError):
    """Failed to parse LLM response."""
    pass


class NoLLMBackendError(LLMError):
    """No LLM backend available."""
    pass


# =============================================================================
# Cloud GPU (cgpu) Errors
# =============================================================================

class CGPUError(MontageError):
    """Error with cgpu cloud GPU service."""
    pass


class CGPUConnectionError(CGPUError):
    """Failed to connect to cgpu service."""
    pass


class CGPUJobError(CGPUError):
    """cgpu job execution failed."""

    def __init__(self, message: str, job_type: str = None, job_id: str = None):
        super().__init__(message)
        self.job_type = job_type
        self.job_id = job_id


class CGPUQuotaError(CGPUError):
    """cgpu quota exceeded."""
    pass


# =============================================================================
# Timeline/Export Errors
# =============================================================================

class TimelineError(MontageError):
    """Error building or exporting timeline."""
    pass


class OTIOExportError(TimelineError):
    """Error exporting to OTIO format."""
    pass


class EDLExportError(TimelineError):
    """Error exporting to EDL format."""
    pass


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(MontageError):
    """Invalid configuration or missing required settings."""
    pass


class InvalidStyleError(ConfigurationError):
    """Unknown or invalid style template."""
    pass


class MissingMediaError(ConfigurationError):
    """Required media files not found."""
    pass


# =============================================================================
# Web UI Errors
# =============================================================================

class WebUIError(MontageError):
    """Error in web UI operations."""
    pass


class SessionError(WebUIError):
    """Session management error."""
    pass


class UploadError(WebUIError):
    """File upload failed."""
    pass
