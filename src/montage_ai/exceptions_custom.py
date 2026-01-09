"""
Custom exceptions for Montage AI with actionable error messages.

Usage:
    from montage_ai.exceptions_custom import OpticalFlowTimeout, ProxyGenerationFailed
    
    try:
        compute_optical_flow(...)
    except OpticalFlowTimeout as e:
        logger.error(e.user_message)
        logger.debug(e.technical_details)
"""

from typing import Optional


class MontageException(Exception):
    """Base exception for all Montage AI errors."""
    
    def __init__(
        self,
        user_message: str,
        technical_details: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        """
        Initialize exception with user-friendly and technical messages.
        
        Args:
            user_message: Human-readable error message
            technical_details: Technical debug info (not shown to users)
            suggestion: How to fix or work around the error
        """
        self.user_message = user_message
        self.technical_details = technical_details or ""
        self.suggestion = suggestion or ""
        
        # Build full message for logging
        full_msg = user_message
        if suggestion:
            full_msg += f"\n💡 Try: {suggestion}"
        
        super().__init__(full_msg)
    
    def __str__(self) -> str:
        """Return user-friendly message."""
        return self.user_message


class OpticalFlowTimeout(MontageException):
    """Optical flow analysis exceeded timeout threshold."""
    
    def __init__(self, duration_seconds: float, scene_number: int, total_scenes: int):
        user_message = (
            f"❌ Optical flow analysis timeout on scene {scene_number}/{total_scenes}.\n"
            f"   Large video (likely {duration_seconds:.0f}s+) requires more time for frame analysis."
        )
        
        suggestion = (
            "1. Increase FFMPEG_LONG_TIMEOUT (currently 600s)\n"
            "   export FFMPEG_LONG_TIMEOUT=1800\n"
            "2. OR disable optical flow for large videos\n"
            "   export ENABLE_OPTICAL_FLOW=false\n"
            "3. OR use proxy mode (auto-enabled for 10+ min videos)\n"
            "   Proxy mode generates 720p analysis video (8-10x faster)"
        )
        
        super().__init__(
            user_message=user_message,
            technical_details=f"Timeout at scene {scene_number}/{total_scenes} after 600s",
            suggestion=suggestion
        )


class ProxyGenerationFailed(MontageException):
    """Proxy video generation failed."""
    
    def __init__(self, reason: str, video_size_mb: Optional[float] = None):
        video_info = f" ({video_size_mb:.0f}MB)" if video_size_mb else ""
        
        user_message = (
            f"❌ Failed to generate 720p analysis proxy{video_info}.\n"
            f"   This would speed up scene detection and optical flow 8-10x."
        )
        
        suggestion = (
            "1. Check disk space in /tmp (need 2-3x video size)\n"
            "2. Verify FFmpeg is installed: ffmpeg -version\n"
            "3. Try disabling proxy mode\n"
            "   export ENABLE_PROXY_ANALYSIS=false\n"
            "   (Will be slower but will work)"
        )
        
        super().__init__(
            user_message=user_message,
            technical_details=f"Proxy generation error: {reason}",
            suggestion=suggestion
        )


class SceneDetectionFailed(MontageException):
    """Scene detection failed or returned no results."""
    
    def __init__(self, video_path: str, reason: str):
        user_message = (
            f"❌ Scene detection failed for {video_path}.\n"
            f"   Could not analyze video structure."
        )
        
        suggestion = (
            "1. Verify video is not corrupted\n"
            "   ffmpeg -i video.mp4 -f null -\n"
            "2. Check video codec is supported (H.264, VP9, etc.)\n"
            "3. Try reducing video resolution\n"
            "   export MAX_ANALYSIS_RESOLUTION=1080"
        )
        
        super().__init__(
            user_message=user_message,
            technical_details=f"Scene detection error: {reason}",
            suggestion=suggestion
        )


class ResourceThresholdExceeded(MontageException):
    """System resource limits exceeded (memory, CPU, etc)."""
    
    def __init__(self, resource_type: str, current: float, limit: float, unit: str):
        user_message = (
            f"❌ {resource_type} usage too high.\n"
            f"   Current: {current:.1f}{unit} / Limit: {limit:.1f}{unit}"
        )
        
        suggestion = (
            "1. Close other applications\n"
            "2. Reduce batch size: export PROCESSING_BATCH_SIZE=2\n"
            "3. Reduce video resolution for analysis\n"
            "4. Enable proxy mode (if not already)\n"
            "   export ENABLE_PROXY_ANALYSIS=true"
        )
        
        super().__init__(
            user_message=user_message,
            technical_details=f"{resource_type} exceeded: {current:.1f}{unit} > {limit:.1f}{unit}",
            suggestion=suggestion
        )


class VideoFormatNotSupported(MontageException):
    """Video format or codec not supported."""
    
    def __init__(self, format_or_codec: str, supported_list: list):
        user_message = (
            f"❌ Video format/codec '{format_or_codec}' not supported.\n"
            f"   Supported: {', '.join(supported_list[:3])}..."
        )
        
        suggestion = (
            f"Convert video to H.264/MP4:\n"
            f"  ffmpeg -i input.{format_or_codec.lower()} -c:v libx264 -c:a aac output.mp4"
        )
        
        super().__init__(
            user_message=user_message,
            technical_details=f"Unsupported format: {format_or_codec}. Supported: {supported_list}",
            suggestion=suggestion
        )
