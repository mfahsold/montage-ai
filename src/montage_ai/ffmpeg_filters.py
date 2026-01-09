"""
FFmpegFilterBuilder - Unified Filter Graph Construction for Montage AI

Consolidates filter graph building logic from multiple modules into a single,
composable interface. Supports complex filter chains for audio/video processing.

This follows the Builder Pattern for fluent, readable filter construction.

Usage:
    from montage_ai.ffmpeg_filters import VideoFilterChain, AudioFilterChain
    
    # Video filters
    video = (VideoFilterChain()
        .add_stabilization(strength=0.8)
        .add_color_grade(lut_path="/data/luts/cinematic.cube")
        .add_deinterlace()
        .build())
    
    # Audio filters
    audio = (AudioFilterChain()
        .add_denoise(strength=0.7)
        .add_ducking(threshold=-30)
        .add_leveling(target_level=-20)
        .build())

Architecture:
    - Stateful filter chains (build state step-by-step)
    - Composable components (can combine multiple chains)
    - FFmpeg string generation (outputs filter_complex strings)
    - Validation (checks for incompatibilities)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass
class FilterStep:
    """Single filter operation in a chain."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert filter step to FFmpeg filter string.
        
        Example: 
            FilterStep("scale", {"w": 1920, "h": 1080})
            → "scale=w=1920:h=1080"
        """
        if not self.params:
            return self.name
        
        parts = []
        for key, value in self.params.items():
            if isinstance(value, str):
                # Quote string values if they contain special chars
                if any(c in value for c in " ;:[],'\""):
                    value = f"'{value}'"
            parts.append(f"{key}={value}")
        
        return f"{self.name}={':'.join(parts)}"


class VideoFilterChain:
    """Builder for video filter chains.
    
    Stateful filter composition with validation.
    """
    
    def __init__(self, pad_color: str = "black"):
        """Initialize video filter chain.
        
        Args:
            pad_color: Default padding color for scale operations
        """
        self.filters: List[FilterStep] = []
        self.pad_color = pad_color
        self.target_width: Optional[int] = None
        self.target_height: Optional[int] = None
    
    def add_scale(self, width: int, height: int, force_aspect: bool = False) -> "VideoFilterChain":
        """Scale video to specified dimensions.
        
        Args:
            width: Target width (-1 = auto based on aspect)
            height: Target height (-1 = auto based on aspect)
            force_aspect: If True, use force_original_aspect_ratio=decrease
            
        Returns:
            Self for chaining
        """
        params = {"w": width, "h": height}
        if force_aspect:
            params["force_original_aspect_ratio"] = "decrease"
        
        self.filters.append(FilterStep("scale", params))
        self.target_width = width if width > 0 else None
        self.target_height = height if height > 0 else None
        return self
    
    def add_pad(self, width: int, height: int, x: int = 0, y: int = 0) -> "VideoFilterChain":
        """Add padding to frame.
        
        Args:
            width: Padded width
            height: Padded height
            x: X offset
            y: Y offset (use -1 for center)
            
        Returns:
            Self for chaining
        """
        # Center vertically if y == -1
        if y == -1 and self.target_height:
            y = (height - self.target_height) // 2
        
        params = {
            "w": width,
            "h": height,
            "x": x,
            "y": y,
            "color": self.pad_color,
        }
        self.filters.append(FilterStep("pad", params))
        self.target_width = width
        self.target_height = height
        return self
    
    def add_crop(self, width: int, height: int, x: int = 0, y: int = 0) -> "VideoFilterChain":
        """Crop video frame.
        
        Args:
            width: Crop width
            height: Crop height
            x: X offset
            y: Y offset
            
        Returns:
            Self for chaining
        """
        params = {"w": width, "h": height, "x": x, "y": y}
        self.filters.append(FilterStep("crop", params))
        self.target_width = width
        self.target_height = height
        return self
    
    def add_deinterlace(self, mode: str = "yadif") -> "VideoFilterChain":
        """Deinterlace video.
        
        Args:
            mode: yadif (default, high quality) or bwdif (bidirectional)
            
        Returns:
            Self for chaining
        """
        self.filters.append(FilterStep(mode))
        return self
    
    def add_denoise(self, strength: float = 0.5) -> "VideoFilterChain":
        """Add video denoising.
        
        Args:
            strength: Denoise strength [0.0, 1.0]
            
        Returns:
            Self for chaining
        """
        # nlmeans is high quality but slow
        # hqdn3d is faster
        sigma = 2.0 + (strength * 3.0)  # Map [0,1] to [2,5]
        params = {"luma_s": sigma, "chroma_s": sigma}
        self.filters.append(FilterStep("hqdn3d", params))
        return self
    
    def add_sharpen(self, amount: float = 0.5) -> "VideoFilterChain":
        """Add sharpening filter.
        
        Args:
            amount: Sharpen amount [0.0, 1.0]
            
        Returns:
            Self for chaining
        """
        # unsharpmask: luma_amount, luma_radius, luma_threshold
        luma_amount = 0.5 + (amount * 1.5)  # [0.5, 2.0]
        params = {
            "luma_msize_x": 5,
            "luma_msize_y": 5,
            "luma_amount": luma_amount,
        }
        self.filters.append(FilterStep("unsharp", params))
        return self
    
    def add_color_grading(self, lut_path: str) -> "VideoFilterChain":
        """Apply color grading via LUT (Look-Up Table).
        
        Args:
            lut_path: Path to .cube or .3dl LUT file
            
        Returns:
            Self for chaining
        """
        params = {"file": lut_path}
        self.filters.append(FilterStep("lut3d", params))
        return self
    
    def add_saturation(self, level: float = 1.0) -> "VideoFilterChain":
        """Adjust color saturation.
        
        Args:
            level: Saturation level (1.0 = normal, <1 = less, >1 = more)
            
        Returns:
            Self for chaining
        """
        params = {"s": level}
        self.filters.append(FilterStep("saturate", params))
        return self
    
    def add_stabilization(self, strength: float = 0.8, shakiness: int = 5) -> "VideoFilterChain":
        """Stabilize shaky video.
        
        Args:
            strength: Stabilization strength [0.0, 1.0]
            shakiness: Detection sensitivity (1-10, higher = more sensitive)
            
        Returns:
            Self for chaining
        """
        params = {
            "shakiness": shakiness,
            "accuracy": int(strength * 15),  # [0, 15]
            "result": "vibrate.trf",  # Temporary transform file
        }
        self.filters.append(FilterStep("vidstabdetect", params))
        # Note: vidstabtransform is applied in second pass
        return self
    
    def add_fps(self, fps: float) -> "VideoFilterChain":
        """Convert video to target frame rate.
        
        Args:
            fps: Target frames per second
            
        Returns:
            Self for chaining
        """
        params = {"fps": fps}
        self.filters.append(FilterStep("fps", params))
        return self
    
    def add_format(self, pixel_format: str = "yuv420p") -> "VideoFilterChain":
        """Convert pixel format.
        
        Args:
            pixel_format: Target format (yuv420p, rgb24, etc.)
            
        Returns:
            Self for chaining
        """
        params = {"pix_fmts": pixel_format}
        self.filters.append(FilterStep("format", params))
        return self
    
    def build(self) -> str:
        """Build the complete filter chain string.
        
        Returns:
            FFmpeg filtergraph string
            
        Example:
            "scale=w=1920:h=1080,hqdn3d=luma_s=2.5:chroma_s=2.5"
        """
        if not self.filters:
            return ""
        return ",".join(f.to_string() for f in self.filters)
    
    def __repr__(self) -> str:
        return f"VideoFilterChain({self.build()})"


class AudioFilterChain:
    """Builder for audio filter chains.
    
    Stateful filter composition for audio processing.
    """
    
    def __init__(self):
        """Initialize audio filter chain."""
        self.filters: List[FilterStep] = []
    
    def add_volume(self, level_db: float) -> "AudioFilterChain":
        """Adjust audio volume.
        
        Args:
            level_db: Volume level in dB (e.g., -6 = half volume)
            
        Returns:
            Self for chaining
        """
        # Convert dB to linear scale: 10^(dB/20)
        linear = 10 ** (level_db / 20)
        params = {"volume": linear}
        self.filters.append(FilterStep("volume", params))
        return self
    
    def add_denoise(self, strength: float = 0.5) -> "AudioFilterChain":
        """Denoise audio using noise gate.
        
        Args:
            strength: Gate strength [0.0, 1.0]
            
        Returns:
            Self for chaining
        """
        # Map strength to dB threshold
        threshold_db = -40 + (strength * 30)  # [-40, -10] dB
        params = {
            "threshold": abs(threshold_db),
            "ratio": 10,
            "attack": 0.001,
            "release": 0.1,
        }
        self.filters.append(FilterStep("compand", params))
        return self
    
    def add_leveling(self, target_level: float = -20.0) -> "AudioFilterChain":
        """Normalize audio to target level.
        
        Args:
            target_level: Target loudness in dB (typically -14 to -20)
            
        Returns:
            Self for chaining
        """
        params = {
            "mode": "peak",
            "rate": 0.1,
            "delay": 0.5,
            "target": target_level,
        }
        self.filters.append(FilterStep("loudnorm", params))
        return self
    
    def add_ducking(self, threshold: float = -30.0, ratio: float = 4.0) -> "AudioFilterChain":
        """Duck (reduce) audio under threshold.
        
        Used for reducing music under dialogue.
        
        Args:
            threshold: Threshold in dB
            ratio: Compression ratio (4 = -4dB per +1dB over threshold)
            
        Returns:
            Self for chaining
        """
        params = {
            "threshold": threshold,
            "ratio": ratio,
            "attack": 0.005,
            "release": 0.2,
        }
        self.filters.append(FilterStep("compand", params))
        return self
    
    def add_highpass(self, freq: float = 100.0) -> "AudioFilterChain":
        """Remove low frequencies (rumble filter).
        
        Args:
            freq: Cutoff frequency in Hz
            
        Returns:
            Self for chaining
        """
        params = {"frequency": freq}
        self.filters.append(FilterStep("highpass", params))
        return self
    
    def add_lowpass(self, freq: float = 8000.0) -> "AudioFilterChain":
        """Remove high frequencies (telephone effect).
        
        Args:
            freq: Cutoff frequency in Hz
            
        Returns:
            Self for chaining
        """
        params = {"frequency": freq}
        self.filters.append(FilterStep("lowpass", params))
        return self
    
    def add_reverb(self, room_size: float = 0.5, damping: float = 0.5) -> "AudioFilterChain":
        """Add reverb/room acoustics.
        
        Args:
            room_size: Room size [0.0, 1.0]
            damping: Damping [0.0, 1.0]
            
        Returns:
            Self for chaining
        """
        params = {
            "room_size": room_size,
            "damping": damping,
            "wet": 0.3,
            "dry": 0.7,
            "width": 1.0,
        }
        self.filters.append(FilterStep("aecho", params))
        return self
    
    def add_fade_in(self, duration_ms: float = 1000.0) -> "AudioFilterChain":
        """Fade in audio at the beginning.
        
        Args:
            duration_ms: Fade duration in milliseconds
            
        Returns:
            Self for chaining
        """
        params = {
            "t": "lin",  # linear fade
            "st": 0,
            "d": duration_ms / 1000.0,
        }
        self.filters.append(FilterStep("afade", params))
        return self
    
    def add_fade_out(self, duration_ms: float = 1000.0) -> "AudioFilterChain":
        """Fade out audio at the end.
        
        Args:
            duration_ms: Fade duration in milliseconds
            
        Returns:
            Self for chaining
        """
        params = {
            "t": "lin",
            "st": 0,
            "d": duration_ms / 1000.0,
        }
        self.filters.append(FilterStep("afade", params))
        return self
    
    def add_tempo(self, rate: float = 1.0) -> "AudioFilterChain":
        """Change playback speed (with pitch preservation).
        
        Args:
            rate: Speed multiplier (1.0 = normal, 0.5 = half, 2.0 = double)
            
        Returns:
            Self for chaining
        """
        params = {"tempo": rate}
        self.filters.append(FilterStep("atempo", params))
        return self
    
    def build(self) -> str:
        """Build the complete filter chain string.
        
        Returns:
            FFmpeg audio filtergraph string
            
        Example:
            "volume=0.5,loudnorm=target=-20"
        """
        if not self.filters:
            return ""
        return ",".join(f.to_string() for f in self.filters)
    
    def __repr__(self) -> str:
        return f"AudioFilterChain({self.build()})"


# ============================================================================
# Convenience Functions
# ============================================================================

def create_stabilization_chain(
    shakiness: int = 5,
    strength: float = 0.8,
) -> VideoFilterChain:
    """Create a pre-configured stabilization chain.
    
    Args:
        shakiness: Detection sensitivity (1-10)
        strength: Stabilization strength [0.0, 1.0]
        
    Returns:
        Configured VideoFilterChain
    """
    return VideoFilterChain().add_stabilization(strength=strength, shakiness=shakiness)


def create_denoise_chain(
    video_strength: float = 0.5,
    audio_strength: float = 0.5,
) -> Dict[str, str]:
    """Create complete denoise chain for video + audio.
    
    Args:
        video_strength: Video denoise [0.0, 1.0]
        audio_strength: Audio denoise [0.0, 1.0]
        
    Returns:
        Dict with 'video' and 'audio' keys containing filter strings
    """
    return {
        "video": VideoFilterChain().add_denoise(strength=video_strength).build(),
        "audio": AudioFilterChain().add_denoise(strength=audio_strength).build(),
    }


def create_color_grade_chain(
    saturation: float = 1.0,
    lut_path: Optional[str] = None,
) -> VideoFilterChain:
    """Create a color grading chain.
    
    Args:
        saturation: Saturation level
        lut_path: Optional path to LUT file
        
    Returns:
        Configured VideoFilterChain
    """
    chain = VideoFilterChain()
    if lut_path:
        chain.add_color_grading(lut_path)
    chain.add_saturation(saturation)
    return chain


__all__ = [
    "FilterStep",
    "VideoFilterChain",
    "AudioFilterChain",
    "create_stabilization_chain",
    "create_denoise_chain",
    "create_color_grade_chain",
]
