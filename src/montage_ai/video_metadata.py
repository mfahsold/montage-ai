"""
Video Metadata Module for Montage AI

Provides video metadata extraction, output profile determination, and FFmpeg parameter building.
Uses ffprobe for metadata extraction and weighted median algorithms for profile selection.

Usage:
    from montage_ai.video_metadata import probe_metadata, determine_output_profile

    metadata = probe_metadata("/path/to/video.mp4")
    profile = determine_output_profile(["/video1.mp4", "/video2.mp4"])
"""

import os
import json
import subprocess
import hashlib
import time
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from threading import Lock
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from .config import get_settings
from .logger import logger
from .ffmpeg_utils import build_ffprobe_cmd
from .ffmpeg_config import (
    get_config as get_ffmpeg_config,
    STANDARD_WIDTH_VERTICAL as DEFAULT_STANDARD_WIDTH,
    STANDARD_HEIGHT_VERTICAL as DEFAULT_STANDARD_HEIGHT,
    STANDARD_FPS as DEFAULT_STANDARD_FPS,
    STANDARD_CODEC as DEFAULT_STANDARD_CODEC,
    STANDARD_PROFILE as DEFAULT_STANDARD_PROFILE,
    STANDARD_LEVEL as DEFAULT_STANDARD_LEVEL,
    STANDARD_PIX_FMT as DEFAULT_STANDARD_PIX_FMT,
)

_settings = get_settings()
_ffmpeg_config = get_ffmpeg_config(hwaccel=_settings.gpu.ffmpeg_hwaccel)


# =============================================================================
# Metadata Cache (Performance Optimization)
# =============================================================================

class MetadataCache:
    """
    In-memory cache for video metadata with file-hash validation.

    Uses file size + first/last 64KB hash for validation,
    which handles rsync/cp -a (preserved mtime) cases.

    Performance: ~10x faster for repeated metadata lookups.
    """

    CACHE_TTL_SECONDS = 3600  # 1 hour TTL
    HASH_CHUNK_SIZE = 65536   # 64KB for hash sampling

    def __init__(self, max_size: int = 500):
        self._cache: Dict[str, Tuple[str, float, 'VideoMetadata']] = {}
        self._lock = Lock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _compute_file_hash(self, path: str) -> str:
        """
        Compute a fast file hash based on size + first/last chunks.

        This is faster than full MD5 while still detecting changes.
        """
        try:
            stat = os.stat(path)
            size = stat.st_size

            hasher = hashlib.md5()
            hasher.update(str(size).encode())

            with open(path, 'rb') as f:
                # First chunk
                hasher.update(f.read(self.HASH_CHUNK_SIZE))

                # Last chunk (if file is large enough)
                if size > self.HASH_CHUNK_SIZE * 2:
                    f.seek(-self.HASH_CHUNK_SIZE, 2)
                    hasher.update(f.read(self.HASH_CHUNK_SIZE))

            return hasher.hexdigest()
        except Exception:
            return ""

    def get(self, path: str) -> Optional['VideoMetadata']:
        """Get cached metadata if valid."""
        with self._lock:
            if path not in self._cache:
                self._misses += 1
                return None

            file_hash, cached_at, metadata = self._cache[path]

            # Check TTL
            if time.time() - cached_at > self.CACHE_TTL_SECONDS:
                del self._cache[path]
                self._misses += 1
                return None

            # Validate hash (file might have changed)
            current_hash = self._compute_file_hash(path)
            if current_hash != file_hash:
                del self._cache[path]
                self._misses += 1
                return None

            self._hits += 1
            return metadata

    def set(self, path: str, metadata: 'VideoMetadata') -> None:
        """Cache metadata for a file."""
        with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                # Remove 10% of oldest entries
                items = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in items[:self._max_size // 10]:
                    del self._cache[key]

            file_hash = self._compute_file_hash(path)
            self._cache[path] = (file_hash, time.time(), metadata)

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0.0
        }


# Global cache instance
_metadata_cache = MetadataCache()


def get_metadata_cache_stats() -> Dict[str, Any]:
    """Get current metadata cache statistics."""
    return _metadata_cache.stats


def clear_metadata_cache() -> None:
    """Clear the metadata cache."""
    _metadata_cache.clear()
    logger.debug("Metadata cache cleared")


# =============================================================================
# RAW Codec Detection
# =============================================================================

# Professional RAW codecs that require special handling
RAW_CODECS = {
    "prores_raw": "ProRes RAW (requires FFmpeg with --enable-libprores_raw)",
    "braw": "Blackmagic RAW (requires Blackmagic RAW SDK)",
    "redcode": "RED RAW (requires REDline SDK or FFmpeg plugin)",
    "cinemadng": "CinemaDNG (requires FFmpeg with --enable-libraw)",
    "arriraw": "ARRIRAW (requires ARRI SDK)",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VideoMetadata:
    """Metadata extracted from a video file via ffprobe."""
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    pix_fmt: str
    bitrate: int
    rotation: int = 0
    # Optional field for tests and extended info; not used in core logic
    filesize: int = 0
    # Path is optional in tests; default to empty string when not provided
    path: str = ""

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio (accounting for rotation)."""
        w, h = self.resolution
        return w / h if h > 0 else 1.0

    @property
    def orientation(self) -> str:
        """Determine orientation: horizontal, vertical, or square."""
        ratio = self.aspect_ratio
        if ratio > 1.1:
            return "horizontal"
        elif ratio < 0.9:
            return "vertical"
        return "square"

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) tuple, swapping if rotated 90/270 degrees."""
        if abs(self.rotation) in (90, 270):
            return (self.height, self.width)
        return (self.width, self.height)

    @property
    def long_side(self) -> int:
        """Longer dimension of the video."""
        return max(self.width, self.height)

    @property
    def short_side(self) -> int:
        """Shorter dimension of the video."""
        return min(self.width, self.height)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (legacy format)."""
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "codec": self.codec,
            "pix_fmt": self.pix_fmt,
            "bitrate": self.bitrate,
            "rotation": self.rotation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMetadata":
        """Create from dictionary."""
        return cls(
            path=data.get("path", ""),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            fps=float(data.get("fps", 0.0)),
            duration=float(data.get("duration", 0.0)),
            codec=str(data.get("codec", "unknown")),
            pix_fmt=str(data.get("pix_fmt", "unknown")),
            bitrate=int(data.get("bitrate", 0)),
            rotation=int(data.get("rotation", 0)),
        )


@dataclass
class OutputProfile:
    """Output encoding profile determined from input footage analysis."""
    width: int
    height: int
    fps: float
    pix_fmt: str
    codec: str
    profile: str
    level: str
    bitrate: int = 0
    orientation: str = "vertical"
    aspect_ratio: str = "9:16"
    reason: str = "defaults"
    source_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) tuple."""
        return (self.width, self.height)

    @property
    def is_4k(self) -> bool:
        """Check if resolution is 4K or higher."""
        return max(self.width, self.height) >= 3840

    @property
    def is_hd(self) -> bool:
        """Check if resolution is HD (1080p) or higher."""
        return max(self.width, self.height) >= 1080

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (legacy format)."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "pix_fmt": self.pix_fmt,
            "codec": self.codec,
            "profile": self.profile,
            "level": self.level,
            "bitrate": self.bitrate,
            "orientation": self.orientation,
            "aspect_ratio": self.aspect_ratio,
            "reason": self.reason,
            "source_summary": self.source_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputProfile":
        """Create from dictionary."""
        return cls(
            width=int(data.get("width", DEFAULT_STANDARD_WIDTH)),
            height=int(data.get("height", DEFAULT_STANDARD_HEIGHT)),
            fps=float(data.get("fps", DEFAULT_STANDARD_FPS)),
            pix_fmt=str(data.get("pix_fmt", DEFAULT_STANDARD_PIX_FMT)),
            codec=str(data.get("codec", DEFAULT_STANDARD_CODEC)),
            profile=str(data.get("profile", DEFAULT_STANDARD_PROFILE)),
            level=str(data.get("level", DEFAULT_STANDARD_LEVEL)),
            bitrate=int(data.get("bitrate", 0)),
            orientation=str(data.get("orientation", "vertical")),
            aspect_ratio=str(data.get("aspect_ratio", "9:16")),
            reason=str(data.get("reason", "defaults")),
            source_summary=data.get("source_summary", {}),
        )

    @classmethod
    def default(cls) -> "OutputProfile":
        """Create default profile with standard settings."""
        return cls(
            width=DEFAULT_STANDARD_WIDTH,
            height=DEFAULT_STANDARD_HEIGHT,
            fps=DEFAULT_STANDARD_FPS,
            pix_fmt=_ffmpeg_config.pix_fmt,
            codec=_ffmpeg_config.effective_codec,
            profile=_ffmpeg_config.profile,
            level=_ffmpeg_config.level,
            reason="defaults",
        )


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_frame_rate(fps_str: str) -> float:
    """Parse FFprobe frame rate strings like '30000/1001'."""
    if not fps_str:
        return 0.0
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            den = float(den)
            return float(num) / den if den else 0.0
        return float(fps_str)
    except Exception:
        return 0.0


def _weighted_median(values: List[float], weights: List[float]) -> float:
    """Compute weighted median; falls back to simple median on error."""
    if not values:
        return 0.0
    try:
        ordered = sorted(zip(values, weights), key=lambda v: v[0])
        cumulative = np.cumsum([w for _, w in ordered])
        threshold = cumulative[-1] / 2.0
        for (val, _), total in zip(ordered, cumulative):
            if total >= threshold:
                return val
        return ordered[-1][0]
    except Exception:
        return float(np.median(values))


def _even_int(value: float) -> int:
    """Ensure integer is even and >= 2."""
    rounded = int(round(value))
    if rounded % 2 != 0:
        rounded += 1
    return max(2, rounded)


def _snap_aspect_ratio(ratio: float) -> Tuple[str, float]:
    """Snap aspect ratio to common presets if within 8%."""
    common = {
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "1:1": 1.0,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
    }
    best_name, best_val = min(common.items(), key=lambda kv: abs(kv[1] - ratio))
    rel_diff = abs(best_val - ratio) / best_val if best_val else 1.0
    return (best_name, best_val) if rel_diff <= 0.08 else ("custom", ratio)


def _snap_resolution(resolution: Tuple[int, int], orientation: str, max_long_side: int) -> Tuple[int, int]:
    """Prefer standard resolutions when they are close to the measured median."""
    presets = {
        "horizontal": [(3840, 2160), (2560, 1440), (1920, 1080), (1600, 900), (1280, 720)],
        "vertical": [(2160, 3840), (1440, 2560), (1080, 1920), (720, 1280)],
        "square": [(1920, 1920), (1080, 1080), (720, 720)]
    }
    target_w, target_h = resolution
    if orientation not in presets:
        return target_w, target_h

    # Avoid choosing a preset that upscales far beyond available footage
    candidates = [p for p in presets[orientation] if max(p) <= max_long_side * 1.05]
    if not candidates:
        candidates = presets[orientation]

    def _diff(p):
        return max(abs(p[0] - target_w) / target_w, abs(p[1] - target_h) / target_h)

    best = min(candidates, key=_diff)
    return best if _diff(best) <= 0.12 else (target_w, target_h)


def _normalize_codec_name(codec: str) -> str:
    """Normalize codec name to standard form (h264, hevc)."""
    c = (codec or "").lower()
    if "265" in c or "hevc" in c:
        return "hevc"
    if "264" in c or "avc" in c:
        return "h264"
    return c or "unknown"


# =============================================================================
# Core Functions
# =============================================================================

def probe_metadata(video_path: str, timeout: Optional[int] = None, use_cache: bool = True) -> Optional[VideoMetadata]:
    """
    Extract video metadata using ffprobe with caching.

    Args:
        video_path: Path to video file
        timeout: Optional ffprobe timeout
        use_cache: Whether to use metadata cache (default: True)

    Returns:
        VideoMetadata object or None if extraction fails

    Performance:
        With caching enabled, repeated lookups are ~10x faster.
        Cache validates using file hash (size + first/last 64KB).
    """
    # Check cache first
    if use_cache:
        cached = _metadata_cache.get(video_path)
        if cached is not None:
            return cached

    try:
        cmd = build_ffprobe_cmd([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,pix_fmt,r_frame_rate,avg_frame_rate,bit_rate:stream_side_data=rotation:format=duration,bit_rate",
            "-of", "json",
            video_path
        ])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout or _settings.processing.ffprobe_timeout,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        stream = (data.get("streams") or [None])[0] or {}
        format_info = data.get("format", {})

        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        fps = _parse_frame_rate(stream.get("r_frame_rate") or stream.get("avg_frame_rate", "0"))
        duration = float(format_info.get("duration") or 0.0)
        codec = (stream.get("codec_name") or "unknown").lower()
        pix_fmt = (stream.get("pix_fmt") or "unknown").lower()

        # Warn on RAW codecs that may need special SDKs
        if codec in RAW_CODECS:
            logger.warning(
                f"⚠️  RAW codec detected: {codec}\n"
                f"   {RAW_CODECS[codec]}\n"
                f"   Consider generating H.264/H.265 proxies for better compatibility."
            )

        # Try to get bitrate from stream or format
        bitrate = int(stream.get("bit_rate") or format_info.get("bit_rate") or 0)

        # Get rotation
        side_data_list = stream.get("side_data_list", [])
        rotation = 0
        for side_data in side_data_list:
            if "rotation" in side_data:
                rotation = int(float(side_data["rotation"]))
                break
        
        # Also check tags for rotate
        if rotation == 0:
            tags = stream.get("tags", {})
            if "rotate" in tags:
                rotation = int(float(tags["rotate"]))

        metadata = VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            duration=duration if duration > 0 else 1.0,  # avoid zero weights
            codec=codec,
            pix_fmt=pix_fmt,
            bitrate=bitrate,
            rotation=rotation,
        )

        # Store in cache for future lookups
        if use_cache:
            _metadata_cache.set(video_path, metadata)

        return metadata
    except Exception as exc:
        logger.warning(f"ffprobe failed for {os.path.basename(video_path)}: {exc}")
        return None


def probe_duration(media_path: str, timeout: Optional[int] = None) -> float:
    """
    Get duration of any media file (audio or video) using ffprobe.
    
    Args:
        media_path: Path to media file
        timeout: Optional timeout in seconds
        
    Returns:
        Duration in seconds, or 0.0 if failed
    """
    try:
        cmd = build_ffprobe_cmd([
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path
        ])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout or _settings.processing.ffprobe_timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return 0.0
    except Exception:
        return 0.0


class MetadataProber:
    """Class-based interface for video metadata extraction."""

    def __init__(self, timeout: Optional[int] = None):
        """
        Initialize metadata prober.

        Args:
            timeout: ffprobe timeout in seconds
        """
        self.timeout = timeout or _settings.processing.ffprobe_timeout

    def probe(self, video_path: str) -> Optional[VideoMetadata]:
        """Extract metadata from video file."""
        return probe_metadata(video_path, timeout=self.timeout)

    def probe_many(self, video_paths: List[str]) -> List[VideoMetadata]:
        """Extract metadata from multiple video files, filtering failures."""
        results = [probe_metadata(p, timeout=self.timeout) for p in video_paths]
        return [m for m in results if m is not None]


class OutputProfileBuilder:
    """Builder for determining output profile from input footage."""

    def __init__(
        self,
        default_width: int = DEFAULT_STANDARD_WIDTH,
        default_height: int = DEFAULT_STANDARD_HEIGHT,
        default_fps: float = DEFAULT_STANDARD_FPS,
    ):
        """Initialize builder with default values."""
        self.default_width = default_width
        self.default_height = default_height
        self.default_fps = default_fps

    def build(self, video_files: List[str]) -> OutputProfile:
        """
        Determine output profile from input video files.

        Args:
            video_files: List of video file paths

        Returns:
            OutputProfile with settings matching dominant input footage
        """
        default_profile = OutputProfile.default()
        default_profile.width = self.default_width
        default_profile.height = self.default_height
        default_profile.fps = self.default_fps

        if not video_files:
            return default_profile

        metadata_list = [probe_metadata(v) for v in video_files]
        metadata_list = [m for m in metadata_list if m is not None]
        if not metadata_list:
            return default_profile

        weights = [m.duration if m.duration > 0 else 1.0 for m in metadata_list]
        aspect_ratios = [m.aspect_ratio for m in metadata_list]
        long_sides = [m.long_side for m in metadata_list]
        short_sides = [m.short_side for m in metadata_list]
        max_long_side = max(long_sides) if long_sides else self.default_height

        # Orientation by weighted duration
        orientation_weights = {"horizontal": 0.0, "vertical": 0.0, "square": 0.0}
        for meta, weight in zip(metadata_list, weights):
            orientation_weights[meta.orientation] += weight
        orientation = max(orientation_weights, key=lambda k: orientation_weights[k])

        # Aspect ratio: snap to common ratios if close
        raw_aspect = _weighted_median(aspect_ratios, weights) or (16 / 9)
        ratio_name, snapped_ratio = _snap_aspect_ratio(raw_aspect)
        aspect_for_calc = snapped_ratio

        long_med = _weighted_median(long_sides, weights) or self.default_height
        short_med = _weighted_median(short_sides, weights) or self.default_width

        if orientation == "vertical":
            target_h = _even_int(long_med)
            target_w = _even_int(target_h * aspect_for_calc)
        elif orientation == "square":
            target_w = target_h = _even_int(min(long_med, short_med))
        else:
            target_w = _even_int(long_med)
            target_h = _even_int(target_w / aspect_for_calc)

        target_w, target_h = _snap_resolution((target_w, target_h), orientation, max_long_side)

        # FPS: choose nearest common value to weighted median
        fps_pairs = [(m.fps, w) for m, w in zip(metadata_list, weights) if m.fps > 0]
        if fps_pairs:
            fps_values, fps_weights = zip(*fps_pairs)
            raw_fps = _weighted_median(list(fps_values), list(fps_weights))
        else:
            raw_fps = self.default_fps
        common_fps = [23.976, 24, 25, 29.97, 30, 50, 59.94, 60]
        target_fps = min(common_fps, key=lambda f: abs(f - raw_fps))

        # Codec selection: honor centralized settings, otherwise follow dominant input
        from .config import get_settings
        settings = get_settings()
        env_codec = settings.encoding.codec
        codec_map = {"hevc": "libx265", "h265": "libx265", "h264": "libx264", "avc": "libx264"}
        input_codec_weights: Dict[str, float] = {}
        for meta, weight in zip(metadata_list, weights):
            norm = _normalize_codec_name(meta.codec)
            input_codec_weights[norm] = input_codec_weights.get(norm, 0.0) + weight

        dominant_input_codec = max(input_codec_weights.items(), key=lambda kv: kv[1])[0] if input_codec_weights else "h264"
        target_codec = codec_map.get(env_codec.lower(), "libx264") if env_codec else codec_map.get(dominant_input_codec, "libx264")

        # Pixel format selection
        env_pix_fmt = settings.encoding.pix_fmt
        pix_fmt_weights: Dict[str, float] = {}
        for meta, weight in zip(metadata_list, weights):
            pf = meta.pix_fmt or "yuv420p"
            pix_fmt_weights[pf] = pix_fmt_weights.get(pf, 0.0) + weight

        if env_pix_fmt:
            target_pix_fmt = env_pix_fmt
        else:
            dominant_pix_fmt = max(pix_fmt_weights.items(), key=lambda kv: kv[1])[0] if pix_fmt_weights else "yuv420p"
            compatible_formats = ["yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p", "yuvj444p"]
            target_pix_fmt = dominant_pix_fmt if dominant_pix_fmt in compatible_formats else "yuv420p"

        # Bitrate estimation
        bitrates = [m.bitrate for m in metadata_list if m.bitrate > 0]
        if bitrates:
            bitrate_weights = [weights[i] for i, m in enumerate(metadata_list) if m.bitrate > 0]
            target_bitrate = int(_weighted_median(bitrates, bitrate_weights))
        else:
            pixels_per_frame = target_w * target_h
            target_bitrate = int(pixels_per_frame * target_fps * 0.1)

        # Profile/level: keep safe defaults, bump level for 4K/high fps
        high_res = max_long_side >= 3200 or target_fps > 60
        if target_codec == "libx265":
            target_profile = settings.encoding.profile or "main"
            target_level = settings.encoding.level or ("5.1" if high_res else "4.0")
        else:
            target_profile = settings.encoding.profile or "high"
            target_level = settings.encoding.level or ("5.1" if high_res else "4.1")

        return OutputProfile(
            width=target_w,
            height=target_h,
            fps=target_fps,
            pix_fmt=target_pix_fmt,
            codec=target_codec,
            profile=target_profile,
            level=target_level,
            bitrate=target_bitrate,
            orientation=orientation,
            aspect_ratio=ratio_name,
            source_summary={
                "orientation_weights": orientation_weights,
                "median_aspect": raw_aspect,
                "median_long_side": long_med,
                "median_short_side": short_med,
                "dominant_codec": dominant_input_codec,
                "dominant_pix_fmt": max(pix_fmt_weights, key=lambda k: pix_fmt_weights[k]) if pix_fmt_weights else "yuv420p",
                "fps_selected": target_fps,
                "bitrate_median": target_bitrate,
                "total_input_files": len(metadata_list),
                "total_input_duration": sum(weights),
            },
            reason=f"{orientation} dominant; snapped to {ratio_name} aspect; {dominant_input_codec} codec with {target_pix_fmt}",
        )


def determine_output_profile(video_files: List[str]) -> OutputProfile:
    """
    Convenience function to determine output profile.

    Args:
        video_files: List of video file paths

    Returns:
        OutputProfile with settings matching dominant input footage
    """
    builder = OutputProfileBuilder()
    return builder.build(video_files)


def build_ffmpeg_params(crf: Optional[int] = None) -> List[str]:
    """
    Build FFmpeg parameters for MoviePy writes.

    Uses centralized FFmpegConfig for DRY. Automatically uses GPU-appropriate
    quality parameters when HW accel is active:
    - NVENC: -cq instead of -crf
    - VAAPI: -qp instead of -crf
    - QSV: -global_quality instead of -crf
    - CPU: -crf (standard)

    Args:
        crf: Quality value (0-51, lower=better). Uses config default if not specified.

    Returns:
        List of FFmpeg parameter strings
    """
    return _ffmpeg_config.moviepy_params(crf=crf)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "VideoMetadata",
    "OutputProfile",
    # Classes
    "MetadataProber",
    "OutputProfileBuilder",
    # Main functions
    "probe_metadata",
    "determine_output_profile",
    "build_ffmpeg_params",
    # Helper functions (exposed for testing)
    "_parse_frame_rate",
    "_weighted_median",
    "_even_int",
    "_snap_aspect_ratio",
    "_snap_resolution",
    "_normalize_codec_name",
]
