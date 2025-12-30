"""
Segment Writer for Progressive Rendering

Writes video segments to disk incrementally to prevent memory buildup.
Replaces in-memory clip accumulation with disk-based segment files.

Key improvements:
- Single-encode pipeline (clips written once, concat with -c copy)
- Stream parameter validation (fps, pix_fmt, profile)
- FFmpeg concat demuxer for memory-efficient final assembly
- Logo/branding support in post-processing pass
- Real crossfades via FFmpeg xfade filter (not fade-to-black)

Memory savings: ~200-400MB depending on project size
"""

import os
import subprocess
import tempfile
import shutil
import json
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Import centralized FFmpeg config (DRY)
from .ffmpeg_config import (
    FFmpegConfig,
    get_config,
    STANDARD_FPS,
    STANDARD_PIX_FMT,
    STANDARD_CODEC,
    STANDARD_PROFILE,
    STANDARD_LEVEL,
    STANDARD_WIDTH_HORIZONTAL,
    STANDARD_HEIGHT_HORIZONTAL,
)

# Default to horizontal (will be overridden by output profile sync)
STANDARD_WIDTH = STANDARD_WIDTH_HORIZONTAL   # 1920
STANDARD_HEIGHT = STANDARD_HEIGHT_HORIZONTAL  # 1080

# Get runtime config (env vars applied)
_ffmpeg_config = get_config()
_ffmpeg_cpu_config: Optional[FFmpegConfig] = None
TARGET_CODEC = _ffmpeg_config.codec
TARGET_PROFILE = _ffmpeg_config.profile
TARGET_LEVEL = _ffmpeg_config.level
TARGET_PIX_FMT = _ffmpeg_config.pix_fmt

# Default crossfade duration in seconds
DEFAULT_XFADE_DURATION = 0.3


def _get_cpu_config() -> FFmpegConfig:
    """Lazy init CPU-only config for cases where GPU filters are incompatible."""
    global _ffmpeg_cpu_config
    if _ffmpeg_cpu_config is None:
        _ffmpeg_cpu_config = FFmpegConfig(hwaccel="none")
    return _ffmpeg_cpu_config

def _moviepy_params(
    config: Optional[FFmpegConfig] = None,
    crf: Optional[int] = None,
    preset: Optional[str] = None,
    target_profile: Optional[str] = None,
    target_level: Optional[str] = None,
    target_pix_fmt: Optional[str] = None,
) -> List[str]:
    """
    Shared ffmpeg parameters for MoviePy writes to keep video streams aligned.
    Uses centralized FFmpegConfig for DRY.
    """
    cfg = config or _ffmpeg_config
    return cfg.moviepy_params(
        crf=crf,
        preset=preset,
        profile_override=target_profile,
        level_override=target_level,
        pix_fmt_override=target_pix_fmt,
    )

def _video_params_for_target(
    config: FFmpegConfig,
    crf: int,
    preset: str,
    target_codec: Optional[str],
    target_profile: Optional[str],
    target_level: Optional[str],
    target_pix_fmt: Optional[str],
) -> List[str]:
    """Build encoder params honoring output overrides when on CPU."""
    return config.video_params(
        crf=crf,
        preset=preset,
        codec_override=target_codec,
        profile_override=target_profile,
        level_override=target_level,
        pix_fmt_override=target_pix_fmt,
    )

def _append_hwupload_vf(vf_chain: str, config: FFmpegConfig) -> str:
    """Append hwupload filter if required by the active encoder."""
    if config.hwupload_filter:
        return f"{vf_chain},{config.hwupload_filter}"
    return vf_chain

def _append_hwupload_filter_complex(
    filter_complex: str,
    config: FFmpegConfig,
    input_label: str,
    output_label: str,
) -> Tuple[str, str]:
    """Append hwupload filter to a filter_complex graph when needed."""
    if config.hwupload_filter:
        return f"{filter_complex};{input_label}{config.hwupload_filter}{output_label}", output_label
    return filter_complex, input_label


@dataclass
class StreamParams:
    """Video stream parameters for validation."""
    fps: float
    pix_fmt: str
    width: int
    height: int
    codec: str
    profile: Optional[str] = None
    duration: float = 0.0  # Added for xfade offset calculation
    
    def is_compatible_with(self, other: 'StreamParams') -> bool:
        """Check if streams are compatible for concat demuxer."""
        return (
            abs(self.fps - other.fps) < 0.1 and
            self.pix_fmt == other.pix_fmt and
            self.width == other.width and
            self.height == other.height and
            self.codec == other.codec
        )


def ffprobe_stream_params(video_path: str) -> Optional[StreamParams]:
    """
    Get video stream parameters using ffprobe.
    
    Returns StreamParams for validation before concat.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,pix_fmt,codec_name,profile:format=duration",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return None
            
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        
        if not streams:
            return None
            
        stream = streams[0]
        format_info = data.get("format", {})
        
        # Parse frame rate (can be "30/1" or "29.97")
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        # Get duration from format section
        duration = float(format_info.get("duration", 0))
        
        return StreamParams(
            fps=fps,
            pix_fmt=stream.get("pix_fmt", "unknown"),
            width=int(stream.get("width", 0)),
            height=int(stream.get("height", 0)),
            codec=stream.get("codec_name", "unknown"),
            profile=stream.get("profile"),
            duration=duration
        )
    except Exception as e:
        print(f"   ⚠️ ffprobe failed for {video_path}: {e}")
        return None


def normalize_clip_ffmpeg(input_path: str, output_path: str,
                          target_fps: float = STANDARD_FPS,
                          target_pix_fmt: str = TARGET_PIX_FMT,
                          target_codec: str = TARGET_CODEC,
                          target_profile: Optional[str] = TARGET_PROFILE,
                          target_level: Optional[str] = TARGET_LEVEL,
                          crf: int = 18,
                          preset: str = "fast") -> bool:
    """
    Normalize a clip to standard parameters for concat compatibility.

    Includes:
    - Resolution scaling with padding
    - Broadcast-safe color levels (16-235)
    - Auto brightness/contrast normalization
    - Format conversion for concat compatibility

    Args:
        input_path: Source video
        output_path: Normalized output
        target_fps: Target frame rate
        target_pix_fmt: Target pixel format
        crf: Quality setting
        preset: Encoding speed preset

    Returns:
        True if successful
    """
    try:
        # Build filter chain:
        # 1. Scale to standard resolution (no cropping)
        # 2. Pad to fill frame (letterbox/pillarbox)
        # 3. Broadcast-safe levels (16-235 range)
        # 4. Normalize brightness for consistency
        # 5. Convert fps and pixel format
        vf_chain = (
            f"scale={STANDARD_WIDTH}:{STANDARD_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={STANDARD_WIDTH}:{STANDARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"colorlevels=rimin=0.063:gimin=0.063:bimin=0.063:rimax=0.922:gimax=0.922:bimax=0.922,"
            f"normalize=blackpt=black:whitept=white:smoothing=10,"
            f"fps={target_fps},"
            f"format={target_pix_fmt}"
        )

        config = _ffmpeg_config
        vf_chain = _append_hwupload_vf(vf_chain, config)

        cmd = ["ffmpeg", "-y"]
        if config.is_gpu_accelerated:
            cmd.extend(config.hwaccel_input_params())
        cmd.extend([
            "-i", input_path,
            "-vf", vf_chain,
        ])
        cmd.extend(_video_params_for_target(
            config,
            crf=crf,
            preset=preset,
            target_codec=target_codec,
            target_profile=target_profile,
            target_level=target_level,
            target_pix_fmt=target_pix_fmt,
        ))
        cmd.extend([
            "-an",  # No audio (added later)
            "-movflags", "+faststart",
            output_path
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0

    except Exception as e:
        print(f"   ❌ Normalization failed: {e}")
        return False


def xfade_two_clips(clip1_path: str, clip2_path: str, output_path: str,
                    xfade_duration: float = DEFAULT_XFADE_DURATION,
                    transition: str = "fade",
                    crf: int = 18,
                    preset: str = "fast",
                    target_codec: str = TARGET_CODEC,
                    target_profile: Optional[str] = TARGET_PROFILE,
                    target_level: Optional[str] = TARGET_LEVEL,
                    target_pix_fmt: str = TARGET_PIX_FMT) -> bool:
    """
    Create a real crossfade between two clips using FFmpeg xfade filter.
    
    This creates an actual overlapping transition, not just fade-in/out.
    
    Args:
        clip1_path: First clip path
        clip2_path: Second clip path  
        output_path: Output path for merged clip
        xfade_duration: Crossfade duration in seconds (default 0.3s)
        transition: Transition type (fade, wipeleft, slideright, etc.)
        crf: Quality setting
        preset: Encoding preset
        
    Returns:
        True if successful
    """
    try:
        # Get clip1 duration to calculate offset
        params1 = ffprobe_stream_params(clip1_path)
        if not params1 or params1.duration <= 0:
            print(f"   ⚠️ Could not get duration for {clip1_path}")
            return False
        
        # xfade offset = clip1_duration - xfade_duration
        offset = max(0, params1.duration - xfade_duration)
        
        # Build xfade filter
        # Format: [0:v][1:v]xfade=transition=fade:duration=0.3:offset=2.7[v]
        config = _ffmpeg_config
        filter_complex = (
            f"[0:v][1:v]xfade=transition={transition}:"
            f"duration={xfade_duration}:offset={offset}[v]"
        )
        filter_complex, out_label = _append_hwupload_filter_complex(
            filter_complex,
            config,
            "[v]",
            "[v_hw]"
        )
        
        cmd = ["ffmpeg", "-y"]
        if config.is_gpu_accelerated:
            cmd.extend(config.hwaccel_input_params())
        cmd.extend([
            "-i", clip1_path,
            "-i", clip2_path,
            "-filter_complex", filter_complex,
            "-map", out_label,
        ])

        cmd.extend(_video_params_for_target(
            config,
            crf=crf,
            preset=preset,
            target_codec=target_codec,
            target_profile=target_profile,
            target_level=target_level,
            target_pix_fmt=target_pix_fmt,
        ))

        cmd.extend([
            "-an",  # No audio - handled separately
            output_path
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"   ⚠️ xfade failed: {result.stderr[:200]}")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ xfade error: {e}")
        return False


@dataclass
class SegmentInfo:
    """Information about a rendered segment."""
    path: str
    index: int
    clip_count: int
    duration: float
    size_bytes: int = 0
    
    def __post_init__(self):
        if os.path.exists(self.path):
            self.size_bytes = os.path.getsize(self.path)


@dataclass
class SegmentWriterStats:
    """Statistics for segment writing operations."""
    total_segments: int = 0
    total_clips: int = 0
    total_duration: float = 0.0
    total_size_bytes: int = 0
    peak_memory_mb: float = 0.0
    concatenation_time: float = 0.0


class SegmentWriter:
    """
    Manages progressive rendering of video segments to disk.
    
    Instead of keeping all clips in memory, renders batches to 
    intermediate segment files and concatenates at the end.
    
    Features:
    - Configurable batch sizes
    - Automatic temp file management
    - FFmpeg-based concatenation with optional xfade transitions
    - Memory-efficient progressive rendering
    """
    
    def __init__(self, 
                 output_dir: str = "/tmp/segments",
                 segment_prefix: str = "segment",
                 ffmpeg_preset: str = "fast",
                 ffmpeg_crf: int = 18,
                 cleanup_on_complete: bool = True,
                 enable_xfade: bool = False,
                 xfade_duration: float = DEFAULT_XFADE_DURATION):
        """
        Initialize segment writer.
        
        Args:
            output_dir: Directory for intermediate segment files
            segment_prefix: Prefix for segment filenames
            ffmpeg_preset: FFmpeg encoding preset
            ffmpeg_crf: FFmpeg quality (lower = better, 18 recommended)
            cleanup_on_complete: Remove temp files after concatenation
            enable_xfade: Enable real crossfades between clips using xfade filter
            xfade_duration: Duration of crossfade transitions (seconds)
        """
        self.output_dir = Path(output_dir)
        self.segment_prefix = segment_prefix
        self.ffmpeg_preset = ffmpeg_preset
        self.ffmpeg_crf = ffmpeg_crf
        self.cleanup_on_complete = cleanup_on_complete
        self.enable_xfade = enable_xfade
        self.xfade_duration = xfade_duration
        
        # State
        self.segments: List[SegmentInfo] = []
        self.stats = SegmentWriterStats()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_segment_path(self, index: int) -> str:
        """Get path for segment file at given index."""
        return str(self.output_dir / f"{self.segment_prefix}_{index:04d}.mp4")
    
    def write_segment(self, 
                      clips: List,  # MoviePy VideoFileClip objects
                      segment_index: int,
                      audio_clip = None) -> Optional[SegmentInfo]:
        """
        Write a batch of clips as a single segment file.
        
        Args:
            clips: List of MoviePy VideoFileClip objects
            segment_index: Index of this segment
            audio_clip: Optional audio to mix (usually None for segments)
            
        Returns:
            SegmentInfo if successful, None if failed
        """
        if not clips:
            return None
        
        from .moviepy_compat import concatenate_videoclips
        
        segment_path = self.get_segment_path(segment_index)
        
        try:
            print(f"   📼 Writing segment {segment_index} ({len(clips)} clips)...")
            
            # Concatenate clips in memory (minimal, just this batch)
            combined = concatenate_videoclips(clips, method="compose")
            
            moviepy_config = _ffmpeg_config
            if moviepy_config.is_gpu_accelerated and moviepy_config.hwupload_filter:
                moviepy_config = _get_cpu_config()
                codec = TARGET_CODEC
            else:
                codec = moviepy_config.effective_codec if moviepy_config.is_gpu_accelerated else TARGET_CODEC

            # Write to disk
            combined.write_videofile(
                segment_path,
                codec=codec,
                preset=self.ffmpeg_preset,
                audio_codec="aac",
                fps=STANDARD_FPS,
                logger=None,  # Suppress MoviePy progress bar
                threads=4,
                ffmpeg_params=_moviepy_params(
                    config=moviepy_config,
                    crf=self.ffmpeg_crf,
                    preset=self.ffmpeg_preset,
                    target_profile=TARGET_PROFILE,
                    target_level=TARGET_LEVEL,
                    target_pix_fmt=TARGET_PIX_FMT,
                )
            )
            
            # Get duration before closing
            duration = combined.duration
            
            # Close to free memory immediately
            combined.close()
            
            # Close individual clips
            for clip in clips:
                try:
                    clip.close()
                except Exception:
                    pass  # Ignore cleanup errors during memory release
            
            # Create segment info
            segment_info = SegmentInfo(
                path=segment_path,
                index=segment_index,
                clip_count=len(clips),
                duration=duration
            )
            
            self.segments.append(segment_info)
            self.stats.total_segments += 1
            self.stats.total_clips += len(clips)
            self.stats.total_duration += duration
            self.stats.total_size_bytes += segment_info.size_bytes
            
            print(f"   ✅ Segment {segment_index} written: {duration:.1f}s, "
                  f"{segment_info.size_bytes / (1024*1024):.1f}MB")
            
            return segment_info
            
        except Exception as e:
            print(f"   ❌ Failed to write segment {segment_index}: {e}")
            return None
    
    def write_segment_ffmpeg(self,
                              clip_paths: List[str],
                              segment_index: int,
                              durations: Optional[List[float]] = None,
                              validate_streams: bool = True) -> Optional[SegmentInfo]:
        """
        Write segment using FFmpeg concat demuxer with -c copy (no re-encoding).
        
        Args:
            clip_paths: List of video file paths (must have compatible streams)
            segment_index: Index of this segment
            durations: Optional list of durations for each clip
            validate_streams: Check stream compatibility before concat
            
        Returns:
            SegmentInfo if successful, None if failed
        """
        if not clip_paths:
            return None
        
        segment_path = self.get_segment_path(segment_index)
        concat_list_path = str(self.output_dir / f"concat_{segment_index}.txt")
        
        try:
            print(f"   📼 Writing segment {segment_index} ({len(clip_paths)} clips via FFmpeg -c copy)...")
            
            # Validate stream parameters if requested
            if validate_streams and len(clip_paths) > 1:
                first_params = ffprobe_stream_params(clip_paths[0])
                if first_params:
                    for path in clip_paths[1:]:
                        params = ffprobe_stream_params(path)
                        if params and not first_params.is_compatible_with(params):
                            print(f"   ⚠️ Stream mismatch in {os.path.basename(path)}")
                            print(f"      Expected: {first_params.fps:.2f}fps, {first_params.pix_fmt}")
                            print(f"      Got: {params.fps:.2f}fps, {params.pix_fmt}")
                            # Fall back to re-encoding for this segment
                            return self._write_segment_ffmpeg_reencode(clip_paths, segment_index, concat_list_path)
            
            # Create concat file list
            with open(concat_list_path, 'w') as f:
                for path in clip_paths:
                    # Escape single quotes in path
                    escaped_path = path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
            
            # FFmpeg concat demuxer with -c copy (no re-encoding, very fast)
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",  # Stream copy - no re-encoding!
                segment_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=120  # Much faster with -c copy
            )
            
            # Cleanup concat list
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
            
            if result.returncode != 0:
                print(f"   ⚠️ Concat copy failed, trying re-encode: {result.stderr[:100]}")
                return self._write_segment_ffmpeg_reencode(clip_paths, segment_index, concat_list_path)
            
            # Get duration from output file
            duration = self._get_video_duration(segment_path)
            
            segment_info = SegmentInfo(
                path=segment_path,
                index=segment_index,
                clip_count=len(clip_paths),
                duration=duration
            )
            
            self.segments.append(segment_info)
            self.stats.total_segments += 1
            self.stats.total_clips += len(clip_paths)
            self.stats.total_duration += duration
            self.stats.total_size_bytes += segment_info.size_bytes
            
            print(f"   ✅ Segment {segment_index} (copy): {duration:.1f}s, "
                  f"{segment_info.size_bytes / (1024*1024):.1f}MB")
            
            return segment_info
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ Segment {segment_index} timed out")
            return None
        except Exception as e:
            print(f"   ❌ Failed to write segment {segment_index}: {e}")
            return None
    
    def _write_segment_ffmpeg_reencode(self,
                                        clip_paths: List[str],
                                        segment_index: int,
                                        concat_list_path: str) -> Optional[SegmentInfo]:
        """
        Fallback: Write segment with re-encoding when streams are incompatible.
        
        Only used when -c copy fails due to stream parameter mismatches.
        """
        segment_path = self.get_segment_path(segment_index)
        
        try:
            print(f"   🔄 Re-encoding segment {segment_index} (stream normalization)...")
            
            # Create concat file list if not exists
            if not os.path.exists(concat_list_path):
                with open(concat_list_path, 'w') as f:
                    for path in clip_paths:
                        escaped_path = path.replace("'", "'\\''")
                        f.write(f"file '{escaped_path}'\n")
            
            # Re-encode with normalized parameters
            # CRITICAL: Also scale to standard resolution to handle dimension mismatches
            # This ensures all clips have identical dimensions before concat
            vf_chain = (
                f"scale={STANDARD_WIDTH}:{STANDARD_HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={STANDARD_WIDTH}:{STANDARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
                f"fps={STANDARD_FPS},format={TARGET_PIX_FMT}"
            )
            config = _ffmpeg_config
            vf_chain = _append_hwupload_vf(vf_chain, config)

            cmd = ["ffmpeg", "-y"]
            if config.is_gpu_accelerated:
                cmd.extend(config.hwaccel_input_params())
            cmd.extend([
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-vf", vf_chain,
            ])

            cmd.extend(_video_params_for_target(
                config,
                crf=self.ffmpeg_crf,
                preset=self.ffmpeg_preset,
                target_codec=TARGET_CODEC,
                target_profile=TARGET_PROFILE,
                target_level=TARGET_LEVEL,
                target_pix_fmt=TARGET_PIX_FMT,
            ))

            cmd.extend([
                "-an",  # No audio in segments
                segment_path
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Cleanup
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
            
            if result.returncode != 0:
                print(f"   ❌ Re-encode failed: {result.stderr[:200]}")
                return None
            
            duration = self._get_video_duration(segment_path)
            
            segment_info = SegmentInfo(
                path=segment_path,
                index=segment_index,
                clip_count=len(clip_paths),
                duration=duration
            )
            
            self.segments.append(segment_info)
            self.stats.total_segments += 1
            self.stats.total_clips += len(clip_paths)
            self.stats.total_duration += duration
            self.stats.total_size_bytes += segment_info.size_bytes
            
            print(f"   ✅ Segment {segment_index} (re-encoded): {duration:.1f}s")
            return segment_info
            
        except Exception as e:
            print(f"   ❌ Re-encode failed: {e}")
            return None
    
    def write_segment_with_xfade(self,
                                  clip_paths: List[str],
                                  segment_index: int,
                                  xfade_duration: float = None) -> Optional[SegmentInfo]:
        """
        Write segment using FFmpeg xfade filter for real crossfade transitions.
        
        Creates overlapping transitions between clips instead of fade-to-black.
        Requires re-encoding but produces professional-looking transitions.
        
        Args:
            clip_paths: List of video file paths
            segment_index: Index of this segment
            xfade_duration: Crossfade duration (uses instance default if None)
            
        Returns:
            SegmentInfo if successful, None if failed
        """
        if not clip_paths:
            return None
            
        if len(clip_paths) == 1:
            # Single clip - no crossfade needed, just copy
            return self.write_segment_ffmpeg(clip_paths, segment_index, validate_streams=False)
        
        xfade_dur = xfade_duration if xfade_duration is not None else self.xfade_duration
        segment_path = self.get_segment_path(segment_index)
        
        try:
            print(f"   📼 Writing segment {segment_index} ({len(clip_paths)} clips with xfade)...")
            
            # For xfade, we need to chain clips together progressively
            # Start with first clip, then xfade each subsequent clip
            current_result = clip_paths[0]
            temp_files = []
            
            for i, next_clip in enumerate(clip_paths[1:], start=1):
                # Create temp output for intermediate xfade result
                if i < len(clip_paths) - 1:
                    temp_output = str(self.output_dir / f"xfade_temp_{segment_index}_{i}.mp4")
                    temp_files.append(temp_output)
                else:
                    # Last clip - output to final segment path
                    temp_output = segment_path
                
                success = xfade_two_clips(
                    current_result, 
                    next_clip, 
                    temp_output,
                    xfade_duration=xfade_dur,
                    transition="fade",
                    crf=self.ffmpeg_crf,
                    preset=self.ffmpeg_preset
                )
                
                if not success:
                    print(f"   ⚠️ xfade failed at clip {i}, falling back to concat")
                    # Cleanup temp files
                    for tf in temp_files:
                        if os.path.exists(tf):
                            os.remove(tf)
                    # Fall back to regular concat
                    return self.write_segment_ffmpeg(clip_paths, segment_index)
                
                # Use this result for next iteration
                current_result = temp_output
            
            # Cleanup intermediate temp files (not the final result)
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)
            
            # Get duration and create segment info
            duration = self._get_video_duration(segment_path)
            
            segment_info = SegmentInfo(
                path=segment_path,
                index=segment_index,
                clip_count=len(clip_paths),
                duration=duration
            )
            
            self.segments.append(segment_info)
            self.stats.total_segments += 1
            self.stats.total_clips += len(clip_paths)
            self.stats.total_duration += duration
            self.stats.total_size_bytes += segment_info.size_bytes
            
            print(f"   ✅ Segment {segment_index} (xfade): {duration:.1f}s, "
                  f"{segment_info.size_bytes / (1024*1024):.1f}MB")
            
            return segment_info
            
        except Exception as e:
            print(f"   ❌ xfade segment failed: {e}")
            # Fall back to regular concat
            return self.write_segment_ffmpeg(clip_paths, segment_index)
    
    def concatenate_segments(self,
                              output_path: str,
                              audio_path: Optional[str] = None,
                              audio_volume: float = 1.0,
                              audio_duration: Optional[float] = None,
                              logo_path: Optional[str] = None,
                              logo_position: str = "top-right") -> bool:
        """
        Concatenate all segments into final output file using -c copy.

        Uses stream copy for video (no re-encoding) and only encodes audio.
        Optional logo overlay as second pass.

        Args:
            output_path: Final output video path
            audio_path: Optional audio track to mix
            audio_volume: Volume multiplier for audio
            audio_duration: Optional duration to trim audio to (in seconds)
            logo_path: Optional logo image to overlay
            logo_position: Logo position (top-right, top-left, bottom-right, bottom-left)

        Returns:
            True if successful
        """
        if not self.segments:
            print("   ❌ No segments to concatenate")
            return False
        
        import time
        start_time = time.time()
        
        # Sort segments by index
        sorted_segments = sorted(self.segments, key=lambda s: s.index)
        
        print(f"\n   🔗 Concatenating {len(sorted_segments)} segments with -c copy...")
        
        # Create concat file
        concat_list_path = str(self.output_dir / "final_concat.txt")
        
        with open(concat_list_path, 'w') as f:
            for seg in sorted_segments:
                escaped_path = seg.path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        # Determine output path (temp if logo overlay needed)
        actual_output = output_path if not logo_path else str(self.output_dir / "pre_logo.mp4")
        
        try:
            if audio_path and os.path.exists(audio_path):
                # Concatenate video (copy) + encode audio only
                # Build audio filter chain
                af_filters = [f"volume={audio_volume}"]

                # Add trim filter if audio_duration is specified
                if audio_duration is not None and audio_duration > 0:
                    af_filters.insert(0, f"atrim=0:{audio_duration}")

                af_chain = ",".join(af_filters)

                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-i", audio_path,
                    "-map", "0:v",
                    "-map", "1:a",
                    "-c:v", "copy",  # Stream copy - no re-encode!
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-af", af_chain,
                    "-shortest",
                    "-movflags", "+faststart",
                    actual_output
                ]
            else:
                # Just concatenate video with copy
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",  # Full stream copy
                    "-movflags", "+faststart",
                    actual_output
                ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # Much faster with -c copy
            )
            
            # Cleanup concat list
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
            
            if result.returncode != 0:
                print(f"   ❌ Final concatenation failed: {result.stderr[:300]}")
                return False
            
            # Apply logo overlay if requested (requires re-encode)
            if logo_path and os.path.exists(logo_path) and os.path.exists(actual_output):
                print(f"   🏷️ Adding logo overlay...")
                logo_success = self._apply_logo_overlay(actual_output, output_path, logo_path, logo_position)
                # Cleanup temp file
                if os.path.exists(actual_output) and actual_output != output_path:
                    os.remove(actual_output)
                if not logo_success:
                    print(f"   ⚠️ Logo overlay failed, using video without logo")
            
            self.stats.concatenation_time = time.time() - start_time
            
            # Cleanup segment files if requested
            if self.cleanup_on_complete:
                self.cleanup_segments()
            
            final_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            print(f"   ✅ Final video: {output_path}")
            print(f"      Duration: {self.stats.total_duration:.1f}s")
            print(f"      Size: {final_size / (1024*1024):.1f}MB")
            print(f"      Concatenation time: {self.stats.concatenation_time:.1f}s")
            
            return True
            
        except subprocess.TimeoutExpired:
            print("   ❌ Final concatenation timed out")
            return False
        except Exception as e:
            print(f"   ❌ Concatenation error: {e}")
            return False
    
    def _apply_logo_overlay(self, input_path: str, output_path: str, 
                            logo_path: str, position: str = "top-right") -> bool:
        """
        Apply logo overlay to video (requires re-encoding).
        
        Args:
            input_path: Source video
            output_path: Output with logo
            logo_path: Logo image path
            position: Position (top-right, top-left, bottom-right, bottom-left)
            
        Returns:
            True if successful
        """
        try:
            # Position mapping
            positions = {
                "top-right": "W-w-50:50",
                "top-left": "50:50",
                "bottom-right": "W-w-50:H-h-50",
                "bottom-left": "50:H-h-50"
            }
            overlay_pos = positions.get(position, positions["top-right"])
            
            config = _ffmpeg_config
            filter_complex = f"[1:v]scale=150:-1[logo];[0:v][logo]overlay={overlay_pos}[v]"
            filter_complex, out_label = _append_hwupload_filter_complex(
                filter_complex,
                config,
                "[v]",
                "[v_hw]"
            )

            cmd = ["ffmpeg", "-y"]
            if config.is_gpu_accelerated:
                cmd.extend(config.hwaccel_input_params())
            cmd.extend([
                "-i", input_path,
                "-i", logo_path,
                "-filter_complex",
                filter_complex,
                "-map", out_label,
                "-map", "0:a?",
            ])

            cmd.extend(_video_params_for_target(
                config,
                crf=self.ffmpeg_crf,
                preset=self.ffmpeg_preset,
                target_codec=TARGET_CODEC,
                target_profile=TARGET_PROFILE,
                target_level=TARGET_LEVEL,
                target_pix_fmt=TARGET_PIX_FMT,
            ))

            cmd.extend([
                "-c:a", "copy",  # Audio copy
                "-movflags", "+faststart",
                output_path
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return result.returncode == 0
            
        except Exception as e:
            print(f"   ❌ Logo overlay error: {e}")
            return False
    
    def cleanup_segments(self) -> int:
        """
        Remove all segment files.
        
        Returns:
            Number of files removed
        """
        removed = 0
        for seg in self.segments:
            try:
                if os.path.exists(seg.path):
                    os.remove(seg.path)
                    removed += 1
            except OSError:
                pass
        
        print(f"   🧹 Cleaned up {removed} segment files")
        return removed
    
    def cleanup_all(self) -> None:
        """Remove all temp files including output directory and temporary clip files."""
        self.cleanup_segments()

        # Remove any remaining files in output dir
        try:
            for f in self.output_dir.iterdir():
                if f.is_file():
                    f.unlink()
        except Exception:
            pass

        # CRITICAL: Also cleanup temporary clip files (clip_*.mp4 and *_norm.mp4)
        # These are created in TEMP_DIR and can accumulate during long runs
        try:
            from pathlib import Path
            temp_dir = Path(os.environ.get("TEMP_DIR", "/tmp"))

            # Find and remove clip files matching pattern clip_*_####.mp4
            for clip_file in temp_dir.glob("clip_*_*.mp4"):
                try:
                    clip_file.unlink()
                    print(f"   🧹 Cleaned up temp clip: {clip_file.name}")
                except Exception:
                    pass

            # Find and remove normalized files (*_norm.mp4)
            for norm_file in temp_dir.glob("*_norm.mp4"):
                try:
                    norm_file.unlink()
                    print(f"   🧹 Cleaned up normalized clip: {norm_file.name}")
                except Exception:
                    pass
        except Exception as e:
            print(f"   ⚠️ Temp file cleanup warning: {e}")
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip()) if result.stdout.strip() else 0.0
        except Exception:
            return 0.0
    
    def get_stats(self) -> Dict:
        """Get statistics as dictionary."""
        return {
            'total_segments': self.stats.total_segments,
            'total_clips': self.stats.total_clips,
            'total_duration': self.stats.total_duration,
            'total_size_mb': self.stats.total_size_bytes / (1024 * 1024),
            'concatenation_time': self.stats.concatenation_time,
            'segment_paths': [s.path for s in self.segments]
        }


class ProgressiveRenderer:
    """
    High-level interface for progressive video rendering.
    
    Coordinates SegmentWriter with memory management for
    optimal memory usage during large montage creation.
    
    Uses FFmpeg concat demuxer for memory-efficient segment writing
    instead of MoviePy in-memory concatenation.
    
    Supports real crossfade transitions via xfade filter when enabled.
    """
    
    def __init__(self,
                 batch_size: int = 25,
                 output_dir: str = "/tmp/progressive_render",
                 memory_manager = None,
                 job_id: str = "default",
                 enable_xfade: bool = False,
                 xfade_duration: float = DEFAULT_XFADE_DURATION,
                 ffmpeg_crf: int = 18,
                 normalize_clips: bool = False):
        """
        Initialize progressive renderer.
        
        Args:
            batch_size: Number of clips per segment
            output_dir: Directory for temp files
            memory_manager: Optional AdaptiveMemoryManager instance
            job_id: Job identifier for temp file naming
            enable_xfade: Enable real crossfades using FFmpeg xfade filter
            xfade_duration: Duration of crossfade transitions (seconds)
            ffmpeg_crf: Quality setting for encoding
            normalize_clips: Force normalize all clips before first flush
        """
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.memory_manager = memory_manager
        self.job_id = job_id
        self.enable_xfade = enable_xfade
        self.xfade_duration = xfade_duration
        self.normalize_clips = normalize_clips
        
        self.segment_writer = SegmentWriter(
            output_dir=output_dir,
            segment_prefix=f"segment_{job_id}",
            enable_xfade=enable_xfade,
            xfade_duration=xfade_duration,
            ffmpeg_crf=ffmpeg_crf
        )
        
        # Track clip file paths (not MoviePy objects - more memory efficient)
        self.current_batch_paths: List[str] = []
        self.current_batch_index = 0
        
        # For clips that need MoviePy processing first
        self.pending_moviepy_clips: List = []
    
    def add_clip_path(self, clip_path: str) -> Optional[SegmentInfo]:
        """
        Add a pre-rendered clip file to the current batch.
        
        This is the memory-efficient path - clips are already on disk.
        
        Args:
            clip_path: Path to video file
            
        Returns:
            SegmentInfo if segment was written, None otherwise
        """
        self.current_batch_paths.append(clip_path)
        
        # Calculate effective batch size based on memory
        effective_batch_size = self.batch_size
        if self.memory_manager:
            effective_batch_size = self.memory_manager.calculate_safe_batch_size(
                self.batch_size, clip_size_mb=60.0
            )
            
            # Force immediate flush on critical memory
            if self.memory_manager.is_critical():
                print(f"   🚨 Critical memory - forcing segment flush")
                effective_batch_size = 1
        
        # Check if batch is full
        if len(self.current_batch_paths) >= effective_batch_size:
            return self.flush_batch()
        
        return None
    
    def add_moviepy_clip(self, clip, temp_path: str) -> Optional[SegmentInfo]:
        """
        Add a MoviePy clip by rendering it to a temp file first.
        
        This converts MoviePy objects to files for FFmpeg processing.
        
        Args:
            clip: MoviePy VideoFileClip object
            temp_path: Path to write the clip to
            
        Returns:
            SegmentInfo if segment was written, None otherwise
        """
        try:
            # Render clip to temp file (with audio preserved)
            moviepy_config = _ffmpeg_config
            if moviepy_config.is_gpu_accelerated and moviepy_config.hwupload_filter:
                moviepy_config = _get_cpu_config()
                codec = TARGET_CODEC
            else:
                codec = moviepy_config.effective_codec if moviepy_config.is_gpu_accelerated else TARGET_CODEC

            clip.write_videofile(
                temp_path,
                codec=codec,
                audio_codec='aac',
                fps=STANDARD_FPS,
                preset='fast',
                logger=None,
                ffmpeg_params=_moviepy_params(
                    config=moviepy_config,
                    preset='fast',
                    target_profile=TARGET_PROFILE,
                    target_level=TARGET_LEVEL,
                    target_pix_fmt=TARGET_PIX_FMT,
                )
            )
            
            # Close MoviePy clip to free memory
            try:
                clip.close()
            except Exception:
                pass  # Ignore cleanup errors during memory release
            
            # Add the file path
            return self.add_clip_path(temp_path)
            
        except Exception as e:
            print(f"   ❌ Failed to render clip: {e}")
            return None
    
    def flush_batch(self) -> Optional[SegmentInfo]:
        """
        Write current batch as a segment using FFmpeg.
        
        Uses xfade for real crossfades if enabled, otherwise concat demuxer.
        
        Returns:
            SegmentInfo if successful
        """
        if not self.current_batch_paths:
            return None
        
        if self.memory_manager:
            self.memory_manager.print_memory_status(f"   [Segment {self.current_batch_index}] Before flush: ")
        
        # Normalize clips if enabled (ensures stream compatibility)
        if self.normalize_clips and len(self.current_batch_paths) > 1:
            self._normalize_batch_clips()
        
        # Choose method based on xfade setting
        if self.enable_xfade:
            # Use xfade for real crossfade transitions (requires re-encoding)
            segment = self.segment_writer.write_segment_with_xfade(
                self.current_batch_paths,
                self.current_batch_index,
                xfade_duration=self.xfade_duration
            )
        else:
            # Use concat demuxer (no re-encoding, fast)
            segment = self.segment_writer.write_segment_ffmpeg(
                self.current_batch_paths,
                self.current_batch_index
            )
        
        # Clear batch paths
        self.current_batch_paths = []
        self.current_batch_index += 1
        
        # Trigger garbage collection
        import gc
        gc.collect()
        
        if self.memory_manager:
            self.memory_manager.trigger_cleanup(force=True)
            self.memory_manager.print_memory_status(f"   [Segment {self.current_batch_index-1}] After flush:  ")
        
        return segment
    
    def _normalize_batch_clips(self) -> None:
        """
        Normalize all clips in current batch to ensure stream compatibility.

        Only called when NORMALIZE_CLIPS is enabled. Converts clips to
        standard fps/pix_fmt/profile/dimensions for concat demuxer compatibility.
        """
        print(f"   🔄 Normalizing {len(self.current_batch_paths)} clips for compatibility...")

        normalized_paths = []
        for clip_path in self.current_batch_paths:
            # Check if clip needs normalization
            params = ffprobe_stream_params(clip_path)

            needs_normalize = (
                params is None or
                abs(params.fps - STANDARD_FPS) >= 0.1 or
                params.pix_fmt != STANDARD_PIX_FMT or
                params.width != STANDARD_WIDTH or
                params.height != STANDARD_HEIGHT
            )

            if needs_normalize:
                # Create normalized version
                normalized_path = clip_path.replace('.mp4', '_norm.mp4')
                success = normalize_clip_ffmpeg(
                    clip_path,
                    normalized_path,
                    crf=self.segment_writer.ffmpeg_crf
                )

                if success:
                    # Remove original, use normalized
                    try:
                        os.remove(clip_path)
                    except OSError:
                        pass  # Ignore file removal errors
                    normalized_paths.append(normalized_path)
                else:
                    # Keep original on failure
                    normalized_paths.append(clip_path)
            else:
                normalized_paths.append(clip_path)

        self.current_batch_paths = normalized_paths
        print(f"   ✅ Normalization complete")
    
    def finalize(self, 
                 output_path: str,
                 audio_path: Optional[str] = None,
                 audio_duration: Optional[float] = None,
                 logo_path: Optional[str] = None) -> bool:
        """
        Finalize rendering - write remaining batch and concatenate.
        
        Uses FFmpeg for final concatenation with audio mixing.
        Optionally adds logo overlay as second pass.
        
        Args:
            output_path: Final output video path
            audio_path: Optional audio track to mix
            audio_duration: Duration to trim audio to (if different from video)
            logo_path: Optional logo image for branding overlay
            
        Returns:
            True if successful
        """
        import time
        self.finalize_start_time = time.time()
        
        # Flush any remaining clips
        if self.current_batch_paths:
            self.flush_batch()

        # Concatenate all segments using FFmpeg (with optional logo and audio trimming)
        success = self.segment_writer.concatenate_segments(
            output_path,
            audio_path=audio_path,
            audio_duration=audio_duration,
            logo_path=logo_path
        )
        
        self.finalize_duration = time.time() - self.finalize_start_time
        return success
    
    def get_finalize_duration(self) -> float:
        """Get duration of finalize() call in seconds."""
        return getattr(self, 'finalize_duration', 0.0)
    
    def get_segment_count(self) -> int:
        """Get number of segments written so far."""
        return len(self.segment_writer.segments)
    
    def get_total_clips(self) -> int:
        """Get total number of clips processed."""
        return self.segment_writer.stats.total_clips + len(self.current_batch_paths)
    
    def get_stats(self) -> Dict:
        """Get rendering statistics."""
        stats = self.segment_writer.get_stats()
        stats['batch_size'] = self.batch_size
        stats['pending_clips'] = len(self.current_batch_paths)
        stats['job_id'] = self.job_id
        return stats
    
    def cleanup(self) -> None:
        """Clean up all temporary files."""
        self.segment_writer.cleanup_all()
