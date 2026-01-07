"""
Preview Generator

Generates fast, low-resolution previews for the Transcript Editor and Shorts Studio.
Prioritizes speed over quality (360p, ultrafast preset).

Performance optimizations:
- Uses ultrafast preset with -tune zerolatency for minimum encode time
- Limits preview duration (first 30s by default)
- Uses multi-threading (-threads 0)
- Enables faststart for quick playback
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .logger import logger
from .ffmpeg_utils import build_ffmpeg_cmd
from .ffmpeg_config import (
    PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_CRF, PREVIEW_PRESET,
    STANDARD_AUDIO_CODEC, STANDARD_AUDIO_BITRATE
)
from .config import get_settings
from .core.hardware import get_best_hwaccel

# Maximum preview duration in seconds (for speed) from centralized settings
_settings = get_settings()
MAX_PREVIEW_DURATION = float(_settings.preview.max_duration)

class PreviewGenerator:
    def __init__(self, output_dir: str, max_duration: float = MAX_PREVIEW_DURATION):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_duration = max_duration

    def _get_encoder_settings(self, encoder: str) -> List[str]:
        """Get speed-optimized settings for the specific encoder."""
        if "libx264" in encoder:
            return ["-preset", PREVIEW_PRESET, "-tune", "zerolatency", "-crf", str(PREVIEW_CRF)]
        elif "nvenc" in encoder:
            # p1 is fastest (lowest quality), ull = ultra low latency
            return ["-preset", "p1", "-tune", "ull", "-rc", "constqp", "-qp", str(PREVIEW_CRF)]
        elif "vaapi" in encoder:
            # VAAPI doesn't use -preset ultrafast usually
            return ["-qp", str(PREVIEW_CRF)]
        elif "videotoolbox" in encoder:
            # 0-100, 60 is roughly CRF 23-28 range
            return ["-realtime", "true", "-q:v", "60"]
        else:
            # Fallback for unknown encoders (e.g. qsv, nvmpi)
            return []

    def generate_transcript_preview(self, source_path: str, segments: List[Tuple[float, float]], output_filename: str) -> str:
        """
        Generate a preview by concatenating kept segments.
        
        Performance optimized:
        - Caps total duration at max_duration
        - Uses multi-threading
        - Uses zerolatency tuning
        - Adds faststart flag for streaming
        - Uses Hardware Acceleration if available
        
        Args:
            source_path: Path to source video.
            segments: List of (start, end) tuples in seconds.
            output_filename: Name of the output file.
            
        Returns:
            Path to the generated preview file.
        """
        output_path = self.output_dir / output_filename
        
        # Limit segments to fit within max_duration
        limited_segments = []
        total_duration = 0.0
        
        for start, end in segments:
            seg_duration = end - start
            if total_duration + seg_duration > self.max_duration:
                # Trim this segment to fit
                remaining = self.max_duration - total_duration
                if remaining > 0.5:  # Minimum meaningful segment
                    limited_segments.append((start, start + remaining))
                break
            limited_segments.append((start, end))
            total_duration += seg_duration
        
        if not limited_segments:
            # Fallback: at least first 5 seconds
            limited_segments = [(0, min(5.0, self.max_duration))]
        
        # Detect hardware acceleration
        hw_config = get_best_hwaccel()

        # Construct complex filter for trimming and concatenation
        filter_complex = ""
        
        for i, (start, end) in enumerate(limited_segments):
            # Video trim + scale
            filter_complex += f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}[v{i}];"
            # Audio trim
            filter_complex += f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"
        
        inputs = "".join([f"[v{i}][a{i}]" for i in range(len(limited_segments))])
        concat_part = f"{inputs}concat=n={len(limited_segments)}:v=1:a=1[outv][outa]"
        filter_complex += concat_part

        # If VAAPI, we need to upload to GPU memory before encoding
        if hw_config.hwupload_filter:
             filter_complex = filter_complex.replace("[outv]", "[outv_sw]")
             filter_complex += f";[outv_sw]{hw_config.hwupload_filter}[outv]"
        
        cmd_args = [
            "-threads", "0",  # Use all available CPU cores
            "-i", source_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
        ]

        # Add encoder args (e.g. -c:v h264_nvenc)
        cmd_args.extend(hw_config.encoder_args)

        # Add optimized settings for the specific encoder
        cmd_args.extend(self._get_encoder_settings(hw_config.encoder))

        cmd_args.extend([
            "-c:a", STANDARD_AUDIO_CODEC, "-b:a", STANDARD_AUDIO_BITRATE,
            "-movflags", "+faststart",  # Enable streaming
            str(output_path)
        ])

        cmd = build_ffmpeg_cmd(cmd_args)
        
        logger.info(f"Generating transcript preview ({len(limited_segments)} segments, max {self.max_duration}s): {output_path}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=60)  # 60s timeout
            return str(output_path)
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg preview generation timed out")
            raise RuntimeError("Preview generation timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg preview generation failed: {e.stderr.decode()}")
            raise RuntimeError(f"Preview generation failed: {e.stderr.decode()}")

    def generate_shorts_preview(self, source_path: str, crop_config: Dict[str, Any], output_filename: str, keyframes: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a vertical preview with crop.
        
        Args:
            source_path: Path to source video.
            crop_config: Dictionary with crop details (x, y, width, height relative to 1.0).
            output_filename: Name of the output file.
            keyframes: Optional list of keyframes for dynamic cropping.
            
        Returns:
            Path to the generated preview file.
        """
        output_path = self.output_dir / output_filename
        cmd_file_path = None

        try:
            if keyframes:
                # Dynamic crop using sendcmd
                # Create a temporary file for sendcmd
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cmd') as tmp:
                    cmd_file_path = tmp.name
                    for kf in keyframes:
                        # kf is dict with time, x, y, width, height (in pixels)
                        t = kf.get('time', 0.0)
                        x = int(kf.get('x', 0))
                        y = int(kf.get('y', 0))
                        w = int(kf.get('width', 0))
                        h = int(kf.get('height', 0))
                        
                        # Write commands: [time] [command] [arg]
                        # crop filter supports x, y, w, h commands
                        tmp.write(f"{t} x {x};\n")
                        tmp.write(f"{t} y {y};\n")
                        if w > 0: tmp.write(f"{t} w {w};\n")
                        if h > 0: tmp.write(f"{t} h {h};\n")
                
                # Use the first keyframe for initial values
                first = keyframes[0]
                x_init = int(first.get('x', 0))
                y_init = int(first.get('y', 0))
                w_init = int(first.get('width', 0))
                h_init = int(first.get('height', 0))
                
                # sendcmd must be before crop? Or we use sendcmd=f=...
                # sendcmd sends commands to all filters.
                # We initialize crop with first keyframe values.
                crop_filter = (
                    f"sendcmd=f='{cmd_file_path}',"
                    f"crop=w={w_init}:h={h_init}:x={x_init}:y={y_init},"
                    f"scale={PREVIEW_HEIGHT*9//16}:{PREVIEW_HEIGHT}"
                )
            else:
                # Static crop
                x = crop_config.get('x', 0.5)
                y = crop_config.get('y', 0.5)
                w = crop_config.get('width', 9/16)
                h = crop_config.get('height', 1.0)
                
                # Convert center-based (x,y) to top-left (left, top)
                # crop=w=iw*W:h=ih*H:x=(iw*X)-(ow/2):y=(ih*Y)-(oh/2)
                crop_filter = (
                    f"crop=w=iw*{w}:h=ih*{h}:"
                    f"x=(iw*{x})-(ow/2):y=(ih*{y})-(oh/2),"
                    f"scale={PREVIEW_HEIGHT*9//16}:{PREVIEW_HEIGHT}"
                )
            
            # Detect hardware acceleration
            hw_config = get_best_hwaccel()

            cmd_args = [
                "-threads", "0",  # Use all available cores
                "-t", str(self.max_duration),  # Limit preview duration
                "-i", source_path,
                "-vf", crop_filter,
            ]

            # Add encoder args (e.g. -c:v h264_nvenc)
            cmd_args.extend(hw_config.encoder_args)

            # Add optimized settings for the specific encoder
            cmd_args.extend(self._get_encoder_settings(hw_config.encoder))

            cmd_args.extend([
                "-c:a", STANDARD_AUDIO_CODEC, "-b:a", STANDARD_AUDIO_BITRATE,
                "-movflags", "+faststart",
                str(output_path)
            ])

            cmd = build_ffmpeg_cmd(cmd_args)
            
            logger.info(f"Generating shorts preview (max {self.max_duration}s): {output_path}")
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            return str(output_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg shorts preview failed: {e.stderr.decode()}")
            raise RuntimeError(f"Shorts preview failed: {e.stderr.decode()}")
        finally:
            if cmd_file_path and os.path.exists(cmd_file_path):
                os.unlink(cmd_file_path)
