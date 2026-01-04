"""
Preview Generator

Generates fast, low-resolution previews for the Transcript Editor and Shorts Studio.
Prioritizes speed over quality (360p, ultrafast preset).
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .logger import logger
from .ffmpeg_config import (
    PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_CRF, PREVIEW_PRESET,
    STANDARD_AUDIO_CODEC, STANDARD_AUDIO_BITRATE
)

class PreviewGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_transcript_preview(self, source_path: str, segments: List[Tuple[float, float]], output_filename: str) -> str:
        """
        Generate a preview by concatenating kept segments.
        
        Args:
            source_path: Path to source video.
            segments: List of (start, end) tuples in seconds.
            output_filename: Name of the output file.
            
        Returns:
            Path to the generated preview file.
        """
        output_path = self.output_dir / output_filename
        
        # Create a temporary concat file for ffmpeg
        # We use the 'concat demuxer' approach which is fastest but requires same codecs.
        # Since we are re-encoding for preview anyway (to downscale), we might need complex filter.
        # Actually, for preview, we want to re-encode to 360p anyway.
        
        # Construct complex filter for trimming and concatenation
        # [0:v]trim=start=0:end=10,setpts=PTS-STARTPTS,scale=640:360[v0];
        # [0:a]atrim=start=0:end=10,asetpts=PTS-STARTPTS[a0];
        # ...
        # [v0][a0][v1][a1]...concat=n=N:v=1:a=1[outv][outa]
        
        # Limit number of segments to avoid command line length limits?
        # For very long edits, this might be an issue. 
        # But for a preview, it's usually fine.
        
        filter_complex = ""
        inputs = []
        
        for i, (start, end) in enumerate(segments):
            # Video trim + scale
            filter_complex += f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}[v{i}];"
            # Audio trim
            filter_complex += f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"
            inputs.append(f"[v{i}][a{i}]")
            
        concat_part = "".join(inputs) + f"concat=n={len(segments)}:v=1:a=1[outv][outa]"
        filter_complex += concat_part
        
        cmd = [
            "ffmpeg", "-y",
            "-i", source_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", PREVIEW_PRESET, "-crf", str(PREVIEW_CRF),
            "-c:a", STANDARD_AUDIO_CODEC, "-b:a", STANDARD_AUDIO_BITRATE,
            str(output_path)
        ]
        
        logger.info(f"Generating transcript preview: {output_path}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(output_path)
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
        
        # Calculate crop in pixels
        # We need source dimensions first. 
        # For now, assume we can get them or use relative crop filter if ffmpeg supports it (it does via iw/ih).
        
        if keyframes:
            # Dynamic crop using sendcmd
            # We need to generate a command file for sendcmd
            # Format: [time] [command] [args]
            # e.g. 0.0 crop x 100; 0.0 crop y 100; ...
            
            # However, ffmpeg's crop filter supports expressions with 't'.
            # But for complex paths, sendcmd is better or generating a complex expression.
            # For simplicity in this preview generator, let's stick to static crop if keyframes are complex,
            # OR implement a basic interpolation expression if possible.
            
            # Actually, let's use the 'sendcmd' filter approach which is robust.
            # But sendcmd requires a file.
            
            # Alternative: Use the 'crop' filter with expressions if we can construct a piecewise function.
            # That's hard.
            
            # Let's fallback to static center crop for MVP of preview if keyframes are too complex,
            # OR just use the first keyframe/crop_config.
            
            # TODO: Implement full dynamic crop preview.
            # For now, we will use the static crop config (which might be the center or a specific frame).
            pass

        x = crop_config.get('x', 0.5)
        y = crop_config.get('y', 0.5)
        w = crop_config.get('width', 9/16)
        h = crop_config.get('height', 1.0)
        
        # Convert center-based (x,y) to top-left (left, top)
        # x, y are center coordinates (0-1)
        # w, h are dimensions (0-1)
        
        # crop=w=iw*W:h=ih*H:x=(iw*X)-(ow/2):y=(ih*Y)-(oh/2)
        
        crop_filter = (
            f"crop=w=iw*{w}:h=ih*{h}:"
            f"x=(iw*{x})-(ow/2):y=(ih*{y})-(oh/2),"
            f"scale={PREVIEW_HEIGHT*9//16}:{PREVIEW_HEIGHT}" # Scale to 360p vertical (approx 202x360)
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", source_path,
            "-vf", crop_filter,
            "-c:v", "libx264", "-preset", PREVIEW_PRESET, "-crf", str(PREVIEW_CRF),
            "-c:a", STANDARD_AUDIO_CODEC, "-b:a", STANDARD_AUDIO_BITRATE,
            str(output_path)
        ]
        
        logger.info(f"Generating shorts preview: {output_path}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg shorts preview failed: {e.stderr.decode()}")
            raise RuntimeError(f"Shorts preview failed: {e.stderr.decode()}")
