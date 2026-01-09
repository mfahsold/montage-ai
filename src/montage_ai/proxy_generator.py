"""
Proxy Generator - Handles creation of lightweight editing proxies.

Based on industry standards (Frame.io, Blackmagic Cloud):
- Generates H.264 proxies for universal compatibility (web/desktop).
- Uses Hardware Acceleration (NVENC/VAAPI/VideoToolbox) where available.
- Optimizes connection structure (Timecode passthrough, GOP alignment).
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Union
import logging

from .ffmpeg_config import FFmpegConfig
from .utils import get_video_duration

logger = logging.getLogger(__name__)

class ProxyGenerator:
    """
    Manages generation of video proxies for performance optimization.
    
    Strategy:
    1. Check if proxy exists.
    2. detailed probing of source (resolution, codec).
    3. If source is already lightweight, skip (optional).
    4. Generate using HW acceleration.
    """

    def __init__(self, proxy_dir: Union[str, Path], config: Optional[FFmpegConfig] = None):
        self.proxy_dir = Path(proxy_dir)
        self.config = config or FFmpegConfig(hwaccel="auto")
        self.proxy_dir.mkdir(parents=True, exist_ok=True)

    def get_proxy_path(self, source_path: Union[str, Path], format: str = "h264") -> Path:
        """Get the expected path for a proxy file."""
        source = Path(source_path)
        ext_map = {
            "h264": ".mp4",
            "prores": ".mov",
            "dnxhr": ".mov"
        }
        ext = ext_map.get(format, ".mp4")
        # Unique mapping: filename_projection.mp4
        # We use a simple strategy: proxy_{stem}.mp4
        return self.proxy_dir / f"proxy_{source.stem}{ext}"

    def generate(self, source_path: Union[str, Path], output_path: Union[str, Path], format: str = "h264") -> bool:
        """Generate a proxy file using ffmpeg."""
        from .ffmpeg_utils import build_ffmpeg_cmd
        from .core.cmd_runner import run_command
        
        # Determine codec settings
        if format == "prores":
            v_codec = ["-c:v", "prores_ks", "-profile:v", "proxy"]
            a_codec = ["-c:a", "pcm_s16le"]
        elif format == "dnxhr":
            v_codec = ["-c:v", "dnxhd", "-profile:v", "dnxhr_lb"]
            a_codec = ["-c:a", "pcm_s16le"]
        else:
            # H.264 (Default)
            # Use 720p or 540p for proxies
            v_codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-vf", "scale=-2:720"]
            a_codec = ["-c:a", "aac", "-b:a", "128k"]
            
        cmd = build_ffmpeg_cmd([
            "-i", str(source_path),
            *v_codec,
            *a_codec,
            str(output_path)
        ])
        
        try:
            run_command(cmd)
            return True
        except Exception as e:
            logger.error(f"Proxy generation failed: {e}")
            raise

    def ensure_proxy(self, source_path: Union[str, Path], format: str = "h264", force: bool = False) -> Optional[Path]:
        """
        Ensure a proxy exists for the given source file.
        Returns path to proxy if successful/exists, None on failure.
        
        Formats: 'h264' (default), 'prores', 'dnxhr'
        """
        source = Path(source_path)
        proxy_path = self.get_proxy_path(source, format)

        if proxy_path.exists() and not force:
            logger.debug(f"Proxy exists: {proxy_path}")
            return proxy_path

        try:
            return self._generate(source, proxy_path, format)
        except Exception as e:
            logger.error(f"Failed to generate proxy for {source}: {e}")
            return None

    def _generate(self, source: Path, output: Path, format: str) -> Path:
        """Internal generation logic with FFmpeg."""
        logger.info(f"Generating proxy [{format}]: {source.name} -> {output.name}")
        
        # 1. Build Command
        cmd = ["ffmpeg", "-y"]
        
        # Input options (HW decode if available)
        cmd.extend(self.config.hwaccel_input_params())
        cmd.extend(["-i", str(source)])

        # Video stream mapping (First Video)
        cmd.extend(["-map", "0:v:0"])
        # Audio stream mapping (First Audio if exists, -map 0:a? is safer)
        cmd.extend(["-map", "0:a?"])

        # Format Logic
        if format == "prores":
            # ProRes Proxy (profile 0)
            # Scale to 960x540 (qscale -1 maintains aspect ratio)
            # ProRes needs pcm audio
            cmd.extend([
                "-c:v", "prores_ks",
                "-profile:v", "0",
                "-vf", "scale=-2:540",
                "-c:a", "pcm_s16le"
            ])
            # ProRes doesn't use GOP settings like H.264
            
        elif format == "dnxhr":
            # DNxHR LB (Low Bandwidth)
            cmd.extend([
                "-c:v", "dnxhd",
                "-profile:v", "dnxhr_lb",
                "-vf", "scale=-2:540,format=yuv422p",
                "-c:a", "pcm_s16le"
            ])
            
        else:
            # SOTA: H.264 540p or 720p, Fixed GOP for scrubbing performance
            # Standard H.264 (Default)
            
            # Audio: AAC for MP4
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])

            # Resolution: Scale to 540p height, maintain aspect ratio
            vf_chain = ["scale=-2:540"]
            
            # HW Upload if needed (for VAAPI/QSV pipeline)
            hw_filter = self.config.hwupload_filter
            # Simplified: relying on SW scaling if filters complex
            
            cmd.extend(["-vf", ",".join(vf_chain)])

            # GOP size optimization for scrubbing (FRAME.IO Keyframe Interval)
            # Frame.io recommends 1-2 seconds. We'll use 30 frames.
            cmd.extend(["-g", "30"])

            # Get Encoder Params (NVENC, etc.) from Config
            # We request "fast" preset and higher CRF (lower quality) for proxies
            # Proxy Quality: CRF 28 (Visual OK, small size)
            base_video_params = self.config.video_params(
                crf=26, 
                preset="fast"
                # We don't override codec here, we let config choose best H.264 encoder
            )
            cmd.extend(base_video_params)

        
        # Timecode Passthrough (CRITICAL for NLE Relinking)
        cmd.extend(["-map_metadata", "0", "-write_tmcd", "1"])
        
        # Output
        cmd.append(str(output))

        # 2. Execute
        with subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        ) as proc:
            # We can parse progress here if we want
            stdout, stderr = proc.communicate()
            
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg exited with {proc.returncode}:\n{stderr}")

        return output

    @staticmethod
    def generate_analysis_proxy(source_path: str, output_path: str, height: int = 720) -> bool:
        """
        Generate a lightweight proxy for fast analysis (scene detection, feature extraction).
        
        Args:
            source_path: Input video file
            output_path: Output proxy file (h264 mp4)
            height: Proxy height in pixels (default 720p); maintains aspect ratio
        
        Returns:
            True if successful, raises on failure
        
        Example:
            ProxyGenerator.generate_analysis_proxy(
                source_path="/data/input/long_raw_video.mp4",
                output_path="/tmp/proxy_long_raw_video.mp4",
                height=720
            )
        """
        import subprocess
        
        # Build FFmpeg command
        # Use hardware acceleration if available (auto-detect)
        cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel", "auto",
            "-i", str(source_path),
            # Scale to target height (maintains aspect ratio)
            "-vf", f"scale=-1:{height}",
            # Fast H.264 encoding (CRF 28 = acceptable quality for analysis)
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            # Copy audio as-is (we don't analyze it)
            "-c:a", "aac",
            "-b:a", "128k",
            # Output
            str(output_path)
        ]
        
        logger.info(f"Generating analysis proxy: {height}p ({source_path} → {output_path})")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour max for proxy generation
            )
            
            if result.returncode != 0:
                logger.error(f"Proxy generation failed: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            logger.info(f"✓ Analysis proxy created: {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Proxy generation timeout after 3600 seconds")
            raise
        except Exception as e:
            logger.error(f"Proxy generation error: {e}")
            raise
