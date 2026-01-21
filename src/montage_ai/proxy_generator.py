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
import time

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
        """Ensure a proxy exists for the given source file (with TTL + cache checks).

        Behavior:
        - Consults `settings.proxy` for cache TTL and size limits.
        - Emits telemetry events: proxy_cache_hit / proxy_cache_miss / proxy_cache_evicted / proxy_generation.
        - Keeps backwards-compatible semantics when settings are not available.
        """
        source = Path(source_path)
        proxy_path = self.get_proxy_path(source, format)

        # Try to read cache policy from settings (fall back to safe defaults)
        try:
            from .config import get_settings
            settings = get_settings()
            ttl = int(settings.proxy.proxy_cache_ttl_seconds)
            max_bytes = int(settings.proxy.proxy_cache_max_bytes)
            min_age = int(settings.proxy.proxy_cache_min_age_seconds)
        except Exception:
            ttl = 86400
            max_bytes = 1 * 1024 * 1024 * 1024
            min_age = 60

        # Fast-path: reuse existing proxy when fresh
        try:
            if proxy_path.exists() and not force:
                src_mtime = source.stat().st_mtime
                proxy_mtime = proxy_path.stat().st_mtime
                proxy_size = proxy_path.stat().st_size

                # TTL check
                if (time.time() - proxy_mtime) <= ttl and proxy_mtime >= src_mtime and proxy_size > 0:
                    logger.debug(f"Proxy cache hit: {proxy_path}")
                    try:
                        from montage_ai import telemetry
                        telemetry.record_event("proxy_cache_hit", {"file": proxy_path.name, "size": proxy_size})
                    except Exception:
                        pass
                    return proxy_path
                else:
                    try:
                        from montage_ai import telemetry
                        telemetry.record_event("proxy_cache_miss", {"file": proxy_path.name, "age_s": int(time.time() - proxy_mtime)})
                    except Exception:
                        pass
        except OSError:
            # Stat failed — fall back to (re)generate
            pass

        # Ensure cache size is within limits before generating (evict if necessary)
        try:
            self._enforce_cache_limits(max_bytes=max_bytes, min_age_seconds=min_age)
        except Exception:
            # Non-fatal — continue to attempt generation
            pass

        # Generate proxy
        try:
            out = self._generate(source, proxy_path, format)

            # Emit telemetry and enforce cache limits after successful creation
            try:
                from montage_ai import telemetry
                telemetry.record_event("proxy_generation", {"file": proxy_path.name, "size": proxy_path.stat().st_size})
            except Exception:
                pass

            try:
                self._enforce_cache_limits(max_bytes=max_bytes, min_age_seconds=min_age)
            except Exception:
                pass

            return out
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

    def _enforce_cache_limits(self, max_bytes: int, min_age_seconds: int = 60) -> None:
        """Evict oldest accessed proxy files until total cache size is <= max_bytes.

        - Respects a minimum age to avoid evicting very recently-created files.
        - Emits `proxy_cache_evicted` telemetry with bytes_freed and files_removed.
        """
        try:
            files = [p for p in self.proxy_dir.iterdir() if p.is_file()]
        except Exception:
            return

        total = 0
        entries = []  # (atime, path, size)
        for p in files:
            try:
                st = p.stat()
                entries.append((st.st_atime, p, st.st_size))
                total += st.st_size
            except Exception:
                continue

        if total <= max_bytes:
            return

        # Sort by access time (oldest first)
        entries.sort(key=lambda e: e[0])
        removed = 0
        freed = 0
        threshold_time = time.time() - min_age_seconds
        for atime, p, sz in entries:
            # Do not evict files younger than min_age_seconds
            if atime > threshold_time:
                continue
            try:
                p.unlink()
                removed += 1
                freed += sz
                total -= sz
            except Exception:
                continue
            if total <= max_bytes:
                break

        if removed:
            try:
                from montage_ai import telemetry
                telemetry.record_event("proxy_cache_evicted", {"files_removed": removed, "bytes_freed": freed})
            except Exception:
                pass

    def ensure_analysis_proxy(self, source_path: Union[str, Path], height: int = 360, force: bool = False) -> Optional[Path]:
        """Ensure a small analysis proxy exists (cached) and return its path.

        This is a convenience wrapper around `generate_analysis_proxy` that uses
        the same filesystem cache as `ensure_proxy` but produces a much smaller
        proxy optimized for scene-detection and metadata extraction.
        """
        src = Path(source_path)
        # Analysis proxies use a distinct filename suffix so they can coexist with edit proxies
        out = self.proxy_dir / f"{src.stem}_analysis_proxy.mp4"

        # Fast reuse when present and not expired
        try:
            if out.exists() and not force:
                if out.stat().st_size > 0:
                    try:
                        from .config import get_settings
                        ttl = int(get_settings().proxy.proxy_cache_ttl_seconds)
                    except Exception:
                        ttl = 86400
                    if (time.time() - out.stat().st_mtime) <= ttl:
                        try:
                            from montage_ai import telemetry
                            telemetry.record_event("analysis_proxy_reuse", {"file": out.name})
                        except Exception:
                            pass
                        return out
        except Exception:
            pass

        # Create parent dir and generate
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.generate_analysis_proxy(str(src), str(out), height=height)
            try:
                from montage_ai import telemetry
                telemetry.record_event("analysis_proxy_created", {"file": out.name, "height": height})
            except Exception:
                pass
            return out
        except Exception:
            return None

    def generate_analysis_proxy(self, source_path: str, output_path: str, height: int = 720) -> bool:
        """
        Generate a lightweight proxy for fast analysis (scene detection, feature extraction).

        FIXES / SAFEGUARDS:
        - Corrected method signature (must accept `self`). A previous implementation
          mistakenly declared this without `self` which caused `self` to be passed
          into `source_path` and produced silently-broken ffmpeg invocations.
        - Validate inputs early and fail-fast (clear error messages).
        - Cap proxy-generation timeout for preview workloads to avoid long hangs.
        """
        import subprocess
        from .ffmpeg_config import FFmpegConfig
        from .config import get_settings
        from pathlib import Path

        # Validate inputs early
        src = Path(source_path)
        if not src.exists():
            logger.error("Analysis proxy requested for missing source: %s", source_path)
            raise FileNotFoundError(f"Source not found: {source_path}")
        try:
            height = int(height)
            if height <= 0 or height > 2160:
                raise ValueError()
        except Exception:
            logger.error("Invalid analysis proxy height: %s", height)
            raise ValueError("height must be a positive integer (reasonable range)")

        # Use CPU for analysis proxy generation (robust & small)
        config = FFmpegConfig(hwaccel="none")

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]
        cmd.extend(config.hwaccel_input_params())
        cmd.extend(["-i", str(src)])

        vf_chain = [f"scale=-2:{height}"]
        cmd.extend(["-vf", ",".join(vf_chain)])

        video_params = config.video_params(crf=28, preset="ultrafast")
        cmd.extend(video_params)

        cmd.extend(["-c:a", "aac", "-b:a", "64k"])
        cmd.append(str(output_path))

        logger.info("Generating analysis proxy: %sp (%s -> %s) using %s", height, src.name, Path(output_path).name, config.effective_codec)

        # Respect configured timeout but bound it for preview so SLOs aren't blocked
        settings = get_settings()
        timeout = int(settings.analysis.proxy_generation_timeout_seconds)
        try:
            if getattr(settings.encoding, "quality_profile", "") == "preview":
                timeout = min(timeout, int(getattr(settings.processing, "preview_job_timeout", 120)))
        except Exception:
            # best-effort; keep configured timeout
            pass

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                logger.error("Proxy generation failed (rc=%s): %s", result.returncode, (result.stderr or "").strip()[:512])
                raise RuntimeError(f"FFmpeg failed: rc={result.returncode}")
            logger.info("✓ Analysis proxy created: %s", output_path)
            return True
        except subprocess.TimeoutExpired:
            logger.error("Proxy generation timed out after %ss for %s", timeout, src)
            raise
        except Exception as e:
            logger.error("Proxy generation error for %s: %s", src, e)
            raise
