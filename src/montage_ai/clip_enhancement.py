"""
Clip Enhancement Module for Montage AI

Centralized 'Pixel Polishing' operations: stabilization, upscaling, color grading.
The Editor says "make it pretty", this module decides HOW (Local vs Cloud, CPU vs GPU).

Usage:
    from montage_ai.clip_enhancement import ClipEnhancer

    enhancer = ClipEnhancer()
    enhancer.enhance("/input.mp4", "/output.mp4")
    enhancer.stabilize("/input.mp4", "/stabilized.mp4")
    enhancer.upscale("/input.mp4", "/upscaled.mp4")
"""

import os
import random
import subprocess
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .config import get_settings, Settings
from .ffmpeg_config import (
    get_config as get_ffmpeg_config,
    STANDARD_CODEC,
    STANDARD_PIX_FMT,
    STANDARD_PROFILE,
    STANDARD_LEVEL,
)
from .logger import logger


_settings = get_settings()
_ffmpeg_config = get_ffmpeg_config(hwaccel=_settings.gpu.ffmpeg_hwaccel)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BrightnessAnalysis:
    """Result of clip brightness analysis."""
    avg_brightness: float
    is_dark: bool
    is_bright: bool
    suggested_brightness: float

    @classmethod
    def neutral(cls) -> "BrightnessAnalysis":
        """Return neutral analysis (no adjustment needed)."""
        return cls(
            avg_brightness=128,
            is_dark=False,
            is_bright=False,
            suggested_brightness=0
        )


@dataclass
class EnhancementResult:
    """Result of clip enhancement operation."""
    input_path: str
    output_path: str
    success: bool
    method: str  # 'enhanced', 'stabilized', 'upscaled', 'original'
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Lazy imports for optional dependencies
# =============================================================================

def _get_color_matcher():
    """Lazy load color-matcher if available."""
    try:
        from color_matcher import ColorMatcher
        from color_matcher.io_handler import load_img_file
        return ColorMatcher, load_img_file
    except ImportError:
        return None, None


def _get_cgpu_upscaler():
    """Lazy load cgpu upscaler if available."""
    try:
        from .cgpu_upscaler import upscale_with_cgpu, is_cgpu_available
        return upscale_with_cgpu, is_cgpu_available
    except ImportError:
        return None, lambda: False


def _get_cgpu_stabilizer():
    """Lazy load cgpu stabilizer if available."""
    try:
        from .cgpu_jobs.stabilize import stabilize_video as cgpu_stabilize_video
        from .cgpu_upscaler import is_cgpu_available
        return cgpu_stabilize_video, is_cgpu_available
    except ImportError:
        return None, lambda: False


# =============================================================================
# Module-level cache
# =============================================================================

_VIDSTAB_AVAILABLE: Optional[bool] = None


def _check_vidstab_available() -> bool:
    """Check if FFmpeg was compiled with libvidstab support."""
    global _VIDSTAB_AVAILABLE
    if _VIDSTAB_AVAILABLE is not None:
        return _VIDSTAB_AVAILABLE

    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, timeout=_settings.processing.ffmpeg_short_timeout
        )
        _VIDSTAB_AVAILABLE = "vidstabdetect" in result.stdout
        if _VIDSTAB_AVAILABLE:
            logger.info("vidstab (libvidstab) available - using 2-pass stabilization")
        else:
            logger.info("vidstab not available - falling back to deshake filter")
    except Exception:
        _VIDSTAB_AVAILABLE = False

    return _VIDSTAB_AVAILABLE


# =============================================================================
# ClipEnhancer Class
# =============================================================================

class ClipEnhancer:
    """
    Centralized clip enhancement with automatic backend selection.

    Handles stabilization, upscaling, color grading, and enhancement.
    Automatically chooses between local and cloud processing.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the ClipEnhancer.

        Args:
            settings: Optional Settings instance (uses global settings if None)
        """
        self.settings = settings or _settings
        self.ffmpeg_config = get_ffmpeg_config(hwaccel=self.settings.gpu.ffmpeg_hwaccel)

        # Feature flags
        self.cgpu_enabled = self.settings.llm.cgpu_gpu_enabled
        self.parallel_enhance = self.settings.processing.parallel_enhance
        self.low_memory_mode = self.settings.features.low_memory_mode
        self.max_parallel_jobs = self.settings.processing.get_adaptive_parallel_jobs(self.low_memory_mode)

        if self.low_memory_mode:
            logger.warning("LOW_MEMORY_MODE: Parallel jobs reduced to 1")

        # FFmpeg settings
        self.ffmpeg_threads = self.ffmpeg_config.threads
        self.ffmpeg_preset = self.ffmpeg_config.preset
        self.output_codec = self.ffmpeg_config.effective_codec
        self.output_pix_fmt = self.ffmpeg_config.pix_fmt
        self.output_profile = self.ffmpeg_config.profile
        self.output_level = self.ffmpeg_config.level
        self.upscale_model = self.settings.upscale.model
        self.upscale_scale = self.settings.upscale.scale
        frame_format = (self.settings.upscale.frame_format or "jpg").strip().lower()
        self.upscale_frame_format = frame_format if frame_format in ("jpg", "png") else "jpg"
        self.upscale_tile_size = self.settings.upscale.tile_size
        self.upscale_crf = self.settings.upscale.crf
        self.ffprobe_timeout = self.settings.processing.ffprobe_timeout
        self.ffmpeg_short_timeout = self.settings.processing.ffmpeg_short_timeout
        self.ffmpeg_timeout = self.settings.processing.ffmpeg_timeout
        self.ffmpeg_long_timeout = self.settings.processing.ffmpeg_long_timeout

        # Temp directory for intermediate files
        self.temp_dir = str(self.settings.paths.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

    # =========================================================================
    # Public API
    # =========================================================================

    def _skip_output_if_present(self, output_path: str, label: str) -> bool:
        """Skip processing if output exists and idempotency is enabled."""
        if self.settings.processing.should_skip_output(output_path):
            logger.info(f"{label} skipped (exists): {os.path.basename(output_path)}")
            return True
        return False

    def enhance_batch(self, clip_jobs: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Enhance multiple clips in parallel.

        Args:
            clip_jobs: List of (input_path, output_path) tuples

        Returns:
            Dict mapping input_path to resulting output_path
        """
        if not self.parallel_enhance or len(clip_jobs) <= 1:
            # Sequential processing
            results = {}
            for input_path, output_path in clip_jobs:
                logger.info(f"Enhancing {os.path.basename(input_path)}...")
                results[input_path] = self.enhance(input_path, output_path)
            return results

        logger.info(f"Parallel enhancement: {len(clip_jobs)} clips with {self.max_parallel_jobs} workers...")
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
            future_to_input = {
                executor.submit(self.enhance, input_path, output_path): input_path
                for input_path, output_path in clip_jobs
            }

            completed = 0
            for future in as_completed(future_to_input):
                input_path = future_to_input[future]
                try:
                    output_path = future.result()
                    results[input_path] = output_path
                    completed += 1
                    if completed % 10 == 0 or completed == len(clip_jobs):
                        logger.info(f"Enhanced {completed}/{len(clip_jobs)} clips...")
                except Exception as e:
                    logger.error(f"Enhancement failed for {os.path.basename(input_path)}: {e}")
                    results[input_path] = input_path

        return results

    def enhance(self, input_path: str, output_path: str) -> str:
        """
        Apply cinematic enhancements with content-aware adjustments.

        Enhancement pipeline:
        1. Analyze clip brightness
        2. Apply adaptive color grading (Teal & Orange)
        3. Apply S-curve contrast
        4. CAS (Contrast Adaptive Sharpening)
        5. Subtle vignette

        Args:
            input_path: Source video file
            output_path: Destination for enhanced video

        Returns:
            Path to enhanced video (or original on failure)
        """
        if self._skip_output_if_present(output_path, "Enhance"):
            return output_path

        analysis = self._analyze_brightness(input_path)

        # Adaptive parameters based on content
        brightness_adj = analysis.suggested_brightness

        # Adjust saturation/contrast based on brightness
        if analysis.is_dark:
            saturation = 1.08
            contrast = 1.02
            shadow_lift = 0.25
        elif analysis.is_bright:
            saturation = 1.12
            contrast = 1.08
            shadow_lift = 0.18
        else:
            saturation = 1.15
            contrast = 1.05
            shadow_lift = 0.20

        # Build filter chain
        filters = ",".join([
            # Teal & Orange color grading (Hollywood look)
            "colorbalance=rs=-0.1:gs=-0.05:bs=0.15:rm=0.05:gm=0:bm=-0.05:rh=0.1:gh=0.05:bh=-0.1",
            # Adaptive S-curve contrast
            f"curves=m='0/0 0.25/{shadow_lift} 0.5/0.5 0.75/0.80 1/1'",
            # Contrast Adaptive Sharpening
            "cas=0.5",
            # Adaptive saturation, contrast, and brightness
            f"eq=saturation={saturation}:contrast={contrast}:brightness={brightness_adj:.3f}",
            # Subtle vignette
            "vignette=PI/4:mode=forward:eval=frame",
            # Fine detail sharpening
            "unsharp=3:3:0.5:3:3:0.3"
        ])

        # Log if content needed adjustment
        if analysis.is_dark or analysis.is_bright:
            exposure_type = "dark" if analysis.is_dark else "bright"
            logger.info(f"Content-aware enhance: {os.path.basename(input_path)} ({exposure_type}, brightness={analysis.avg_brightness:.0f})")

        # Use fewer threads per job when running in parallel
        threads_per_job = "2" if self.parallel_enhance else self.ffmpeg_threads

        cmd = [
            "ffmpeg", "-y",
            "-threads", threads_per_job,
            "-i", input_path,
            "-vf", filters,
            "-c:v", self.output_codec,
            "-preset", self.ffmpeg_preset,
            "-crf", str(self.settings.encoding.crf),
            "-threads", threads_per_job,
        ]

        if self.output_profile:
            cmd.extend(["-profile:v", self.output_profile])
        if self.output_level:
            cmd.extend(["-level", self.output_level])
        cmd.extend(["-pix_fmt", self.output_pix_fmt, "-c:a", "copy", output_path])

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_path
        except Exception:
            return input_path

    def stabilize(self, input_path: str, output_path: str) -> str:
        """
        Stabilize a video clip using the best available method.

        Priority:
        1. cgpu Cloud GPU (if enabled and available)
        2. Local vidstab 2-pass (professional quality)
        3. Local deshake (basic fallback)

        Args:
            input_path: Source video file
            output_path: Destination for stabilized video

        Returns:
            Path to stabilized video (or original on failure)
        """
        logger.info(f"Stabilizing {os.path.basename(input_path)}...")
        if self._skip_output_if_present(output_path, "Stabilize"):
            return output_path

        # Try cloud GPU first if enabled
        if self.cgpu_enabled:
            cgpu_stabilize, is_available = _get_cgpu_stabilizer()
            if cgpu_stabilize and is_available():
                logger.info("Attempting cgpu cloud GPU stabilization...")
                result = cgpu_stabilize(input_path, output_path)
                if result:
                    logger.info("Cloud stabilization complete")
                    return result
                logger.warning("Cloud stabilization failed, falling back to local...")

        # Local stabilization
        if _check_vidstab_available():
            return self._stabilize_vidstab(input_path, output_path)
        else:
            return self._stabilize_deshake(input_path, output_path)

    def upscale(self, input_path: str, output_path: str, scale: Optional[int] = None) -> str:
        """
        Upscale video using the best available method.

        Priority:
        1. cgpu Cloud GPU (if enabled) - Free cloud GPU via Google Colab
        2. Real-ESRGAN local (if Vulkan GPU available) - SOTA quality
        3. FFmpeg Lanczos + Sharpening (CPU fallback)

        Args:
            input_path: Source video file
            output_path: Destination for upscaled video
            scale: Upscaling factor (default: 2)

        Returns:
            Path to upscaled video (or original on failure)
        """
        scale_value = scale if scale is not None else self.upscale_scale
        scale_value = max(2, min(int(scale_value), 4))
        model = self.upscale_model
        logger.info(f"Upscaling {os.path.basename(input_path)}...")
        if self._skip_output_if_present(output_path, "Upscale"):
            return output_path

        # Priority 1: cgpu Cloud GPU
        if self.cgpu_enabled:
            cgpu_upscale, is_available = _get_cgpu_upscaler()
            if cgpu_upscale and is_available():
                logger.info("Attempting cgpu cloud GPU upscaling...")
                result = cgpu_upscale(
                    input_path,
                    output_path,
                    scale=scale_value,
                    model=model,
                    frame_format=self.upscale_frame_format,
                    crf=self.upscale_crf,
                    tile_size=self.upscale_tile_size,
                )
                if result:
                    return result
                logger.warning("cgpu upscaling failed, falling back to local methods...")

        # Priority 2: Local Vulkan GPU (Real-ESRGAN)
        real_esrgan_available = self._check_realesrgan_available()

        if real_esrgan_available:
            logger.info("Attempting Real-ESRGAN with Vulkan GPU...")
            return self._upscale_realesrgan(input_path, output_path, model=model, scale=scale_value)
        else:
            logger.info("Using FFmpeg Lanczos upscaling (Vulkan GPU not available)")
            return self._upscale_ffmpeg(input_path, output_path, scale=scale_value)

    def color_match(
        self,
        clip_paths: List[str],
        reference_clip: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Match colors across multiple clips for visual consistency.

        Uses color-matcher library for histogram-based color transfer.

        Args:
            clip_paths: List of video file paths to match
            reference_clip: Path to reference clip (first clip if None)
            output_dir: Directory for matched clips

        Returns:
            Dict mapping original paths to color-matched paths
        """
        ColorMatcher, load_img_file = _get_color_matcher()

        if ColorMatcher is None:
            logger.warning("color-matcher not installed - skipping shot matching")
            return {p: p for p in clip_paths}

        if len(clip_paths) < 2:
            return {p: p for p in clip_paths}

        output_dir = output_dir or self.temp_dir
        os.makedirs(output_dir, exist_ok=True)

        ref_path = reference_clip or clip_paths[0]
        logger.info(f"Color matching {len(clip_paths)} clips to reference...")

        try:
            # Extract reference frame
            ref_frame_path = os.path.join(output_dir, "ref_frame.png")
            self._extract_middle_frame(ref_path, ref_frame_path)

            if not os.path.exists(ref_frame_path):
                logger.warning("Could not extract reference frame")
                return {p: p for p in clip_paths}

            ref_img = load_img_file(ref_frame_path)
            cm = ColorMatcher()
            results = {}

            for i, clip_path in enumerate(clip_paths):
                if clip_path == ref_path:
                    results[clip_path] = clip_path
                    continue

                try:
                    # Extract source frame
                    src_frame_path = os.path.join(output_dir, f"src_frame_{i}.png")
                    self._extract_middle_frame(clip_path, src_frame_path)

                    if not os.path.exists(src_frame_path):
                        results[clip_path] = clip_path
                        continue

                    # Calculate color transfer
                    src_img = load_img_file(src_frame_path)
                    matched = cm.transfer(src=src_img, ref=ref_img, method='mkl')

                    # Calculate per-channel adjustments
                    src_mean = np.mean(src_img, axis=(0, 1))
                    matched_mean = np.mean(matched, axis=(0, 1))

                    r_adj = (matched_mean[0] - src_mean[0]) / 255.0
                    g_adj = (matched_mean[1] - src_mean[1]) / 255.0
                    b_adj = (matched_mean[2] - src_mean[2]) / 255.0

                    r_bal = np.clip(r_adj * 2, -0.3, 0.3)
                    g_bal = np.clip(g_adj * 2, -0.3, 0.3)
                    b_bal = np.clip(b_adj * 2, -0.3, 0.3)

                    filter_str = f"colorbalance=rs={r_bal:.3f}:gs={g_bal:.3f}:bs={b_bal:.3f}:rm={r_bal:.3f}:gm={g_bal:.3f}:bm={b_bal:.3f}:rh={r_bal:.3f}:gh={g_bal:.3f}:bh={b_bal:.3f}"

                    output_path = os.path.join(output_dir, f"matched_{os.path.basename(clip_path)}")

                    cmd = ["ffmpeg", "-y", "-i", clip_path, "-vf", filter_str]
                    cmd.extend(["-c:v", self.output_codec, "-preset", "fast", "-crf", "18"])
                    if self.output_profile:
                        cmd.extend(["-profile:v", self.output_profile])
                    if self.output_level:
                        cmd.extend(["-level", self.output_level])
                    cmd.extend(["-pix_fmt", self.output_pix_fmt, "-c:a", "copy", output_path])

                    subprocess.run(cmd, check=True, capture_output=True, timeout=self.ffmpeg_timeout)
                    results[clip_path] = output_path

                    # Cleanup temp frame
                    if os.path.exists(src_frame_path):
                        os.remove(src_frame_path)

                except Exception as e:
                    logger.error(f"Color match failed for {os.path.basename(clip_path)}: {e}")
                    results[clip_path] = clip_path

            # Cleanup reference frame
            if os.path.exists(ref_frame_path):
                os.remove(ref_frame_path)

            matched_count = sum(1 for k, v in results.items() if k != v)
            logger.info(f"Color matched {matched_count}/{len(clip_paths)-1} clips")

            return results

        except Exception as e:
            logger.error(f"Color matching failed: {e}")
            return {p: p for p in clip_paths}

    # =========================================================================
    # Private Workers
    # =========================================================================

    def _analyze_brightness(self, input_path: str) -> BrightnessAnalysis:
        """
        Analyze clip brightness/exposure using FFmpeg signalstats.

        Returns BrightnessAnalysis with exposure data and suggested adjustments.
        """
        try:
            # Get duration for sampling points
            dur_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_path
            ]
            dur_result = subprocess.run(
                dur_cmd,
                capture_output=True,
                text=True,
                timeout=self.ffprobe_timeout,
            )
            duration = float(dur_result.stdout.strip()) if dur_result.stdout.strip() else 2.0

            # Analyze mean brightness at 3 points
            sample_points = [duration * 0.25, duration * 0.5, duration * 0.75]
            brightnesses = []

            for t in sample_points:
                analyze_cmd = [
                    "ffmpeg", "-ss", str(t), "-i", input_path,
                    "-vframes", "1",
                    "-vf", "signalstats=stat=tout+vrep+brng,metadata=print:file=-",
                    "-f", "null", "-"
                ]
                result = subprocess.run(
                    analyze_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.ffmpeg_short_timeout,
                )

                # Parse YAVG (luma average) from output
                for line in result.stderr.split('\n'):
                    if 'YAVG' in line:
                        try:
                            yavg = float(line.split('=')[-1].strip())
                            brightnesses.append(yavg)
                        except Exception:
                            pass

            if not brightnesses:
                return BrightnessAnalysis.neutral()

            avg = sum(brightnesses) / len(brightnesses)

            # Video range: 16-235
            is_dark = avg < 70
            is_bright = avg > 180

            # Calculate suggested adjustment
            if is_dark:
                suggested = min(0.15, (100 - avg) / 500)
            elif is_bright:
                suggested = max(-0.10, (120 - avg) / 500)
            else:
                suggested = 0

            return BrightnessAnalysis(
                avg_brightness=avg,
                is_dark=is_dark,
                is_bright=is_bright,
                suggested_brightness=suggested
            )

        except Exception:
            return BrightnessAnalysis.neutral()

    def _stabilize_vidstab(self, input_path: str, output_path: str) -> str:
        """
        Professional 2-pass video stabilization using vidstab.
        """
        transform_file = f"{output_path}.trf"

        try:
            # PASS 1: Motion Analysis
            logger.info("Pass 1/2: Analyzing motion...")
            cmd_detect = [
                "ffmpeg", "-y",
                "-threads", self.ffmpeg_threads,
                "-i", input_path,
                "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={transform_file}",
                "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd_detect,
                capture_output=True,
                text=True,
                timeout=self.ffmpeg_long_timeout,
            )

            if result.returncode != 0 or not os.path.exists(transform_file):
                logger.warning("Motion analysis failed, falling back to deshake")
                return self._stabilize_deshake(input_path, output_path)

            # PASS 2: Apply Stabilization
            logger.info("Pass 2/2: Applying stabilization...")
            cmd_transform = [
                "ffmpeg", "-y",
                "-threads", self.ffmpeg_threads,
                "-i", input_path,
                "-vf", f"vidstabtransform=input={transform_file}:smoothing=30:crop=black:zoom=0:interpol=bicubic",
                "-c:v", self.output_codec,
                "-preset", "fast",
                "-crf", "18",
            ]

            if self.output_profile:
                cmd_transform.extend(["-profile:v", self.output_profile])
            if self.output_level:
                cmd_transform.extend(["-level", self.output_level])
            cmd_transform.extend(["-pix_fmt", self.output_pix_fmt, "-c:a", "copy", output_path])

            subprocess.run(
                cmd_transform,
                check=True,
                capture_output=True,
                timeout=self.ffmpeg_long_timeout,
            )

            # Cleanup transform file
            if os.path.exists(transform_file):
                os.remove(transform_file)

            logger.info("Stabilization complete (vidstab 2-pass)")
            return output_path

        except subprocess.TimeoutExpired:
            logger.warning("Stabilization timed out, using original")
            return input_path
        except subprocess.CalledProcessError as e:
            logger.error(f"vidstab failed: {e.stderr[:200] if e.stderr else 'unknown error'}")
            return self._stabilize_deshake(input_path, output_path)
        except Exception as e:
            logger.error(f"Stabilization error: {e}")
            return input_path
        finally:
            if os.path.exists(transform_file):
                try:
                    os.remove(transform_file)
                except Exception:
                    pass

    def _stabilize_deshake(self, input_path: str, output_path: str) -> str:
        """
        Fallback stabilization using FFmpeg's built-in deshake filter.
        """
        cmd = [
            "ffmpeg", "-y",
            "-threads", self.ffmpeg_threads,
            "-i", input_path,
            "-vf", "deshake=rx=32:ry=32:blocksize=8:contrast=125",
            "-c:v", self.output_codec,
            "-preset", "fast",
            "-crf", "20",
        ]

        if self.output_profile:
            cmd.extend(["-profile:v", self.output_profile])
        if self.output_level:
            cmd.extend(["-level", self.output_level])
        cmd.extend(["-pix_fmt", self.output_pix_fmt, "-c:a", "copy", output_path])

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=self.ffmpeg_timeout,
            )
            logger.info("Stabilization complete (deshake fallback)")
            return output_path
        except subprocess.CalledProcessError:
            logger.warning("Stabilization failed (ffmpeg error). Using original.")
            return input_path
        except Exception as e:
            logger.error(f"Stabilization failed: {e}")
            return input_path

    def _check_realesrgan_available(self) -> bool:
        """Check if Real-ESRGAN with Vulkan GPU is available."""
        try:
            test_result = subprocess.run(
                ["realesrgan-ncnn-vulkan", "-i", "/dev/null", "-o", "/dev/null"],
                capture_output=True,
                timeout=self.ffmpeg_short_timeout,
            )
            if b"invalid gpu" not in test_result.stderr:
                vulkan_info = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True,
                    text=True,
                    timeout=self.ffmpeg_short_timeout,
                )
                if vulkan_info.returncode == 0:
                    output = vulkan_info.stdout.lower()
                    # Skip software renderers
                    if any(sw in output for sw in ["llvmpipe", "lavapipe", "swiftshader", "cpu"]):
                        logger.warning("Detected software Vulkan renderer - skipping Real-ESRGAN")
                        return False
                    # Skip GPUs without compute shader support
                    if "adreno" in output:
                        logger.warning("Detected Qualcomm Adreno GPU - no compute shader support")
                        return False
                    return True
        except Exception:
            pass
        return False

    def _upscale_realesrgan(self, input_path: str, output_path: str, model: str, scale: int) -> str:
        """Upscale using Real-ESRGAN-ncnn-vulkan (requires Vulkan GPU)."""
        frame_dir = os.path.join(self.temp_dir, f"frames_{random.randint(0, 99999)}")
        out_frame_dir = os.path.join(self.temp_dir, f"out_frames_{random.randint(0, 99999)}")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(out_frame_dir, exist_ok=True)

        try:
            # Check for rotation metadata
            probe_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height:stream_side_data=rotation",
                "-of", "json", input_path
            ]
            probe_output = subprocess.check_output(probe_cmd).decode().strip()
            probe_data = json.loads(probe_output)
            stream = probe_data.get('streams', [{}])[0]

            rotation = 0
            for side_data in stream.get('side_data_list', []):
                if 'rotation' in side_data:
                    rotation = int(side_data['rotation'])
                    break

            # Build filter based on rotation
            if rotation == -90 or rotation == 270:
                vf_filter = "transpose=1"
                logger.info(f"Detected {rotation} rotation, applying transpose=1")
            elif rotation == 90 or rotation == -270:
                vf_filter = "transpose=2"
                logger.info(f"Detected {rotation} rotation, applying transpose=2")
            elif rotation == 180 or rotation == -180:
                vf_filter = "vflip,hflip"
                logger.info(f"Detected {rotation} rotation, applying vflip,hflip")
            else:
                vf_filter = None

            # Extract frames with rotation correction
            extract_cmd = ["ffmpeg", "-i", input_path]
            if vf_filter:
                extract_cmd += ["-vf", vf_filter]
            extract_cmd += ["-q:v", "2", f"{frame_dir}/frame_%08d.jpg"]

            subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Upscale frames with Real-ESRGAN
            upscale_cmd = [
                "realesrgan-ncnn-vulkan",
                "-i", frame_dir,
                "-o", out_frame_dir,
                "-n", model,
                "-s", str(scale),
                "-g", "0",
                "-m", "/usr/local/share/realesrgan-models"
            ]
            subprocess.run(upscale_cmd, check=True)

            # Get original FPS
            fps_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_path
            ]
            fps_str = subprocess.check_output(fps_cmd).decode().strip()

            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps_val = num / den
                fps_arg = f"{fps_val:.2f}"
            else:
                fps_arg = fps_str

            # Reassemble video
            esrgan_cmd = [
                "ffmpeg", "-y", "-framerate", fps_arg,
                "-i", f"{out_frame_dir}/frame_%08d.png",
                "-c:v", self.output_codec, "-crf", str(self.upscale_crf)
            ]
            if self.output_profile:
                esrgan_cmd.extend(["-profile:v", self.output_profile])
            if self.output_level:
                esrgan_cmd.extend(["-level", self.output_level])
            esrgan_cmd.extend(["-pix_fmt", self.output_pix_fmt, output_path])

            subprocess.run(esrgan_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            return output_path

        except Exception as e:
            logger.warning(f"Real-ESRGAN failed: {e}")
            logger.info("Falling back to FFmpeg upscaling...")
            return self._upscale_ffmpeg(input_path, output_path, scale=scale_value)
        finally:
            import shutil
            if os.path.exists(frame_dir):
                shutil.rmtree(frame_dir)
            if os.path.exists(out_frame_dir):
                shutil.rmtree(out_frame_dir)

    def _upscale_ffmpeg(self, input_path: str, output_path: str, scale: int) -> str:
        """
        High-quality FFmpeg-based upscaling using Lanczos + sharpening.
        """
        try:
            # Get original dimensions with rotation handling
            probe_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height:stream_side_data=rotation",
                "-of", "json", input_path
            ]
            probe_output = subprocess.check_output(probe_cmd).decode().strip()
            probe_data = json.loads(probe_output)
            stream = probe_data.get('streams', [{}])[0]
            orig_w = int(stream.get('width', 0))
            orig_h = int(stream.get('height', 0))

            rotation = 0
            for side_data in stream.get('side_data_list', []):
                if 'rotation' in side_data:
                    rotation = abs(int(side_data['rotation']))
                    break

            if rotation in (90, 270):
                orig_w, orig_h = orig_h, orig_w
                logger.info(f"Detected {rotation} rotation, adjusted dimensions")

            new_w, new_h = orig_w * scale, orig_h * scale
            logger.info(f"Upscaling {orig_w}x{orig_h} -> {new_w}x{new_h}")

            # Build filter chain
            filter_chain = (
                f"scale={new_w}:{new_h}:flags=lanczos,"
                f"unsharp=5:5:0.8:5:5:0.0,"
                f"cas=0.4"
            )

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", filter_chain,
                "-c:v", self.output_codec,
                "-preset", "slow",
                "-crf", str(self.upscale_crf),
            ]

            if self.output_profile:
                ffmpeg_cmd.extend(["-profile:v", self.output_profile])
            if self.output_level:
                ffmpeg_cmd.extend(["-level", self.output_level])
            ffmpeg_cmd.extend(["-pix_fmt", self.output_pix_fmt, "-c:a", "copy", output_path])

            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("FFmpeg upscaling complete")
            return output_path

        except Exception as e:
            logger.error(f"FFmpeg upscaling failed: {e}")
            return input_path

    def _extract_middle_frame(self, video_path: str, output_path: str) -> bool:
        """Extract a frame from the middle of a video for color analysis."""
        try:
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=self.ffprobe_timeout,
            )
            duration = float(result.stdout.strip()) if result.stdout.strip() else 1.0

            middle = duration / 2
            extract_cmd = [
                "ffmpeg", "-y",
                "-ss", str(middle),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                output_path
            ]
            subprocess.run(
                extract_cmd,
                check=True,
                capture_output=True,
                timeout=self.ffmpeg_short_timeout,
            )
            return os.path.exists(output_path)

        except Exception:
            return False


# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

# Module-level enhancer instance (lazy initialization)
_default_enhancer: Optional[ClipEnhancer] = None


def _get_default_enhancer() -> ClipEnhancer:
    """Get or create the default ClipEnhancer instance."""
    global _default_enhancer
    if _default_enhancer is None:
        _default_enhancer = ClipEnhancer()
    return _default_enhancer


def stabilize_clip(input_path: str, output_path: str) -> str:
    """Legacy function: Stabilize a video clip."""
    return _get_default_enhancer().stabilize(input_path, output_path)


def enhance_clip(input_path: str, output_path: str) -> str:
    """Legacy function: Apply cinematic enhancements."""
    return _get_default_enhancer().enhance(input_path, output_path)


def upscale_clip(input_path: str, output_path: str) -> str:
    """Legacy function: Upscale video."""
    return _get_default_enhancer().upscale(input_path, output_path)


def enhance_clips_parallel(clip_jobs: List[Tuple[str, str]]) -> Dict[str, str]:
    """Legacy function: Enhance multiple clips in parallel."""
    return _get_default_enhancer().enhance_batch(clip_jobs)


def color_match_clips(
    clip_paths: List[str],
    reference_clip: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """Legacy function: Match colors across clips."""
    return _get_default_enhancer().color_match(clip_paths, reference_clip, output_dir)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "BrightnessAnalysis",
    "EnhancementResult",
    # Main class
    "ClipEnhancer",
    # Legacy functions
    "stabilize_clip",
    "enhance_clip",
    "upscale_clip",
    "enhance_clips_parallel",
    "color_match_clips",
    # Utility
    "_check_vidstab_available",
]
