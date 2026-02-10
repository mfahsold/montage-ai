"""
Clip Processor Module

Handles the heavy lifting of processing individual clips:
- FFmpeg extraction
- Auto-reframing (Smart Crop)
- Enhancement (Stabilization, Upscaling, Color Grading)
- Normalization (Frame rate/Size standardization)

This function is designed to be run in a separate thread/process.
"""

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from ..logger import logger
from ..utils import file_exists_and_valid
from ..ffmpeg_utils import build_ffmpeg_cmd

def process_clip_task(
    scene_path: str,
    clip_start: float,
    cut_duration: float,
    temp_dir: str,
    temp_clip_name: str,
    ctx_stabilize: bool,
    ctx_upscale: bool,
    ctx_enhance: bool,
    ctx_color_grade: str,
    ctx_denoise: bool,
    ctx_sharpen: bool,
    ctx_film_grain: str,
    enhancer: Any,
    output_profile: Any,
    settings: Any,
    resource_manager: Any
) -> Tuple[str, Dict[str, bool], List[str]]:
    """
    Process a single clip: extract, enhance, normalize.
    Executed in a thread pool.
    """
    
    base_temp_dir = Path(temp_dir)
    clip_temp_dir = base_temp_dir / f"clip_{uuid.uuid4().hex}"
    clip_temp_dir.mkdir(parents=True, exist_ok=True)

    temp_clip_path = str(clip_temp_dir / temp_clip_name)
    temp_files = [temp_clip_path, str(clip_temp_dir)]
    
    # 1. Extract subclip
    if settings.encoding.extract_reencode:
        target_fps = output_profile.fps if output_profile else 24.0
        cmd = build_ffmpeg_cmd([
            "-ss", str(clip_start),
            "-i", scene_path,
            "-t", str(cut_duration),
            "-vf", f"fps={target_fps}",
            "-c:v", settings.encoding.codec,
            "-preset", settings.encoding.preset,
            "-crf", str(settings.encoding.crf),
            "-pix_fmt", settings.encoding.pix_fmt,
            "-an",
            temp_clip_path,
        ])
    else:
        cmd = build_ffmpeg_cmd([
            "-ss", str(clip_start),
            "-i", scene_path,
            "-t", str(cut_duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            temp_clip_path
        ])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.processing.ffmpeg_short_timeout,
    )
    if result.returncode != 0 or not os.path.exists(temp_clip_path) or os.path.getsize(temp_clip_path) == 0:
        err_lines = (result.stderr or "").strip().splitlines()
        err = err_lines[-1] if err_lines else "unknown error"
        raise RuntimeError(f"ffmpeg extract failed: {err}")
    
    # 1.5 Auto Reframe (Shorts Mode)
    if settings.features.shorts_mode:
        try:
            from ..auto_reframe import AutoReframeEngine
            # Initialize reframer (target 9:16)
            reframer = AutoReframeEngine(target_aspect=9/16)
            
            # Analyze the extracted subclip
            crops = reframer.analyze(temp_clip_path)
            
            shorts_clip_name = f"shorts_{temp_clip_name}"
            shorts_clip_path = os.path.join(temp_dir, shorts_clip_name)
            
            # Apply dynamic crop
            reframer.apply(crops, temp_clip_path, shorts_clip_path)
            
            # Update path and track temp file
            if os.path.exists(shorts_clip_path) and os.path.getsize(shorts_clip_path) > 0:
                temp_clip_path = shorts_clip_path
                temp_files.append(shorts_clip_path)
            else:
                logger.warning("Smart reframing failed, using original crop")
                
        except Exception as e:
            logger.warning(f"Smart reframing error: {e}")

    # 2. Enhance
    current_path = temp_clip_path
    stabilize_applied = False
    upscale_applied = False
    enhance_applied = False
    denoise_applied = False
    sharpen_applied = False
    film_grain_applied = False
    enhancement_timings: Dict[str, Any] = {}

    if enhancer:
        # 2a. Upscale (BEFORE stabilization per request)
        if ctx_upscale:
            stage_input = current_path
            upscale_path = str(clip_temp_dir / f"upscale_{temp_clip_name}")
            stage_start = time.perf_counter()
            result = enhancer.upscale(stage_input, upscale_path)
            stage_duration = time.perf_counter() - stage_start
            upscale_applied = result != stage_input
            enhancement_timings["upscale"] = {
                "seconds": round(stage_duration, 4),
                "status": "applied" if upscale_applied else "skipped",
            }
            if upscale_applied:
                current_path = result
                temp_files.append(upscale_path)

        # 2b. Stabilize (AFTER upscale)
        if ctx_stabilize:
            stage_input = current_path
            stab_path = str(clip_temp_dir / f"stab_{temp_clip_name}")
            stage_start = time.perf_counter()
            result = enhancer.stabilize(stage_input, stab_path)
            stage_duration = time.perf_counter() - stage_start
            stabilize_applied = result != stage_input
            enhancement_timings["stabilize"] = {
                "seconds": round(stage_duration, 4),
                "status": "applied" if stabilize_applied else "skipped",
            }
            if stabilize_applied:
                current_path = result
                temp_files.append(stab_path)

        # 2c. Enhance (color grading)
        if ctx_enhance:
            stage_input = current_path
            enhance_path = str(clip_temp_dir / f"enhance_{temp_clip_name}")
            stage_start = time.perf_counter()
            result = enhancer.enhance(stage_input, enhance_path, color_grade=ctx_color_grade)
            stage_duration = time.perf_counter() - stage_start
            enhance_applied = result != stage_input
            enhancement_timings["enhance"] = {
                "seconds": round(stage_duration, 4),
                "status": "applied" if enhance_applied else "skipped",
            }
            if enhance_applied:
                current_path = result
                temp_files.append(enhance_path)

        # 2d. Denoise
        if ctx_denoise:
            stage_input = current_path
            denoise_path = str(clip_temp_dir / f"denoise_{temp_clip_name}")
            stage_start = time.perf_counter()
            result = enhancer.denoise(stage_input, denoise_path)
            stage_duration = time.perf_counter() - stage_start
            denoise_applied = result != stage_input
            enhancement_timings["denoise"] = {
                "seconds": round(stage_duration, 4),
                "status": "applied" if denoise_applied else "skipped",
            }
            if denoise_applied:
                current_path = result
                temp_files.append(denoise_path)

        # 2e. Sharpen
        if ctx_sharpen:
            stage_input = current_path
            sharpen_path = str(clip_temp_dir / f"sharpen_{temp_clip_name}")
            stage_start = time.perf_counter()
            result = enhancer.sharpen(stage_input, sharpen_path)
            stage_duration = time.perf_counter() - stage_start
            sharpen_applied = result != stage_input
            enhancement_timings["sharpen"] = {
                "seconds": round(stage_duration, 4),
                "status": "applied" if sharpen_applied else "skipped",
            }
            if sharpen_applied:
                current_path = result
                temp_files.append(sharpen_path)

        # 2f. Film Grain
        if ctx_film_grain and ctx_film_grain != "none":
            stage_input = current_path
            grain_path = str(clip_temp_dir / f"grain_{temp_clip_name}")
            stage_start = time.perf_counter()
            result = enhancer.add_film_grain(stage_input, grain_path, preset=ctx_film_grain)
            stage_duration = time.perf_counter() - stage_start
            film_grain_applied = result != stage_input
            enhancement_timings["film_grain"] = {
                "seconds": round(stage_duration, 4),
                "status": "applied" if film_grain_applied else "skipped",
            }
            if film_grain_applied:
                current_path = result
                temp_files.append(grain_path)

    # 3. Normalize (optional)
    final_clip_path = current_path

    if not settings.encoding.normalize_clips:
        if not file_exists_and_valid(final_clip_path):
            raise RuntimeError("source clip missing before normalize")
    elif not output_profile:
        final_clip_path = str(clip_temp_dir / f"norm_{temp_clip_name}")
        if not file_exists_and_valid(current_path):
            raise RuntimeError("source clip missing before normalize")
        shutil.copy(current_path, final_clip_path)
        if not file_exists_and_valid(final_clip_path):
            raise RuntimeError("failed to write normalized clip")
    else:
        final_clip_path = str(clip_temp_dir / f"norm_{temp_clip_name}")
        # Get optimal encoder
        encoder_config = None
        if resource_manager:
            encoder_config = resource_manager.get_encoder(prefer_gpu=True)
            ffmpeg_params = encoder_config.video_params(
                crf=settings.encoding.crf,
                preset=settings.encoding.preset,
                codec_override=output_profile.codec,
                profile_override=output_profile.profile,
                level_override=output_profile.level,
                pix_fmt_override=output_profile.pix_fmt,
            )
        else:
            ffmpeg_params = [
                "-c:v", output_profile.codec,
                "-pix_fmt", output_profile.pix_fmt,
                "-crf", str(settings.encoding.crf),
                "-preset", settings.encoding.preset,
            ]
            if output_profile.profile:
                ffmpeg_params.extend(["-profile:v", output_profile.profile])
            if output_profile.level:
                ffmpeg_params.extend(["-level", output_profile.level])

        vf_filters = [
            f"scale={output_profile.width}:{output_profile.height}:force_original_aspect_ratio=decrease",
            f"pad={output_profile.width}:{output_profile.height}:(ow-iw)/2:(oh-ih)/2",
            # "setsar=1",  # Ensure square pixels - REMOVED temporarily if causing issues, or keep if critical
        ]
        # RE-ADD setsar=1 to match original code
        vf_filters.append("setsar=1")

        # Optimization: Skip expensive filters in preview mode
        is_preview = settings.encoding.quality_profile == "preview"
        
        if getattr(settings.features, "colorlevels", True) and not is_preview:
            vf_filters.append(
                "colorlevels=rimin=0.063:gimin=0.063:bimin=0.063:"
                "rimax=0.922:gimax=0.922:bimax=0.922"
            )
        
        # Ensure output pixel format matches encoder expectation
        if output_profile.pix_fmt:
            vf_filters.append(f"format={output_profile.pix_fmt}")

        vf_chain = ",".join(vf_filters)

        def run_normalize(cfg, params, label: str) -> bool:
            vf = vf_chain
            if cfg and cfg.hwupload_filter:
                vf = f"{vf_chain},{cfg.hwupload_filter}"
            cmd = build_ffmpeg_cmd([])
            if cfg and cfg.is_gpu_accelerated:
                cmd.extend(cfg.hwaccel_input_params())
            
            # Log normalization start (only once per clip)
            # logger.info(f"   ⚙️ Normalizing {label}: {os.path.basename(current_path)}")

            cmd.extend([
                "-i", current_path,
                "-vf", vf,
                "-r", str(output_profile.fps),
                *params,
                "-an",
                final_clip_path
            ])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=settings.processing.ffmpeg_timeout,
            )
            if result.returncode == 0 and file_exists_and_valid(final_clip_path):
                return True
            err_lines = (result.stderr or "").strip().splitlines()
            # Log full error for debugging
            logger.warning(f"Normalize ({label}) failed details:\n{result.stderr}")
            err = err_lines[-1] if err_lines else "unknown error"
            logger.warning(f"Normalize ({label}) failed: {err}")
            if os.path.exists(final_clip_path):
                try:
                    os.remove(final_clip_path)
                except OSError:
                    pass
            return False

        label = "gpu" if encoder_config and encoder_config.is_gpu_accelerated else "cpu"
        success = run_normalize(encoder_config, ffmpeg_params, label)

        if not success and resource_manager:
            cpu_config = resource_manager.get_encoder(prefer_gpu=False, cache_key="cpu_normalize")
            cpu_params = cpu_config.video_params(
                crf=settings.encoding.crf,
                preset=settings.encoding.preset,
                codec_override=output_profile.codec,
                profile_override="high" if output_profile.codec == "libx264" else output_profile.profile,
                level_override=output_profile.level,
                pix_fmt_override="yuv420p" if output_profile.codec == "libx264" else output_profile.pix_fmt,
            )
            success = run_normalize(cpu_config, cpu_params, "cpu_fallback")

        if not success:
            if file_exists_and_valid(current_path):
                logger.warning("Normalization failed; using unnormalized clip")
                final_clip_path = current_path
            else:
                raise RuntimeError("Normalization failed and source clip missing")

    enhancements = {
        'stabilized': stabilize_applied,
        'upscaled': upscale_applied,
        'enhanced': enhance_applied,
        'denoised': denoise_applied,
        'sharpened': sharpen_applied,
        'film_grain': film_grain_applied,
        'timings': enhancement_timings,
    }

    return final_clip_path, enhancements, temp_files
