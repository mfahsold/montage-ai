"""
Montage AI - AI-Powered Video Montage Creator

Features:
- Advanced 2024/2025 cutting techniques (Fibonacci pacing, match cuts, invisible cuts)
- Natural language control via Creative Director LLM
- Energy-aware beat-synced editing
- Cinematic style templates (Hitchcock, Wes Anderson, MTV, Documentary, etc.)

Architecture:
  Natural Language → Creative Director (LLM) → JSON Instructions →
  Editing Engine (This) → FFmpeg/MoviePy → Final Video

Version: 0.2.0 (Natural Language Control)

.. note::
    This module acts as a **Facade** for the legacy procedural API.
    New code should prefer using `src/montage_ai/core/montage_builder.py` directly.
    This file maintains backward compatibility for scripts like `montage-ai.sh`.
"""

import os
import shutil
import random
import json
import time
import gc
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

from .config import settings
from .logger import logger
from .ffmpeg_utils import build_ffmpeg_cmd

# Disable TQDM progress bars globally - they create chaotic output when mixed with logs
# This affects librosa, moviepy, and any other library using tqdm
if not settings.features.verbose:
    os.environ["TQDM_DISABLE"] = "true"

import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Any, List, Tuple

# MoviePy 1.x/2.x compatibility layer
from .moviepy_compat import (
    VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, TextClip,
    concatenate_videoclips,
    subclip, set_audio, set_duration, set_position, resize, crop, rotate,
    crossfadein, crossfadeout,
    enforce_dimensions, log_clip_info, ensure_even_dimensions, pad_to_target,
)
from tqdm import tqdm

# Alias for backward compatibility
subclip_compat = subclip


# Import Creative Director for natural language control
try:
    from .creative_director import CreativeDirector, interpret_natural_language
    from .style_templates import get_style_template, list_available_styles
    CREATIVE_DIRECTOR_AVAILABLE = True
except ImportError:
    logger.warning("Creative Director not available (missing creative_director.py)")
    CREATIVE_DIRECTOR_AVAILABLE = False

# Import Intelligent Clip Selector for ML-enhanced selection
try:
    from .clip_selector import IntelligentClipSelector, ClipCandidate
    INTELLIGENT_SELECTOR_AVAILABLE = True
except ImportError:
    logger.warning("Intelligent Clip Selector not available")
    INTELLIGENT_SELECTOR_AVAILABLE = False

# Import Footage Manager for professional clip management
from .footage_manager import integrate_footage_manager, select_next_clip
from . import segment_writer as segment_writer_module

# Import color-matcher for shot-to-shot color consistency
try:
    from color_matcher import ColorMatcher
    from color_matcher.io_handler import load_img_file, save_img_file
    COLOR_MATCHER_AVAILABLE = True
except ImportError:
    COLOR_MATCHER_AVAILABLE = False
    logger.debug("color-matcher not available - shot matching disabled")

# Import Memory Management modules
try:
    from .memory_monitor import AdaptiveMemoryManager, MemoryMonitorContext, get_memory_manager
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    logger.debug("Memory monitor not available")

# Import Metadata Cache for pre-computed scene analysis
try:
    from .metadata_cache import MetadataCache, get_metadata_cache
    METADATA_CACHE_AVAILABLE = True
except ImportError:
    METADATA_CACHE_AVAILABLE = False
    logger.debug("Metadata cache not available")

# Import Segment Writer for progressive rendering
try:
    from .segment_writer import (
        SegmentWriter, ProgressiveRenderer,
    )
    SEGMENT_WRITER_AVAILABLE = True
except ImportError:
    SEGMENT_WRITER_AVAILABLE = False
    logger.debug("Segment writer not available")

# Import centralized FFmpeg config (DRY)
from .ffmpeg_config import (
    FFmpegConfig,
    get_config as get_ffmpeg_config,
    get_moviepy_params,
    get_best_gpu_encoder,
    print_gpu_status,
    STANDARD_WIDTH_VERTICAL as STANDARD_WIDTH,
    STANDARD_HEIGHT_VERTICAL as STANDARD_HEIGHT,
    STANDARD_FPS,
    STANDARD_CODEC,
    STANDARD_PROFILE,
    STANDARD_LEVEL,
    STANDARD_PIX_FMT,
)

# Import centralized configuration (Single Source of Truth)
from .config import get_settings
_settings = get_settings()

from .core.montage_builder import MontageBuilder

# Import audio analysis module (extracted for modularity)
from .audio_analysis import (
    analyze_music_energy as _analyze_music_energy_new,
    get_beat_times as _get_beat_times_new,
    calculate_dynamic_cut_length,
    BeatInfo,
    EnergyProfile,
)
# Re-export for backward compatibility
analyze_music_energy = _analyze_music_energy_new
get_beat_times = _get_beat_times_new

# Import scene analysis module (extracted for modularity)
from .scene_analysis import (
    detect_scenes as _detect_scenes_new,
    analyze_scene_content as _analyze_scene_content_new,
    calculate_visual_similarity,
    detect_motion_blur,
    find_best_start_point,
    Scene,
    SceneAnalysis,
    SceneDetector,
)
# Re-export for backward compatibility
detect_scenes = _detect_scenes_new
analyze_scene_content = _analyze_scene_content_new

# Import video metadata module (extracted for modularity)
from .video_metadata import (
    probe_metadata,
    determine_output_profile as _determine_output_profile_new,
    build_ffmpeg_params as _build_ffmpeg_params_new,
    VideoMetadata,
    OutputProfile,
    _parse_frame_rate,
    _weighted_median,
    _even_int,
    _snap_aspect_ratio,
    _snap_resolution,
    _normalize_codec_name,
)
# Re-export for backward compatibility
determine_output_profile = _determine_output_profile_new
build_video_ffmpeg_params = _build_ffmpeg_params_new

# Import clip enhancement module (extracted for modularity)
from .clip_enhancement import (
    ClipEnhancer,
    BrightnessAnalysis,
    stabilize_clip,
    enhance_clip,
    upscale_clip,
    enhance_clips_parallel,
    color_match_clips,
    _check_vidstab_available,
)

# Import MontageBuilder for new pipeline architecture
from .core import MontageBuilder, MontageResult

# Get runtime FFmpeg config (env vars applied, with GPU auto-detection)
_ffmpeg_config = get_ffmpeg_config(hwaccel=_settings.gpu.ffmpeg_hwaccel)

# ============================================================================
# CONFIGURATION (from centralized config module - Single Source of Truth)
# ============================================================================

# Directories (from _settings.paths)
INPUT_DIR = str(_settings.paths.input_dir)
MUSIC_DIR = str(_settings.paths.music_dir)
ASSETS_DIR = str(_settings.paths.assets_dir)
OUTPUT_DIR = str(_settings.paths.output_dir)
TEMP_DIR = str(_settings.paths.temp_dir)

# Job Identification (for parallel runs)
JOB_ID = _settings.job_id

# LLM / AI Configuration (from _settings.llm)
# Priority: OpenAI-compatible > Google AI > Ollama
OPENAI_API_BASE = _settings.llm.openai_api_base
OPENAI_API_KEY = _settings.llm.openai_api_key
OPENAI_MODEL = _settings.llm.openai_model
OPENAI_VISION_MODEL = _settings.llm.openai_vision_model

OLLAMA_HOST = _settings.llm.ollama_host
OLLAMA_MODEL = _settings.llm.ollama_model
DIRECTOR_MODEL = _settings.llm.director_model
ENABLE_AI_FILTER = _settings.features.enable_ai_filter

# OpenAI Vision Client Cache (initialized on first use)
_vision_client = None

# Natural Language Control (from _settings.creative)
CREATIVE_PROMPT = _settings.creative.creative_prompt
# Example prompts:
#   "Edit this like a Hitchcock thriller"
#   "Make it fast-paced like an MTV music video"
#   "Calm and meditative with long shots"
#   "Documentary realism with natural pacing"

# Legacy Cut Style (backwards compatible, overridden by CREATIVE_PROMPT if set)
CUT_STYLE = _settings.creative.cut_style

# Output encoding defaults (from centralized config, overridable by heuristics)
# Use effective_codec to get GPU encoder when HW accel is active
OUTPUT_CODEC = _ffmpeg_config.effective_codec
OUTPUT_PIX_FMT = _ffmpeg_config.pix_fmt
OUTPUT_PROFILE = _ffmpeg_config.profile
OUTPUT_LEVEL = _ffmpeg_config.level

VERSION = "0.2.0"

def _log_startup_backends():
    """Log GPU encoding and Vision AI backend once at import."""
    if _ffmpeg_config.is_gpu_accelerated:
        logger.info(f"🎮 GPU Encoding: {OUTPUT_CODEC} ({_ffmpeg_config.gpu_encoder_type})")
    else:
        logger.info(f"💻 CPU Encoding: {OUTPUT_CODEC}")

    if OPENAI_API_BASE and OPENAI_VISION_MODEL:
        logger.info(f"👁️  Vision AI: {OPENAI_VISION_MODEL} @ {OPENAI_API_BASE}")
    elif ENABLE_AI_FILTER:
        logger.info(f"👁️  Vision AI: {OLLAMA_MODEL} (Ollama fallback)")

# Emit startup logs
_log_startup_backends()

# Visual Enhancement (from _settings.features)
STABILIZE = _settings.features.stabilize
UPSCALE = _settings.features.upscale
ENHANCE = _settings.features.enhance

# Aspect Ratio Handling
# If true: letterbox/pillarbox horizontal clips to preserve full content
# If false (default): crop to fill frame (may cut content at edges)
PRESERVE_ASPECT = _settings.features.preserve_aspect

# Logging / Debug
VERBOSE = _settings.features.verbose

# Output Control
NUM_VARIANTS = _settings.creative.num_variants

# Timeline Export (from _settings.features)
EXPORT_TIMELINE = _settings.features.export_timeline
GENERATE_PROXIES = _settings.features.generate_proxies

# Performance: CPU Threading & Parallelization (from centralized config)
FFMPEG_THREADS = _ffmpeg_config.threads
FFMPEG_PRESET = _ffmpeg_config.preset
PARALLEL_ENHANCE = _settings.processing.parallel_enhance

# Low-memory mode: use adaptive values for constrained hardware
_LOW_MEMORY_MODE = _settings.features.low_memory_mode
MAX_PARALLEL_JOBS = _settings.processing.get_adaptive_parallel_jobs(_LOW_MEMORY_MODE)

# Quality: CRF for final encoding (18 = visually lossless, 23 = good balance for tests)
FINAL_CRF = _settings.encoding.crf

# Stream normalization: Ensure all clips have identical parameters for concat demuxer
NORMALIZE_CLIPS = _settings.encoding.normalize_clips

# Memory Management: Batch processing to prevent OOM
# Process clips in batches, render each batch to disk, then concatenate
BATCH_SIZE = _settings.processing.get_adaptive_batch_size(_LOW_MEMORY_MODE)
FORCE_GC = _settings.processing.force_gc

if _LOW_MEMORY_MODE:
    logger.info(f"⚠️ LOW_MEMORY_MODE active: batch={BATCH_SIZE}, parallel={MAX_PARALLEL_JOBS}")

# Crossfade Configuration: Real FFmpeg xfade vs simple fade-to-black
# xfade creates real overlapping transitions but requires re-encoding (slower)
ENABLE_XFADE = _settings.creative.enable_xfade
XFADE_DURATION = _settings.creative.xfade_duration

# Clip reuse control
MAX_SCENE_REUSE = _settings.processing.max_scene_reuse

# Target Duration & Music Trimming (from _settings.creative)
TARGET_DURATION = _settings.creative.target_duration
MUSIC_START = _settings.creative.music_start
MUSIC_END = _settings.creative.music_end

# GPU/Hardware Acceleration (from _settings.gpu)
USE_GPU = _settings.gpu.use_gpu

# Optional FFmpeg MCP offload (from _settings.gpu)
USE_FFMPEG_MCP = _settings.gpu.use_ffmpeg_mcp
FFMPEG_MCP_ENDPOINT = _settings.gpu.ffmpeg_mcp_endpoint

# cgpu Cloud GPU Configuration (from _settings.llm)
CGPU_GPU_ENABLED = _settings.llm.cgpu_gpu_enabled

# Reuse a single MCP HTTP session to reduce connection overhead in cluster runs
_FFMPEG_MCP_SESSION = None


def _get_ffmpeg_mcp_session() -> requests.Session:
    global _FFMPEG_MCP_SESSION
    if _FFMPEG_MCP_SESSION is None:
        session = requests.Session()
        # Setup HTTP adapter with hardware-aware pool sizing
        from .config_pools import PoolConfig
        pool_conn, pool_maxsize = PoolConfig.http_pool_size()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_conn,
            pool_maxsize=pool_maxsize
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _FFMPEG_MCP_SESSION = session
    return _FFMPEG_MCP_SESSION

# ============================================================================
# GLOBAL EDITING INSTRUCTIONS (Set by Creative Director)
# ============================================================================
EDITING_INSTRUCTIONS = None  # Will be populated by interpret_creative_prompt()

# GPU/Hardware Capability Detection (Deprecated)
GPU_CAPABILITY = None

# Import cgpu Cloud Upscaler (unified job-based architecture)
try:
    from .cgpu_upscaler import upscale_with_cgpu, is_cgpu_available
    CGPU_UPSCALER_AVAILABLE = True
except ImportError:
    CGPU_UPSCALER_AVAILABLE = False
    is_cgpu_available = lambda: False
    upscale_with_cgpu = None

# Import cgpu StabilizeJob for cloud stabilization
try:
    from .cgpu_jobs import StabilizeJob
    from .cgpu_jobs.stabilize import stabilize_video as cgpu_stabilize_video
    CGPU_STABILIZE_AVAILABLE = True
except ImportError:
    CGPU_STABILIZE_AVAILABLE = False
    cgpu_stabilize_video = None

# Import Timeline Exporter if available
try:
    from .timeline_exporter import export_timeline_from_montage
    TIMELINE_EXPORT_AVAILABLE = True
except ImportError as exc:
    logger.debug(f"Timeline Exporter not available: {exc}")
    TIMELINE_EXPORT_AVAILABLE = False

# Import Live Monitoring System
try:
    from .monitoring import Monitor, init_monitor, get_monitor
    MONITORING_AVAILABLE = True
except ImportError:
    logger.debug("Monitoring not available")
    MONITORING_AVAILABLE = False
    Monitor = None
    get_monitor = lambda: None
    init_monitor = lambda *args, **kwargs: None

# Import Deep Footage Analyzer
try:
    from .footage_analyzer import DeepFootageAnalyzer, SceneAnalysis
    DEEP_ANALYSIS_AVAILABLE = True
except ImportError as exc:
    logger.debug(f"Deep Footage Analyzer not available: {exc}")
    DEEP_ANALYSIS_AVAILABLE = False
    DeepFootageAnalyzer = None

# Deep Analysis Configuration (from _settings.features)
DEEP_ANALYSIS = _settings.features.deep_analysis


def get_files(directory, extensions):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]


# =============================================================================
# Video Metadata Functions (delegating to video_metadata module)
# =============================================================================

# Note: Helper functions (_parse_frame_rate, _weighted_median, _even_int,
# _snap_aspect_ratio, _snap_resolution, _normalize_codec_name) are now
# imported directly from video_metadata module for backward compatibility.


def determine_output_profile(video_files: List[str]) -> Dict[str, Any]:
    """
    Pick output dimensions/fps/codec that match dominant input footage.

    Wrapper for backward compatibility - returns Dict instead of OutputProfile.
    """
    profile = _determine_output_profile_new(video_files)
    return profile.to_dict()


def apply_output_profile(profile: Dict[str, Any]) -> None:
    """Propagate chosen output format into module-level constants."""
    global STANDARD_WIDTH, STANDARD_HEIGHT, STANDARD_FPS
    global OUTPUT_CODEC, OUTPUT_PIX_FMT, OUTPUT_PROFILE, OUTPUT_LEVEL

    STANDARD_WIDTH = int(profile.get("width", STANDARD_WIDTH))
    STANDARD_HEIGHT = int(profile.get("height", STANDARD_HEIGHT))
    STANDARD_FPS = float(profile.get("fps", STANDARD_FPS))

    OUTPUT_CODEC = profile.get("codec", OUTPUT_CODEC)
    OUTPUT_PIX_FMT = profile.get("pix_fmt", OUTPUT_PIX_FMT)
    OUTPUT_PROFILE = profile.get("profile", OUTPUT_PROFILE)
    OUTPUT_LEVEL = profile.get("level", OUTPUT_LEVEL)

    # Update segment_writer defaults so normalization/xfade stay in sync
    segment_writer_module.STANDARD_WIDTH = STANDARD_WIDTH
    segment_writer_module.STANDARD_HEIGHT = STANDARD_HEIGHT
    segment_writer_module.STANDARD_FPS = STANDARD_FPS
    segment_writer_module.STANDARD_PIX_FMT = OUTPUT_PIX_FMT
    segment_writer_module.TARGET_PIX_FMT = OUTPUT_PIX_FMT
    segment_writer_module.TARGET_CODEC = OUTPUT_CODEC
    segment_writer_module.TARGET_PROFILE = OUTPUT_PROFILE
    segment_writer_module.TARGET_LEVEL = OUTPUT_LEVEL


def build_video_ffmpeg_params(crf: Optional[int] = None) -> List[str]:
    """
    FFmpeg params for MoviePy writes that mirror the selected output profile.

    Wrapper for backward compatibility.
    """
    return _build_ffmpeg_params_new(crf=crf)


def interpret_creative_prompt():
    """
    Interpret CREATIVE_PROMPT environment variable and set global EDITING_INSTRUCTIONS.

    Called once at startup to configure editing style from natural language.
    Falls back to legacy CUT_STYLE if no prompt provided.
    """
    global EDITING_INSTRUCTIONS

    logger.info(f"\n{'='*60}")
    logger.info(f"🎬 Montage AI v{VERSION}")
    logger.info(f"{'='*60}")

    # Show system configuration
    if VERBOSE:
        logger.info(f"\n📊 SYSTEM CONFIGURATION:")
        logger.info(f"   CPU Cores:        {multiprocessing.cpu_count()}")
        logger.info(f"   Parallel Jobs:    {MAX_PARALLEL_JOBS}")
        logger.info(f"   FFmpeg Preset:    {FFMPEG_PRESET}")
        logger.info(f"   FFmpeg Threads:   {FFMPEG_THREADS}")
        logger.info(f"   Variants:         {NUM_VARIANTS}")

        # GPU Encoder Status
        if _ffmpeg_config.is_gpu_accelerated:
            gpu_type = _ffmpeg_config.gpu_encoder_type.upper()
            effective = _ffmpeg_config.effective_codec
            logger.info(f"   🎮 GPU Encoder:   {gpu_type} → {effective}")
        else:
            best_gpu = get_best_gpu_encoder()
            if best_gpu:
                logger.info(f"   🎮 GPU Available: {best_gpu.upper()} (use FFMPEG_HWACCEL=auto to enable)")
            else:
                logger.info(f"   🎮 GPU Encoder:   None (using {OUTPUT_CODEC})")
        logger.info("")
        logger.info(f"📊 ENHANCEMENT SETTINGS (from config):")
        logger.info(f"   STABILIZE:        {_settings.features.stabilize}")
        logger.info(f"   UPSCALE:          {_settings.features.upscale}")
        logger.info(f"   ENHANCE:          {_settings.features.enhance}")
        logger.info(f"   PARALLEL_ENHANCE: {_settings.processing.parallel_enhance}")
        logger.info("")

    if CREATIVE_PROMPT and CREATIVE_DIRECTOR_AVAILABLE:
        logger.info(f"\n🎯 Creative Prompt: '{CREATIVE_PROMPT}'")
        EDITING_INSTRUCTIONS = interpret_natural_language(CREATIVE_PROMPT)

        if EDITING_INSTRUCTIONS:
            style_name = EDITING_INSTRUCTIONS['style']['name']
            logger.info(f"   ✅ Style Applied: {style_name}")
            
            # Save decisions for transparency/Web UI
            job_id = os.environ.get("JOB_ID", "unknown")
            if job_id != "unknown":
                try:
                    decisions_path = Path(settings.paths.output_dir) / f"decisions_{job_id}.json"
                    with open(decisions_path, 'w') as f:
                        json.dump(EDITING_INSTRUCTIONS, f, indent=2)
                    logger.info(f"   📝 Director's decisions saved to {decisions_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to save decisions: {e}")

            # Show full style details in verbose mode
            if VERBOSE:
                logger.info(f"\n📋 STYLE TEMPLATE DETAILS:")
                if "director_commentary" in EDITING_INSTRUCTIONS:
                    logger.info(f"   🗣️  Director's Note: {EDITING_INSTRUCTIONS['director_commentary']}")
                
                style = EDITING_INSTRUCTIONS.get('style', {})
                pacing = EDITING_INSTRUCTIONS.get('pacing', {})
                effects = EDITING_INSTRUCTIONS.get('effects', {})
                transitions = EDITING_INSTRUCTIONS.get('transitions', {})

                logger.info(f"   Style Name:       {style.get('name') or 'unknown'}")
                desc = style.get('description') or 'N/A'
                logger.info(f"   Description:      {desc[:60]}...")
                logger.info(f"   Pacing Speed:     {pacing.get('speed', 'N/A')}")
                logger.info(f"   Cut Duration:     {pacing.get('min_cut_duration', 'N/A')}-{pacing.get('max_cut_duration', 'N/A')}s")
                logger.info(f"   Beat Sync:        {pacing.get('beat_sync', 'N/A')}")
                logger.info(f"   Transition Style: {transitions.get('preferred', ['N/A'])}")
                logger.info(f"   Template Effects:")
                logger.info(f"      - Stabilization: {effects.get('stabilization', 'N/A')}")
                logger.info(f"      - Upscale:       {effects.get('upscale', 'N/A')}")
                logger.info(f"      - Sharpness:     {effects.get('sharpness_boost', 'N/A')}")
                logger.info(f"      - Contrast:      {effects.get('contrast_boost', 'N/A')}")
                logger.info(f"      - Color Grade:   {effects.get('color_grade', 'N/A')}")
        else:
            logger.warning(f"Falling back to legacy CUT_STYLE={CUT_STYLE}")
            EDITING_INSTRUCTIONS = None

    elif CREATIVE_PROMPT and not CREATIVE_DIRECTOR_AVAILABLE:
        logger.warning(f"Creative Prompt ignored (Creative Director not available)")
        logger.info(f"   Using legacy CUT_STYLE={CUT_STYLE}")
        EDITING_INSTRUCTIONS = None

    else:
        logger.info(f"   ℹ️ Using legacy CUT_STYLE={CUT_STYLE}")
        EDITING_INSTRUCTIONS = None

    logger.info(f"{'='*60}\n")

# =============================================================================
# Scene Analysis (delegated to scene_analysis module)
# =============================================================================

def detect_scenes(video_path, threshold=30.0):
    """Use PySceneDetect to find cuts in the raw video.

    Delegated to scene_analysis module. Returns list of (start, end) tuples for backward compatibility.
    """
    return _detect_scenes_new(video_path, threshold=threshold)


def analyze_scene_content(video_path, time_point):
    """Extract a frame and ask AI to describe it.

    Delegated to scene_analysis module. Returns dict for backward compatibility.
    """
    return _analyze_scene_content_new(video_path, time_point)


# calculate_visual_similarity, detect_motion_blur, find_best_start_point
# are imported directly from scene_analysis module


def extract_subclip_ffmpeg(input_path: str, start: float, duration: float, output_path: str):
    """
    Extract a subclip via FFmpeg MCP (if enabled) with fallback to local ffmpeg.
    """
    if USE_FFMPEG_MCP:
        try:
            session = _get_ffmpeg_mcp_session()
            from .config import get_settings
            preset = get_settings().encoding.preset
            resp = session.post(
                f"{FFMPEG_MCP_ENDPOINT}/clip",
                json={
                    "input": input_path,
                    "start": start,
                    "duration": duration,
                    "output": output_path,
                    "video_codec": OUTPUT_CODEC,
                    "preset": preset,
                    "copy_audio": True
                },
                timeout=settings.processing.ffmpeg_timeout
            )
            resp.raise_for_status()
            return
        except Exception as exc:
            logger.warning(f"MCP clip failed, falling back to local ffmpeg: {exc}")

    cmd_extract = build_ffmpeg_cmd([
        "-ss", str(start), "-i", input_path,
        "-t", str(duration), "-c:v", OUTPUT_CODEC, "-preset", FFMPEG_PRESET,
        "-c:a", "copy", output_path
    ])
    try:
        subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract subclip {input_path}: {e}")

# =============================================================================
# Audio Analysis (delegated to audio_analysis module)
# =============================================================================

def analyze_music_energy(audio_path):
    """Get RMS energy curve.

    Delegated to audio_analysis module. Returns (times, rms) for backward compatibility.
    """
    profile = _analyze_music_energy_new(audio_path, verbose=VERBOSE)
    return profile.times, profile.rms


def get_beat_times(audio_path):
    """Use Librosa to find beat times.

    Delegated to audio_analysis module. Returns (beat_times, tempo) for backward compatibility.
    """
    info = _get_beat_times_new(audio_path, verbose=VERBOSE)
    return info.beat_times, info.tempo


# calculate_dynamic_cut_length is imported directly from audio_analysis module


# =============================================================================
# Post-Processing: Captions & Voice Isolation
# =============================================================================

def _apply_captions(video_path: str, settings) -> str:
    """
    Apply burn-in captions to rendered video.

    Pipeline:
    1. Transcribe audio using Whisper (via cgpu if available)
    2. Burn captions into video using caption_burner

    Args:
        video_path: Path to rendered video
        settings: Settings object with caption config

    Returns:
        Path to video with captions (or original if failed)
    """
    from pathlib import Path

    style = settings.features.captions_style
    logger.info(f"\n📝 Applying captions ({style} style)...")

    video = Path(video_path)
    output_path = str(video.parent / f"{video.stem}_captioned.mp4")

    try:
        # Step 1: Transcribe audio
        transcript_path = _transcribe_for_captions(video_path)
        if not transcript_path:
            logger.warning("   Transcription failed, skipping captions")
            return video_path

        # Step 2: Burn captions
        from .caption_burner import burn_captions
        result = burn_captions(
            video_path=video_path,
            caption_path=transcript_path,
            style=style,
            output_path=output_path
        )
        logger.info(f"   ✅ Captions applied: {result}")
        return result

    except Exception as e:
        logger.error(f"   Caption application failed: {e}")
        return video_path


def _transcribe_for_captions(video_path: str) -> Optional[str]:
    """
    Transcribe video audio for caption generation.

    Uses cgpu/Whisper if available, returns JSON with word-level timestamps.

    Args:
        video_path: Path to video file

    Returns:
        Path to transcript JSON, or None if failed
    """
    from pathlib import Path

    # Check if cgpu is available for transcription
    try:
        from .cgpu_utils import is_cgpu_available
        if is_cgpu_available():
            from .transcriber import transcribe_audio
            logger.info("   🎤 Transcribing with Whisper (via cgpu)...")
            transcript = transcribe_audio(video_path, output_format="json")
            if transcript:
                logger.info(f"   Transcript: {transcript}")
                return transcript
    except ImportError:
        pass

    # Fallback: Check for existing transcript
    video = Path(video_path)
    for ext in [".json", ".srt", ".vtt"]:
        transcript = video.with_suffix(ext)
        if transcript.exists():
            logger.info(f"   Using existing transcript: {transcript}")
            return str(transcript)

    logger.warning("   No transcription available (cgpu not enabled, no existing transcript)")
    return None


def _export_timeline_for_nle(builder, result, settings) -> None:
    """
    Export timeline data for professional NLE import.

    Kept as a separate function to avoid bloating MontageBuilder.
    Called as post-processing step when EXPORT_TIMELINE is enabled.

    Args:
        builder: MontageBuilder instance with clips_metadata
        result: MontageResult with output info
        settings: Settings object
    """
    from .timeline_exporter import export_timeline_from_montage

    # Convert ClipMetadata objects to dictionaries for timeline export
    clips_data = []
    for clip in builder.ctx.clips_metadata:
        clips_data.append({
            "source_path": clip.source_path,
            "start_time": clip.start_time,
            "duration": clip.duration,
            "timeline_start": clip.timeline_start,
            "metadata": {
                "energy": getattr(clip, 'energy', None),
                "scene_type": getattr(clip, 'scene_type', None),
            }
        })

    if not clips_data:
        logger.warning("   No clips metadata for timeline export")
        return

    # Get audio path from context
    audio_path = builder.ctx.audio_result.music_path if builder.ctx.audio_result else ""

    # Generate project name from output
    project_name = Path(result.output_path).stem if result.output_path else "montage_ai"

    logger.info(f"\n📽️ Exporting timeline for NLE...")

    exported = export_timeline_from_montage(
        clips_data=clips_data,
        audio_path=audio_path,
        total_duration=result.duration,
        output_dir=str(settings.paths.output_dir),
        project_name=project_name,
        generate_proxies=settings.features.generate_proxies,
        fps=builder.ctx.output_profile.fps if builder.ctx.output_profile else 24.0,
    )

    for fmt, path in exported.items():
        logger.info(f"   ✅ {fmt.upper()}: {path}")


def create_montage(variant_id: int = 1) -> Optional[str]:
    """
    Create a video montage from input footage and music.

    This is the main entry point for montage creation. It serves as a facade
    that delegates to MontageBuilder while maintaining backward compatibility
    with the legacy procedural API.

    Args:
        variant_id: Variant number (1-based) for this montage

    Returns:
        Output path on success, None on failure
    """
    global EDITING_INSTRUCTIONS

    # Build using the new MontageBuilder pipeline
    builder = None  # Will be set for timeline export
    try:
        # Check if creative loop is enabled
        if _settings.features.creative_loop:
            from .creative_evaluator import run_creative_loop
            logger.info(f"Creative Loop enabled (max {_settings.features.creative_loop_max_iterations} iterations)")
            result = run_creative_loop(
                builder_class=MontageBuilder,
                variant_id=variant_id,
                initial_instructions=EDITING_INSTRUCTIONS,
                max_iterations=_settings.features.creative_loop_max_iterations,
                settings=_settings,
            )
            # Note: builder not available with creative_loop, timeline export skipped
        else:
            builder = MontageBuilder(
                variant_id=variant_id,
                settings=_settings,
                editing_instructions=EDITING_INSTRUCTIONS,
            )
            result = builder.build()

        if result.success:
            logger.info(f"\n✅ Variant #{variant_id} Done!")
            logger.info(f"   Output: {result.output_path}")
            logger.info(f"   Duration: {result.duration:.1f}s")
            logger.info(f"   Cuts: {result.cut_count}")
            logger.info(f"   Render time: {result.render_time:.1f}s")
            if result.file_size_mb > 0:
                logger.info(f"   File size: {result.file_size_mb:.1f} MB")

            # Post-processing: Apply captions if enabled
            final_output = result.output_path
            if _settings.features.captions:
                final_output = _apply_captions(final_output, _settings)

            # Organize output into project folder
            if result.project_package_path and os.path.exists(result.project_package_path):
                try:
                    # Move final video
                    if final_output and os.path.exists(final_output):
                        dest = os.path.join(result.project_package_path, os.path.basename(final_output))
                        shutil.move(final_output, dest)
                        final_output = dest
                        logger.info(f"   📦 Moved video to project package: {dest}")
                    
                    # Copy log file if it exists
                    log_file = os.path.join(_settings.paths.output_dir, "render.log")
                    if os.path.exists(log_file):
                        shutil.copy2(log_file, result.project_package_path)
                        logger.info(f"   📝 Copied log to project package")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to organize project files: {e}")

            return final_output
        else:
            logger.error(f"Variant #{variant_id} Failed: {result.error}")
            return None

    except Exception as e:
        logger.error(f"Variant #{variant_id} Failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point for python -m montage_ai or python -m montage_ai.editor"""
    # Initialize monitoring
    monitor = None
    if MONITORING_AVAILABLE:
        monitor = init_monitor(JOB_ID, VERBOSE)
        monitor.start_phase("initialization")

    # Interpret creative prompt (if provided)
    interpret_creative_prompt()

    if monitor:
        monitor.end_phase({"style": EDITING_INSTRUCTIONS.get('style', {}).get('name', CUT_STYLE) if EDITING_INSTRUCTIONS else CUT_STYLE})

    shard_index, shard_count = _settings.processing.get_cluster_shard()
    if shard_count > 1:
        logger.info(f"   🧩 Cluster shard: {shard_index + 1}/{shard_count}")

    variant_ids = _settings.processing.get_sharded_variants(NUM_VARIANTS)
    if not variant_ids:
        logger.warning("No variants assigned to this shard; exiting")
        if monitor:
            monitor.log_resources()
            monitor.print_summary()
        raise SystemExit(0)

    # Generate variants
    for i in variant_ids:
        if monitor:
            monitor.start_phase(f"variant_{i}")
        create_montage(i)
        if monitor:
            monitor.end_phase()

    # Print final summary
    if monitor:
        monitor.log_resources()
        monitor.print_summary()
        # Export monitoring data as JSON
        monitor.export_json(os.path.join(OUTPUT_DIR, f"monitoring_{JOB_ID}.json"))


if __name__ == "__main__":
    main()
