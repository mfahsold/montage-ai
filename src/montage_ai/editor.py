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
"""

import os
import random
import json
import time
import gc
import requests
import numpy as np
from datetime import datetime

from .config import settings
from .logger import logger

# Disable TQDM progress bars globally - they create chaotic output when mixed with logs
# This affects librosa, moviepy, and any other library using tqdm
if not settings.features.verbose:
    os.environ["TQDM_DISABLE"] = "true"

import librosa
import cv2
import base64
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
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
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
    detect_gpu_encoders,
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
MAX_PARALLEL_JOBS = _settings.processing.max_parallel_jobs

# Quality: CRF for final encoding (18 = visually lossless, 23 = good balance for tests)
FINAL_CRF = _settings.encoding.crf

# Stream normalization: Ensure all clips have identical parameters for concat demuxer
NORMALIZE_CLIPS = _settings.encoding.normalize_clips

# Memory Management: Batch processing to prevent OOM
# Process clips in batches, render each batch to disk, then concatenate
BATCH_SIZE = _settings.processing.batch_size
FORCE_GC = _settings.processing.force_gc

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

# ============================================================================
# GLOBAL EDITING INSTRUCTIONS (Set by Creative Director)
# ============================================================================
EDITING_INSTRUCTIONS = None  # Will be populated by interpret_creative_prompt()

# GPU/Hardware Capability Detection (set at runtime)
GPU_CAPABILITY = None  # Will be set by detect_gpu_capabilities()

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


def detect_gpu_capabilities() -> Dict[str, Any]:
    """
    Detect available GPU/hardware acceleration capabilities.

    Checks for:
    - Vulkan video encoding (h264_vulkan, hevc_vulkan)
    - V4L2 mem2mem encoding (ARM hardware encoders)
    - DRM devices (/dev/dri/)

    Returns:
        Dict with 'encoder' (best encoder to use) and 'hwaccel' (decode acceleration)
    """
    logger.debug("Detecting GPU capabilities...")
    capabilities = {
        'encoder': 'libx264',  # Default: CPU encoding
        'hwaccel': None,
        'available': [],
        'gpu_name': 'CPU only'
    }

    # Check for DRI devices
    dri_path = '/dev/dri'
    if os.path.exists(dri_path):
        devices = os.listdir(dri_path)
        if 'renderD128' in devices or 'card0' in devices or 'card1' in devices:
            capabilities['available'].append('drm')
            logger.debug(f"DRM device found: {devices}")

    # Test Vulkan encoder (experimental - often crashes on ARM)
    if USE_GPU in ['auto', 'vulkan'] and 'drm' in capabilities['available']:
        try:
            # Quick test: Try to initialize Vulkan encoder
            test_cmd = [
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1',
                '-c:v', 'h264_vulkan', '-f', 'null', '-'
            ]
            result = subprocess.run(test_cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                capabilities['encoder'] = 'h264_vulkan'
                capabilities['available'].append('vulkan')
                capabilities['gpu_name'] = 'Vulkan GPU'
                logger.debug("Vulkan h264 encoder available")
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Vulkan encoder not usable: {e}")

    # Test V4L2 encoder (common on Raspberry Pi, some ARM SoCs)
    if USE_GPU in ['auto', 'v4l2']:
        try:
            test_cmd = [
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1',
                '-c:v', 'h264_v4l2m2m', '-f', 'null', '-'
            ]
            result = subprocess.run(test_cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                capabilities['encoder'] = 'h264_v4l2m2m'
                capabilities['available'].append('v4l2m2m')
                capabilities['gpu_name'] = 'V4L2 Hardware Encoder'
                logger.debug("V4L2 mem2mem encoder available")
        except (subprocess.TimeoutExpired, Exception) as e:
            pass  # V4L2 not available, use CPU

    # Summary
    if capabilities['encoder'] == 'libx264':
        logger.debug(f"Using CPU encoding (libx264) with {multiprocessing.cpu_count()} cores")
        logger.debug(f"Parallel enhancement: {MAX_PARALLEL_JOBS} workers")
    else:
        logger.debug(f"Using hardware encoder: {capabilities['encoder']}")

    return capabilities


def get_video_rotation(video_path: str) -> int:
    """
    Get video rotation metadata using ffprobe.
    
    Many phone videos are stored with rotation metadata (e.g., -90° for portrait).
    MoviePy doesn't automatically apply this rotation, so we need to handle it manually.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Rotation angle in degrees (0, 90, 180, 270, or -90, -180, -270)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream_side_data=rotation',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            if streams:
                side_data_list = streams[0].get('side_data_list', [])
                for side_data in side_data_list:
                    if 'rotation' in side_data:
                        rotation = int(side_data['rotation'])
                        return rotation
    except Exception as e:
        logger.debug(f"Could not get rotation metadata: {e}")

    return 0


def get_ffmpeg_encoder_params() -> List[str]:
    """
    Get optimized FFmpeg encoder parameters based on detected capabilities.
    
    Returns:
        List of FFmpeg parameters for video encoding
    """
    global GPU_CAPABILITY
    if GPU_CAPABILITY is None:
        GPU_CAPABILITY = detect_gpu_capabilities()
    
    encoder = GPU_CAPABILITY.get('encoder', 'libx264')
    
    if encoder == 'h264_vulkan':
        # Vulkan GPU encoding - let GPU handle quality
        return [
            '-c:v', 'h264_vulkan',
            '-qp', '20',  # Quality parameter for Vulkan
        ]
    
    elif encoder == 'h264_v4l2m2m':
        # V4L2 hardware encoding (Raspberry Pi, etc.)
        return [
            '-c:v', 'h264_v4l2m2m',
            '-b:v', '8M',  # Target bitrate
        ]
    
    else:
        # CPU encoding with NEON optimization (ARM64)
        # FFmpeg automatically uses NEON SIMD on ARM64
        return [
            '-c:v', 'libx264',
            '-preset', FFMPEG_PRESET,
            '-crf', '20',
            '-tune', 'film',
            '-profile:v', 'high',
            '-level', '4.1',
        ]


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

            # Show full style details in verbose mode
            if VERBOSE:
                logger.info(f"\n📋 STYLE TEMPLATE DETAILS:")
                style = EDITING_INSTRUCTIONS.get('style', {})
                pacing = EDITING_INSTRUCTIONS.get('pacing', {})
                effects = EDITING_INSTRUCTIONS.get('effects', {})
                transitions = EDITING_INSTRUCTIONS.get('transitions', {})

                logger.info(f"   Style Name:       {style.get('name', 'unknown')}")
                logger.info(f"   Description:      {style.get('description', 'N/A')[:60]}...")
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
            resp = requests.post(
                f"{FFMPEG_MCP_ENDPOINT}/clip",
                json={
                    "input": input_path,
                    "start": start,
                    "duration": duration,
                    "output": output_path,
                    "video_codec": OUTPUT_CODEC,
                    "preset": "ultrafast",
                    "copy_audio": True
                },
                timeout=120
            )
            resp.raise_for_status()
            return
        except Exception as exc:
            logger.warning(f"MCP clip failed, falling back to local ffmpeg: {exc}")

    cmd_extract = [
        "ffmpeg", "-y", "-ss", str(start), "-i", input_path,
        "-t", str(duration), "-c:v", OUTPUT_CODEC, "-preset", "ultrafast",
        "-c:a", "copy", output_path
    ]
    subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
            return result.output_path
        else:
            logger.error(f"Variant #{variant_id} Failed: {result.error}")
            return None

    except Exception as e:
        logger.error(f"Variant #{variant_id} Failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Initialize monitoring
    monitor = None
    if MONITORING_AVAILABLE:
        monitor = init_monitor(JOB_ID, VERBOSE)
        monitor.start_phase("initialization")
    
    # Interpret creative prompt (if provided)
    interpret_creative_prompt()
    
    if monitor:
        monitor.end_phase({"style": EDITING_INSTRUCTIONS.get('style', {}).get('name', CUT_STYLE) if EDITING_INSTRUCTIONS else CUT_STYLE})

    # Generate variants
    for i in range(1, NUM_VARIANTS + 1):
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
