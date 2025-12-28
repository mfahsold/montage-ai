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

# Disable TQDM progress bars globally - they create chaotic output when mixed with logs
# This affects librosa, moviepy, and any other library using tqdm
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
    print("⚠️ Creative Director not available (missing creative_director.py)")
    CREATIVE_DIRECTOR_AVAILABLE = False

# Import Intelligent Clip Selector for ML-enhanced selection
try:
    from .clip_selector import IntelligentClipSelector, ClipCandidate
    INTELLIGENT_SELECTOR_AVAILABLE = True
except ImportError:
    print("⚠️ Intelligent Clip Selector not available")
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
    print("⚠️ color-matcher not available - shot matching disabled")

# Import Memory Management modules
try:
    from .memory_monitor import AdaptiveMemoryManager, MemoryMonitorContext, get_memory_manager
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    print("⚠️ Memory monitor not available")

# Import Metadata Cache for pre-computed scene analysis
try:
    from .metadata_cache import MetadataCache, get_metadata_cache
    METADATA_CACHE_AVAILABLE = True
except ImportError:
    METADATA_CACHE_AVAILABLE = False
    print("⚠️ Metadata cache not available")

# Import Segment Writer for progressive rendering
try:
    from .segment_writer import (
        SegmentWriter, ProgressiveRenderer,
    )
    SEGMENT_WRITER_AVAILABLE = True
except ImportError:
    SEGMENT_WRITER_AVAILABLE = False
    print("⚠️ Segment writer not available")

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

# Import audio analysis module (extracted for modularity)
from .audio_analysis import (
    analyze_music_energy as _analyze_music_energy_new,
    get_beat_times as _get_beat_times_new,
    calculate_dynamic_cut_length,
    BeatInfo,
    EnergyProfile,
)

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

# Import video metadata module (extracted for modularity)
from .video_metadata import (
    probe_metadata,
    ffprobe_video_metadata,
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
        print(f"🎮 GPU Encoding: {OUTPUT_CODEC} ({_ffmpeg_config.gpu_encoder_type})")
    else:
        print(f"💻 CPU Encoding: {OUTPUT_CODEC}")

    if OPENAI_API_BASE and OPENAI_VISION_MODEL:
        print(f"👁️  Vision AI: {OPENAI_VISION_MODEL} @ {OPENAI_API_BASE}")
    elif ENABLE_AI_FILTER:
        print(f"👁️  Vision AI: {OLLAMA_MODEL} (Ollama fallback)")

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
    print(f"⚠️ Timeline Exporter not available (missing timeline_exporter.py): {exc}")
    TIMELINE_EXPORT_AVAILABLE = False

# Import Live Monitoring System
try:
    from .monitoring import Monitor, init_monitor, get_monitor
    MONITORING_AVAILABLE = True
except ImportError:
    print("⚠️ Monitoring not available (missing monitoring.py)")
    MONITORING_AVAILABLE = False
    Monitor = None
    get_monitor = lambda: None
    init_monitor = lambda *args, **kwargs: None

# Import Deep Footage Analyzer
try:
    from .footage_analyzer import DeepFootageAnalyzer, SceneAnalysis
    DEEP_ANALYSIS_AVAILABLE = True
except ImportError as exc:
    print(f"⚠️ Deep Footage Analyzer not available (missing footage_analyzer.py): {exc}")
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
    print("🔍 Detecting GPU capabilities...")
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
            print(f"   ✅ DRM device found: {devices}")
    
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
                print("   ✅ Vulkan h264 encoder available")
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"   ⚠️ Vulkan encoder not usable: {e}")
    
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
                print("   ✅ V4L2 mem2mem encoder available")
        except (subprocess.TimeoutExpired, Exception) as e:
            pass  # V4L2 not available, use CPU
    
    # Summary
    if capabilities['encoder'] == 'libx264':
        print(f"   ℹ️ Using CPU encoding (libx264) with {multiprocessing.cpu_count()} cores")
        print(f"   ℹ️ Parallel enhancement: {MAX_PARALLEL_JOBS} workers")
    else:
        print(f"   ✅ Using hardware encoder: {capabilities['encoder']}")
    
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
        print(f"   ⚠️ Could not get rotation metadata: {e}")
    
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

# Note: ffprobe_video_metadata is imported directly from video_metadata module.


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

    print(f"\n{'='*60}")
    print(f"🎬 Montage AI v{VERSION}")
    print(f"{'='*60}")
    
    # Show system configuration
    if VERBOSE:
        print(f"\n📊 SYSTEM CONFIGURATION:")
        print(f"   CPU Cores:        {multiprocessing.cpu_count()}")
        print(f"   Parallel Jobs:    {MAX_PARALLEL_JOBS}")
        print(f"   FFmpeg Preset:    {FFMPEG_PRESET}")
        print(f"   FFmpeg Threads:   {FFMPEG_THREADS}")
        print(f"   Variants:         {NUM_VARIANTS}")
        
        # GPU Encoder Status
        if _ffmpeg_config.is_gpu_accelerated:
            gpu_type = _ffmpeg_config.gpu_encoder_type.upper()
            effective = _ffmpeg_config.effective_codec
            print(f"   🎮 GPU Encoder:   {gpu_type} → {effective}")
        else:
            best_gpu = get_best_gpu_encoder()
            if best_gpu:
                print(f"   🎮 GPU Available: {best_gpu.upper()} (use FFMPEG_HWACCEL=auto to enable)")
            else:
                print(f"   🎮 GPU Encoder:   None (using {OUTPUT_CODEC})")
        print(f"")
        print(f"📊 ENHANCEMENT SETTINGS (from config):")
        print(f"   STABILIZE:        {_settings.features.stabilize}")
        print(f"   UPSCALE:          {_settings.features.upscale}")
        print(f"   ENHANCE:          {_settings.features.enhance}")
        print(f"   PARALLEL_ENHANCE: {_settings.processing.parallel_enhance}")
        print(f"")

    if CREATIVE_PROMPT and CREATIVE_DIRECTOR_AVAILABLE:
        print(f"\n🎯 Creative Prompt: '{CREATIVE_PROMPT}'")
        EDITING_INSTRUCTIONS = interpret_natural_language(CREATIVE_PROMPT)

        if EDITING_INSTRUCTIONS:
            style_name = EDITING_INSTRUCTIONS['style']['name']
            print(f"   ✅ Style Applied: {style_name}")
            
            # Show full style details in verbose mode
            if VERBOSE:
                print(f"\n📋 STYLE TEMPLATE DETAILS:")
                style = EDITING_INSTRUCTIONS.get('style', {})
                pacing = EDITING_INSTRUCTIONS.get('pacing', {})
                effects = EDITING_INSTRUCTIONS.get('effects', {})
                transitions = EDITING_INSTRUCTIONS.get('transitions', {})
                
                print(f"   Style Name:       {style.get('name', 'unknown')}")
                print(f"   Description:      {style.get('description', 'N/A')[:60]}...")
                print(f"   Pacing Speed:     {pacing.get('speed', 'N/A')}")
                print(f"   Cut Duration:     {pacing.get('min_cut_duration', 'N/A')}-{pacing.get('max_cut_duration', 'N/A')}s")
                print(f"   Beat Sync:        {pacing.get('beat_sync', 'N/A')}")
                print(f"   Transition Style: {transitions.get('preferred', ['N/A'])}")
                print(f"   Template Effects:")
                print(f"      - Stabilization: {effects.get('stabilization', 'N/A')}")
                print(f"      - Upscale:       {effects.get('upscale', 'N/A')}")
                print(f"      - Sharpness:     {effects.get('sharpness_boost', 'N/A')}")
                print(f"      - Contrast:      {effects.get('contrast_boost', 'N/A')}")
                print(f"      - Color Grade:   {effects.get('color_grade', 'N/A')}")
        else:
            print(f"   ⚠️ Falling back to legacy CUT_STYLE={CUT_STYLE}")
            EDITING_INSTRUCTIONS = None

    elif CREATIVE_PROMPT and not CREATIVE_DIRECTOR_AVAILABLE:
        print(f"   ⚠️ Creative Prompt ignored (Creative Director not available)")
        print(f"   Using legacy CUT_STYLE={CUT_STYLE}")
        EDITING_INSTRUCTIONS = None

    else:
        print(f"   ℹ️ Using legacy CUT_STYLE={CUT_STYLE}")
        EDITING_INSTRUCTIONS = None

    print(f"{'='*60}\n")

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
            print(f"   ⚠️ MCP clip failed, falling back to local ffmpeg: {exc}")

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

def create_montage(variant_id=1):
    print(f"\n🎬 Starting Montage Variant #{variant_id}")
    
    # Get monitor instance
    monitor = get_monitor()
    montage_start_time = time.time() if 'time' in dir() else None
    
    if monitor:
        monitor.log_info("montage_start", f"Beginning Variant #{variant_id}", {
            "job_id": JOB_ID,
            "variant": variant_id
        })

    # 🎬 CREATIVE DIRECTOR INTEGRATION: Override effects based on EDITING_INSTRUCTIONS
    # IMPORTANT: Environment variables take PRECEDENCE over style templates!
    # This allows users to force STABILIZE=true even if the template says false.
    global STABILIZE, UPSCALE, ENHANCE

    # Get current values from centralized config (_settings.features)
    env_stabilize = _settings.features.stabilize
    env_upscale = _settings.features.upscale
    env_enhance = _settings.features.enhance

    # Apply style template defaults ONLY if settings are false (not explicitly enabled)
    if EDITING_INSTRUCTIONS is not None:
        effects = EDITING_INSTRUCTIONS.get('effects', {})

        # Only use template value if not explicitly enabled via settings
        # ENV=true always wins, ENV=false or unset uses template
        if not env_stabilize and 'stabilization' in effects:
            STABILIZE = effects['stabilization']
        else:
            STABILIZE = env_stabilize

        if not env_upscale and 'upscale' in effects:
            UPSCALE = effects['upscale']
        else:
            UPSCALE = env_upscale

        if 'sharpness_boost' in effects and not env_enhance:
            ENHANCE = effects['sharpness_boost']
        else:
            ENHANCE = env_enhance

    print(f"   🎨 Effects: STABILIZE={STABILIZE}, UPSCALE={UPSCALE}, ENHANCE={ENHANCE}")

    # Initialize Intelligent Clip Selector if available
    intelligent_selector = None
    if INTELLIGENT_SELECTOR_AVAILABLE:
        try:
            # Determine style from EDITING_INSTRUCTIONS or default
            style = "dynamic"
            if EDITING_INSTRUCTIONS is not None:
                style = EDITING_INSTRUCTIONS.get('style', {}).get('template', 'dynamic')

            intelligent_selector = IntelligentClipSelector(style=style)
            print(f"   🧠 Intelligent Clip Selector initialized (style={style})")
        except Exception as e:
            print(f"   ⚠️ Failed to initialize Intelligent Clip Selector: {e}")
            intelligent_selector = None

    # 1. Load Music
    music_files = get_files(MUSIC_DIR, ('.mp3', '.wav'))
    if not music_files:
        print("❌ No music found in /data/music")
        return
    
    # Pick a random track if multiple exist, or rotate through them based on variant_id
    music_index = (variant_id - 1) % len(music_files)
    music_path = music_files[music_index]
    
    if monitor:
        monitor.start_phase("music_analysis")
    
    beat_times, tempo = get_beat_times(music_path)
    energy_times, energy_values = analyze_music_energy(music_path)
    
    if monitor:
        # Determine energy profile
        avg_energy = np.mean(energy_values) if len(energy_values) > 0 else 0.5
        energy_profile = "high" if avg_energy > 0.6 else "low" if avg_energy < 0.4 else "mixed"
        monitor.log_beat_analysis(music_path, tempo, len(beat_times), energy_profile)
        monitor.end_phase({
            "bpm": tempo,
            "beats": len(beat_times),
            "energy_profile": energy_profile
        })
    
    # 2. Load Videos & Detect Scenes
    video_files = get_files(INPUT_DIR, ('.mp4', '.mov'))
    if not video_files:
        print("❌ No videos found in /data/input")
        return

    # 🔧 Determine ideal output format from incoming footage
    output_profile = determine_output_profile(video_files)
    apply_output_profile(output_profile)

    print("\n🧭 Output format heuristic:")
    print(f"   Orientation: {output_profile.get('orientation')} ({output_profile.get('aspect_ratio')})")
    print(f"   Resolution:  {STANDARD_WIDTH}x{STANDARD_HEIGHT}")
    print(f"   Frame rate:  {STANDARD_FPS:.2f} fps")
    print(f"   Codec:       {OUTPUT_CODEC} (profile={OUTPUT_PROFILE or 'auto'}, level={OUTPUT_LEVEL or 'auto'})")
    print(f"   Pixel fmt:   {OUTPUT_PIX_FMT}")
    if output_profile.get('bitrate', 0) > 0:
        bitrate_mbps = output_profile['bitrate'] / 1_000_000
        print(f"   Bitrate:     {bitrate_mbps:.1f} Mbps (from input median)")

    if VERBOSE:
        summary = output_profile.get('source_summary', {})
        print(f"\n   📊 Input analysis:")
        print(f"      Files:     {summary.get('total_input_files', 0)} videos ({summary.get('total_input_duration', 0):.1f}s total)")
        print(f"      Dominant:  {summary.get('dominant_codec', 'n/a')} @ {summary.get('dominant_pix_fmt', 'n/a')}")
        print(f"      Reason:    {output_profile.get('reason', 'default')}")

    if monitor:
        monitor.log_info("output_profile", "Selected output format", output_profile)

    all_scenes = [] # List of (video_path, start, end, duration)
    
    if monitor:
        monitor.start_phase("scene_detection")
    
    scene_detect_start = time.time()
    for v_idx, v_path in enumerate(video_files):
        scenes = detect_scenes(v_path)
        
        if monitor:
            monitor.log_scene_detection(v_path, [{"duration": e-s} for s, e in scenes], 30.0)
            monitor.log_progress(v_idx + 1, len(video_files), "videos scanned")
        
        for start, end in scenes:
            duration = end - start
            if duration > 1.0: # Ignore tiny clips
                all_scenes.append({
                    'path': v_path,
                    'start': start,
                    'end': end,
                    'duration': duration
                })
    
    if monitor:
        monitor.end_phase({
            "videos_processed": len(video_files),
            "scenes_found": len(all_scenes),
            "duration_s": time.time() - scene_detect_start
        })
    
    # AI Director: Analyze scenes
    print("🤖 AI Director is watching footage...")
    # Limit to first 20 scenes for speed in this demo, or all if few
    scenes_to_analyze = all_scenes[:20] if len(all_scenes) > 20 else all_scenes
    for scene in tqdm(scenes_to_analyze):
        mid_point = scene['start'] + (scene['duration'] / 2)
        meta = analyze_scene_content(scene['path'], mid_point)
        scene['meta'] = meta
    
    # Show footage analysis summary
    total_footage_duration = sum(s['duration'] for s in all_scenes)
    unique_videos = len(set(s['path'] for s in all_scenes))
    avg_scene_duration = total_footage_duration / len(all_scenes) if all_scenes else 0
    
    # Scene duration distribution
    short_scenes = sum(1 for s in all_scenes if s['duration'] < 3)
    medium_scenes = sum(1 for s in all_scenes if 3 <= s['duration'] < 8)
    long_scenes = sum(1 for s in all_scenes if s['duration'] >= 8)
    
    if monitor:
        monitor.log_data_flow(
            "footage_analysis",
            f"{len(video_files)} videos",
            f"{len(all_scenes)} usable scenes",
            "scene_detection + AI_analysis",
            {
                "total_footage_s": total_footage_duration,
                "avg_scene_duration": avg_scene_duration,
                "distribution": {"short": short_scenes, "medium": medium_scenes, "long": long_scenes}
            }
        )
    
    if VERBOSE:
        print(f"\n📊 FOOTAGE ANALYSIS SUMMARY:")
        print(f"   Total Videos:       {len(video_files)}")
        print(f"   Total Scenes:       {len(all_scenes)}")
        print(f"   Unique Sources:     {unique_videos}")
        print(f"   Total Footage:      {total_footage_duration:.1f}s ({total_footage_duration/60:.1f} min)")
        print(f"   Avg Scene Duration: {avg_scene_duration:.1f}s")
        print(f"   Scene Distribution: Short(<3s)={short_scenes}, Medium(3-8s)={medium_scenes}, Long(>8s)={long_scenes}")
        print(f"")
    
    # 🔬 DEEP FOOTAGE ANALYSIS (if enabled)
    deep_analyzer = None
    if DEEP_ANALYSIS and DEEP_ANALYSIS_AVAILABLE:
        print(f"\n🔬 Running Deep Footage Analysis...")
        if monitor:
            monitor.start_phase("deep_analysis")
        
        deep_analyzer = DeepFootageAnalyzer(sample_frames=8, verbose=VERBOSE)
        
        # Analyze each unique video file
        analyzed_paths = set()
        for scene in all_scenes:
            if scene['path'] not in analyzed_paths:
                analysis = deep_analyzer.analyze_clip(
                    scene['path'], 
                    scene['start'], 
                    scene['end']
                )
                # Store analysis in scene for later use
                scene['deep_analysis'] = analysis
                analyzed_paths.add(scene['path'])
        
        # Print comprehensive summary
        deep_analyzer.print_footage_summary()
        
        # Export analysis to JSON
        analysis_file = os.path.join(OUTPUT_DIR, f"footage_analysis_{JOB_ID}.json")
        deep_analyzer.export_analysis(analysis_file)
        
        if monitor:
            summary = deep_analyzer.get_footage_summary()
            monitor.end_phase({
                "clips_analyzed": summary.get("total_clips", 0),
                "establishing_shots": len(summary.get("narrative_inventory", {}).get("establishing_shots", [])),
                "climax_candidates": len(summary.get("narrative_inventory", {}).get("climax_candidates", []))
            })
    
    # OPTIMIZATION FOR SHORT/LIMITED FOOTAGE
    # If we have very few scenes, chop them up to create more options
    if len(all_scenes) < 10 or unique_videos == 1:
        print(f"   ✨ Optimizing for limited footage ({len(all_scenes)} scenes, {unique_videos} sources)...")
        new_subscenes = []
        for scene in all_scenes:
            # Only chop if scene is long enough (e.g. > 5s)
            if scene['duration'] > 5.0:
                # Create 4s chunks with randomized overlap for variety
                chunk_size = 4.0
                # Base step size 3.0 (1s overlap), plus random jitter +/- 0.5s
                step_size = 3.0 + random.uniform(-0.5, 0.5)
                
                current_start = scene['start']
                while current_start + chunk_size <= scene['end']:
                    # Create new sub-scene
                    sub = scene.copy()
                    sub['start'] = current_start
                    sub['end'] = current_start + chunk_size
                    sub['duration'] = chunk_size
                    # Inherit meta
                    sub['meta'] = scene.get('meta', {}).copy()
                    sub['meta']['is_subclip'] = True
                    new_subscenes.append(sub)
                    
                    current_start += step_size
        
        if new_subscenes:
            print(f"   ✨ Generated {len(new_subscenes)} sub-clips to increase variety")
            all_scenes.extend(new_subscenes)

    random.shuffle(all_scenes) # Mix it up

    # 📚 Initialize Professional Footage Pool Manager
    print("📚 Initializing Footage Pool Manager...")
    footage_pool = integrate_footage_manager(
        all_scenes,
        strict_once=False  # Allow AI to reuse when it improves flow; capped by MAX_SCENE_REUSE
    )
    if VERBOSE:
        print(f"   ✓ Footage Pool: {len(all_scenes)} clips ready")

    # 3. Assemble Timeline
    clips_metadata = []  # Track metadata for timeline export (OTIO/EDL/CSV)
    current_time = 0
    beat_idx = 0
    cut_number = 0  # For monitoring
    
    audio_clip = AudioFileClip(music_path)
    original_audio_duration = audio_clip.duration
    
    # Log UI settings for debugging (helps confirm values arrived correctly)
    if VERBOSE:
        print(f"\n   📋 Duration Settings (from UI):")
        print(f"      TARGET_DURATION: {TARGET_DURATION}s {'(override active)' if TARGET_DURATION > 0 else '(use full audio)'}")
        print(f"      MUSIC_START: {MUSIC_START}s")
        print(f"      MUSIC_END: {MUSIC_END}s" if MUSIC_END else "      MUSIC_END: (not set, will use full track or target)")
        print(f"      Original audio: {original_audio_duration:.1f}s")
    
    # Apply music trimming (from Web UI)
    music_start = MUSIC_START
    music_end = MUSIC_END if MUSIC_END else audio_clip.duration
    
    # Validate and apply trimming
    if music_start > 0 or music_end < audio_clip.duration:
        # Ensure valid range
        music_start = max(0, min(music_start, audio_clip.duration - 1))
        music_end = max(music_start + 1, min(music_end, audio_clip.duration))
        if VERBOSE:
            print(f"   🎵 Music trim: {music_start:.1f}s → {music_end:.1f}s (from {original_audio_duration:.1f}s)")
        audio_clip = subclip_compat(audio_clip, music_start, music_end)
    
    # Determine target duration
    # Priority: TARGET_DURATION env > trimmed audio duration
    if TARGET_DURATION > 0:
        target_duration = TARGET_DURATION
        if VERBOSE:
            print(f"   ⏱️ Target duration: {target_duration:.1f}s (from UI setting)")
    else:
        target_duration = audio_clip.duration
        if VERBOSE:
            print(f"   ⏱️ Target duration: {target_duration:.1f}s (from trimmed audio)")
    
    # Estimate total cuts for progress tracking
    # Use tempo-aware estimate: faster tempo = more cuts
    # Average 4 beats per cut, so cuts = beats / 4 = (duration * tempo / 60) / 4
    avg_beats_per_cut = 4.0  # Common average
    estimated_total_cuts = int((target_duration * tempo / 60) / avg_beats_per_cut)
    
    # Initialize Memory Manager for adaptive batch sizing
    memory_manager = None
    if MEMORY_MONITOR_AVAILABLE:
        memory_manager = get_memory_manager()
        memory_manager.print_memory_status("   ")
    
    # Determine if crossfades should use xfade (real overlap) or simple fade
    # Priority: ENV override > Creative Director settings > default (off)
    enable_xfade = False
    xfade_duration = XFADE_DURATION  # From ENV, default 0.3
    
    # Check ENV override first
    if ENABLE_XFADE == "true":
        enable_xfade = True
        print(f"   ⚠️ ENABLE_XFADE=true: Real crossfades enabled (slower, higher quality)")
    elif ENABLE_XFADE == "false":
        enable_xfade = False
        # No log needed, this is the expected default
    elif EDITING_INSTRUCTIONS is not None:
        # No ENV override - check Creative Director settings
        transitions = EDITING_INSTRUCTIONS.get('transitions', {})
        transition_type = transitions.get('type', 'energy_aware')
        xfade_duration = transitions.get('crossfade_duration_sec', xfade_duration)
        # Only enable xfade if explicitly requested, NOT automatically for all non-hard_cuts
        # This prevents unexpected re-encoding slowdowns
        if transition_type == 'crossfade':
            enable_xfade = True
            print(f"   ⚠️ Style requests crossfades: enabling xfade (slower render)")
    
    # Initialize Progressive Renderer for memory-efficient batch processing
    # Uses FFmpeg concat demuxer instead of MoviePy in-memory concatenation
    progressive_renderer = None
    if SEGMENT_WRITER_AVAILABLE:
        progressive_renderer = ProgressiveRenderer(
            batch_size=BATCH_SIZE,
            output_dir=os.path.join(TEMP_DIR, f"segments_{JOB_ID}"),
            memory_manager=memory_manager,
            job_id=JOB_ID,
            enable_xfade=enable_xfade,
            xfade_duration=xfade_duration,
            ffmpeg_crf=FINAL_CRF,
            normalize_clips=NORMALIZE_CLIPS
        )
        xfade_status = f"xfade={enable_xfade}" if enable_xfade else "concat"
        print(f"   ✅ Progressive Renderer initialized (batch={BATCH_SIZE}, {xfade_status}, crf={FINAL_CRF})")
    else:
        print(f"   ⚠️ Progressive Renderer not available - using legacy batching")
    
    # Initialize clips and batch_files for legacy path (used when progressive_renderer is None)
    clips = []
    batch_files = []
    
    # Log batch processing config
    if VERBOSE:
        print(f"   ⚙️ Memory management: BATCH_SIZE={BATCH_SIZE}, FORCE_GC={FORCE_GC}")
        if memory_manager:
            safe_batch = memory_manager.calculate_safe_batch_size(BATCH_SIZE)
            if safe_batch < BATCH_SIZE:
                print(f"   ⚠️ Adaptive batch size: {safe_batch} (reduced from {BATCH_SIZE} due to memory pressure)")
    
    # Start assembling phase monitoring
    monitor = get_monitor()
    if monitor:
        monitor.start_phase("assembling")
        monitor.log_assembling_start(
            total_clips=len(all_scenes),
            target_duration=target_duration,
            tempo=tempo
        )
    else:
        print("✂️  Assembling cuts...")
    
    last_used_path = None
    last_shot_type = None
    last_clip_end_time = None  # For match cut detection

    # ADVANCED 2024/2025 PACING: Position-aware patterns with Fibonacci & Cinematic Rhythms
    # Pattern pool for dynamic selection (fallback for non-dynamic modes)
    cut_patterns = [
        [4, 4, 4, 4],       # Steady pace (The March)
        [2, 2, 4, 8],       # Accelerate then hold (The Breath)
        [8, 4, 2, 2],       # Long shot into fast cuts (The Drop)
        [2, 2, 2, 2],       # Energetic steady (The Pulse)
        [1, 1, 2, 4],       # Rapid fire start (The Spark)
        [4, 2, 1, 1, 8],    # Complex phrase (The Jazz)
        [1, 1, 2, 3, 5],    # Fibonacci magic (The Golden Ratio)
        [6, 2, 4, 4],       # Syncopated (The Offbeat)
        [3, 3, 3, 3],       # Triplet feel (The Waltz)
        [12, 4],            # Establishing -> Detail (The Context)
    ]
    current_pattern = None
    pattern_idx = 0


    while current_time < target_duration and all_scenes:
        # Determine cut length based on beats and style

        # Get current energy
        # Find index in energy_times closest to current_time
        if len(energy_times) > 0:
            idx = (np.abs(energy_times - current_time)).argmin()
            current_energy = energy_values[idx]
        else:
            current_energy = 0.5

        # 🎬 CREATIVE DIRECTOR INTEGRATION: Use EDITING_INSTRUCTIONS if available
        if EDITING_INSTRUCTIONS is not None:
            pacing_speed = EDITING_INSTRUCTIONS.get('pacing', {}).get('speed', 'dynamic')

            # Map Creative Director pacing to cut behavior
            if pacing_speed == "very_fast":
                # MTV style: 1-beat cuts
                beats_per_cut = 1
            elif pacing_speed == "fast":
                # Energetic: 2 beats
                beats_per_cut = 2 if tempo < 130 else 4
            elif pacing_speed == "medium":
                # Moderate: 4 beats
                beats_per_cut = 4
            elif pacing_speed == "slow":
                # Gallery style: 8 beats
                beats_per_cut = 8
            elif pacing_speed == "very_slow":
                # Minimalist: 16 beats
                beats_per_cut = 16 if tempo < 100 else 8
            else:  # "dynamic" - ADVANCED 2024/2025 PACING
                # Get new pattern if needed
                if current_pattern is None or pattern_idx >= len(current_pattern):
                    current_pattern = calculate_dynamic_cut_length(
                        current_energy, tempo, current_time, target_duration, cut_patterns
                    )
                    pattern_idx = 0

                beats_per_cut = current_pattern[pattern_idx]
                pattern_idx += 1

        # Legacy CUT_STYLE fallback (if no Creative Director instructions)
        elif CUT_STYLE == "fast":
            # Energetic: Cut every 2 beats (half bar), or 4 if tempo is very fast
            beats_per_cut = 2
            if tempo > 130: beats_per_cut = 4

        elif CUT_STYLE == "hyper":
            # TikTok Style: Cut every beat or every 2 beats
            beats_per_cut = 1 if random.random() > 0.5 else 2

        elif CUT_STYLE == "slow":
            # Gallery/Museum Style: Long takes (8 or 16 beats)
            beats_per_cut = 8
            if tempo > 100: beats_per_cut = 16

        else: # "dynamic" (Default) - ADVANCED 2024/2025 PACING
            # Get new pattern if needed
            if current_pattern is None or pattern_idx >= len(current_pattern):
                current_pattern = calculate_dynamic_cut_length(
                    current_energy, tempo, current_time, target_duration, cut_patterns
                )
                pattern_idx = 0

            beats_per_cut = current_pattern[pattern_idx]
            pattern_idx += 1
        
        # Find the time of the next cut
        target_beat_idx = beat_idx + beats_per_cut
        
        # Helper for sub-beat interpolation (handles float indices)
        def get_beat_time_interp(idx):
            idx_int = int(idx)
            if idx_int >= len(beat_times) - 1:
                return beat_times[-1]
            frac = idx - idx_int
            # Ensure we don't go out of bounds
            if idx_int + 1 >= len(beat_times):
                return beat_times[-1]
            return beat_times[idx_int] + (beat_times[idx_int+1] - beat_times[idx_int]) * frac

        if target_beat_idx >= len(beat_times):
            cut_duration = target_duration - current_time
        else:
            # Use interpolation to handle fractional beats (e.g. 0.5, 1.5)
            t_start = get_beat_time_interp(beat_idx)
            t_end = get_beat_time_interp(target_beat_idx)
            cut_duration = t_end - t_start
            
            # 🎲 MICRO-TIMING JITTER (Humanization)
            # Add slight imperfection to cut timing (-0.05s to +0.05s)
            # This prevents the "robotic" feel of perfectly quantized cuts
            jitter = random.uniform(-0.05, 0.05)
            # Only apply if it doesn't make the cut too short
            if cut_duration + jitter > 0.5:
                cut_duration += jitter
            
        # Find a scene that fits using Footage Pool Manager
        # Get available clips that haven't been overused
        # Relax min_duration to 50% of cut_duration to handle fast tempos
        min_dur = cut_duration * 0.5
        available_footage = footage_pool.get_available_clips(min_duration=min_dur)

        # DEBUG: Show what's happening
        if VERBOSE and not available_footage:
            all_durations = [c.duration for c in footage_pool.clips.values()]
            print(f"   🐛 DEBUG: cut_duration={cut_duration:.2f}s, min_duration={min_dur:.2f}s")
            print(f"   🐛 DEBUG: beats_per_cut={beats_per_cut}, tempo={tempo:.1f} BPM")
            print(f"   🐛 DEBUG: Available footage durations: min={min(all_durations):.2f}s, max={max(all_durations):.2f}s, median={sorted(all_durations)[len(all_durations)//2]:.2f}s")
            print(f"   🐛 DEBUG: Total clips in pool: {len(footage_pool.clips)}, Used clips: {len(footage_pool.used_clips)}")

        if not available_footage:
            print("   ⚠️ No more footage available. Stopping.")
            break

        # Convert FootageClip objects back to scene dicts for compatibility
        # (using clip_id to match original scenes)
        valid_scenes = [s for s in all_scenes if id(s) in [c.clip_id for c in available_footage]]

        if not valid_scenes:
            print(f"   🐛 DEBUG: available_footage has {len(available_footage)} clips but valid_scenes is empty!")
            print(f"   🐛 DEBUG: First available clip_id: {available_footage[0].clip_id if available_footage else 'N/A'}")
            print(f"   🐛 DEBUG: First all_scenes id: {id(all_scenes[0]) if all_scenes else 'N/A'}")
            print(f"   🐛 DEBUG: Do they match? {id(all_scenes[0]) in [c.clip_id for c in available_footage] if all_scenes and available_footage else 'N/A'}")
            print("   ⚠️ No more footage long enough for this cut. Stopping.")
            break

        # INTELLIGENT SELECTION (AI Director):
        # Score candidates based on rules

        # Consider top candidates from footage pool
        candidates = valid_scenes[:20]  # Increased to allow better selection
        
        best_score = -1000
        selected_scene = candidates[0] # Default
        candidate_scores = {}  # Track all scores for intelligent selection

        for scene in candidates:
            score = 0

            # Rule 1: Story Arc Energy Matching (Professional Footage Management)
            # Get clip from footage pool to check consumption status
            scene_id = id(scene)
            footage_clip = next((c for c in available_footage if c.clip_id == scene_id), None)
            if footage_clip:
                # Bonus for fresh (unconsumed) clips
                if footage_clip.usage_count == 0:
                    score += 50  # Strong preference for unused footage
                elif footage_clip.usage_count == 1:
                    score += 20  # Secondary preference
                else:
                    score -= footage_clip.usage_count * 10  # Penalty for overuse

                # Story Arc phase matching (professional pacing)
                story_position = current_time / target_duration
                if 0 <= story_position < 0.15:  # INTRO phase
                    if current_energy < 0.4: score += 15
                elif 0.15 <= story_position < 0.40:  # BUILD phase
                    if 0.4 <= current_energy < 0.7: score += 15
                elif 0.40 <= story_position < 0.70:  # CLIMAX phase
                    if current_energy >= 0.7: score += 15
                elif 0.70 <= story_position < 0.90:  # SUSTAIN phase
                    if current_energy >= 0.6: score += 15
                else:  # OUTRO phase
                    if current_energy < 0.5: score += 15

            # Rule 2: Avoid jump cuts (same video file back-to-back)
            if scene['path'] == last_used_path:
                if unique_videos > 1:
                    score -= 50
                else:
                    # If we only have 1 source, we MUST jump cut. 
                    # Small penalty to prefer non-adjacent segments if possible.
                    score -= 5

            # Rule 3: AI Content Matching
            meta = scene.get('meta', {})
            action = meta.get('action', 'medium')
            shot = meta.get('shot', 'medium')

            # 🎬 CREATIVE DIRECTOR INTEGRATION: Cinematography preferences
            if EDITING_INSTRUCTIONS is not None:
                cinematography = EDITING_INSTRUCTIONS.get('cinematography', {})

                # Prefer high action scenes (if requested)
                if cinematography.get('prefer_high_action', False):
                    if action == 'high': score += 30
                    elif action == 'low': score -= 10
                # Prefer low action scenes (calm/meditative)
                elif cinematography.get('prefer_high_action', True) is False:
                    if action == 'low': score += 30
                    elif action == 'high': score -= 10
                else:
                    # Default: Energy-based matching
                    if current_energy > 0.6 and action == 'high': score += 20
                    if current_energy < 0.4 and action == 'low': score += 20

                # Prefer wide shots (if requested)
                if cinematography.get('prefer_wide_shots', False):
                    if shot == 'wide': score += 20
            else:
                # Legacy: High energy -> High action
                if current_energy > 0.6 and action == 'high': score += 20
                # Low energy -> Low action
                if current_energy < 0.4 and action == 'low': score += 20

            # Rule 4: Shot Variation (Cinematic Grammar)
            # 🎬 CREATIVE DIRECTOR INTEGRATION: Shot variation priority
            if EDITING_INSTRUCTIONS is not None:
                cinematography = EDITING_INSTRUCTIONS.get('cinematography', {})
                variation_priority = cinematography.get('shot_variation_priority', 'medium')

                # Avoid same shot type twice in a row (e.g. Wide -> Wide)
                if last_shot_type and shot == last_shot_type:
                    if variation_priority == 'high':
                        score -= 20  # Strong penalty
                    elif variation_priority == 'medium':
                        score -= 10  # Moderate penalty
                    # Low priority: minimal penalty (skip)
                else:
                    if variation_priority == 'high':
                        score += 15  # Strong bonus for variation
                    else:
                        score += 10  # Standard bonus
            else:
                # Legacy: Moderate variation
                if last_shot_type and shot == last_shot_type:
                    score -= 10
                else:
                    score += 10

            # Rule 5: MATCH CUT DETECTION (2024/2025 technique)
            # 🎬 CREATIVE DIRECTOR INTEGRATION: Match cuts enable/disable
            match_cuts_enabled = True  # Default
            if EDITING_INSTRUCTIONS is not None:
                cinematography = EDITING_INSTRUCTIONS.get('cinematography', {})
                match_cuts_enabled = cinematography.get('match_cuts_enabled', True)

            if match_cuts_enabled and last_clip_end_time is not None and last_used_path is not None:
                try:
                    similarity = calculate_visual_similarity(
                        last_used_path, last_clip_end_time,
                        scene['path'], scene['start']
                    )
                    # High similarity = great match cut opportunity
                    if similarity > 0.7:
                        score += 30  # Strong bonus for match cuts
                except:
                    pass  # Skip if analysis fails

            # Rule 6: INVISIBLE CUT (motion blur detection)
            # 🎬 CREATIVE DIRECTOR INTEGRATION: Invisible cuts enable/disable
            invisible_cuts_enabled = True  # Default
            if EDITING_INSTRUCTIONS is not None:
                cinematography = EDITING_INSTRUCTIONS.get('cinematography', {})
                invisible_cuts_enabled = cinematography.get('invisible_cuts_enabled', True)

            if invisible_cuts_enabled and last_clip_end_time is not None and last_used_path is not None:
                try:
                    blur_score = detect_motion_blur(last_used_path, last_clip_end_time)
                    # High blur = good moment for invisible cut
                    if blur_score > 0.6:
                        score += 15  # Bonus for cutting during motion
                except:
                    pass

            # Randomness factor to keep it fresh (Increased variance)
            score += random.randint(-15, 15)

            # Store score for intelligent selection
            scene['_heuristic_score'] = score

        # PROBABILISTIC SELECTION (Variance Injection)
        # Sort candidates by score descending
        candidates.sort(key=lambda x: x.get('_heuristic_score', -1000), reverse=True)
        
        # Take top 3 (or fewer)
        top_n = min(3, len(candidates))
        if top_n > 0:
            top_candidates = candidates[:top_n]
            # Weighted random selection: higher score = higher chance
            min_score = min(c.get('_heuristic_score', 0) for c in top_candidates)
            # Shift scores to be positive for weighting (min_score + 10 base)
            weights = [c.get('_heuristic_score', 0) - min_score + 10 for c in top_candidates]
            
            selected_scene = random.choices(top_candidates, weights=weights, k=1)[0]
            best_score = selected_scene.get('_heuristic_score', 0)
            
            if VERBOSE:
                print(f"   🎲 Probabilistic selection: Picked score {best_score} from top {top_n} scores {[c.get('_heuristic_score') for c in top_candidates]}")
        else:
            selected_scene = candidates[0] if candidates else None

        # 🧠 INTELLIGENT CLIP SELECTION: Use LLM reasoning if available
        selection_reasoning = "Heuristic selection"
        if intelligent_selector is not None and len(candidates) > 1:
            try:
                # Get top 3 candidates by heuristic score
                sorted_candidates = sorted(
                    candidates,
                    key=lambda s: s.get('_heuristic_score', -1000),
                    reverse=True
                )[:3]

                # Convert to ClipCandidate objects
                clip_candidates = []
                for scene in sorted_candidates:
                    clip_candidates.append(ClipCandidate(
                        path=scene['path'],
                        start_time=scene['start'],
                        duration=scene['duration'],
                        heuristic_score=scene.get('_heuristic_score', 0),
                        metadata=scene.get('meta', {})
                    ))

                # Build context for LLM
                story_position = current_time / target_duration
                position = "intro" if story_position < 0.15 else \
                          "build" if story_position < 0.40 else \
                          "climax" if story_position < 0.70 else \
                          "sustain" if story_position < 0.90 else "outro"

                # Get previous clips info (last 2)
                previous_clips_info = []
                if len(clips) > 0:
                    for clip in clips[-2:]:
                        previous_clips_info.append({
                            'meta': clip.get('meta', {}),
                            'duration': clip.get('duration', 0)
                        })

                context = {
                    'current_energy': current_energy,
                    'position': position,
                    'previous_clips': previous_clips_info,
                    'beat_position': cut_number % 4  # Which beat in the bar
                }

                # Query LLM for best clip
                best_candidate, reasoning = intelligent_selector.select_best_clip(
                    clip_candidates, context, top_n=3
                )

                # Override selection with LLM choice
                for scene in sorted_candidates:
                    if scene['path'] == best_candidate.path and scene['start'] == best_candidate.start_time:
                        selected_scene = scene
                        selection_reasoning = reasoning
                        break

                if monitor:
                    monitor.log_info("llm_clip_selection", reasoning, {
                        "clip_path": best_candidate.path,
                        "cut_number": cut_number,
                        "heuristic_score": best_candidate.heuristic_score
                    })

            except Exception as e:
                print(f"   ⚠️ LLM clip selection failed: {e}")
                selection_reasoning = f"Heuristic fallback (error: {str(e)[:50]})"

        # Increment cut counter
        cut_number += 1

        # === MONITORING: Log cut decision ===
        if monitor:
            # Determine selection reason based on score components
            selection_reasons = []
            if selected_scene.get('meta', {}).get('action') == 'high' and current_energy > 0.6:
                selection_reasons.append("energy_match")
            if selected_scene['path'] != last_used_path:
                selection_reasons.append("variety")
            # Check footage pool for freshness
            selected_id = id(selected_scene)
            selected_footage = next((c for c in available_footage if c.clip_id == selected_id), None)
            if selected_footage and selected_footage.usage_count == 0:
                selection_reasons.append("fresh_clip")
            reason_str = ", ".join(selection_reasons) if selection_reasons else "best_available"
            
            monitor.log_cut_placed(
                cut_num=cut_number,
                total_cuts=estimated_total_cuts,
                clip_name=selected_scene['path'],
                start=clip_start if 'clip_start' in dir() else selected_scene['start'],
                duration=cut_duration,
                beat_idx=beat_idx,
                beats_per_cut=beats_per_cut,
                energy=current_energy,
                score=best_score,
                reason=reason_str
            )
        
        # Update state
        last_used_path = selected_scene['path']
        last_shot_type = selected_scene.get('meta', {}).get('shot', 'medium')

        # OPTIMIZATION: Use Optical Flow to find best start point (Perfect Cut)
        # Falls back to random if analysis fails
        clip_start = find_best_start_point(
            selected_scene['path'],
            selected_scene['start'],
            selected_scene['end'],
            cut_duration
        )

        # Track clip end time for match cut / invisible cut detection
        clip_end = clip_start + cut_duration
        last_clip_end_time = clip_end

        # 📚 Mark clip as consumed in Footage Pool Manager
        footage_pool.consume_clip(
            clip_id=id(selected_scene),
            timeline_position=current_time,
            used_in_point=clip_start,
            used_out_point=clip_end
        )
        
        # Track enhancements applied for monitoring
        enhancements_applied = []
        enhancement_start_time = time.time()
        
        # Track which enhancements were actually applied (for metadata)
        stabilize_applied = False
        upscale_applied = False
        enhance_applied = False
            
        # Create Clip
        # If stabilization is requested, we need to process the subclip first
        # But stabilizing a 2-second clip is fast.
        
        if STABILIZE:
            enhancements_applied.append("stabilize")
            stabilize_applied = True
            # Extract subclip to temp file first
            temp_clip_name = f"temp_clip_{beat_idx}_{random.randint(0,9999)}.mp4"
            temp_clip_path = os.path.join(TEMP_DIR, temp_clip_name)
            temp_stab_path = os.path.join(TEMP_DIR, f"stab_{temp_clip_name}")
            
            # Use ffmpeg to extract exactly the part we need (faster than loading full video)
            # ffmpeg -ss START -i INPUT -t DURATION -c copy TEMP
            extract_subclip_ffmpeg(selected_scene['path'], clip_start, cut_duration, temp_clip_path)
            
            # Stabilize
            if STABILIZE:
                temp_stab_path = os.path.join(TEMP_DIR, f"stab_{temp_clip_name}")
                final_clip_path = stabilize_clip(temp_clip_path, temp_stab_path)
                temp_clip_path = final_clip_path # Chain it
            
            # AI Upscale (Optional - Slow!)
            if UPSCALE:
                enhancements_applied.append("upscale")
                upscale_applied = True
                temp_upscale_path = os.path.join(TEMP_DIR, f"upscale_{temp_clip_name}")
                final_clip_path = upscale_clip(temp_clip_path, temp_upscale_path)
                temp_clip_path = final_clip_path

            # Enhance (Color/Sharpness) - Do this LAST for best quality on upscaled result
            if ENHANCE:
                enhancements_applied.append("enhance")
                enhance_applied = True
                temp_enhance_path = os.path.join(TEMP_DIR, f"enhance_{temp_clip_name}")
                final_clip_path = enhance_clip(temp_clip_path, temp_enhance_path)
                temp_clip_path = final_clip_path
            
            # Load into MoviePy
            v_clip = VideoFileClip(temp_clip_path)

            # === MEMORY MANAGEMENT: Track temp files for cleanup ===
            # Store temp file path for later cleanup
            if not hasattr(v_clip, '_temp_files'):
                v_clip._temp_files = []
            v_clip._temp_files.append(temp_clip_path)
            if 'temp_stab_path' in locals() and os.path.exists(temp_stab_path):
                v_clip._temp_files.append(temp_stab_path)
            if 'temp_upscale_path' in locals() and os.path.exists(temp_upscale_path):
                v_clip._temp_files.append(temp_upscale_path)
            if 'temp_enhance_path' in locals() and os.path.exists(temp_enhance_path):
                v_clip._temp_files.append(temp_enhance_path)
        else:
            # Even if not stabilizing, we might want to enhance/upscale
            if ENHANCE or UPSCALE:
                # Extract subclip first
                temp_clip_name = f"temp_clip_{beat_idx}_{random.randint(0,9999)}.mp4"
                temp_clip_path = os.path.join(TEMP_DIR, temp_clip_name)
                
                extract_subclip_ffmpeg(selected_scene['path'], clip_start, cut_duration, temp_clip_path)
                
                if UPSCALE:
                    enhancements_applied.append("upscale")
                    upscale_applied = True
                    temp_upscale_path = os.path.join(TEMP_DIR, f"upscale_{temp_clip_name}")
                    temp_clip_path = upscale_clip(temp_clip_path, temp_upscale_path)

                if ENHANCE:
                    enhancements_applied.append("enhance")
                    enhance_applied = True
                    temp_enhance_path = os.path.join(TEMP_DIR, f"enhance_{temp_clip_name}")
                    temp_clip_path = enhance_clip(temp_clip_path, temp_enhance_path)
                    
                v_clip = VideoFileClip(temp_clip_path)
                # Track temp files for cleanup
                if not hasattr(v_clip, '_temp_files'):
                    v_clip._temp_files = []
                v_clip._temp_files.append(temp_clip_path)
                if 'temp_upscale_path' in locals() and os.path.exists(temp_upscale_path):
                    v_clip._temp_files.append(temp_upscale_path)
                if 'temp_enhance_path' in locals() and os.path.exists(temp_enhance_path):
                    v_clip._temp_files.append(temp_enhance_path)
            else:
                v_clip = subclip_compat(VideoFileClip(selected_scene['path']), clip_start, clip_start + cut_duration)
        
        # Handle video rotation metadata
        # MoviePy's automatic rotation handling is inconsistent, so we check and apply manually
        rotation = get_video_rotation(selected_scene['path'])
        if rotation != 0:
            if VERBOSE:
                print(f"  🔄 Rotation metadata: {rotation}°")
            # Apply rotation correction using compat wrapper
            if rotation == 90 or rotation == -270:
                v_clip = rotate(v_clip, -90, expand=True)
            elif rotation == -90 or rotation == 270:
                v_clip = rotate(v_clip, 90, expand=True)
            elif rotation == 180 or rotation == -180:
                v_clip = rotate(v_clip, 180, expand=False)
        
        # Get video dimensions (after rotation correction)
        w, h = v_clip.size
        
        # Resize/crop to target dimensions using DRY helper
        # PRESERVE_ASPECT: letterbox horizontal clips instead of cropping
        target_w, target_h = STANDARD_WIDTH, STANDARD_HEIGHT
        v_clip = enforce_dimensions(
            v_clip, target_w, target_h, 
            verbose=True, 
            preserve_aspect=PRESERVE_ASPECT
        )

        # 🎬 CREATIVE DIRECTOR INTEGRATION: Transitions control
        # Progressive path with xfade: Real crossfades handled by xfade filter in SegmentWriter
        # Progressive path without xfade: No per-clip fades (hard cuts)
        # Legacy path: Apply per-clip fade-in/out (limited to fade-to-black)
        apply_per_clip_fade = False
        crossfade_duration = xfade_duration  # Use global setting
        
        # CRITICAL: Per-clip fades create fade-to-black artifacts when combined with xfade
        # Only apply per-clip fades if:
        # 1. Using legacy path (no progressive_renderer), AND
        # 2. xfade is NOT enabled (otherwise xfade handles transitions)
        if progressive_renderer and enable_xfade:
            # xfade enabled: SegmentWriter handles real crossfades, no per-clip fades
            apply_per_clip_fade = False
        elif not progressive_renderer:
            # Legacy path: use per-clip fades based on style
            if EDITING_INSTRUCTIONS is not None:
                transitions = EDITING_INSTRUCTIONS.get('transitions', {})
                transition_type = transitions.get('type', 'energy_aware')
                crossfade_duration = transitions.get('crossfade_duration_sec', crossfade_duration)
                
                if transition_type != 'hard_cuts':
                    if transition_type == "crossfade":
                        apply_per_clip_fade = True
                    elif transition_type == "mixed":
                        apply_per_clip_fade = random.random() > 0.5
                    elif transition_type == "energy_aware":
                        apply_per_clip_fade = current_energy < 0.3
            else:
                # Legacy default: Energy-aware crossfade
                if current_energy < 0.3:
                    apply_per_clip_fade = True
        
        # Apply per-clip fades only when appropriate (legacy path, no xfade)
        if apply_per_clip_fade:
            fade_duration = min(crossfade_duration, cut_duration * 0.3)
            if cut_number > 0:
                v_clip = crossfadein(v_clip, fade_duration)
                if VERBOSE:
                    print(f"   ↗️ Crossfade-in: {fade_duration:.2f}s")
            v_clip = crossfadeout(v_clip, fade_duration)
            if VERBOSE:
                print(f"   ↘️ Crossfade-out: {fade_duration:.2f}s")

        # === PROGRESSIVE RENDERING: Write clip to disk immediately ===
        # This is memory-efficient: each clip is rendered and freed before next
        if progressive_renderer:
            # Generate temp file path for this clip
            clip_temp_path = os.path.join(TEMP_DIR, f"clip_{JOB_ID}_{cut_number:04d}.mp4")
            
            # Render clip to temp file with normalized parameters for concat compatibility
            try:
                v_clip.write_videofile(
                    clip_temp_path,
                    codec=OUTPUT_CODEC,
                    audio=False,  # Audio added in final composition
                    fps=STANDARD_FPS,
                    preset='fast',
                    ffmpeg_params=build_video_ffmpeg_params(crf=FINAL_CRF),
                    logger=None
                )
                
                # Close MoviePy clip to free memory
                if hasattr(v_clip, '_temp_files'):
                    for tf in v_clip._temp_files:
                        try:
                            if os.path.exists(tf):
                                os.remove(tf)
                        except:
                            pass
                try:
                    v_clip.close()
                except:
                    pass
                
                # Add to progressive renderer (handles batching automatically)
                segment_info = progressive_renderer.add_clip_path(clip_temp_path)
                
                if segment_info:
                    print(f"   ✅ Segment {segment_info.index} written ({segment_info.clip_count} clips)")
                    
            except Exception as e:
                print(f"   ❌ Failed to render clip {cut_number}: {e}")
                try:
                    v_clip.close()
                except:
                    pass
        else:
            # Legacy fallback: keep clips in memory (will batch later)
            clips.append(v_clip)
            
            # Legacy batch processing (only used if ProgressiveRenderer unavailable)
            if len(clips) >= BATCH_SIZE:
                batch_num = len(batch_files) + 1
                batch_file = os.path.join(TEMP_DIR, f"batch_{JOB_ID}_{batch_num:03d}.mp4")
                
                print(f"   💾 Legacy: Rendering batch {batch_num} ({len(clips)} clips)...")
                
                batch_video = concatenate_videoclips(clips, method="compose")
                batch_video.write_videofile(
                    batch_file,
                    codec=OUTPUT_CODEC,
                    audio=False,
                    fps=STANDARD_FPS,
                    preset='fast',
                    ffmpeg_params=build_video_ffmpeg_params(crf=FINAL_CRF),
                    threads=int(FFMPEG_THREADS) if FFMPEG_THREADS != "0" else None,
                    logger=None
                )
                batch_files.append(batch_file)
                
                for clip in clips:
                    try:
                        clip.close()
                    except:
                        pass
                batch_video.close()
                clips = []
                
                if FORCE_GC:
                    gc.collect()

        # === COLLECT METADATA FOR TIMELINE EXPORT ===
        # Store clip metadata for OTIO/EDL/CSV export
        clip_metadata = {
            'source_path': selected_scene['path'],
            'start_time': clip_start,
            'duration': cut_duration,
            'timeline_start': current_time,
            'metadata': {
                'energy': current_energy,
                'action': selected_scene.get('meta', {}).get('action', 'medium'),
                'shot': selected_scene.get('meta', {}).get('shot', 'medium'),
                'beat_idx': beat_idx,
                'beats_per_cut': beats_per_cut,
                'selection_score': best_score if 'best_score' in dir() else 0,
                'enhancements': {
                    'stabilized': stabilize_applied,
                    'upscaled': upscale_applied,
                    'enhanced': enhance_applied
                }
            }
        }
        clips_metadata.append(clip_metadata)

        current_time += cut_duration
        beat_idx += beats_per_cut

        # === MONITORING: Log progress every 5 cuts ===
        if monitor and cut_number % 5 == 0:
            monitor.log_cut_summary(cut_number, estimated_total_cuts, current_time)
    
    # End assembling phase
    if monitor:
        total_clips_processed = progressive_renderer.get_total_clips() if progressive_renderer else len(clips) + (len(batch_files) * BATCH_SIZE)
        monitor.end_phase({
            "total_cuts": cut_number,
            "timeline_duration": f"{current_time:.1f}s",
            "clips_in_sequence": total_clips_processed
        })
        
    # 4. Final Composition
    if monitor:
        monitor.start_phase("composition")
    
    # 🎬 CREATIVE DIRECTOR INTEGRATION: Use style name in filename
    if EDITING_INSTRUCTIONS is not None:
        style_name = EDITING_INSTRUCTIONS.get('style', {}).get('name', CUT_STYLE)
    else:
        style_name = CUT_STYLE

    output_filename = os.path.join(OUTPUT_DIR, f"gallery_montage_{JOB_ID}_v{variant_id}_{style_name}.mp4")
    
    # Check for logo files (for branding overlay)
    logo_files = get_files(ASSETS_DIR, ('.png', '.jpg'))
    logo_path = logo_files[0] if logo_files else None
    
    # === PROGRESSIVE RENDERING: Final composition using FFmpeg ===
    if progressive_renderer:
        print(f"\n   🔗 Finalizing with Progressive Renderer ({progressive_renderer.get_segment_count()} segments)...")
        
        # Start timing BEFORE finalize (this is the actual render time)
        render_start_time = time.time()
        
        if monitor:
            method = "ffmpeg_xfade" if enable_xfade else "ffmpeg_concat_copy"
            monitor.log_render_start(output_filename, {
                "codec": OUTPUT_CODEC,
                "method": method,
                "segments": progressive_renderer.get_segment_count(),
                "total_clips": progressive_renderer.get_total_clips(),
                "xfade_enabled": enable_xfade,
                "crf": FINAL_CRF,
                "logo": os.path.basename(logo_path) if logo_path else None
            })
        
        # Finalize: flush remaining clips and concatenate all segments with audio + logo
        # Pass audio_duration for proper trimming when target duration is set
        audio_duration = target_duration if TARGET_DURATION > 0 else current_time
        success = progressive_renderer.finalize(
            output_path=output_filename,
            audio_path=music_path,
            audio_duration=audio_duration,  # Trim audio to match video
            logo_path=logo_path  # Logo overlay (if available)
        )
        
        # Calculate render duration consistently with legacy path
        render_duration = time.time() - render_start_time
        
        if success:
            method_str = "xfade" if enable_xfade else "-c copy"
            print(f"   ✅ Final video rendered via FFmpeg ({method_str}) in {render_duration:.1f}s")
            if logo_path:
                print(f"   🏷️ Logo overlay: {os.path.basename(logo_path)}")
            
            # Get stats for logging
            stats = progressive_renderer.get_stats()
            if monitor:
                monitor.log_info("composition", f"Progressive render complete", stats)
        else:
            print(f"   ❌ Progressive render failed - attempting legacy fallback")
            progressive_renderer = None  # Fall through to legacy path
    
    # === LEGACY FALLBACK: MoviePy-based composition ===
    if not progressive_renderer:
        # Initialize clips list if not already (for legacy path)
        if 'clips' not in dir() or clips is None:
            clips = []
            
        if batch_files:
            print(f"   🔗 Legacy: Combining {len(batch_files)} batch files + {len(clips)} remaining clips...")
            
            # Render any remaining clips to a final batch
            if clips:
                final_batch_file = os.path.join(TEMP_DIR, f"batch_{JOB_ID}_final.mp4")
                final_batch = concatenate_videoclips(clips, method="compose")
                final_batch.write_videofile(
                    final_batch_file,
                    codec=OUTPUT_CODEC,
                    audio=False,
                    fps=STANDARD_FPS,
                    preset='fast',
                    ffmpeg_params=build_video_ffmpeg_params(crf=FINAL_CRF),
                    threads=int(FFMPEG_THREADS) if FFMPEG_THREADS != "0" else None,
                    logger=None
                )
                batch_files.append(final_batch_file)
                
                # Free memory
                for clip in clips:
                    try:
                        clip.close()
                    except:
                        pass
                final_batch.close()
                clips = []
                if FORCE_GC:
                    gc.collect()
            
            # Load all batch files and concatenate (legacy - memory intensive!)
            batch_clips = [VideoFileClip(bf) for bf in batch_files]
            final_video = concatenate_videoclips(batch_clips, method="compose")
            
        else:
            # No batching needed (small project) - use clips directly
            if clips:
                final_video = concatenate_videoclips(clips, method="compose")
            else:
                print("   ❌ No clips to compose!")
                return None
        
        final_video = set_audio(final_video, subclip(audio_clip, 0, current_time))
        
        # Render legacy way
        print(f"   🎬 Rendering (legacy MoviePy)...")
        final_video.write_videofile(
            output_filename,
            codec=OUTPUT_CODEC,
            audio_codec='aac',
            fps=STANDARD_FPS,
            preset=FFMPEG_PRESET,
            ffmpeg_params=build_video_ffmpeg_params() + ["-crf", str(FINAL_CRF)],
            threads=int(FFMPEG_THREADS) if FFMPEG_THREADS != "0" else None,
            logger=None
        )
        
        # Cleanup legacy
        try:
            final_video.close()
            for bf in batch_files:
                if os.path.exists(bf):
                    os.remove(bf)
        except:
            pass
    
    # 5. Legacy Logo Overlay (only needed if progressive didn't handle it)
    # Progressive renderer handles logo in finalize(), legacy needs separate pass
    if not progressive_renderer and logo_path and 'final_video' in dir():
        try:
            logo = set_position(resize(set_duration(ImageClip(logo_path), final_video.duration), height=150), ("right", "top"))
            final_video = CompositeVideoClip([final_video, logo])
            if monitor:
                monitor.log_info("composition", f"Logo overlay added: {os.path.basename(logo_path)}")
        except Exception as e:
            print(f"   ⚠️ Logo overlay failed: {e}")
    
    if monitor:
        if progressive_renderer:
            stats = progressive_renderer.get_stats()
            monitor.end_phase({"final_duration": f"{stats.get('total_duration', current_time):.1f}s", "method": "progressive_ffmpeg_copy"})
        elif 'final_video' in dir() and final_video:
            monitor.end_phase({"final_duration": f"{final_video.duration:.1f}s", "method": "legacy_moviepy"})
        else:
            monitor.end_phase({"method": "unknown"})
    
    # 6. Final Render (only needed for legacy path - progressive already rendered)
    if progressive_renderer:
        # Progressive render already completed in finalize() above
        # Just cleanup temp files
        progressive_renderer.cleanup()
        
    else:
        # Legacy MoviePy render path
        # === MONITORING: Render phase ===
        if monitor:
            monitor.log_render_start(output_filename, {
                "codec": OUTPUT_CODEC,
                "preset": FFMPEG_PRESET,
                "crf": FINAL_CRF,
                "fps": STANDARD_FPS,
                "duration": f"{final_video.duration:.1f}s" if 'final_video' in dir() else "unknown"
            })
        else:
            print(f"🚀 Rendering Variant #{variant_id} to {output_filename}...")
            print(f"   Job-ID: {JOB_ID}")
        
        render_start_time = time.time()
        
        # Multi-threaded rendering with quality settings (LEGACY PATH ONLY)
        # Using pixel format yuv420p for maximum compatibility
        final_video.write_videofile(
            output_filename,
            codec=OUTPUT_CODEC,
            audio_codec='aac',
            fps=STANDARD_FPS,
            preset=FFMPEG_PRESET,
            threads=int(FFMPEG_THREADS) if FFMPEG_THREADS != "0" else None,  # None = auto
            ffmpeg_params=build_video_ffmpeg_params() + [
                "-crf", str(FINAL_CRF),  # Quality from env var
                "-tune", "film",  # Optimize for film content
                "-movflags", "+faststart"  # Web-optimized
            ],
            logger=None  # Suppress tqdm progress bars
        )
        
        render_duration = time.time() - render_start_time
    
    # === MONITORING: Render complete ===
    # Use correct render_duration based on path
    if progressive_renderer:
        render_duration_ms = render_duration * 1000  # Already set from finalize
    else:
        render_duration_ms = render_duration * 1000 if 'render_duration' in dir() else 0
    
    if monitor:
        # Get file size
        try:
            file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
        except:
            file_size_mb = 0
        monitor.log_render_complete(output_filename, file_size_mb, render_duration_ms)
    else:
        print(f"✅ Variant #{variant_id} Done!")

    # === CLEANUP: Free memory and delete temp files ===
    print(f"\n🧹 Cleaning up resources...")
    cleanup_count = 0
    cleanup_size_mb = 0

    # Clean up remaining clips (if any weren't batched) - LEGACY PATH
    if 'clips' in dir() and clips:
        for clip in clips:
            # Delete temp files associated with this clip
            if hasattr(clip, '_temp_files'):
                for temp_file in clip._temp_files:
                    try:
                        if os.path.exists(temp_file):
                            size = os.path.getsize(temp_file) / (1024 * 1024)
                            os.remove(temp_file)
                            cleanup_count += 1
                            cleanup_size_mb += size
                    except Exception as e:
                        if monitor:
                            monitor.log_warning("cleanup", f"Failed to delete {temp_file}: {e}")

            # Close clip to free memory
            try:
                clip.close()
            except Exception as e:
                if monitor:
                    monitor.log_warning("cleanup", f"Failed to close clip: {e}")

    # Clean up batch files (if batch processing was used) - LEGACY PATH
    if 'batch_files' in dir() and batch_files:
        for bf in batch_files:
            try:
                if os.path.exists(bf):
                    size = os.path.getsize(bf) / (1024 * 1024)
                    os.remove(bf)
                    cleanup_count += 1
                    cleanup_size_mb += size
            except Exception as e:
                if monitor:
                    monitor.log_warning("cleanup", f"Failed to delete batch file {bf}: {e}")
        
        # Close batch clips if they exist
        if 'batch_clips' in dir():
            for bc in batch_clips:
                try:
                    bc.close()
                except:
                    pass

    # Close audio clip
    try:
        audio_clip.close()
    except:
        pass

    # Close final video (LEGACY PATH)
    if 'final_video' in dir() and final_video:
        try:
            final_video.close()
        except:
            pass

    # Force garbage collection
    if FORCE_GC:
        gc.collect()

    print(f"   ✅ Deleted {cleanup_count} temp files ({cleanup_size_mb:.1f} MB freed)")
    if monitor:
        monitor.log_success("cleanup", f"Resources freed: {cleanup_count} files, {cleanup_size_mb:.1f}MB")
        monitor.log_resources()  # Log final memory state

    # 7. Timeline Export (if enabled)
    if EXPORT_TIMELINE and TIMELINE_EXPORT_AVAILABLE:
        print(f"\n📽️ Exporting Timeline for NLE import...")

        try:
            # Determine style name for project naming
            style_name = "dynamic"
            if EDITING_INSTRUCTIONS is not None:
                style_name = EDITING_INSTRUCTIONS.get('style_name', 'custom')

            # Get final video duration from appropriate source
            # Progressive path: use stats, Legacy path: use final_video.duration
            if progressive_renderer:
                final_video_duration = progressive_renderer.get_stats().get('total_duration', current_time)
            elif 'final_video' in dir() and final_video:
                final_video_duration = final_video.duration
            else:
                # Fallback to tracked timeline position
                final_video_duration = current_time

            # Export to OTIO/EDL/CSV for professional NLE software
            exported_files = export_timeline_from_montage(
                clips_metadata,
                music_path,
                final_video_duration,
                output_dir=OUTPUT_DIR,
                project_name=f"montage_{JOB_ID}_v{variant_id}_{style_name}",
                generate_proxies=GENERATE_PROXIES,
                resolution=(STANDARD_WIDTH, STANDARD_HEIGHT),
                fps=STANDARD_FPS
            )

            print(f"   ✅ Timeline exported successfully!")
            print(f"   📁 Files: {', '.join(exported_files.keys())}")

            if monitor:
                monitor.log_info("timeline_export", "Timeline exported", {
                    "formats": list(exported_files.keys()),
                    "project_name": f"montage_{JOB_ID}_v{variant_id}_{style_name}"
                })

        except Exception as exc:
            print(f"   ⚠️ Timeline export failed: {exc}")
            if monitor:
                monitor.log_warning("timeline_export", f"Export failed: {exc}")

# Global flag to cache vidstab availability (checked once per process)
_VIDSTAB_AVAILABLE = None

def _check_vidstab_available():
    """Check if FFmpeg was compiled with libvidstab support."""
    global _VIDSTAB_AVAILABLE
    if _VIDSTAB_AVAILABLE is not None:
        return _VIDSTAB_AVAILABLE
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, timeout=5
        )
        _VIDSTAB_AVAILABLE = "vidstabdetect" in result.stdout
        if _VIDSTAB_AVAILABLE:
            print("   ✅ vidstab (libvidstab) available - using 2-pass stabilization")
        else:
            print("   ⚠️ vidstab not available - falling back to deshake filter")
    except Exception:
        _VIDSTAB_AVAILABLE = False
    
    return _VIDSTAB_AVAILABLE


def stabilize_clip(input_path, output_path):
    """
    Stabilize a video clip using the best available method.

    Priority:
    1. cgpu Cloud GPU (if CGPU_GPU_ENABLED and available)
    2. Local vidstab 2-pass (professional quality)
    3. Local deshake (basic fallback)

    vidstab (libvidstab) provides superior stabilization:
    - Pass 1: Analyzes motion vectors and stores transform data
    - Pass 2: Applies smooth transformations with configurable smoothing

    Parameters tuned for handheld/action footage:
    - shakiness=5: Medium shake detection (1-10, higher = more sensitive)
    - accuracy=15: High accuracy motion detection (1-15)
    - smoothing=30: ~1 second smoothing window (frames)
    - crop=black: Fill borders with black (vs. keep=zoom which loses resolution)
    """
    print(f"   ⚖️ Stabilizing {os.path.basename(input_path)}...")

    # Try cloud GPU stabilization first if enabled
    if CGPU_GPU_ENABLED and CGPU_STABILIZE_AVAILABLE and is_cgpu_available():
        print(f"   ☁️ Attempting cgpu cloud GPU stabilization...")
        result = cgpu_stabilize_video(input_path, output_path)
        if result:
            print(f"   ✅ Cloud stabilization complete")
            return result
        print(f"   ⚠️ Cloud stabilization failed, falling back to local...")

    # Local stabilization
    if _check_vidstab_available():
        return _stabilize_vidstab(input_path, output_path)
    else:
        return _stabilize_deshake(input_path, output_path)


def _stabilize_vidstab(input_path, output_path):
    """
    Professional 2-pass video stabilization using vidstab.
    
    This is the same algorithm used in professional tools like:
    - DaVinci Resolve (stabilizer)
    - Adobe Premiere (Warp Stabilizer basic mode)
    - Kdenlive, Shotcut
    """
    # Transform data file (motion vectors)
    transform_file = f"{output_path}.trf"
    
    try:
        # PASS 1: Motion Analysis
        # Detect camera motion and save transform vectors
        print(f"      Pass 1/2: Analyzing motion...")
        cmd_detect = [
            "ffmpeg", "-y",
            "-threads", FFMPEG_THREADS,
            "-i", input_path,
            "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={transform_file}",
            "-f", "null", "-"
        ]
        
        result = subprocess.run(
            cmd_detect,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout for analysis
        )
        
        if result.returncode != 0 or not os.path.exists(transform_file):
            print(f"      ⚠️ Motion analysis failed, falling back to deshake")
            return _stabilize_deshake(input_path, output_path)
        
        # PASS 2: Apply Stabilization
        # Transform video using analyzed motion vectors
        print(f"      Pass 2/2: Applying stabilization...")
        cmd_transform = [
            "ffmpeg", "-y",
            "-threads", FFMPEG_THREADS,
            "-i", input_path,
            "-vf", f"vidstabtransform=input={transform_file}:smoothing=30:crop=black:zoom=0:interpol=bicubic",
        ]
        
        cmd_transform.extend(["-c:v", OUTPUT_CODEC, "-preset", "fast", "-crf", "18"])
        if OUTPUT_PROFILE:
            cmd_transform.extend(["-profile:v", OUTPUT_PROFILE])
        if OUTPUT_LEVEL:
            cmd_transform.extend(["-level", OUTPUT_LEVEL])
        cmd_transform.extend(["-pix_fmt", OUTPUT_PIX_FMT, "-c:a", "copy", output_path])
        
        subprocess.run(cmd_transform, check=True, capture_output=True, timeout=300)
        
        # Cleanup transform file
        if os.path.exists(transform_file):
            os.remove(transform_file)
        
        print(f"      ✅ Stabilization complete (vidstab 2-pass)")
        return output_path
        
    except subprocess.TimeoutExpired:
        print(f"      ⚠️ Stabilization timed out, using original")
        return input_path
    except subprocess.CalledProcessError as e:
        print(f"      ⚠️ vidstab failed: {e.stderr[:200] if e.stderr else 'unknown error'}")
        return _stabilize_deshake(input_path, output_path)
    except Exception as e:
        print(f"      ⚠️ Stabilization error: {e}")
        return input_path
    finally:
        # Always cleanup transform file
        if os.path.exists(transform_file):
            try:
                os.remove(transform_file)
            except:
                pass


def _stabilize_deshake(input_path, output_path):
    """
    Fallback stabilization using FFmpeg's built-in deshake filter.
    
    Less effective than vidstab but always available.
    Good for minor camera shake, not recommended for heavy motion.
    """
    cmd = [
        "ffmpeg", "-y",
        "-threads", FFMPEG_THREADS,
        "-i", input_path,
        "-vf", "deshake=rx=32:ry=32:blocksize=8:contrast=125",  # Tuned parameters
    ]

    cmd.extend(["-c:v", OUTPUT_CODEC, "-preset", "fast", "-crf", "20"])
    if OUTPUT_PROFILE:
        cmd.extend(["-profile:v", OUTPUT_PROFILE])
    if OUTPUT_LEVEL:
        cmd.extend(["-level", OUTPUT_LEVEL])
    cmd.extend(["-pix_fmt", OUTPUT_PIX_FMT, "-c:a", "copy", output_path])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        print(f"      ✅ Stabilization complete (deshake fallback)")
        return output_path
    except subprocess.CalledProcessError:
        print("      ⚠️ Stabilization failed (ffmpeg error). Using original.")
        return input_path
    except Exception as e:
        print(f"      ⚠️ Stabilization failed: {e}")
        return input_path

def _analyze_clip_brightness(input_path):
    """
    Analyze clip brightness/exposure using FFmpeg signalstats.
    
    Returns dict with:
    - avg_brightness: 0-255 (16=black, 235=white in video range)
    - is_dark: True if clip appears underexposed
    - is_bright: True if clip appears overexposed
    - suggested_brightness: adjustment value for eq filter
    """
    try:
        # Sample 3 frames from the clip for quick analysis
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_pts_time",
            "-of", "json",
            input_path
        ]
        # Get duration for sampling points
        dur_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ]
        dur_result = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=5)
        duration = float(dur_result.stdout.strip()) if dur_result.stdout.strip() else 2.0
        
        # Analyze mean brightness at 3 points using signalstats
        sample_points = [duration * 0.25, duration * 0.5, duration * 0.75]
        brightnesses = []
        
        for t in sample_points:
            analyze_cmd = [
                "ffmpeg", "-ss", str(t), "-i", input_path,
                "-vframes", "1",
                "-vf", "signalstats=stat=tout+vrep+brng,metadata=print:file=-",
                "-f", "null", "-"
            ]
            result = subprocess.run(analyze_cmd, capture_output=True, text=True, timeout=10)
            
            # Parse YAVG (luma average) from output
            for line in result.stderr.split('\n'):
                if 'YAVG' in line:
                    try:
                        yavg = float(line.split('=')[-1].strip())
                        brightnesses.append(yavg)
                    except:
                        pass
        
        if not brightnesses:
            return {"avg_brightness": 128, "is_dark": False, "is_bright": False, "suggested_brightness": 0}
        
        avg = sum(brightnesses) / len(brightnesses)
        
        # Video range: 16-235 (not 0-255)
        # Target: ~120-130 for well-exposed footage
        is_dark = avg < 70
        is_bright = avg > 180
        
        # Calculate suggested adjustment
        if is_dark:
            suggested = min(0.15, (100 - avg) / 500)  # Max +0.15
        elif is_bright:
            suggested = max(-0.10, (120 - avg) / 500)  # Max -0.10
        else:
            suggested = 0
        
        return {
            "avg_brightness": avg,
            "is_dark": is_dark,
            "is_bright": is_bright,
            "suggested_brightness": suggested
        }
        
    except Exception as e:
        # On any error, return neutral values
        return {"avg_brightness": 128, "is_dark": False, "is_bright": False, "suggested_brightness": 0}


def enhance_clip(input_path, output_path):
    """
    Apply CINEMATIC enhancements using ffmpeg filters with Content-Aware adjustments.
    
    Enhancement pipeline:
    1. Analyze clip brightness (fast FFmpeg signalstats)
    2. Adjust parameters based on content:
       - Dark clips: Boost brightness, lift shadows
       - Bright clips: Reduce brightness, protect highlights
       - Normal clips: Standard cinematic grade
    3. Apply filter chain:
       - Teal & Orange color grading (Hollywood look)
       - S-curve contrast for filmic depth
       - CAS (Contrast Adaptive Sharpening)
       - Content-aware exposure correction
       - Subtle vignette
    
    This function is thread-safe and can be called from ThreadPoolExecutor.
    """
    # Analyze clip content for adaptive enhancement
    analysis = _analyze_clip_brightness(input_path)
    
    # Adaptive parameters based on content
    brightness_adj = analysis["suggested_brightness"]
    
    # Adjust saturation: less for dark clips (avoids noise), more for normal
    if analysis["is_dark"]:
        saturation = 1.08  # Subtle - dark clips show noise with high saturation
        contrast = 1.02    # Less contrast to preserve shadow detail
        shadow_lift = 0.25  # Stronger shadow lift: 0/0 → 0.1/0.25
    elif analysis["is_bright"]:
        saturation = 1.12
        contrast = 1.08    # More contrast to add depth
        shadow_lift = 0.18  # Standard
    else:
        saturation = 1.15  # Standard cinematic
        contrast = 1.05
        shadow_lift = 0.20
    
    # Build filter chain with adaptive values
    filters = ",".join([
        # Teal & Orange color grading (Hollywood blockbuster look)
        "colorbalance=rs=-0.1:gs=-0.05:bs=0.15:rm=0.05:gm=0:bm=-0.05:rh=0.1:gh=0.05:bh=-0.1",
        # Adaptive S-curve contrast (shadow lift depends on clip brightness)
        f"curves=m='0/0 0.25/{shadow_lift} 0.5/0.5 0.75/0.80 1/1'",
        # Contrast Adaptive Sharpening
        "cas=0.5",
        # Adaptive saturation, contrast, and brightness
        f"eq=saturation={saturation}:contrast={contrast}:brightness={brightness_adj:.3f}",
        # Subtle vignette (cinematic framing)
        "vignette=PI/4:mode=forward:eval=frame",
        # Fine detail sharpening
        "unsharp=3:3:0.5:3:3:0.3"
    ])
    
    # Log adaptive parameters if clip needed adjustment
    if analysis["is_dark"] or analysis["is_bright"]:
        exposure_type = "dark" if analysis["is_dark"] else "bright"
        print(f"   🎬 Content-aware enhance: {os.path.basename(input_path)} ({exposure_type}, brightness={analysis['avg_brightness']:.0f})")
    
    # Use fewer threads per job when running in parallel to avoid CPU contention
    threads_per_job = "2" if PARALLEL_ENHANCE else FFMPEG_THREADS
    
    cmd = [
        "ffmpeg", "-y",
        "-threads", threads_per_job,
        "-i", input_path,
        "-vf", filters,
    ]

    cmd.extend(["-c:v", OUTPUT_CODEC, "-preset", FFMPEG_PRESET, "-crf", "18", "-threads", threads_per_job])
    if OUTPUT_PROFILE:
        cmd.extend(["-profile:v", OUTPUT_PROFILE])
    if OUTPUT_LEVEL:
        cmd.extend(["-level", OUTPUT_LEVEL])
    cmd.extend(["-pix_fmt", OUTPUT_PIX_FMT, "-c:a", "copy", output_path])
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        return input_path


def color_match_clips(clip_paths: List[str], reference_clip: str = None, output_dir: str = None) -> Dict[str, str]:
    """
    Match colors across multiple clips for visual consistency.
    
    Uses color-matcher library for histogram-based color transfer.
    This ensures all clips in a montage have consistent color/exposure,
    similar to professional color grading workflows.
    
    Methods:
    - mkl: Monge-Kantorovitch Linear (best quality, slower)
    - hm: Histogram Matching (fast, good for similar content)
    - reinhard: Reinhard color transfer (classic, fast)
    
    Args:
        clip_paths: List of video file paths to match
        reference_clip: Path to reference clip (first clip if None)
        output_dir: Directory for matched clips (TEMP_DIR if None)
        
    Returns:
        Dict mapping original paths to color-matched paths
    """
    if not COLOR_MATCHER_AVAILABLE:
        print("   ⚠️ color-matcher not installed - skipping shot matching")
        return {p: p for p in clip_paths}
    
    if len(clip_paths) < 2:
        return {p: p for p in clip_paths}
    
    output_dir = output_dir or TEMP_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Use first clip as reference if not specified
    ref_path = reference_clip or clip_paths[0]
    
    print(f"   🎨 Color matching {len(clip_paths)} clips to reference...")
    
    try:
        # Extract reference frame (middle of clip)
        ref_frame_path = os.path.join(output_dir, "ref_frame.png")
        _extract_middle_frame(ref_path, ref_frame_path)
        
        if not os.path.exists(ref_frame_path):
            print("   ⚠️ Could not extract reference frame")
            return {p: p for p in clip_paths}
        
        ref_img = load_img_file(ref_frame_path)
        cm = ColorMatcher()
        
        results = {}
        
        for i, clip_path in enumerate(clip_paths):
            if clip_path == ref_path:
                # Reference clip stays unchanged
                results[clip_path] = clip_path
                continue
            
            try:
                # Extract source frame
                src_frame_path = os.path.join(output_dir, f"src_frame_{i}.png")
                _extract_middle_frame(clip_path, src_frame_path)
                
                if not os.path.exists(src_frame_path):
                    results[clip_path] = clip_path
                    continue
                
                # Calculate color transfer matrix
                src_img = load_img_file(src_frame_path)
                
                # Use mkl method (Monge-Kantorovitch Linear) for best quality
                # Could also use 'hm' (histogram) or 'reinhard' for speed
                matched = cm.transfer(src=src_img, ref=ref_img, method='mkl')
                
                # Apply color transfer to video via FFmpeg LUT
                # Generate a 1D LUT from the color transfer
                output_path = os.path.join(output_dir, f"matched_{os.path.basename(clip_path)}")
                
                # For simplicity, apply color correction via FFmpeg curves
                # Calculate average color shift
                src_mean = np.mean(src_img, axis=(0, 1))
                matched_mean = np.mean(matched, axis=(0, 1))
                
                # Calculate per-channel adjustments
                r_adj = (matched_mean[0] - src_mean[0]) / 255.0
                g_adj = (matched_mean[1] - src_mean[1]) / 255.0
                b_adj = (matched_mean[2] - src_mean[2]) / 255.0
                
                # Apply via FFmpeg colorbalance (simplified transfer)
                # Scale adjustments to colorbalance range (-1 to 1)
                r_bal = np.clip(r_adj * 2, -0.3, 0.3)
                g_bal = np.clip(g_adj * 2, -0.3, 0.3)
                b_bal = np.clip(b_adj * 2, -0.3, 0.3)
                
                filter_str = f"colorbalance=rs={r_bal:.3f}:gs={g_bal:.3f}:bs={b_bal:.3f}:rm={r_bal:.3f}:gm={g_bal:.3f}:bm={b_bal:.3f}:rh={r_bal:.3f}:gh={g_bal:.3f}:bh={b_bal:.3f}"
                
                cmd = [
                    "ffmpeg", "-y", "-i", clip_path,
                    "-vf", filter_str,
                ]
                
                cmd.extend(["-c:v", OUTPUT_CODEC, "-preset", "fast", "-crf", "18"])
                if OUTPUT_PROFILE:
                    cmd.extend(["-profile:v", OUTPUT_PROFILE])
                if OUTPUT_LEVEL:
                    cmd.extend(["-level", OUTPUT_LEVEL])
                cmd.extend(["-pix_fmt", OUTPUT_PIX_FMT, "-c:a", "copy", output_path])
                
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                results[clip_path] = output_path
                
                # Cleanup temp frame
                if os.path.exists(src_frame_path):
                    os.remove(src_frame_path)
                    
            except Exception as e:
                print(f"   ⚠️ Color match failed for {os.path.basename(clip_path)}: {e}")
                results[clip_path] = clip_path
        
        # Cleanup reference frame
        if os.path.exists(ref_frame_path):
            os.remove(ref_frame_path)
        
        matched_count = sum(1 for k, v in results.items() if k != v)
        print(f"   ✅ Color matched {matched_count}/{len(clip_paths)-1} clips")
        
        return results
        
    except Exception as e:
        print(f"   ⚠️ Color matching failed: {e}")
        return {p: p for p in clip_paths}


def _extract_middle_frame(video_path: str, output_path: str) -> bool:
    """Extract a frame from the middle of a video for color analysis."""
    try:
        # Get duration
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
        duration = float(result.stdout.strip()) if result.stdout.strip() else 1.0
        
        # Extract frame at middle
        middle = duration / 2
        extract_cmd = [
            "ffmpeg", "-y",
            "-ss", str(middle),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True, timeout=10)
        return os.path.exists(output_path)
        
    except Exception:
        return False


def enhance_clips_parallel(clip_jobs: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Enhance multiple clips in parallel using ThreadPoolExecutor.
    
    Args:
        clip_jobs: List of (input_path, output_path) tuples
        
    Returns:
        Dict mapping input_path to resulting output_path (enhanced or original on failure)
    """
    if not PARALLEL_ENHANCE or len(clip_jobs) <= 1:
        # Sequential processing
        results = {}
        for input_path, output_path in clip_jobs:
            print(f"   🎨 Enhancing {os.path.basename(input_path)}...")
            results[input_path] = enhance_clip(input_path, output_path)
        return results
    
    print(f"   🚀 Parallel enhancement: {len(clip_jobs)} clips with {MAX_PARALLEL_JOBS} workers...")
    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        # Submit all jobs
        future_to_input = {
            executor.submit(enhance_clip, input_path, output_path): input_path
            for input_path, output_path in clip_jobs
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_input):
            input_path = future_to_input[future]
            try:
                output_path = future.result()
                results[input_path] = output_path
                completed += 1
                if completed % 10 == 0 or completed == len(clip_jobs):
                    print(f"   🎨 Enhanced {completed}/{len(clip_jobs)} clips...")
            except Exception as e:
                print(f"   ⚠️ Enhancement failed for {os.path.basename(input_path)}: {e}")
                results[input_path] = input_path  # Fallback to original
    
    return results

def upscale_clip(input_path, output_path):
    """
    Upscale video using the best available method.
    
    Priority:
    1. cgpu Cloud GPU (if CGPU_GPU_ENABLED) - Free cloud GPU via Google Colab
    2. Real-ESRGAN local (if Vulkan GPU is available) - SOTA quality
    3. FFmpeg Lanczos + Sharpening (CPU fallback) - Good quality, always works
    
    NOTE: realesrgan-ncnn-vulkan does NOT support CPU mode (-g -1 is invalid).
    On systems without Vulkan compute (like Qualcomm Adreno), we use FFmpeg.
    """
    print(f"   ✨ Upscaling {os.path.basename(input_path)}...")
    
    # Priority 1: cgpu Cloud GPU
    if CGPU_GPU_ENABLED and CGPU_UPSCALER_AVAILABLE and is_cgpu_available():
        print(f"   ☁️ Attempting cgpu cloud GPU upscaling...")
        result = upscale_with_cgpu(input_path, output_path, scale=2)
        if result:
            return result
        print(f"   ⚠️ cgpu upscaling failed, falling back to local methods...")
    
    # Priority 2: Local Vulkan GPU
    # NOTE: realesrgan-ncnn-vulkan requires a proper Vulkan GPU with compute support.
    # Software renderers (llvmpipe) and GPUs without compute (Adreno) will crash.
    real_esrgan_available = False
    
    try:
        # Test: Try to run realesrgan with a tiny test to verify Vulkan works
        # The -i /dev/null test only checks if the binary starts, not Vulkan compute
        test_result = subprocess.run(
            ["realesrgan-ncnn-vulkan", "-i", "/dev/null", "-o", "/dev/null"],
            capture_output=True, timeout=5
        )
        # If it doesn't crash immediately with "invalid gpu", Vulkan might work
        if b"invalid gpu" not in test_result.stderr:
            # Additional check: Verify we have a real GPU, not software rendering
            vulkan_info = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=5)
            if vulkan_info.returncode == 0:
                output = vulkan_info.stdout.lower()
                # Skip software renderers (llvmpipe, lavapipe, swiftshader)
                if any(sw in output for sw in ["llvmpipe", "lavapipe", "swiftshader", "cpu"]):
                    print(f"   ⚠️ Detected software Vulkan renderer - skipping Real-ESRGAN")
                    real_esrgan_available = False
                # Skip GPUs without compute shader support (Adreno, Mali on some devices)
                elif "adreno" in output:
                    print(f"   ⚠️ Detected Qualcomm Adreno GPU - no compute shader support")
                    real_esrgan_available = False
                else:
                    real_esrgan_available = True
    except Exception:
        pass
    
    if real_esrgan_available:
        print(f"   🎮 Attempting Real-ESRGAN with Vulkan GPU...")
        return _upscale_with_realesrgan(input_path, output_path)
    else:
        print(f"   🖥️ Using FFmpeg Lanczos upscaling (Vulkan GPU not available)")
        return _upscale_with_ffmpeg(input_path, output_path)


def _upscale_with_realesrgan(input_path, output_path):
    """
    Upscale using Real-ESRGAN-ncnn-vulkan (requires Vulkan GPU).
    """
    frame_dir = os.path.join(TEMP_DIR, f"frames_{random.randint(0,99999)}")
    out_frame_dir = os.path.join(TEMP_DIR, f"out_frames_{random.randint(0,99999)}")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(out_frame_dir, exist_ok=True)
    
    try:
        # 1. Check for rotation metadata and build appropriate filter
        # FFmpeg does NOT auto-apply rotation when extracting to images without a filter
        import json
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:stream_side_data=rotation",
            "-of", "json", input_path
        ]
        probe_output = subprocess.check_output(probe_cmd).decode().strip()
        probe_data = json.loads(probe_output)
        stream = probe_data.get('streams', [{}])[0]
        
        # Check for rotation in side_data
        rotation = 0
        for side_data in stream.get('side_data_list', []):
            if 'rotation' in side_data:
                rotation = int(side_data['rotation'])
                break
        
        # Build filter based on rotation
        # transpose=1: 90° CW, transpose=2: 90° CCW, vflip+hflip: 180°
        if rotation == -90 or rotation == 270:
            vf_filter = "transpose=1"  # 90° CW to correct -90° (270°) rotation
            print(f"   🔄 Detected {rotation}° rotation, applying transpose=1")
        elif rotation == 90 or rotation == -270:
            vf_filter = "transpose=2"  # 90° CCW to correct 90° rotation  
            print(f"   🔄 Detected {rotation}° rotation, applying transpose=2")
        elif rotation == 180 or rotation == -180:
            vf_filter = "vflip,hflip"  # 180° flip
            print(f"   🔄 Detected {rotation}° rotation, applying vflip,hflip")
        else:
            vf_filter = None  # No rotation needed
        
        # 2. Extract frames with rotation correction if needed
        extract_cmd = ["ffmpeg", "-i", input_path]
        if vf_filter:
            extract_cmd += ["-vf", vf_filter]
        extract_cmd += ["-q:v", "2", f"{frame_dir}/frame_%08d.jpg"]
        
        subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Upscale frames with Real-ESRGAN (GPU 0)
        upscale_cmd = [
            "realesrgan-ncnn-vulkan",
            "-i", frame_dir,
            "-o", out_frame_dir,
            "-n", "realesr-animevideov3",  # Good for general video
            "-s", "2",                      # 2x upscale
            "-g", "0",                      # Use GPU 0
            "-m", "/usr/local/share/realesrgan-models"
        ]
        subprocess.run(upscale_cmd, check=True)
                      
        # 3. Reassemble video
        fps_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
                   "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", input_path]
        fps_str = subprocess.check_output(fps_cmd).decode().strip()
        
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps_val = num / den
            fps_arg = f"{fps_val:.2f}"
        else:
            fps_arg = fps_str

        # Real-ESRGAN writes PNG frames by default; re-encode from PNGs
        esrgan_cmd = ["ffmpeg", "-y", "-framerate", fps_arg, "-i", f"{out_frame_dir}/frame_%08d.png",
                      "-c:v", OUTPUT_CODEC, "-crf", "18"]
        if OUTPUT_PROFILE:
            esrgan_cmd.extend(["-profile:v", OUTPUT_PROFILE])
        if OUTPUT_LEVEL:
            esrgan_cmd.extend(["-level", OUTPUT_LEVEL])
        esrgan_cmd.extend(["-pix_fmt", OUTPUT_PIX_FMT, output_path])

        subprocess.run(esrgan_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                       
        return output_path
        
    except Exception as e:
        print(f"   ⚠️ Real-ESRGAN failed: {e}")
        print(f"   ↩️ Falling back to FFmpeg upscaling...")
        return _upscale_with_ffmpeg(input_path, output_path)
    finally:
        import shutil
        if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
        if os.path.exists(out_frame_dir): shutil.rmtree(out_frame_dir)


def _upscale_with_ffmpeg(input_path, output_path):
    """
    High-quality FFmpeg-based 2x upscaling using Lanczos + sharpening.
    
    This is the CPU fallback when Real-ESRGAN/Vulkan is not available.
    Uses a multi-filter chain for best quality:
    1. Lanczos resampling (best interpolation algorithm)
    2. Unsharp mask (recover detail lost in scaling)
    3. CAS (Contrast Adaptive Sharpening) for extra clarity
    """
    try:
        # Get original dimensions AFTER rotation is applied
        # We need to check rotation metadata and swap w/h if rotated 90/270
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:stream_side_data=rotation",
            "-of", "json", input_path
        ]
        import json
        probe_output = subprocess.check_output(probe_cmd).decode().strip()
        probe_data = json.loads(probe_output)
        stream = probe_data.get('streams', [{}])[0]
        orig_w = int(stream.get('width', 0))
        orig_h = int(stream.get('height', 0))
        
        # Check for rotation in side_data
        rotation = 0
        for side_data in stream.get('side_data_list', []):
            if 'rotation' in side_data:
                rotation = abs(int(side_data['rotation']))
                break
        
        # Swap dimensions if rotated 90 or 270 degrees
        if rotation in (90, 270):
            orig_w, orig_h = orig_h, orig_w
            print(f"   🔄 Detected {rotation}° rotation, adjusted dimensions")
        
        new_w, new_h = orig_w * 2, orig_h * 2
        
        print(f"   📐 Upscaling {orig_w}x{orig_h} → {new_w}x{new_h}")
        
        # Build filter chain for high-quality upscaling
        # 1. scale: Lanczos resampling (best quality interpolation)
        # 2. unsharp: Restore some sharpness lost in scaling
        # 3. cas: Contrast Adaptive Sharpening (0.5 = moderate)
        filter_chain = (
            f"scale={new_w}:{new_h}:flags=lanczos,"  # Lanczos interpolation
            f"unsharp=5:5:0.8:5:5:0.0,"              # Sharpen luma, not chroma
            f"cas=0.4"                                # Contrast adaptive sharpening
        )
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", filter_chain,
        ]

        ffmpeg_cmd.extend(["-c:v", OUTPUT_CODEC, "-preset", "slow", "-crf", "18"])
        if OUTPUT_PROFILE:
            ffmpeg_cmd.extend(["-profile:v", OUTPUT_PROFILE])
        if OUTPUT_LEVEL:
            ffmpeg_cmd.extend(["-level", OUTPUT_LEVEL])
        ffmpeg_cmd.extend(["-pix_fmt", OUTPUT_PIX_FMT, "-c:a", "copy", output_path])

        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"   ✅ FFmpeg upscaling complete")
        return output_path
        
    except Exception as e:
        print(f"   ⚠️ FFmpeg upscaling failed: {e}")
        return input_path

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
