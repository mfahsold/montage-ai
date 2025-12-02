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
import requests
import numpy as np
from datetime import datetime
import librosa
import cv2
import base64
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Any, List, Tuple
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    ImageClip, CompositeVideoClip, TextClip
)
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm

# Import Creative Director for natural language control
try:
    from .creative_director import CreativeDirector, interpret_natural_language
    from .style_templates import get_style_template, list_available_styles
    CREATIVE_DIRECTOR_AVAILABLE = True
except ImportError:
    print("⚠️ Creative Director not available (missing creative_director.py)")
    CREATIVE_DIRECTOR_AVAILABLE = False

# Import Footage Manager for professional clip management
from .footage_manager import integrate_footage_manager, select_next_clip

VERSION = "0.2.0"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
INPUT_DIR = "/data/input"
MUSIC_DIR = "/data/music"
ASSETS_DIR = "/data/assets"
OUTPUT_DIR = "/data/output"
TEMP_DIR = "/tmp"

# Job Identification (for parallel runs)
JOB_ID = os.environ.get("JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))

# LLM / AI Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava")  # For scene analysis
DIRECTOR_MODEL = os.environ.get("DIRECTOR_MODEL", "llama3.1:70b")  # For creative direction
ENABLE_AI_FILTER = os.environ.get("ENABLE_AI_FILTER", "false").lower() == "true"

# Natural Language Control (NEW!)
CREATIVE_PROMPT = os.environ.get("CREATIVE_PROMPT", "").strip()
# Example prompts:
#   "Edit this like a Hitchcock thriller"
#   "Make it fast-paced like an MTV music video"
#   "Calm and meditative with long shots"
#   "Documentary realism with natural pacing"

# Legacy Cut Style (backwards compatible, overridden by CREATIVE_PROMPT if set)
CUT_STYLE = os.environ.get("CUT_STYLE", "dynamic").lower()  # fast, hyper, slow, dynamic

# Visual Enhancement
STABILIZE = os.environ.get("STABILIZE", "false").lower() == "true"
UPSCALE = os.environ.get("UPSCALE", "false").lower() == "true"  # AI Upscaling (Slow!)
ENHANCE = os.environ.get("ENHANCE", "true").lower() == "true"  # Color/Sharpness (Fast)

# Logging / Debug
VERBOSE = os.environ.get("VERBOSE", "true").lower() == "true"  # Show detailed analysis

# Output Control
NUM_VARIANTS = int(os.environ.get("NUM_VARIANTS", "1"))

# Timeline Export (NEW!)
EXPORT_TIMELINE = os.environ.get("EXPORT_TIMELINE", "false").lower() == "true"
GENERATE_PROXIES = os.environ.get("GENERATE_PROXIES", "false").lower() == "true"

# Performance: CPU Threading & Parallelization
FFMPEG_THREADS = os.environ.get("FFMPEG_THREADS", "0")  # 0 = auto (all cores)
FFMPEG_PRESET = os.environ.get("FFMPEG_PRESET", "medium")  # ultrafast/fast/medium/slow/veryslow
PARALLEL_ENHANCE = os.environ.get("PARALLEL_ENHANCE", "true").lower() == "true"  # Parallel clip enhancement
MAX_PARALLEL_JOBS = int(os.environ.get("MAX_PARALLEL_JOBS", str(max(1, multiprocessing.cpu_count() - 2))))  # Leave 2 cores free

# Clip reuse control
MAX_SCENE_REUSE = int(os.environ.get("MAX_SCENE_REUSE", "3"))

# GPU/Hardware Acceleration (auto-detected)
USE_GPU = os.environ.get("USE_GPU", "auto").lower()  # auto, vulkan, v4l2, none

# Optional FFmpeg MCP offload
USE_FFMPEG_MCP = os.environ.get("USE_FFMPEG_MCP", "false").lower() == "true"
FFMPEG_MCP_ENDPOINT = os.environ.get("FFMPEG_MCP_ENDPOINT", "http://ffmpeg-mcp.montage-ai.svc.cluster.local:8080")

# cgpu Cloud GPU Configuration
CGPU_GPU_ENABLED = os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true"

# ============================================================================
# GLOBAL EDITING INSTRUCTIONS (Set by Creative Director)
# ============================================================================
EDITING_INSTRUCTIONS = None  # Will be populated by interpret_creative_prompt()

# GPU/Hardware Capability Detection (set at runtime)
GPU_CAPABILITY = None  # Will be set by detect_gpu_capabilities()

# Import cgpu Cloud Upscaler if available
try:
    from .cgpu_upscaler import upscale_with_cgpu, is_cgpu_available
    CGPU_UPSCALER_AVAILABLE = True
except ImportError:
    CGPU_UPSCALER_AVAILABLE = False
    is_cgpu_available = lambda: False
    upscale_with_cgpu = None

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

# Deep Analysis Configuration
DEEP_ANALYSIS = os.environ.get("DEEP_ANALYSIS", "false").lower() == "true"


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
        print(f"")
        print(f"📊 ENHANCEMENT SETTINGS (from ENV):")
        print(f"   STABILIZE:        {os.environ.get('STABILIZE', 'false')}")
        print(f"   UPSCALE:          {os.environ.get('UPSCALE', 'false')}")
        print(f"   ENHANCE:          {os.environ.get('ENHANCE', 'true')}")
        print(f"   PARALLEL_ENHANCE: {os.environ.get('PARALLEL_ENHANCE', 'true')}")
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

def detect_scenes(video_path, threshold=30.0):
    """Use PySceneDetect to find cuts in the raw video."""
    print(f"🎬 Detecting scenes in {os.path.basename(video_path)}...")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    # Convert to (start, end) in seconds
    scenes = []
    for scene in scene_list:
        start, end = scene
        scenes.append((start.get_seconds(), end.get_seconds()))
    print(f"   Found {len(scenes)} scenes.")
    
    # If no scenes were detected (single shot), use the whole video
    if not scenes:
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            print(f"   ⚠️ No cuts detected. Using full video ({duration:.1f}s).")
            return [(0.0, duration)]
        except Exception as e:
            print(f"   ❌ Could not read video duration: {e}")
            return []
            
    return scenes

def analyze_scene_content(video_path, time_point):
    """Extract a frame and ask AI to describe it."""
    if not ENABLE_AI_FILTER:
        return {"quality": "YES", "description": "unknown", "action": "medium", "shot": "medium"}
        
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {"quality": "NO", "description": "read error", "action": "low", "shot": "medium"}
            
        _, buffer = cv2.imencode('.jpg', frame)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        prompt = (
            "Analyze this image for a video editor. Return a JSON object with these keys: "
            "quality (YES/NO), description (5 words), action (low/medium/high), shot (close-up/medium/wide). "
            "Example: {\"quality\": \"YES\", \"description\": \"Man running in park\", \"action\": \"high\", \"shot\": \"wide\"}"
        )
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [b64_img],
            "stream": False,
            "format": "json"
        }
        
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        if response.status_code == 200:
            try:
                res_json = json.loads(response.json().get("response", "{}"))
                # Normalize keys
                return {
                    "quality": res_json.get("quality", "YES"),
                    "description": res_json.get("description", "unknown"),
                    "action": res_json.get("action", "medium").lower(),
                    "shot": res_json.get("shot", "medium").lower()
                }
            except:
                text = response.json().get("response", "")
                return {"quality": "YES", "description": text[:50], "action": "medium", "shot": "medium"}
                
    except Exception as e:
        print(f"   ⚠️ AI Analysis failed: {e}")
        return {"quality": "YES", "description": "error", "action": "medium", "shot": "medium"}
        
    return {"quality": "YES", "description": "unknown", "action": "medium", "shot": "medium"}

def calculate_visual_similarity(frame1_path, frame1_time, frame2_path, frame2_time):
    """
    Calculate visual similarity between two frames using color histogram comparison.
    Used for 'Match Cut' detection - finding visually similar moments for seamless transitions.

    Returns:
        Similarity score 0-1 (1 = identical)
    """
    try:
        cap1 = cv2.VideoCapture(frame1_path)
        cap1.set(cv2.CAP_PROP_POS_MSEC, frame1_time * 1000)
        ret1, frame1 = cap1.read()
        cap1.release()

        cap2 = cv2.VideoCapture(frame2_path)
        cap2.set(cv2.CAP_PROP_POS_MSEC, frame2_time * 1000)
        ret2, frame2 = cap2.read()
        cap2.release()

        if not ret1 or not ret2:
            return 0.0

        # Convert to HSV for better color similarity
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # Calculate histograms
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

        # Normalize
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Compare using correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, similarity)  # Correlation can be negative

    except Exception as e:
        print(f"   ⚠️ Visual similarity failed: {e}")
        return 0.0

def detect_motion_blur(video_path, time_point):
    """
    Detect motion blur intensity at a specific time point.
    Used for 'Invisible Cut' placement - cuts work best during motion blur.

    Returns:
        Blur score 0-1 (1 = heavy blur, good for cuts)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return 0.0

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance (low = blurry)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize to 0-1 (invert: high variance = sharp, low = blurry)
        # Typical sharp video has variance > 500, blurry < 100
        blur_score = 1.0 - min(1.0, laplacian_var / 500.0)
        return blur_score

    except Exception as e:
        print(f"   ⚠️ Motion blur detection failed: {e}")
        return 0.0

def find_best_start_point(video_path, scene_start, scene_end, target_duration):
    """
    Find the most 'interesting' segment within a scene using Optical Flow.

    This is the 'Perfect Cut' optimization - instead of random start points,
    we find the moment with highest action/motion.

    Args:
        video_path: Path to video file
        scene_start: Scene start time in seconds
        scene_end: Scene end time in seconds
        target_duration: Desired clip duration

    Returns:
        Best start timestamp (float)
    """
    try:
        cap = cv2.VideoCapture(video_path)

        # Jump to scene start
        cap.set(cv2.CAP_PROP_POS_MSEC, scene_start * 1000)

        motion_scores = []
        timestamps = []

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            # Fallback to random
            max_start = scene_end - target_duration
            return random.uniform(scene_start, max(scene_start, max_start))

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Sample every 500ms for performance
        frame_interval = 0.5
        current_time = scene_start

        while current_time < scene_end - target_duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow magnitude (motion intensity)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Compute motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_score = np.mean(magnitude)

            motion_scores.append(motion_score)
            timestamps.append(current_time)

            prev_gray = gray
            current_time += frame_interval

        cap.release()

        # Find peak motion (action climax)
        if motion_scores and len(motion_scores) > 0:
            # Use 75th percentile to avoid noise spikes
            threshold = np.percentile(motion_scores, 75)
            high_motion_indices = [i for i, score in enumerate(motion_scores) if score >= threshold]

            if high_motion_indices:
                # Pick first high-motion moment (natural story progression)
                peak_idx = high_motion_indices[0]
                best_start = timestamps[peak_idx]

                # Ensure we don't exceed scene boundaries
                if best_start + target_duration > scene_end:
                    best_start = scene_end - target_duration

                return max(scene_start, best_start)

    except Exception as e:
        print(f"   ⚠️ Optical Flow analysis failed: {e}")

    # Fallback: Random start (existing behavior)
    max_start = scene_end - target_duration
    if max_start <= scene_start:
        return scene_start
    return random.uniform(scene_start, max_start)


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
                    "video_codec": "libx264",
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
        "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "copy", output_path
    ]
    subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def analyze_music_energy(audio_path):
    """Get RMS energy curve."""
    print(f"🎵 Analyzing energy levels of {os.path.basename(audio_path)}...")
    y, sr = librosa.load(audio_path)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    # Normalize energy 0-1
    rms = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
    
    if VERBOSE:
        # Show energy statistics
        avg_energy = np.mean(rms)
        max_energy = np.max(rms)
        min_energy = np.min(rms)
        high_energy_pct = np.sum(rms > 0.7) / len(rms) * 100
        print(f"   📊 Energy Stats: avg={avg_energy:.2f}, max={max_energy:.2f}, min={min_energy:.2f}")
        print(f"   📊 High Energy (>70%): {high_energy_pct:.1f}% of track")
    
    return times, rms

def get_beat_times(audio_path):
    """Use Librosa to find beat times."""
    print(f"🎵 Analyzing beat structure of {os.path.basename(audio_path)}...")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Handle tempo being an array (newer librosa versions)
    if isinstance(tempo, np.ndarray):
        tempo = tempo.item()

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Additional music analysis for verbose output
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"   Tempo: {tempo:.1f} BPM, Detected {len(beat_times)} beats.")
    
    if VERBOSE:
        print(f"   📊 Track Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"   📊 Sample Rate: {sr} Hz")
        print(f"   📊 Beat Interval: {60/tempo:.2f}s avg")
        
        # Detect if tempo is fast/slow
        if tempo > 140:
            print(f"   🚀 Fast Tempo (>140 BPM) - will use longer beat groups to avoid seizure cuts")
        elif tempo < 80:
            print(f"   🐢 Slow Tempo (<80 BPM) - will use shorter beat groups for variety")
        else:
            print(f"   ⚖️ Medium Tempo - balanced pacing")
    
    return beat_times, tempo
    return beat_times, tempo

def calculate_dynamic_cut_length(current_energy, tempo, current_time, total_duration, pattern_pool):
    """
    Advanced 2024/2025 pacing algorithm with position-aware intelligence.

    Based on research from:
    - Film Editing Pro 2024: Track-position pacing theory
    - Premiere Gal: Fibonacci rhythm patterns
    - Industry standard: Intro/Build/Climax/Outro structure

    Args:
        current_energy: Audio RMS energy (0-1)
        tempo: BPM of music
        current_time: Current position in track (seconds)
        total_duration: Total track duration (seconds)
        pattern_pool: List of available cut patterns

    Returns:
        List of beat counts for next cuts (Fibonacci or custom pattern)
    """
    # Calculate track position (0-1)
    progress = current_time / total_duration

    # PHASE 1: INTRO (0-20%) - Establish atmosphere, longer cuts
    if progress < 0.2:
        if current_energy < 0.3:
            # Calm intro: Very long takes to set mood
            base_pattern = [8, 8, 8, 4]
        else:
            # Energetic intro: Steady rhythm
            base_pattern = [4, 4, 4, 4]

    # PHASE 2: BUILD-UP (20-40%) - Increasing tension and variation
    elif progress < 0.4:
        if current_energy > 0.6:
            # Energy rising: Fibonacci acceleration
            base_pattern = [8, 5, 3, 2, 1, 1]  # Fibonacci descent
        else:
            # Gentle build: Classic pattern
            base_pattern = [4, 4, 2, 2]

    # PHASE 3: CLIMAX (40-75%) - Peak energy, maximum variation
    elif progress < 0.75:
        if current_energy > 0.8:
            # Hyper-energy peak: Rapid cuts (TikTok style)
            base_pattern = [1, 1, 1, 2, 1, 1, 3]  # Syncopated
        elif current_energy > 0.6:
            # High energy: Fibonacci magic
            base_pattern = [1, 1, 2, 3, 5]  # Classic Fibonacci
        else:
            # Medium energy: Varied but controlled
            base_pattern = [2, 2, 4, 2]

    # PHASE 4: OUTRO/RESOLUTION (75-100%) - Wind down, return to calm
    else:
        if current_energy > 0.7:
            # High-energy ending: Sustain excitement then resolve
            base_pattern = [2, 2, 4, 8]
        else:
            # Calm ending: Long reflective cuts
            base_pattern = [8, 8, 4, 8]

    # TEMPO MODULATION: Adjust for BPM
    # Fast tempos need longer beat counts to avoid seizure-inducing cuts
    if tempo > 140:
        base_pattern = [max(2, b) for b in base_pattern]  # Minimum 2 beats
    elif tempo < 80:
        base_pattern = [max(1, b // 2) for b in base_pattern]  # Halve for slow songs

    return base_pattern

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
    
    # Store original ENV values (these have higher priority)
    env_stabilize = os.environ.get("STABILIZE", "").lower() == "true"
    env_upscale = os.environ.get("UPSCALE", "").lower() == "true"
    env_enhance = os.environ.get("ENHANCE", "true").lower() == "true"
    
    # Apply style template defaults ONLY if ENV was not explicitly set
    if EDITING_INSTRUCTIONS is not None:
        effects = EDITING_INSTRUCTIONS.get('effects', {})

        # Only use template value if ENV wasn't explicitly set to "true"
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
    clips = []
    clips_metadata = []  # Track metadata for timeline export (OTIO/EDL/CSV)
    current_time = 0
    beat_idx = 0
    cut_number = 0  # For monitoring
    
    audio_clip = AudioFileClip(music_path)
    target_duration = audio_clip.duration
    
    # Estimate total cuts for progress tracking
    avg_cut_duration = 2.0  # Rough estimate
    estimated_total_cuts = int(target_duration / avg_cut_duration)
    
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

    # ADVANCED 2024/2025 PACING: Position-aware patterns with Fibonacci
    # Pattern pool for dynamic selection (fallback for non-dynamic modes)
    cut_patterns = [
        [4, 4, 4, 4],       # Steady pace
        [2, 2, 4, 8],       # Accelerate then hold (build-up)
        [8, 4, 2, 2],       # Long shot into fast cuts
        [2, 2, 2, 2],       # Energetic steady
        [1, 1, 2, 4],       # Rapid fire start
        [4, 2, 1, 1, 8],    # Complex phrase
        [1, 1, 2, 3, 5],    # Fibonacci magic
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
        if target_beat_idx >= len(beat_times):
            cut_duration = target_duration - current_time
        else:
            cut_duration = beat_times[target_beat_idx] - beat_times[beat_idx]
            
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
                score -= 50

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

            # Randomness factor to keep it fresh
            score += random.randint(-5, 5)

            if score > best_score:
                best_score = score
                selected_scene = scene
        
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
            
            # Cleanup later? For now let /tmp fill up, it's a container.
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
            else:
                v_clip = VideoFileClip(selected_scene['path']).subclip(clip_start, clip_start + cut_duration)
        
        # Get video dimensions
        # NOTE: MoviePy automatically handles rotation metadata since recent versions
        # So we should NOT manually apply rotation - the video is already correctly oriented
        w, h = v_clip.size
        
        # Debug: Show original dimensions
        print(f"  📏 Video dimensions: {w}x{h}")
        
        # Resize/Crop logic for 9:16 vertical video (preserving aspect ratio)
        target_ratio = 9/16  # 0.5625
        current_ratio = w/h
        
        print(f"  📏 Current ratio: {current_ratio:.3f}, target: {target_ratio:.3f}")
        
        if current_ratio > target_ratio + 0.01:  # Add tolerance for floating point
            # Video is wider than 9:16 - crop width to fit
            new_w = int(h * target_ratio)
            crop_x = (w - new_w) // 2
            print(f"  ✂️ Cropping width: {w} -> {new_w} (crop_x={crop_x})")
            v_clip = v_clip.crop(x1=crop_x, x2=crop_x + new_w)
        elif current_ratio < target_ratio - 0.01:
            # Video is taller than 9:16 - crop height to fit
            new_h = int(w / target_ratio)
            crop_y = (h - new_h) // 2
            print(f"  ✂️ Cropping height: {h} -> {new_h} (crop_y={crop_y})")
            v_clip = v_clip.crop(y1=crop_y, y2=crop_y + new_h)
        # else: already ~9:16, no crop needed
            
        # Scale to 1080x1920 (standard vertical HD)
        # This preserves aspect ratio since we already cropped to 9:16
        v_clip = v_clip.resize(height=1920)
        print(f"  📐 Final size: {v_clip.size}")
        
        # Ensure dimensions are even (required by h264 encoder)
        clip_w, clip_h = v_clip.size
        final_w = clip_w if clip_w % 2 == 0 else clip_w - 1
        final_h = clip_h if clip_h % 2 == 0 else clip_h - 1
        if final_w != clip_w or final_h != clip_h:
            v_clip = v_clip.crop(x2=final_w, y2=final_h)

        # 🎬 CREATIVE DIRECTOR INTEGRATION: Transitions control
        if EDITING_INSTRUCTIONS is not None:
            transitions = EDITING_INSTRUCTIONS.get('transitions', {})
            transition_type = transitions.get('type', 'energy_aware')
            crossfade_duration_sec = transitions.get('crossfade_duration_sec', 0.5)

            if transition_type == "crossfade":
                # Always crossfade
                if len(clips) > 0:
                    fade_duration = min(crossfade_duration_sec, cut_duration * 0.3)
                    v_clip = v_clip.crossfadein(fade_duration)
                    clips[-1] = clips[-1].crossfadeout(fade_duration)

            elif transition_type == "mixed":
                # Random crossfade (50% of the time)
                if len(clips) > 0 and random.random() > 0.5:
                    fade_duration = min(crossfade_duration_sec, cut_duration * 0.3)
                    v_clip = v_clip.crossfadein(fade_duration)
                    clips[-1] = clips[-1].crossfadeout(fade_duration)

            elif transition_type == "energy_aware":
                # Crossfade on low energy (default)
                if len(clips) > 0 and current_energy < 0.3:
                    fade_duration = min(crossfade_duration_sec, cut_duration * 0.2)
                    v_clip = v_clip.crossfadein(fade_duration)
                    clips[-1] = clips[-1].crossfadeout(fade_duration)

            # "hard_cuts" = no crossfade (skip)

        else:
            # Legacy: Energy-aware crossfade
            if len(clips) > 0 and current_energy < 0.3:
                fade_duration = min(0.5, cut_duration * 0.2)  # Max 0.5s or 20% of clip
                v_clip = v_clip.crossfadein(fade_duration)
                clips[-1] = clips[-1].crossfadeout(fade_duration)

        clips.append(v_clip)

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
        monitor.end_phase({
            "total_cuts": cut_number,
            "timeline_duration": f"{current_time:.1f}s",
            "clips_in_sequence": len(clips)
        })
        
    # 4. Final Composition
    if monitor:
        monitor.start_phase("composition")
    
    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.set_audio(audio_clip.subclip(0, current_time))
    
    # 5. Add Logo (Overlay)
    logo_files = get_files(ASSETS_DIR, ('.png', '.jpg'))
    if logo_files:
        logo = ImageClip(logo_files[0]).set_duration(final_video.duration).resize(height=150).margin(right=50, top=50, opacity=0).set_pos(("right", "top"))
        final_video = CompositeVideoClip([final_video, logo])
        if monitor:
            monitor.log_info("composition", f"Logo overlay added: {os.path.basename(logo_files[0])}")
    
    if monitor:
        monitor.end_phase({"final_duration": f"{final_video.duration:.1f}s"})
        
    # 6. Render
    # 🎬 CREATIVE DIRECTOR INTEGRATION: Use style name in filename
    if EDITING_INSTRUCTIONS is not None:
        style_name = EDITING_INSTRUCTIONS.get('style', {}).get('name', CUT_STYLE)
    else:
        style_name = CUT_STYLE

    output_filename = os.path.join(OUTPUT_DIR, f"gallery_montage_{JOB_ID}_v{variant_id}_{style_name}.mp4")
    
    # === MONITORING: Render phase ===
    if monitor:
        monitor.log_render_start(output_filename, {
            "codec": "libx264",
            "preset": FFMPEG_PRESET,
            "crf": 18,
            "fps": 30,
            "duration": f"{final_video.duration:.1f}s"
        })
    else:
        print(f"🚀 Rendering Variant #{variant_id} to {output_filename}...")
        print(f"   Job-ID: {JOB_ID}")
    
    render_start_time = time.time()
    
    # Multi-threaded rendering with quality settings
    # Using pixel format yuv420p for maximum compatibility
    final_video.write_videofile(
        output_filename,
        codec='libx264',
        audio_codec='aac',
        fps=30,
        preset=FFMPEG_PRESET,
        threads=int(FFMPEG_THREADS) if FFMPEG_THREADS != "0" else None,  # None = auto
        ffmpeg_params=[
            "-crf", "18",  # High quality (lower = better, 18 is visually lossless)
            "-tune", "film",  # Optimize for film content
            "-pix_fmt", "yuv420p",  # Standard pixel format for wide compatibility
            "-level", "4.1",  # Compatibility
            "-movflags", "+faststart"  # Web-optimized
        ]
    )
    
    # === MONITORING: Render complete ===
    render_duration_ms = (time.time() - render_start_time) * 1000
    if monitor:
        # Get file size
        try:
            file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
        except:
            file_size_mb = 0
        monitor.log_render_complete(output_filename, file_size_mb, render_duration_ms)
    else:
        print(f"✅ Variant #{variant_id} Done!")

    # 7. Timeline Export (if enabled)
    if EXPORT_TIMELINE and TIMELINE_EXPORT_AVAILABLE:
        print(f"\n📽️ Exporting Timeline for NLE import...")

        try:
            # Determine style name for project naming
            style_name = "dynamic"
            if EDITING_INSTRUCTIONS is not None:
                style_name = EDITING_INSTRUCTIONS.get('style_name', 'custom')

            # Export to OTIO/EDL/CSV for professional NLE software
            exported_files = export_timeline_from_montage(
                clips_metadata,
                music_path,
                final_video.duration,
                output_dir=OUTPUT_DIR,
                project_name=f"montage_{JOB_ID}_v{variant_id}_{style_name}",
                generate_proxies=GENERATE_PROXIES
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

def stabilize_clip(input_path, output_path):
    """
    Stabilize a video clip using ffmpeg's vidstabtransform (if available) or a simple deshake.
    Since the conda ffmpeg build might lack libvidstab, we'll try a basic 'deshake' filter first
    which is built-in, or skip if too slow.
    
    Actually, for 'SOTA' without libvidstab, we can use a Python-based approach with OpenCV 
    if we really want to, but that's heavy.
    
    Let's try the 'deshake' filter which is standard in ffmpeg.
    """
    print(f"   ⚖️ Stabilizing {os.path.basename(input_path)}...")
    
    # Check if we can use vidstab (requires libvidstab)
    # If not, use 'deshake' (standard open source filter)
    
    # Simple deshake command with multi-threading
    cmd = [
        "ffmpeg", "-y",
        "-threads", FFMPEG_THREADS,  # Use all CPU cores
        "-i", input_path,
        "-vf", "deshake",
        "-c:v", "libx264",
        "-preset", "fast",  # Good balance for stabilization
        "-crf", "20",
        "-threads", FFMPEG_THREADS,
        "-c:a", "copy",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except subprocess.CalledProcessError:
        print("   ⚠️ Stabilization failed (ffmpeg error). Using original.")
        return input_path
    except Exception as e:
        print(f"   ⚠️ Stabilization failed: {e}")
        return input_path

def enhance_clip(input_path, output_path):
    """
    Apply CINEMATIC enhancements using ffmpeg filters:
    - Filmic color grading (Teal & Orange look)
    - S-curve contrast for cinematic depth
    - CAS (Contrast Adaptive Sharpening)
    - Subtle vignette for focus
    - Film grain option
    
    This function is thread-safe and can be called from ThreadPoolExecutor.
    """
    # CINEMATIC FILTER CHAIN:
    # 1. colorbalance: Teal shadows + Orange highlights (Hollywood look)
    # 2. curves: S-curve for filmic contrast (lift blacks, compress highlights)
    # 3. cas: Contrast Adaptive Sharpening (crisp details)
    # 4. eq: Fine-tune saturation and contrast
    # 5. vignette: Subtle edge darkening (draws eye to center)
    # 6. unsharp: Final detail enhancement
    
    filters = ",".join([
        # Teal & Orange color grading (Hollywood blockbuster look)
        "colorbalance=rs=-0.1:gs=-0.05:bs=0.15:rm=0.05:gm=0:bm=-0.05:rh=0.1:gh=0.05:bh=-0.1",
        # S-curve contrast (filmic look - lifted blacks, soft highlights)
        "curves=m='0/0 0.25/0.20 0.5/0.5 0.75/0.80 1/1'",
        # Contrast Adaptive Sharpening
        "cas=0.5",
        # Saturation and contrast boost
        "eq=saturation=1.15:contrast=1.05:brightness=0.02",
        # Subtle vignette (cinematic framing)
        "vignette=PI/4:mode=forward:eval=frame",
        # Fine detail sharpening
        "unsharp=3:3:0.5:3:3:0.3"
    ])
    
    # Use fewer threads per job when running in parallel to avoid CPU contention
    threads_per_job = "2" if PARALLEL_ENHANCE else FFMPEG_THREADS
    
    cmd = [
        "ffmpeg", "-y",
        "-threads", threads_per_job,
        "-i", input_path,
        "-vf", filters,
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", "18",  # Higher quality for cinematic output
        "-threads", threads_per_job,
        "-c:a", "copy",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        return input_path


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
    real_esrgan_available = False
    
    try:
        # Quick test: Try to initialize Vulkan with realesrgan
        test_result = subprocess.run(
            ["realesrgan-ncnn-vulkan", "-i", "/dev/null", "-o", "/dev/null"],
            capture_output=True, timeout=5
        )
        # If it doesn't crash immediately with "invalid gpu", Vulkan might work
        if b"invalid gpu" not in test_result.stderr:
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
        # 1. Extract frames
        subprocess.run(["ffmpeg", "-i", input_path, "-q:v", "2", f"{frame_dir}/frame_%08d.jpg"], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
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
        subprocess.run(["ffmpeg", "-y", "-framerate", fps_arg, "-i", f"{out_frame_dir}/frame_%08d.png", 
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                       
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
        # Get original dimensions
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", input_path
        ]
        dimensions = subprocess.check_output(probe_cmd).decode().strip()
        orig_w, orig_h = map(int, dimensions.split('x'))
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
            "-c:v", "libx264",
            "-preset", "slow",      # Better quality encoding
            "-crf", "18",           # High quality
            "-c:a", "copy",         # Keep original audio
            output_path
        ]
        
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
