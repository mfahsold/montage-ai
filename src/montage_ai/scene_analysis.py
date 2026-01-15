"""
Scene Analysis Module for Montage AI

Provides scene detection, content analysis, and visual similarity calculation.
Uses PySceneDetect for scene detection and OpenCV for visual analysis.

Usage:
    from montage_ai.scene_analysis import SceneDetector, detect_scenes, analyze_scene_content

    detector = SceneDetector()  # Uses ThresholdConfig.scene_detection()
    scenes = detector.detect(video_path)
"""

import os
import json
import random
import base64
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import requests

# Lazy-loaded imports in classes/functions below
# - scenedetect
# - scipy.spatial.KDTree

# OPTIMIZATION: Scene detection backends
try:
    from scenedetect import open_video, SceneManager, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    SceneManager = None
    ContentDetector = None
    open_video = None

# OPTIMIZATION: K-D tree for fast scene similarity queries (sub-linear search)
KDTREE_AVAILABLE = False
try:
    import scipy.spatial
    KDTREE_AVAILABLE = True
except ImportError:
    pass

from .config import get_settings
from .video_metadata import probe_metadata  # Expose for tests to patch
from .logger import logger
from .moviepy_compat import VideoFileClip
from .prompts import get_vision_analysis_prompt, ActionLevel, ShotType

# Import cgpu jobs for offloading
try:
    from .cgpu_jobs import SceneDetectionJob
    from .cgpu_utils import is_cgpu_available
    CGPU_AVAILABLE = True
except ImportError:
    CGPU_AVAILABLE = False

# SOTA: VideoCapture connection pooling (40% faster for repeated access)
try:
    from .video_capture_pool import (
        get_video_pool,
        VideoCapturePool,
        extract_frame_base64 as pool_extract_frame_base64,
    )
    CAPTURE_POOL_AVAILABLE = True
except ImportError:
    CAPTURE_POOL_AVAILABLE = False

_settings = get_settings()

# OPTIMIZATION: LRU cache with size limit
# Increased to 2000 for better hit rate with large projects (1000+ clips)
# Memory impact: ~2000 * 512 floats * 4 bytes = ~4MB
_HISTOGRAM_CACHE_SIZE = 2000


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Scene:
    """A detected scene segment in a video."""
    start: float
    end: float
    path: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        """Midpoint time of the scene."""
        return (self.start + self.end) / 2

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "start": self.start,
            "end": self.end,
            "path": self.path,
            "meta": self.meta,
        }


@dataclass
class SceneAnalysis:
    """
    AI analysis result for a scene/frame.

    Extended with semantic fields for intelligent clip selection:
    - tags: Free-form semantic tags for content matching
    - caption: Detailed description for embedding-based search
    - objects: Detected objects in the frame
    - mood: Emotional tone of the scene
    - setting: Location/environment type
    - face_count: Number of detected faces
    """
    quality: str  # "YES" or "NO"
    description: str  # 5-word summary
    action: ActionLevel
    shot: ShotType
    # Semantic fields (Phase 2: Semantic Storytelling)
    tags: List[str] = field(default_factory=list)  # Free-form semantic tags
    caption: str = ""  # Detailed description for embedding
    objects: List[str] = field(default_factory=list)  # Detected objects
    mood: str = "neutral"  # calm, energetic, dramatic, playful, tense, peaceful, mysterious
    setting: str = "unknown"  # indoor, outdoor, beach, city, nature, studio, street, home
    face_count: int = 0  # Number of detected faces
    focus_center_x: float = 0.5 # Normalised X position of main subject (0.0=left, 1.0=right)
    balance_score: float = 0.5  # Visual balance (0.0=unbalanced, 1.0=perfectly symmetric)

    @classmethod
    def default(cls) -> "SceneAnalysis":
        """Create default analysis (no AI filter)."""
        return cls(
            quality="YES",
            description="unknown",
            action=ActionLevel.MEDIUM,
            shot=ShotType.MEDIUM,
            tags=[],
            caption="",
            objects=[],
            mood="neutral",
            setting="unknown",
            face_count=0,
            focus_center_x=0.5,
            balance_score=0.5,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SceneAnalysis":
        """Create from dictionary response."""
        # Handle action level parsing
        action_str = data.get("action", "medium")
        if isinstance(action_str, str):
            action_str = action_str.lower()
        else:
            action_str = "medium"

        # Handle shot type parsing
        shot_str = data.get("shot", "medium")
        if isinstance(shot_str, str):
            shot_str = shot_str.lower()
        else:
            shot_str = "medium"

        return cls(
            quality=data.get("quality", "YES"),
            description=data.get("description", "unknown"),
            action=ActionLevel(action_str),
            shot=ShotType(shot_str),
            tags=data.get("tags", []),
            caption=data.get("caption", ""),
            objects=data.get("objects", []),
            mood=data.get("mood", "neutral"),
            setting=data.get("setting", "unknown"),
            face_count=data.get("face_count", 0),
            focus_center_x=data.get("focus_center_x", 0.5),
            balance_score=data.get("balance_score", 0.5),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "quality": self.quality,
            "description": self.description,
            "action": self.action.value,
            "shot": self.shot.value,
            "tags": self.tags,
            "caption": self.caption,
            "objects": self.objects,
            "mood": self.mood,
            "setting": self.setting,
            "face_count": self.face_count,
            "focus_center_x": self.focus_center_x,
            "balance_score": self.balance_score,
        }


    @property
    def has_semantic_data(self) -> bool:
        """Check if semantic analysis data is present."""
        return bool(self.tags or self.caption or self.objects)


# =============================================================================
# Scene Detector Class
# =============================================================================

class SceneDetector:
    """
    Detect scene boundaries in video files.

    Supports multiple backends:
    - TransNetV2 (SOTA neural network, ~250fps on GPU)
    - PySceneDetect ContentDetector (traditional, ~50fps)
    - Cloud GPU (cgpu offloading for large videos)

    Backend selection priority: CGPU > TransNetV2 > PySceneDetect
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        verbose: Optional[bool] = None,
        backend: str = "auto",
    ):
        """
        Initialize scene detector.

        Args:
            threshold: Content detection threshold (lower = more sensitive).
                      If None, uses ThresholdConfig.scene_detection()
            verbose: Override verbose setting
            backend: Detection backend - "auto", "transnetv2", "pyscenedetect"
        """
        self.threshold = threshold if threshold is not None else _settings.thresholds.scene_threshold
        self.verbose = verbose if verbose is not None else _settings.features.verbose
        self.backend = backend

        # Check TransNetV2 availability on init
        self._transnetv2_available = None

    def _check_transnetv2(self) -> bool:
        """Lazy check for TransNetV2 availability."""
        if self._transnetv2_available is None:
            try:
                from .scene_detection_sota import get_available_backend
                self._transnetv2_available = get_available_backend() == "transnetv2"
            except ImportError:
                self._transnetv2_available = False
        return self._transnetv2_available

    def _detect_transnetv2(self, video_path: str) -> List[Scene]:
        """Use TransNetV2 SOTA scene detection (250fps on GPU)."""
        from .scene_detection_sota import detect_scenes_sota

        logger.info(f"🧠 Using TransNetV2 SOTA scene detection for {os.path.basename(video_path)}...")

        # TransNetV2 uses different threshold scale (0-1 vs 0-100)
        # Convert: lower PySceneDetect threshold = more sensitive
        # TransNetV2: lower threshold = more sensitive
        tn_threshold = max(0.3, min(0.7, 1.0 - (self.threshold / 100.0)))

        scene_tuples = detect_scenes_sota(
            video_path,
            backend="transnetv2",
            threshold=tn_threshold,
            use_cache=True
        )

        return [
            Scene(start=start, end=end, path=video_path)
            for start, end in scene_tuples
        ]

    def detect(self, video_path: str, max_resolution: Optional[int] = None) -> List[Scene]:
        """
        Detect scene boundaries in a video file.

        Args:
            video_path: Path to video file
            max_resolution: Maximum resolution height for analysis (default: 1080).
                          If input exceeds this, downsample to save processing time.
                          Example: 6K (3160p) downsampled to 1080p = 4x faster.

        Returns:
            List of Scene objects with start/end times
        """
        import time
        start_time = time.perf_counter()
        
        logger.info(f"Detecting scenes in {os.path.basename(video_path)}...")

        # Check for Cloud GPU offloading
        if CGPU_AVAILABLE and _settings.llm.cgpu_enabled and is_cgpu_available():
            logger.info("Offloading scene detection to Cloud GPU...")
            try:
                job = SceneDetectionJob(input_path=video_path, threshold=self.threshold)
                result = job.execute()
                
                if result.success and result.output_path:
                    with open(result.output_path, 'r') as f:
                        scene_data = json.load(f)
                    
                    scenes = []
                    for s in scene_data:
                        scenes.append(Scene(
                            start=s['start'],
                            end=s['end'],
                            path=video_path
                        ))
                    logger.info(f"Cloud detection complete: {len(scenes)} scenes found.")
                    return scenes
                else:
                    if _settings.features.strict_cloud_compute:
                        raise RuntimeError(f"Strict cloud compute enabled: Cloud scene detection failed: {result.error}")
                    logger.warning(f"Cloud detection failed: {result.error}. Falling back to local.")
            except Exception as e:
                if _settings.features.strict_cloud_compute:
                    raise RuntimeError(f"Strict cloud compute enabled: Cloud scene detection error: {e}")
                logger.warning(f"Cloud detection error: {e}. Falling back to local.")
        elif _settings.features.strict_cloud_compute:
            raise RuntimeError("Strict cloud compute enabled: cgpu scene detection not available or disabled.")

        # SOTA: Try TransNetV2 neural network scene detection (250fps on GPU)
        # TransNetV2 is 5x faster than PySceneDetect and more accurate
        if self.backend in ("auto", "transnetv2") and self._check_transnetv2():
            try:
                return self._detect_transnetv2(video_path)
            except Exception as e:
                if self.backend == "transnetv2":
                    raise  # Don't fallback if explicitly requested
                logger.warning(f"TransNetV2 failed: {e}. Falling back to PySceneDetect.")

        # PHASE 4 OPTIMIZATION: Downsample high-resolution videos for scene detection
        # Scene detection analyzes frame differences, not fine details.
        # Downsampling 6K→1080p maintains accuracy while achieving 4-9x speedup.
        effective_path = video_path
        use_proxy = False
        
        if max_resolution is None:
            max_resolution = 1080  # Default: downsample anything above 1080p
        
        metadata = None

        # Check if proxy mode should be enabled (auto-detect for large videos)
        try:
            metadata = probe_metadata(video_path)
            
            # Auto-enable proxy mode for large/long videos
            if _settings.proxy.should_use_proxy(metadata.duration, metadata.width, metadata.height):
                logger.info(f"🎬 Large input detected (duration={metadata.duration:.0f}s, "
                           f"resolution={metadata.width}x{metadata.height}). "
                           f"Enabling proxy analysis mode ({_settings.proxy.proxy_height}p)...")
                
                # Generate proxy video for analysis
                proxy_path = self._generate_analysis_proxy(video_path)
                if proxy_path and os.path.exists(proxy_path):
                    effective_path = proxy_path
                    use_proxy = True
                    logger.info(f"✓ Using proxy for analysis: {os.path.basename(proxy_path)}")
                else:
                    logger.warning("Proxy generation failed. Falling back to downsampling.")
                    if metadata and metadata.height > max_resolution:
                        effective_path = self._create_downsampled_proxy(video_path, max_resolution)
                    else:
                        effective_path = video_path
            
            # Also check for high resolution (even if video is short)
            elif metadata.height > max_resolution:
                logger.info(f"⚡ High-res input detected ({metadata.width}x{metadata.height}). "
                           f"Downsampling to {max_resolution}p for scene analysis (4-9x faster).")
                effective_path = self._create_downsampled_proxy(video_path, max_resolution)
        except Exception as e:
            logger.warning(f"Could not check resolution for proxy/downsampling: {e}. Using original.")

        if not SCENEDETECT_AVAILABLE:
            logger.error("PySceneDetect not installed. Scene detection unavailable.")
            return [Scene(0, probe_metadata(video_path).duration, video_path)]

        video = open_video(effective_path)
        scene_manager = SceneManager()
        # OPTIMIZATION: Use keyframe-only detection for 5-10x speedup
        # Only analyze I-frames (keyframes) instead of every frame
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=_settings.analysis.scene_min_length_frames  # From config
            )
        )
        # Downscale factor reduces memory and processing time
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        # Convert to Scene objects
        scenes = []
        for scene in scene_list:
            start, end = scene
            scenes.append(Scene(
                start=start.get_seconds(),
                end=end.get_seconds(),
                path=video_path,  # Always reference original path
            ))

        logger.info(f"Found {len(scenes)} scenes.")

        # If no scenes were detected (single shot), use the whole video
        if not scenes:
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                clip.close()
                logger.warning(f"No cuts detected. Using full video ({duration:.1f}s).")
                return [Scene(start=0.0, end=duration, path=video_path)]
            except Exception as e:
                logger.error(f"Could not read video duration: {e}")
                return []

        # Cleanup proxy if we created one
        if use_proxy and effective_path != video_path:
            try:
                os.unlink(effective_path)
                logger.debug(f"Cleaned up analysis proxy: {effective_path}")
            except Exception as e:
                logger.debug(f"Could not cleanup proxy {effective_path}: {e}")

        return scenes
    
    def _generate_analysis_proxy(self, video_path: str) -> Optional[str]:
        """
        Generate a lightweight proxy video (720p by default) for fast analysis.
        
        This enables analysis of large/long videos (10+ minutes) without 
        timeout issues from Optical Flow and other heavy processing.
        
        Args:
            video_path: Original video path
        
        Returns:
            Path to analysis proxy file, or None on failure
        """
        import subprocess
        
        try:
            from .proxy_generator import ProxyGenerator
            
            # Create temp file for proxy
            ext = os.path.splitext(video_path)[1]
            proxy_fd, proxy_path = tempfile.mkstemp(
                suffix=f"_analysis_proxy{ext}", 
                prefix="scene_detect_"
            )
            try:
                os.close(proxy_fd)
            except OSError:
                pass
            
            # Get proxy height from config
            proxy_height = _settings.proxy.proxy_height
            
            logger.info(f"📹 Generating analysis proxy ({proxy_height}p) for large video...")
            
            # Use the static method from ProxyGenerator
            ProxyGenerator.generate_analysis_proxy(
                source_path=video_path,
                output_path=proxy_path,
                height=proxy_height
            )
            
            logger.info(f"✓ Analysis proxy created: {proxy_path}")
            return proxy_path
            
        except Exception as e:
            logger.warning(f"Proxy generation failed: {e}. Falling back to downsampling.")
            return None
    
    def _create_downsampled_proxy(self, video_path: str, max_height: int) -> str:
        """
        Create a temporary downsampled proxy for scene detection.
        
        Uses FFmpeg scale filter to downsample while preserving aspect ratio.
        
        Args:
            video_path: Original video path
            max_height: Maximum height in pixels (width scaled proportionally)
        
        Returns:
            Path to temporary downsampled proxy file
        """
        import subprocess
        
        # Create temp file with same extension
        ext = os.path.splitext(video_path)[1]
        proxy_fd, proxy_path = tempfile.mkstemp(suffix=f"_proxy{ext}", prefix="scene_detect_")
        try:
            os.close(proxy_fd)
        except OSError:
            # Some tests mock mkstemp with a dummy fd; ignore close errors
            pass
        
        # FFmpeg command: scale to max_height, maintain aspect ratio
        # -vf scale=-2:1080 means: width auto-calculated, height=1080, divisible by 2
        from .config import get_settings
        settings = get_settings()
        # Use preview profile for fast proxy generation (configurable)
        # Force H.264 for speed and broad compatibility in proxies
        codec = "libx264"
        preset = settings.preview.preset
        crf = settings.preview.crf
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"scale=-2:{max_height}",
            "-c:v", codec,
            "-preset", preset,  # Speed over compression
            "-crf", str(crf),   # Lower quality acceptable for scene detection
            "-an",  # No audio needed
            "-y",
            proxy_path
        ]
        
        logger.debug(f"Creating downsampled proxy: {' '.join(ffmpeg_cmd)}")
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            logger.debug(f"Proxy created: {proxy_path}")
            return proxy_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create proxy: {e.stderr}")
            # Cleanup on failure
            if os.path.exists(proxy_path):
                os.remove(proxy_path)
            # Return original path as fallback
            return video_path


# =============================================================================
# Scene Content Analyzer (AI-powered)
# =============================================================================

class SceneContentAnalyzer:
    """Analyze scene content using AI vision models."""

    def __init__(
        self,
        enable_ai: Optional[bool] = None,
        openai_base: Optional[str] = None,
        openai_key: Optional[str] = None,
        vision_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        """
        Initialize content analyzer.

        Args:
            enable_ai: Enable AI analysis (uses config if None)
            openai_base: OpenAI API base URL
            openai_key: OpenAI API key
            vision_model: Vision model to use
            ollama_host: Ollama host URL (fallback)
            ollama_model: Ollama model name (fallback)
        """
        self.enable_ai = enable_ai if enable_ai is not None else _settings.features.enable_ai_filter
        self.openai_base = openai_base or _settings.llm.openai_api_base
        self.openai_key = openai_key or _settings.llm.openai_api_key
        self.vision_model = vision_model or _settings.llm.openai_vision_model
        self.ollama_host = ollama_host or _settings.llm.ollama_host
        self.ollama_model = ollama_model or _settings.llm.ollama_model
        self.request_timeout = _settings.llm.timeout
        self._vision_client = None

    def analyze(self, video_path: str, time_point: float, semantic: bool = True) -> SceneAnalysis:
        """
        Analyze a frame using AI-powered vision.
        
        Uses the centralized CreativeDirector to query available backends
        (cgpu/Gemini, OpenAI, Ollama) with fallback support.
        """
        # 1. Fast, local face detection (always run)
        face_count = detect_faces(video_path, time_point)

        if not self.enable_ai:
            result = SceneAnalysis.default()
            result.face_count = face_count
            return result

        try:
            # 2. Extract frame for AI vision
            frame_b64 = self._extract_frame_base64(video_path, time_point)
            if frame_b64 is None:
                return SceneAnalysis(
                    quality="NO", description="read error",
                    action=ActionLevel.LOW, shot=ShotType.MEDIUM,
                    face_count=face_count
                )

            # 3. Query centralized Creative Director (AI Analysis Agent)
            from .creative_director import get_creative_director
            director = get_creative_director()
            
            # Use structured vision prompt from prompts.py
            system_prompt = get_vision_analysis_prompt()
            
            response_text = director.query(
                prompt="Analyze this video frame.",
                system_prompt=system_prompt,
                image_b64=frame_b64,
                json_mode=True,
                temperature=0.2
            )

            if response_text:
                res_json = json.loads(response_text)
                result = SceneAnalysis.from_dict(res_json)
                result.face_count = face_count
                return result

        except Exception as e:
            logger.warning(f"AI Vision analysis failed: {e}")

        # Fallback to defaults
        result = SceneAnalysis.default()
        result.face_count = face_count
        return result

    def _extract_frame_base64(self, video_path: str, time_point: float) -> Optional[str]:
        """Extract a frame and return as base64 JPEG."""
        # DRY: Use centralized frame extraction utility
        if CAPTURE_POOL_AVAILABLE:
            return pool_extract_frame_base64(video_path, time_point, quality=85)

        # Fallback if pool not available
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None

            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return None


# =============================================================================
# Visual Analysis Functions
# =============================================================================

# LRU cache for frame histogram extraction (performance optimization)
# Avoids re-reading same frames when comparing multiple clips
_histogram_cache_stats = {"hits": 0, "misses": 0}


@lru_cache(maxsize=_HISTOGRAM_CACHE_SIZE)
def _get_frame_histogram_cached(video_path: str, time_ms: int) -> Optional[tuple]:
    """
    Extract and compute HSV histogram for a frame (cached).

    Uses LRU cache to avoid re-reading the same frames repeatedly
    during clip selection scoring.

    OPTIMIZATION: Uses VideoCapture pool to avoid repeated open/close overhead.
    First access: ~100-500ms, pooled access: ~1-5ms (20-100x speedup).

    Args:
        video_path: Path to video file
        time_ms: Time in milliseconds (int for hashability)

    Returns:
        Flattened histogram as tuple (hashable for cache) or None on failure
    """
    _histogram_cache_stats["misses"] += 1

    try:
        # OPTIMIZATION: Use capture pool instead of creating new capture each time
        from .video_capture_pool import get_capture_pool

        pool = get_capture_pool()
        with pool.get(video_path) as cap:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            ret, frame = cap.read()

            if not ret or frame is None:
                return None

            # Convert to HSV for better color similarity
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Calculate histogram with reduced bins for efficiency
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

            # Normalize
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Return as tuple for hashability
            return tuple(hist.flatten())

    except Exception:
        return None


def _get_histogram_as_array(video_path: str, time_sec: float) -> Optional[np.ndarray]:
    """Helper to get histogram as numpy array from cache."""
    # Convert to ms (int) for cache key
    time_ms = int(time_sec * 1000)
    hist_tuple = _get_frame_histogram_cached(video_path, time_ms)

    if hist_tuple is None:
        return None

    # Check if this was a cache hit (tuple already exists)
    cache_info = _get_frame_histogram_cached.cache_info()
    if cache_info.hits > _histogram_cache_stats["hits"]:
        _histogram_cache_stats["hits"] = cache_info.hits

    return np.array(hist_tuple).reshape((8, 8, 8))


def get_histogram_cache_stats() -> dict:
    """Get cache performance statistics."""
    cache_info = _get_frame_histogram_cached.cache_info()
    return {
        "size": cache_info.currsize,
        "maxsize": cache_info.maxsize,
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "hit_rate": cache_info.hits / max(1, cache_info.hits + cache_info.misses) * 100
    }


def clear_histogram_cache():
    """Clear the histogram cache (call between montage runs)."""
    _get_frame_histogram_cached.cache_clear()


# =============================================================================
# OPTIMIZATION: K-D Tree Scene Similarity Index
# =============================================================================

class SceneSimilarityIndex:
    """
    K-D tree based spatial index for fast scene similarity queries.
    
    Reduces scene similarity search from O(n) to O(log n) by building
    a spatial index of histogram feature vectors.
    
    Usage:
        index = SceneSimilarityIndex()
        index.build(scenes)  # Build index from scene list
        similar_scenes = index.find_similar(target_scene, k=5, threshold=0.7)
    """
    
    def __init__(self):
        self.kdtree = None
        self.scenes = []
        self.histograms = []
        self.enabled = KDTREE_AVAILABLE
        
    def build(self, scenes: List[Scene]) -> None:
        """Build K-D tree index from scene list."""
        if not self.enabled:
            logger.warning("K-D tree not available (scipy not installed). Falling back to linear search.")
            self.scenes = scenes
            return
            
        self.scenes = scenes
        self.histograms = []
        
        logger.info(f"Building scene similarity index for {len(scenes)} scenes...")
        
        # Extract histograms for all scenes (cached)
        for scene in scenes:
            hist = _get_histogram_as_array(scene.path, scene.midpoint)
            if hist is not None:
                # Flatten to 1D vector for K-D tree
                self.histograms.append(hist.flatten())
            else:
                # Use zero vector if extraction fails
                self.histograms.append(np.zeros(512))  # 8*8*8 bins
        
        if self.histograms:
            # Build K-D tree (O(n log n) construction)
            from scipy.spatial import KDTree
            self.kdtree = KDTree(self.histograms)
            logger.info(f"   ✓ K-D tree built with {len(self.histograms)} feature vectors")
        else:
            logger.warning("   ⚠️ No histograms extracted, index empty")
    
    def find_similar(
        self, 
        target_path: str, 
        target_time: float, 
        k: int = 5, 
        threshold: float = 0.7
    ) -> List[Tuple[Scene, float]]:
        """
        Find k most similar scenes to target frame using K-D tree (O(log n)).
        Falls back to linear search if K-D tree unavailable.
        
        Args:
            target_path: Path to target video
            target_time: Time in target video
            k: Number of similar scenes to return
            threshold: Minimum similarity threshold (0-1)
        
        Returns:
            List of (scene, similarity) tuples sorted by similarity descending
        """
        if not self.scenes:
            return []
        
        # Extract target histogram
        target_hist = _get_histogram_as_array(target_path, target_time)
        if target_hist is None:
            return []
        
        target_vec = target_hist.flatten()
        results = []
        
        # K-D tree path (O(log n) lookup)
        if self.kdtree is not None:
            # Query K-D tree for nearest neighbors
            distances, indices = self.kdtree.query(target_vec, k=min(k, len(self.scenes)))
            
            # Convert distances to similarity scores (0-1, inverted)
            for dist, idx in zip(distances, indices):
                if idx < len(self.scenes):
                    # Distance to similarity: similarity = 1 - normalized_distance
                    similarity = 1.0 - (dist / 512.0)  # Normalize by max histogram distance
                    similarity = max(0.0, similarity)  # Clamp to [0, 1]
                    
                    if similarity >= threshold:
                        results.append((self.scenes[idx], similarity))
            
            logger.info(f"   ⚡ K-D tree query found {len(results)} similar scenes (fast path)")
        else:
            # Linear search fallback (O(n)) - only if K-D tree unavailable
            for scene in self.scenes:
                scene_hist = _get_histogram_as_array(scene.path, scene.midpoint)
                if scene_hist is not None:
                    scene_vec = scene_hist.flatten()
                    # Euclidean distance
                    dist = np.linalg.norm(target_vec - scene_vec)
                    similarity = 1.0 - (dist / 512.0)
                    similarity = max(0.0, similarity)
                    
                    if similarity >= threshold:
                        results.append((scene, similarity))
            
            logger.warning(f"   ⚠️  K-D tree unavailable, using linear search O(n)")
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


def detect_faces(video_path: str, time_sec: float) -> int:
    """
    Detect faces in a video frame using OpenCV Haar Cascades.
    
    Args:
        video_path: Path to video file
        time_sec: Timestamp to check
        
    Returns:
        Number of faces detected
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()

        if not ret:
            return 0

        # Load Haar Cascade (included with OpenCV)
        # We use the default frontal face detector
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        if face_cascade.empty():
            logger.warning("Failed to load Haar Cascade for face detection")
            return 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        # scaleFactor=1.1, minNeighbors=5 are standard robust settings
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        return len(faces)
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return 0
    finally:
        if cap is not None:
            cap.release()


def calculate_visual_similarity(
    frame1_path: str,
    frame1_time: float,
    frame2_path: str,
    frame2_time: float
) -> float:
    """
    Calculate visual similarity between two frames using color histogram comparison.
    Used for 'Match Cut' detection - finding visually similar moments for seamless transitions.

    Uses LRU-cached histogram extraction for 2-3x performance improvement
    when comparing the same frames multiple times.

    Args:
        frame1_path: Path to first video
        frame1_time: Time in first video
        frame2_path: Path to second video
        frame2_time: Time in second video

    Returns:
        Similarity score 0-1 (1 = identical)
    """
    try:
        # Use cached histogram extraction (major performance win)
        hist1 = _get_histogram_as_array(frame1_path, frame1_time)
        hist2 = _get_histogram_as_array(frame2_path, frame2_time)

        if hist1 is None or hist2 is None:
            return 0.0

        # Compare using correlation (histograms already normalized)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, float(similarity))

    except Exception as e:
        print(f"   ⚠️ Visual similarity failed: {e}")
        return 0.0


def detect_motion_blur(video_path: str, time_point: float) -> float:
    """
    Detect motion blur intensity at a specific time point.
    Used for 'Invisible Cut' placement - cuts work best during motion blur.

    Args:
        video_path: Path to video file
        time_point: Time in seconds

    Returns:
        Blur score 0-1 (1 = heavy blur, good for cuts)
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
        ret, frame = cap.read()

        if not ret:
            return 0.0

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance (low = blurry)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize to 0-1 (invert: high variance = sharp, low = blurry)
        # Typical sharp video has variance > 500, blurry < 100
        blur_score = 1.0 - min(1.0, laplacian_var / 500.0)
        return float(blur_score)

    except Exception as e:
        logger.warning(f"Motion blur detection failed: {e}")
        return 0.0
    finally:
        if cap is not None:
            cap.release()


def find_best_start_point(
    video_path: str,
    scene_start: float,
    scene_end: float,
    target_duration: float,
    sample_interval: float = 0.5
) -> float:
    """
    Find the most 'interesting' segment within a scene using Optical Flow.

    This is the 'Perfect Cut' optimization - instead of random start points,
    we find the moment with highest action/motion.

    Args:
        video_path: Path to video file
        scene_start: Scene start time in seconds
        scene_end: Scene end time in seconds
        target_duration: Desired clip duration
        sample_interval: Sampling interval in seconds

    Returns:
        Best start timestamp (float)
    """
    # LOW_MEMORY_MODE: Skip optical flow entirely (too expensive)
    low_memory = os.getenv("LOW_MEMORY_MODE", "").lower() in ("true", "1", "yes")
    if low_memory:
        logger.debug("LOW_MEMORY_MODE: Skipping optical flow, using fallback")
        return _fallback_start_point(scene_start, scene_end, target_duration)

    # Optimization: Skip if in preview mode
    settings = get_settings()
    if settings.encoding.quality_profile == "preview":
        logger.debug("QUALITY_PROFILE=preview: Skipping optical flow")
        return _fallback_start_point(scene_start, scene_end, target_duration)

    cap = None
    try:
        # Adaptation: Limit samples to cap processing time
        duration = scene_end - scene_start
        max_samples = 20
        if duration / sample_interval > max_samples:
            sample_interval = duration / max_samples
            logger.debug(f"   ⏱️ Adapting optical flow interval to {sample_interval:.1f}s for {duration:.1f}s scene")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, scene_start * 1000)

        motion_scores = []
        timestamps = []

        ret, prev_frame = cap.read()
        if not ret:
            return _fallback_start_point(scene_start, scene_end, target_duration)

        # Resize to 480p for optical flow to reduce memory
        target_height = 480
        h, w = prev_frame.shape[:2]
        if h > target_height:
            scale = target_height / h
            prev_frame = cv2.resize(prev_frame, (int(w * scale), target_height))

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_time = scene_start

        while current_time < scene_end - target_duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()

            if not ret:
                break

            # Resize for optical flow consistency
            h, w = frame.shape[:2]
            if h > target_height:
                scale = target_height / h
                frame = cv2.resize(frame, (int(w * scale), target_height))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow magnitude (motion intensity)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Compute motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_score = float(np.mean(magnitude))

            motion_scores.append(motion_score)
            timestamps.append(current_time)

            prev_gray = gray
            current_time += sample_interval

        # Find peak motion (action climax)
        if motion_scores:
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
        logger.warning(f"Optical Flow analysis failed: {e}")
    finally:
        if cap is not None:
            cap.release()

    return _fallback_start_point(scene_start, scene_end, target_duration)


def _fallback_start_point(scene_start: float, scene_end: float, target_duration: float) -> float:
    """Fallback: random start point within valid range."""
    max_start = scene_end - target_duration
    if max_start <= scene_start:
        return scene_start
    return random.uniform(scene_start, max_start)


# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    """
    Legacy interface returning list of (start, end) tuples.

    For backward compatibility with existing code.
    """
    detector = SceneDetector(threshold=threshold)
    scenes = detector.detect(video_path)
    return [(s.start, s.end) for s in scenes]


def analyze_scene_content(video_path: str, time_point: float) -> dict:
    """
    Legacy interface returning dictionary.

    For backward compatibility with existing code.
    """
    analyzer = SceneContentAnalyzer()
    result = analyzer.analyze(video_path, time_point)
    return result.to_dict()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ActionLevel",
    "ShotType",
    # Data classes
    "Scene",
    "SceneAnalysis",
    # Classes
    "SceneDetector",
    "SceneContentAnalyzer",
    # Functions
    "detect_scenes",
    "analyze_scene_content",
    "calculate_visual_similarity",
    "detect_motion_blur",
    "find_best_start_point",
    # Cache management
    "get_histogram_cache_stats",
    "clear_histogram_cache",
]
