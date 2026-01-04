"""
Scene Analysis Module for Montage AI

Provides scene detection, content analysis, and visual similarity calculation.
Uses PySceneDetect for scene detection and OpenCV for visual analysis.

Usage:
    from montage_ai.scene_analysis import SceneDetector, detect_scenes, analyze_scene_content

    detector = SceneDetector(threshold=30.0)
    scenes = detector.detect(video_path)
"""

import os
import json
import random
import base64
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import requests

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from .config import get_settings
from .logger import logger
from .moviepy_compat import VideoFileClip

# Import cgpu jobs for offloading
try:
    from .cgpu_jobs import SceneDetectionJob
    from .cgpu_utils import is_cgpu_available
    CGPU_AVAILABLE = True
except ImportError:
    CGPU_AVAILABLE = False

_settings = get_settings()


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ActionLevel(Enum):
    """Action/motion intensity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ShotType(Enum):
    """Camera shot type."""
    CLOSE = "close"
    MEDIUM = "medium"
    WIDE = "wide"


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
        }


    @property
    def has_semantic_data(self) -> bool:
        """Check if semantic analysis data is present."""
        return bool(self.tags or self.caption or self.objects)


# =============================================================================
# Scene Detector Class
# =============================================================================

class SceneDetector:
    """Detect scene boundaries in video files using PySceneDetect."""

    def __init__(self, threshold: float = 30.0, verbose: Optional[bool] = None):
        """
        Initialize scene detector.

        Args:
            threshold: Content detection threshold (lower = more sensitive)
            verbose: Override verbose setting
        """
        self.threshold = threshold
        self.verbose = verbose if verbose is not None else _settings.features.verbose

    def detect(self, video_path: str) -> List[Scene]:
        """
        Detect scene boundaries in a video file.

        Args:
            video_path: Path to video file

        Returns:
            List of Scene objects with start/end times
        """
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

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        # Convert to Scene objects
        scenes = []
        for scene in scene_list:
            start, end = scene
            scenes.append(Scene(
                start=start.get_seconds(),
                end=end.get_seconds(),
                path=video_path,
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

        return scenes


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
        Extract a frame and analyze it with AI.

        Backend priority:
        1. cgpu/Gemini (if CGPU_ENABLED=true) - fast, multimodal
        2. OpenAI-compatible API
        3. Ollama (local fallback)
        4. Default (no AI)

        Args:
            video_path: Path to video file
            time_point: Time in seconds to analyze
            semantic: If True, extract semantic tags (default: True)

        Returns:
            SceneAnalysis with quality, description, action, shot, and semantic fields
        """
        # Always run basic CV face detection (fast, local)
        face_count = detect_faces(video_path, time_point)

        if not self.enable_ai:
            result = SceneAnalysis.default()
            result.face_count = face_count
            return result

        try:
            # Extract frame
            frame_b64 = self._extract_frame_base64(video_path, time_point)
            if frame_b64 is None:
                result = SceneAnalysis(
                    quality="NO",
                    description="read error",
                    action=ActionLevel.LOW,
                    shot=ShotType.MEDIUM,
                    face_count=face_count
                )
                return result

            # Priority 1: cgpu/Gemini (fast, multimodal, cheap)
            if semantic:
                result = self._analyze_cgpu(frame_b64)
                if result:
                    result.face_count = face_count
                    return result

            # Priority 2: OpenAI-compatible API with semantic prompt
            if self.openai_base and self.vision_model:
                result = self._analyze_openai_semantic(frame_b64) if semantic else self._analyze_openai(frame_b64)
                if result:
                    result.face_count = face_count
                    return result

            # Priority 3: Ollama fallback
            result = self._analyze_ollama_semantic(frame_b64) if semantic else self._analyze_ollama(frame_b64)
            if result:
                result.face_count = face_count
                return result

        except Exception as e:
            logger.warning(f"AI Analysis failed: {e}")

        result = SceneAnalysis.default()
        result.face_count = face_count
        return result

    def _extract_frame_base64(self, video_path: str, time_point: float) -> Optional[str]:
        """Extract a frame and return as base64 JPEG."""
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

    def _analyze_openai(self, frame_b64: str) -> Optional[SceneAnalysis]:
        """Analyze frame using OpenAI-compatible API."""
        try:
            if self._vision_client is None:
                from openai import OpenAI
                self._vision_client = OpenAI(
                    base_url=self.openai_base,
                    api_key=self.openai_key
                )

            prompt = (
                "Analyze this image for a video editor. Return a JSON object with these keys: "
                "quality (YES/NO), description (5 words), action (low/medium/high), shot (close/medium/wide). "
                'Example: {"quality": "YES", "description": "Man running in park", "action": "high", "shot": "wide"}'
            )

            response = self._vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
                    ]
                }],
                max_tokens=100,
                temperature=0.2,
                timeout=self.request_timeout
            )

            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                # Clean markdown wrapping
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]

                res_json = json.loads(content.strip())
                return SceneAnalysis.from_dict(res_json)

        except Exception as e:
            logger.warning(f"OpenAI Vision API failed, falling back to Ollama: {e}")

        return None

    def _analyze_ollama(self, frame_b64: str) -> Optional[SceneAnalysis]:
        """Analyze frame using Ollama."""
        if not self.ollama_host:
            return None
        try:
            prompt = (
                "Analyze this image for a video editor. Return a JSON object with these keys: "
                "quality (YES/NO), description (5 words), action (low/medium/high), shot (close/medium/wide). "
                'Example: {"quality": "YES", "description": "Man running in park", "action": "high", "shot": "wide"}'
            )

            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "images": [frame_b64],
                "stream": False,
                "format": "json"
            }

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                try:
                    res_json = json.loads(response.json().get("response", "{}"))
                    return SceneAnalysis.from_dict(res_json)
                except json.JSONDecodeError:
                    text = response.json().get("response", "")
                    return SceneAnalysis(
                        quality="YES",
                        description=text[:50],
                        action=ActionLevel.MEDIUM,
                        shot=ShotType.MEDIUM
                    )

        except Exception:
            pass

        return None

    # -------------------------------------------------------------------------
    # Semantic Analysis Methods (Phase 2: Semantic Storytelling)
    # -------------------------------------------------------------------------

    def _build_semantic_prompt(self) -> str:
        """Build prompt for semantic scene analysis with free-form tags."""
        return '''Analyze this video frame for intelligent video editing. Return a JSON object with these keys:

{
  "quality": "YES" or "NO" (is the frame usable, not blurry or corrupted?),
  "description": "5-word scene summary",
  "action": "low" | "medium" | "high" (motion intensity),
  "shot": "close" | "medium" | "wide" (camera shot type),
  "tags": ["tag1", "tag2", ...],
  "caption": "One detailed sentence describing what is happening in this frame",
  "objects": ["object1", "object2", ...],
  "mood": "calm" | "energetic" | "dramatic" | "playful" | "tense" | "peaceful" | "mysterious",
  "setting": "indoor" | "outdoor" | "beach" | "city" | "nature" | "studio" | "street" | "home"
}

TAGS should include 3-8 relevant descriptive keywords covering:
- Visual elements (colors, lighting, composition)
- Scene content (what is happening)
- Emotional/tonal descriptors
- Motion characteristics (static, moving, fast, slow)
- Searchable keywords for content matching

Example:
{"quality": "YES", "description": "Woman dancing on beach sunset", "action": "high", "shot": "wide", "tags": ["beach", "sunset", "dancing", "golden hour", "silhouette", "waves", "energetic", "freedom"], "caption": "A woman dances freely on a sandy beach during a vibrant sunset with waves in the background", "objects": ["woman", "beach", "ocean", "sun"], "mood": "energetic", "setting": "beach"}

Return ONLY valid JSON, no markdown code blocks.'''

    def _clean_json_response(self, content: str) -> str:
        """Clean JSON response by removing markdown formatting."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def _analyze_cgpu(self, frame_b64: str) -> Optional[SceneAnalysis]:
        """
        Analyze frame using cgpu/Gemini Vision API.

        Uses the cgpu cloud GPU service with Gemini Flash for fast, cheap vision analysis.
        This is the primary backend for semantic scene analysis.
        """
        try:
            from .cgpu_utils import CGPU_ENABLED, CGPU_HOST, CGPU_PORT, CGPU_MODEL

            if not CGPU_ENABLED:
                return None
            
            # Skip local API key check if using cgpu-server (it handles auth)
            # if not (
            #     os.environ.get("GEMINI_API_KEY")
            #     or os.environ.get("GOOGLE_API_KEY")
            #     or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
            #     or os.environ.get("GOOGLE_GENAI_USE_GCA")
            # ):
            #     return None

            from openai import OpenAI

            # cgpu serve exposes OpenAI-compatible Responses API at /v1
            client = OpenAI(
                base_url=f"http://{CGPU_HOST}:{CGPU_PORT}/v1",
                api_key="unused"
            )

            response = client.responses.create(
                model=CGPU_MODEL,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self._build_semantic_prompt()},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{frame_b64}"}
                    ]
                }],
                max_output_tokens=400,
                temperature=0.3,
                timeout=self.request_timeout
            )

            content = getattr(response, "output_text", None)
            if not content and getattr(response, "output", None):
                for item in response.output:
                    for part in getattr(item, "content", []) or []:
                        if getattr(part, "type", "") == "output_text" and getattr(part, "text", None):
                            content = part.text
                            break
                    if content:
                        break

            if content:
                cleaned = self._clean_json_response(content)
                res_json = json.loads(cleaned)
                return SceneAnalysis.from_dict(res_json)

        except ImportError:
            pass  # cgpu_utils not available
        except Exception as e:
            print(f"   ⚠️ cgpu Vision failed: {e}")

        return None

    def _analyze_openai_semantic(self, frame_b64: str) -> Optional[SceneAnalysis]:
        """Analyze frame using OpenAI-compatible API with semantic prompt."""
        try:
            if self._vision_client is None:
                from openai import OpenAI
                self._vision_client = OpenAI(
                    base_url=self.openai_base,
                    api_key=self.openai_key
                )

            response = self._vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_semantic_prompt()},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
                    ]
                }],
                max_tokens=400,
                temperature=0.3,
                timeout=self.request_timeout
            )

            if response.choices and response.choices[0].message.content:
                content = self._clean_json_response(response.choices[0].message.content)
                res_json = json.loads(content)
                return SceneAnalysis.from_dict(res_json)

        except Exception as e:
            print(f"   ⚠️ OpenAI Vision (semantic) failed: {e}")

        return None

    def _analyze_ollama_semantic(self, frame_b64: str) -> Optional[SceneAnalysis]:
        """Analyze frame using Ollama with semantic prompt."""
        if not self.ollama_host:
            return None
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": self._build_semantic_prompt(),
                "images": [frame_b64],
                "stream": False,
                "format": "json"
            }

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                response_text = response.json().get("response", "{}")
                res_json = json.loads(response_text)
                return SceneAnalysis.from_dict(res_json)

        except Exception as e:
            print(f"   ⚠️ Ollama Vision (semantic) failed: {e}")

        return None


# =============================================================================
# Visual Analysis Functions
# =============================================================================

# LRU cache for frame histogram extraction (performance optimization)
# Avoids re-reading same frames when comparing multiple clips
_histogram_cache_stats = {"hits": 0, "misses": 0}


@lru_cache(maxsize=256)
def _get_frame_histogram_cached(video_path: str, time_ms: int) -> Optional[tuple]:
    """
    Extract and compute HSV histogram for a frame (cached).

    Uses LRU cache to avoid re-reading the same frames repeatedly
    during clip selection scoring.

    Args:
        video_path: Path to video file
        time_ms: Time in milliseconds (int for hashability)

    Returns:
        Flattened histogram as tuple (hashable for cache) or None on failure
    """
    _histogram_cache_stats["misses"] += 1

    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ret, frame = cap.read()
        cap.release()

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


def detect_faces(video_path: str, time_sec: float) -> int:
    """
    Detect faces in a video frame using OpenCV Haar Cascades.
    
    Args:
        video_path: Path to video file
        time_sec: Timestamp to check
        
    Returns:
        Number of faces detected
    """
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        cap.release()

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
        return float(blur_score)

    except Exception as e:
        print(f"   ⚠️ Motion blur detection failed: {e}")
        return 0.0


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
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, scene_start * 1000)

        motion_scores = []
        timestamps = []

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return _fallback_start_point(scene_start, scene_end, target_duration)

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
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
            motion_score = float(np.mean(magnitude))

            motion_scores.append(motion_score)
            timestamps.append(current_time)

            prev_gray = gray
            current_time += sample_interval

        cap.release()

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
        print(f"   ⚠️ Optical Flow analysis failed: {e}")

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
