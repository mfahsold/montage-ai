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
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import requests

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from .config import get_settings
from .moviepy_compat import VideoFileClip

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
    """AI analysis result for a scene/frame."""
    quality: str  # "YES" or "NO"
    description: str
    action: ActionLevel
    shot: ShotType

    @classmethod
    def default(cls) -> "SceneAnalysis":
        """Create default analysis (no AI filter)."""
        return cls(
            quality="YES",
            description="unknown",
            action=ActionLevel.MEDIUM,
            shot=ShotType.MEDIUM
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SceneAnalysis":
        """Create from dictionary response."""
        return cls(
            quality=data.get("quality", "YES"),
            description=data.get("description", "unknown"),
            action=ActionLevel(data.get("action", "medium").lower()),
            shot=ShotType(data.get("shot", "medium").lower())
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "quality": self.quality,
            "description": self.description,
            "action": self.action.value,
            "shot": self.shot.value
        }


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
        print(f"🎬 Detecting scenes in {os.path.basename(video_path)}...")

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

        print(f"   Found {len(scenes)} scenes.")

        # If no scenes were detected (single shot), use the whole video
        if not scenes:
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                clip.close()
                print(f"   ⚠️ No cuts detected. Using full video ({duration:.1f}s).")
                return [Scene(start=0.0, end=duration, path=video_path)]
            except Exception as e:
                print(f"   ❌ Could not read video duration: {e}")
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
        self._vision_client = None

    def analyze(self, video_path: str, time_point: float) -> SceneAnalysis:
        """
        Extract a frame and analyze it with AI.

        Args:
            video_path: Path to video file
            time_point: Time in seconds to analyze

        Returns:
            SceneAnalysis with quality, description, action, and shot type
        """
        if not self.enable_ai:
            return SceneAnalysis.default()

        try:
            # Extract frame
            frame_b64 = self._extract_frame_base64(video_path, time_point)
            if frame_b64 is None:
                return SceneAnalysis(
                    quality="NO",
                    description="read error",
                    action=ActionLevel.LOW,
                    shot=ShotType.MEDIUM
                )

            # Try OpenAI-compatible API first
            if self.openai_base and self.vision_model:
                result = self._analyze_openai(frame_b64)
                if result:
                    return result

            # Fallback to Ollama
            result = self._analyze_ollama(frame_b64)
            if result:
                return result

        except Exception as e:
            print(f"   ⚠️ AI Analysis failed: {e}")

        return SceneAnalysis.default()

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
                timeout=30
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
            print(f"   ⚠️ OpenAI Vision API failed, falling back to Ollama: {e}")

        return None

    def _analyze_ollama(self, frame_b64: str) -> Optional[SceneAnalysis]:
        """Analyze frame using Ollama."""
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
                timeout=30
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


# =============================================================================
# Visual Analysis Functions
# =============================================================================

def calculate_visual_similarity(
    frame1_path: str,
    frame1_time: float,
    frame2_path: str,
    frame2_time: float
) -> float:
    """
    Calculate visual similarity between two frames using color histogram comparison.
    Used for 'Match Cut' detection - finding visually similar moments for seamless transitions.

    Args:
        frame1_path: Path to first video
        frame1_time: Time in first video
        frame2_path: Path to second video
        frame2_time: Time in second video

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
]
