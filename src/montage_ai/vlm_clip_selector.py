"""
VLM-based Intelligent Clip Selection

Uses Vision-Language Models (VLMs) for intelligent, query-based clip selection.
Supports Qwen2.5-VL, VILA, and InternVideo2 backends.

Features:
- Query-based clip search ("find action scenes", "find emotional moments")
- Automatic highlight detection
- Scene-aware clip ranking
- Multi-modal understanding (visual + audio cues)

Research References:
- Qwen2.5-VL: https://huggingface.co/Qwen (1h+ video understanding)
- VILA/NVILA: https://github.com/NVlabs/VILA (Edge-deployable, Jetson support)
- InternVideo2.5: https://github.com/OpenGVLab/InternVideo (60+ tasks)
- F2C: Frames-to-Clips temporal coherent selection (+8.1% on Video-MME)

Usage:
    from montage_ai.vlm_clip_selector import VLMClipSelector

    selector = VLMClipSelector()

    # Query-based selection
    best_clips = selector.select_by_query(scenes, "find exciting action moments")

    # Automatic highlight detection
    highlights = selector.detect_highlights(video_path)
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache

import numpy as np

from .logger import logger
from .config import get_settings

# Lazy imports for optional VLM dependencies
_VLM_BACKEND: Optional[str] = None
_VLM_MODEL = None


@dataclass
class ClipScore:
    """A clip with VLM-based relevance scoring."""
    path: str
    start: float
    end: float
    score: float  # 0-1 relevance score
    caption: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass
class HighlightMoment:
    """A detected highlight moment in video."""
    timestamp: float
    duration: float
    confidence: float
    description: str
    category: str  # "action", "emotional", "scenic", "funny", etc.


def _check_vlm_available() -> Optional[str]:
    """Check which VLM backend is available."""
    global _VLM_BACKEND
    if _VLM_BACKEND is not None:
        return _VLM_BACKEND

    # Priority 1: Qwen2.5-VL (best for long videos, 1h+ support)
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        _VLM_BACKEND = "qwen2-vl"
        logger.debug("Qwen2.5-VL available (SOTA video understanding)")
        return _VLM_BACKEND
    except ImportError:
        pass

    # Priority 2: VILA (NVIDIA, good for Jetson deployment)
    try:
        from vila import VILA
        _VLM_BACKEND = "vila"
        logger.debug("VILA available (NVIDIA VLM)")
        return _VLM_BACKEND
    except ImportError:
        pass

    # Priority 3: InternVideo2 (60+ tasks)
    try:
        from internvideo2 import InternVideo2
        _VLM_BACKEND = "internvideo2"
        logger.debug("InternVideo2 available")
        return _VLM_BACKEND
    except ImportError:
        pass

    # Priority 4: LLaVA (widely available)
    try:
        from llava.model import LlavaLlamaForCausalLM
        _VLM_BACKEND = "llava"
        logger.debug("LLaVA available")
        return _VLM_BACKEND
    except ImportError:
        pass

    _VLM_BACKEND = None
    logger.debug("No VLM backend available")
    return None


def _get_pytorch_device() -> str:
    """Get best available PyTorch device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class VLMClipSelector:
    """
    Vision-Language Model based intelligent clip selector.

    Uses VLMs for semantic understanding of video content to enable:
    - Query-based clip search
    - Automatic highlight detection
    - Scene categorization and tagging
    """

    def __init__(self, model_name: str = "auto", device: str = "auto"):
        """
        Initialize VLM clip selector.

        Args:
            model_name: "auto", "qwen2-vl", "vila", "internvideo2", or "llava"
            device: "auto", "cuda", "mps", or "cpu"
        """
        self.settings = get_settings()
        self.device = device if device != "auto" else _get_pytorch_device()

        # Select model
        if model_name == "auto":
            self.backend = _check_vlm_available()
        else:
            self.backend = model_name

        self.model = None
        self.processor = None

        if self.backend:
            logger.info(f"VLM Clip Selector initialized with {self.backend} on {self.device}")
        else:
            logger.warning("No VLM backend available - clip selection will use fallback scoring")

    def _load_model(self):
        """Lazy load the VLM model."""
        global _VLM_MODEL

        if self.model is not None:
            return

        if _VLM_MODEL is not None:
            self.model = _VLM_MODEL
            return

        if self.backend == "qwen2-vl":
            self._load_qwen2_vl()
        elif self.backend == "vila":
            self._load_vila()
        elif self.backend == "internvideo2":
            self._load_internvideo2()
        elif self.backend == "llava":
            self._load_llava()

        _VLM_MODEL = self.model

    def _load_qwen2_vl(self):
        """Load Qwen2.5-VL model."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        model_id = "Qwen/Qwen2-VL-7B-Instruct"

        logger.info(f"Loading Qwen2.5-VL ({model_id})...")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()
        logger.info("Qwen2.5-VL loaded successfully")

    def _load_vila(self):
        """Load VILA model (NVIDIA)."""
        from vila import VILA

        logger.info("Loading VILA...")
        self.model = VILA.from_pretrained("nvidia/VILA-7B")
        self.model.to(self.device)
        self.model.eval()
        logger.info("VILA loaded successfully")

    def _load_internvideo2(self):
        """Load InternVideo2 model."""
        from internvideo2 import InternVideo2

        logger.info("Loading InternVideo2...")
        self.model = InternVideo2.from_pretrained("OpenGVLab/InternVideo2-Stage2_1B-224p-f4")
        self.model.to(self.device)
        self.model.eval()
        logger.info("InternVideo2 loaded successfully")

    def _load_llava(self):
        """Load LLaVA model."""
        from llava.model import LlavaLlamaForCausalLM
        from transformers import AutoTokenizer
        import torch

        model_id = "liuhaotian/llava-v1.5-7b"

        logger.info(f"Loading LLaVA ({model_id})...")

        self.processor = AutoTokenizer.from_pretrained(model_id)
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.model.to(self.device)
        self.model.eval()
        logger.info("LLaVA loaded successfully")

    def _extract_frame(self, video_path: str, timestamp: float) -> np.ndarray:
        """Extract a single frame from video at timestamp."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()

        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def _extract_frames_for_clip(
        self,
        video_path: str,
        start: float,
        end: float,
        num_frames: int = 8
    ) -> List[np.ndarray]:
        """Extract evenly spaced frames from a clip."""
        duration = end - start
        if duration <= 0:
            return [self._extract_frame(video_path, start)]

        timestamps = [start + (i / (num_frames - 1)) * duration for i in range(num_frames)]
        return [self._extract_frame(video_path, t) for t in timestamps]

    def _score_clip_vlm(
        self,
        video_path: str,
        start: float,
        end: float,
        query: Optional[str] = None
    ) -> Tuple[float, str, List[str]]:
        """
        Score a clip using VLM.

        Returns:
            Tuple of (score, caption, tags)
        """
        if self.backend is None:
            return self._score_clip_fallback(video_path, start, end)

        self._load_model()

        # Extract frames
        frames = self._extract_frames_for_clip(video_path, start, end)

        # Build prompt
        if query:
            prompt = f"Rate how well this video clip matches the query '{query}' on a scale of 0-10. Also provide a brief description and relevant tags."
        else:
            prompt = "Rate the visual interest and engagement potential of this video clip on a scale of 0-10. Describe what's happening and provide relevant tags."

        try:
            if self.backend == "qwen2-vl":
                score, caption, tags = self._query_qwen2_vl(frames, prompt)
            elif self.backend == "vila":
                score, caption, tags = self._query_vila(frames, prompt)
            elif self.backend == "internvideo2":
                score, caption, tags = self._query_internvideo2(frames, prompt)
            else:
                score, caption, tags = self._score_clip_fallback(video_path, start, end)

            return score, caption, tags

        except Exception as e:
            logger.warning(f"VLM scoring failed: {e}")
            return self._score_clip_fallback(video_path, start, end)

    def _query_qwen2_vl(
        self,
        frames: List[np.ndarray],
        prompt: str
    ) -> Tuple[float, str, List[str]]:
        """Query Qwen2.5-VL model."""
        from PIL import Image
        import torch

        # Prepare images
        images = [Image.fromarray(f) for f in frames]

        # Build conversation
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images[:4]],  # Limit to 4 frames
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images[:4], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)

        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Parse response
        score, caption, tags = self._parse_vlm_response(response)
        return score, caption, tags

    def _query_vila(
        self,
        frames: List[np.ndarray],
        prompt: str
    ) -> Tuple[float, str, List[str]]:
        """Query VILA model."""
        # VILA API varies by version
        response = self.model.generate(frames, prompt, max_tokens=200)
        return self._parse_vlm_response(response)

    def _query_internvideo2(
        self,
        frames: List[np.ndarray],
        prompt: str
    ) -> Tuple[float, str, List[str]]:
        """Query InternVideo2 model."""
        response = self.model.generate(frames, prompt)
        return self._parse_vlm_response(response)

    def _parse_vlm_response(self, response: str) -> Tuple[float, str, List[str]]:
        """Parse VLM response to extract score, caption, and tags."""
        import re

        # Extract score (look for numbers 0-10)
        score_match = re.search(r'\b(\d+(?:\.\d+)?)\s*/?\s*10\b|\bscore[:\s]+(\d+(?:\.\d+)?)\b', response.lower())
        if score_match:
            score_str = score_match.group(1) or score_match.group(2)
            score = min(1.0, float(score_str) / 10.0)
        else:
            # Default to moderate score if not found
            score = 0.5

        # Extract tags (look for comma-separated words or hashtags)
        tags = []
        tag_match = re.search(r'tags?[:\s]+([^.]+)', response.lower())
        if tag_match:
            tags = [t.strip().strip('#') for t in tag_match.group(1).split(',') if t.strip()]

        # Caption is the first sentence or the response itself
        caption = response.split('.')[0].strip()
        if len(caption) > 200:
            caption = caption[:200] + "..."

        return score, caption, tags

    def _score_clip_fallback(
        self,
        video_path: str,
        start: float,
        end: float
    ) -> Tuple[float, str, List[str]]:
        """Fallback scoring without VLM (uses motion and contrast)."""
        import cv2

        frames = self._extract_frames_for_clip(video_path, start, end, num_frames=4)

        # Calculate motion score (frame differences)
        motion_scores = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
            motion_scores.append(np.mean(diff) / 255.0)

        motion_score = np.mean(motion_scores) if motion_scores else 0.0

        # Calculate contrast/interest score
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        contrast_scores = [np.std(f) / 128.0 for f in gray_frames]
        contrast_score = np.mean(contrast_scores)

        # Combined score
        score = 0.4 * motion_score + 0.6 * contrast_score
        score = min(1.0, max(0.0, score))

        return score, "Fallback scoring (no VLM)", ["motion", "visual"]

    def select_by_query(
        self,
        scenes: List[Dict[str, Any]],
        query: str,
        top_k: int = 10
    ) -> List[ClipScore]:
        """
        Select clips matching a natural language query.

        Args:
            scenes: List of scene dicts with 'path', 'start', 'end' keys
            query: Natural language query (e.g., "find action scenes")
            top_k: Number of top clips to return

        Returns:
            List of ClipScore objects sorted by relevance
        """
        logger.info(f"VLM query-based selection: '{query}' ({len(scenes)} candidates)")

        scored_clips = []

        for scene in scenes:
            score, caption, tags = self._score_clip_vlm(
                scene['path'],
                scene['start'],
                scene['end'],
                query=query
            )

            scored_clips.append(ClipScore(
                path=scene['path'],
                start=scene['start'],
                end=scene['end'],
                score=score,
                caption=caption,
                tags=tags
            ))

        # Sort by score descending
        scored_clips.sort(key=lambda c: c.score, reverse=True)

        logger.info(f"Top {top_k} clips selected (scores: {[f'{c.score:.2f}' for c in scored_clips[:top_k]]})")

        return scored_clips[:top_k]

    def detect_highlights(
        self,
        video_path: str,
        min_duration: float = 2.0,
        max_highlights: int = 10
    ) -> List[HighlightMoment]:
        """
        Automatically detect highlight moments in a video.

        Uses VLM to identify engaging/interesting moments.

        Args:
            video_path: Path to video file
            min_duration: Minimum highlight duration
            max_highlights: Maximum number of highlights to return

        Returns:
            List of HighlightMoment objects
        """
        import cv2

        logger.info(f"Detecting highlights in {os.path.basename(video_path)}...")

        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if duration <= 0:
            return []

        # Sample video at regular intervals
        sample_interval = max(2.0, duration / 50)  # ~50 samples max
        timestamps = np.arange(0, duration, sample_interval)

        highlights = []

        for t in timestamps:
            score, caption, tags = self._score_clip_vlm(
                video_path,
                t,
                min(t + min_duration, duration)
            )

            # Threshold for highlights
            if score >= 0.6:
                # Categorize based on tags
                category = "general"
                if any(tag in ["action", "fast", "movement", "sports"] for tag in tags):
                    category = "action"
                elif any(tag in ["emotion", "face", "expression", "reaction"] for tag in tags):
                    category = "emotional"
                elif any(tag in ["scenic", "landscape", "nature", "beautiful"] for tag in tags):
                    category = "scenic"
                elif any(tag in ["funny", "humor", "comedy", "laugh"] for tag in tags):
                    category = "funny"

                highlights.append(HighlightMoment(
                    timestamp=t,
                    duration=min_duration,
                    confidence=score,
                    description=caption,
                    category=category
                ))

        # Sort by confidence and limit
        highlights.sort(key=lambda h: h.confidence, reverse=True)
        highlights = highlights[:max_highlights]

        logger.info(f"Detected {len(highlights)} highlights")

        return highlights

    def categorize_scenes(
        self,
        scenes: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize scenes by content type.

        Returns:
            Dict mapping category names to lists of scenes
        """
        categories: Dict[str, List[Dict[str, Any]]] = {
            "action": [],
            "dialogue": [],
            "scenic": [],
            "emotional": [],
            "transition": [],
            "other": []
        }

        for scene in scenes:
            _, _, tags = self._score_clip_vlm(
                scene['path'],
                scene['start'],
                scene['end']
            )

            # Categorize based on tags
            categorized = False
            for category in categories.keys():
                if category in tags or any(t in category for t in tags):
                    categories[category].append(scene)
                    categorized = True
                    break

            if not categorized:
                categories["other"].append(scene)

        return categories


# Convenience functions

def is_vlm_available() -> bool:
    """Check if any VLM backend is available."""
    return _check_vlm_available() is not None


def get_available_vlm_backend() -> Optional[str]:
    """Get the name of the available VLM backend."""
    return _check_vlm_available()


# Module exports
__all__ = [
    "VLMClipSelector",
    "ClipScore",
    "HighlightMoment",
    "is_vlm_available",
    "get_available_vlm_backend",
]
