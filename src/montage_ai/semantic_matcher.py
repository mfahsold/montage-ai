"""
Semantic Matcher - Embedding-based semantic matching for clip selection.

Phase 2: Semantic Storytelling
Uses sentence-transformers (all-MiniLM-L6-v2) for embedding-based similarity.

Features:
- Query-to-clip semantic matching
- Weighted scoring (tags 40%, caption 30%, mood 15%, setting 15%)
- Keyword-based mood/setting matching
- Singleton pattern for model reuse
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

# Check for sentence_transformers availability
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SemanticScore:
    """Result of semantic matching between query and clip."""
    overall_score: float      # 0.0-1.0 weighted combination
    tag_score: float          # Similarity to tags
    caption_score: float      # Similarity to caption
    mood_match: float         # Mood match score (0.0 or 1.0)
    setting_match: float      # Setting match score (0.0 or 1.0)
    matched_tags: List[str] = field(default_factory=list)  # Tags that matched query

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "tag_score": self.tag_score,
            "caption_score": self.caption_score,
            "mood_match": self.mood_match,
            "setting_match": self.setting_match,
            "matched_tags": self.matched_tags,
        }


# ============================================================================
# Mood and Setting Keywords
# ============================================================================

# Keyword mappings for mood detection
MOOD_KEYWORDS: Dict[str, List[str]] = {
    "calm": ["calm", "peaceful", "serene", "tranquil", "relaxed", "gentle", "quiet", "still", "slow"],
    "energetic": ["energetic", "dynamic", "fast", "action", "exciting", "intense", "active", "lively", "upbeat"],
    "dramatic": ["dramatic", "epic", "intense", "powerful", "emotional", "cinematic", "striking", "bold"],
    "playful": ["playful", "fun", "happy", "joyful", "cheerful", "light", "cute", "whimsical"],
    "tense": ["tense", "suspense", "thriller", "scary", "dark", "ominous", "mysterious", "danger"],
    "peaceful": ["peaceful", "calm", "nature", "zen", "meditative", "ambient", "soothing"],
    "mysterious": ["mysterious", "enigmatic", "unknown", "hidden", "secret", "foggy", "shadow"],
    "neutral": [],  # Default, matches nothing specific
}

# Keyword mappings for setting detection
SETTING_KEYWORDS: Dict[str, List[str]] = {
    "indoor": ["indoor", "inside", "interior", "room", "house", "building", "office", "home"],
    "outdoor": ["outdoor", "outside", "exterior", "open", "sky", "nature"],
    "beach": ["beach", "ocean", "sea", "sand", "waves", "coast", "shore", "tropical", "surf"],
    "city": ["city", "urban", "street", "downtown", "building", "skyline", "traffic", "metropolitan"],
    "nature": ["nature", "forest", "mountain", "tree", "landscape", "wilderness", "green", "park"],
    "studio": ["studio", "stage", "production", "set", "backdrop", "lighting"],
    "street": ["street", "road", "sidewalk", "pavement", "crosswalk", "pedestrian"],
    "home": ["home", "house", "apartment", "living room", "bedroom", "kitchen", "domestic"],
    "unknown": [],  # Default, matches nothing specific
}


# ============================================================================
# Semantic Matcher
# ============================================================================

class SemanticMatcher:
    """
    Embedding-based semantic matcher for clip selection.

    Uses all-MiniLM-L6-v2 (same as VideoAgent) for text embeddings.
    Implements weighted scoring with configurable weights.

    Default weights:
    - Tags: 40% (primary match signal)
    - Caption: 30% (contextual understanding)
    - Mood: 15% (emotional alignment)
    - Setting: 15% (location/environment alignment)
    """

    _model: Optional[Any] = None  # Singleton model instance

    def __init__(
        self,
        tag_weight: float = 0.4,
        caption_weight: float = 0.3,
        mood_weight: float = 0.15,
        setting_weight: float = 0.15,
    ):
        """
        Initialize matcher with custom weights.

        Args:
            tag_weight: Weight for tag similarity (default 0.4)
            caption_weight: Weight for caption similarity (default 0.3)
            mood_weight: Weight for mood match (default 0.15)
            setting_weight: Weight for setting match (default 0.15)
        """
        self.tag_weight = tag_weight
        self.caption_weight = caption_weight
        self.mood_weight = mood_weight
        self.setting_weight = setting_weight

        # Validate weights sum to 1.0
        total = tag_weight + caption_weight + mood_weight + setting_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Semantic weights sum to {total}, expected 1.0")

        # Initialize embedding model (singleton)
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the embedding model (singleton pattern)."""
        if SemanticMatcher._model is not None:
            return

        if not EMBEDDINGS_AVAILABLE:
            logger.warning("sentence_transformers not available, semantic matching disabled")
            return

        try:
            SemanticMatcher._model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded all-MiniLM-L6-v2 embedding model")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            SemanticMatcher._model = None

    @property
    def model(self) -> Optional[Any]:
        """Get the embedding model instance."""
        return SemanticMatcher._model

    @property
    def is_available(self) -> bool:
        """Check if semantic matching is available."""
        return SemanticMatcher._model is not None

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if model unavailable
        """
        if self.model is None or not text.strip():
            return None

        try:
            return self.model.encode(text)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        if a is None or b is None:
            return 0.0

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(a, b) / (norm_a * norm_b)
        # Clamp to [0, 1] range (cosine can be negative for opposite meanings)
        return max(0.0, float(similarity))

    def _check_mood_match(self, query: str, clip_mood: Optional[str]) -> float:
        """
        Check if query implies same mood as clip.

        Args:
            query: User query
            clip_mood: Clip's detected mood

        Returns:
            1.0 if mood matches, 0.0 otherwise
        """
        if not clip_mood or clip_mood == "neutral":
            return 0.0

        query_lower = query.lower()

        # Check if query contains keywords for the clip's mood
        mood_keywords = MOOD_KEYWORDS.get(clip_mood, [])
        for keyword in mood_keywords:
            if keyword in query_lower:
                return 1.0

        return 0.0

    def _check_setting_match(self, query: str, clip_setting: Optional[str]) -> float:
        """
        Check if query implies same setting as clip.

        Args:
            query: User query
            clip_setting: Clip's detected setting

        Returns:
            1.0 if setting matches, 0.0 otherwise
        """
        if not clip_setting or clip_setting == "unknown":
            return 0.0

        query_lower = query.lower()

        # Check if query contains keywords for the clip's setting
        setting_keywords = SETTING_KEYWORDS.get(clip_setting, [])
        for keyword in setting_keywords:
            if keyword in query_lower:
                return 1.0

        return 0.0

    def _find_matched_tags(self, query: str, tags: List[str]) -> List[str]:
        """
        Find tags that match the query (substring or semantic).

        Args:
            query: User query
            tags: Clip's tags

        Returns:
            List of matching tags
        """
        if not tags:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        matched = []

        for tag in tags:
            tag_lower = tag.lower()
            # Direct substring match
            if tag_lower in query_lower or any(word in tag_lower for word in query_words):
                matched.append(tag)

        return matched

    def match_query_to_clip(
        self,
        query: str,
        clip_analysis: Dict[str, Any],
    ) -> SemanticScore:
        """
        Match user query against clip's semantic analysis.

        Args:
            query: User search query (e.g., "beach scenes with action")
            clip_analysis: Dict with tags, caption, mood, setting from SceneAnalysis

        Returns:
            SemanticScore with weighted scores and matched tags

        Example:
            >>> matcher = SemanticMatcher()
            >>> clip = {"tags": ["beach", "surfing"], "caption": "Person surfing on ocean", "mood": "energetic", "setting": "beach"}
            >>> score = matcher.match_query_to_clip("beach action", clip)
            >>> print(score.overall_score)
        """
        # Extract clip metadata
        tags = clip_analysis.get("tags", [])
        caption = clip_analysis.get("caption", "")
        mood = clip_analysis.get("mood", "neutral")
        setting = clip_analysis.get("setting", "unknown")

        # Compute embedding-based similarities
        query_emb = self._get_embedding(query)

        # Tag similarity (embed all tags as single string)
        tags_text = " ".join(tags) if tags else ""
        tags_emb = self._get_embedding(tags_text)
        tag_score = self._cosine_similarity(query_emb, tags_emb) if query_emb is not None and tags_emb is not None else 0.0

        # Caption similarity
        caption_emb = self._get_embedding(caption)
        caption_score = self._cosine_similarity(query_emb, caption_emb) if query_emb is not None and caption_emb is not None else 0.0

        # Keyword-based mood/setting matching
        mood_match = self._check_mood_match(query, mood)
        setting_match = self._check_setting_match(query, setting)

        # Find matched tags
        matched_tags = self._find_matched_tags(query, tags)

        # Compute weighted overall score
        overall_score = (
            self.tag_weight * tag_score +
            self.caption_weight * caption_score +
            self.mood_weight * mood_match +
            self.setting_weight * setting_match
        )

        return SemanticScore(
            overall_score=overall_score,
            tag_score=tag_score,
            caption_score=caption_score,
            mood_match=mood_match,
            setting_match=setting_match,
            matched_tags=matched_tags,
        )

    def match_query_to_clips(
        self,
        query: str,
        clips: List[Dict[str, Any]],
        min_score: float = 0.0,
    ) -> List[tuple]:
        """
        Match query against multiple clips, returning sorted results.

        Args:
            query: User search query
            clips: List of clip dicts with 'meta' containing semantic analysis
            min_score: Minimum score threshold (default 0.0)

        Returns:
            List of (clip, SemanticScore) tuples sorted by score descending
        """
        results = []

        for clip in clips:
            meta = clip.get("meta", clip)  # Support both nested and flat structure
            score = self.match_query_to_clip(query, meta)

            if score.overall_score >= min_score:
                results.append((clip, score))

        # Sort by overall score descending
        results.sort(key=lambda x: x[1].overall_score, reverse=True)

        return results


# ============================================================================
# Singleton Accessor
# ============================================================================

_semantic_matcher: Optional[SemanticMatcher] = None


def get_semantic_matcher() -> SemanticMatcher:
    """
    Get singleton SemanticMatcher instance.

    Returns:
        SemanticMatcher instance
    """
    global _semantic_matcher

    if _semantic_matcher is None:
        _semantic_matcher = SemanticMatcher()

    return _semantic_matcher


def reset_semantic_matcher() -> None:
    """Reset the singleton instance (for testing)."""
    global _semantic_matcher
    _semantic_matcher = None
