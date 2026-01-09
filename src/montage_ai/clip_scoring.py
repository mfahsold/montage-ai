"""
Unified Clip Scoring Engine for Montage AI

Consolidates clip scoring logic from multiple modules into a single,
testable interface. This follows DRY principles and ensures consistent
scoring across the codebase.

Usage:
    from montage_ai.clip_scoring import ClipScorer
    
    scorer = ClipScorer()
    variety_score = scorer.score_shot_variety(clips, ordered_indices)
    match_score = scorer.score_match_quality(scene, clip_metadata)
    broll_score = scorer.score_broll_relevance(scene, clip_metadata)

Scoring Philosophy:
    - All scores return floats in range [0.0, 1.0]
    - 1.0 = perfect match, 0.0 = no match
    - Scores are composable (can be weighted and combined)
    - Thread-safe (stateless functions)
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ScoringWeights:
    """Configurable weights for score composition.
    
    Adjust these to tune clip selection behavior across the pipeline.
    """
    shot_variety_weight: float = 0.2  # Penalty for consecutive same shots
    match_quality_weight: float = 0.3  # How well clip matches scene
    broll_relevance_weight: float = 0.2  # Semantic relevance for B-roll
    face_confidence_weight: float = 0.15  # Face quality/presence
    novelty_weight: float = 0.15  # Visual uniqueness


class ClipScorer:
    """Unified scoring interface for clip selection decisions."""
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        """Initialize scorer with optional custom weights.
        
        Args:
            weights: Custom scoring weights. If None, uses defaults.
        """
        self.weights = weights or ScoringWeights()
    
    # =========================================================================
    # Shot Variety Scoring
    # =========================================================================
    
    @staticmethod
    def score_shot_variety(
        clips: List[Dict[str, Any]],
        ordered: List[int],
    ) -> float:
        """Calculate penalty for consecutive identical shot types.
        
        A low score means the clip sequence has many repeated shot types.
        A high score means good shot diversity.
        
        Args:
            clips: List of clip metadata dicts with 'shot_type' keys
            ordered: Indices of clips in the sequence
            
        Returns:
            Score from 0.0 (all same) to 1.0 (all different)
            
        Example:
            >>> clips = [
            ...     {"shot_type": "wide"},
            ...     {"shot_type": "close"},
            ...     {"shot_type": "close"},  # Repeat
            ...     {"shot_type": "wide"},
            ... ]
            >>> ordered = [0, 1, 2, 3]
            >>> score = ClipScorer.score_shot_variety(clips, ordered)
            >>> score  # 1 repeat out of 3 transitions
            0.667
        """
        if len(ordered) < 2:
            return 1.0
        
        repeats = 0
        for i in range(1, len(ordered)):
            prev_type = clips[ordered[i - 1]].get("shot_type")
            curr_type = clips[ordered[i]].get("shot_type")
            if prev_type and curr_type and prev_type == curr_type:
                repeats += 1
        
        max_repeats = len(ordered) - 1
        return 1.0 - (repeats / max_repeats) if max_repeats > 0 else 1.0
    
    # =========================================================================
    # Match Quality Scoring (Scene-Clip Affinity)
    # =========================================================================
    
    @staticmethod
    def score_match_quality(
        scene: Dict[str, Any],
        clip_metadata: Dict[str, Any],
    ) -> float:
        """Score how well a clip matches a scene's requirements.
        
        Considers semantic similarity, visual characteristics, and metadata.
        
        Args:
            scene: Scene dict with 'description', 'visual_cues', 'pacing'
            clip_metadata: Clip dict with 'embedding', 'visual_features'
            
        Returns:
            Match score from 0.0 (poor) to 1.0 (excellent)
        """
        score = 0.0
        count = 0
        
        # Semantic similarity (if embeddings available)
        if "embedding" in clip_metadata and "embedding" in scene:
            try:
                import numpy as np
                clip_emb = np.array(clip_metadata["embedding"])
                scene_emb = np.array(scene["embedding"])
                # Cosine similarity
                sim = np.dot(clip_emb, scene_emb) / (
                    np.linalg.norm(clip_emb) * np.linalg.norm(scene_emb) + 1e-8
                )
                score += float(max(0.0, sim))  # Clamp to [0, 1]
                count += 1
            except (ImportError, ValueError, TypeError):
                pass
        
        # Visual feature matching
        if "visual_features" in clip_metadata:
            features = clip_metadata["visual_features"]
            if isinstance(features, dict):
                # Count matching features
                matches = 0
                total = len(features)
                for key, value in features.items():
                    if scene.get(key) == value:
                        matches += 1
                if total > 0:
                    score += matches / total
                    count += 1
        
        # Pacing compatibility (energy level match)
        if "pacing" in scene and "energy_level" in clip_metadata:
            scene_pacing = scene.get("pacing", 0.5)  # 0=slow, 1=fast
            clip_energy = clip_metadata.get("energy_level", 0.5)
            # Reward clips that roughly match scene pacing
            pacing_match = 1.0 - abs(scene_pacing - clip_energy)
            score += pacing_match
            count += 1
        
        return score / count if count > 0 else 0.5
    
    # =========================================================================
    # B-Roll Relevance Scoring
    # =========================================================================
    
    @staticmethod
    def score_broll_relevance(
        scene: Dict[str, Any],
        clip_metadata: Dict[str, Any],
    ) -> float:
        """Score semantic relevance of a B-roll clip to a scene.
        
        Used for supplemental footage that should reinforce the narrative.
        Higher scores indicate better thematic alignment.
        
        Args:
            scene: Scene dict with narrative context
            clip_metadata: B-roll clip with semantic tags
            
        Returns:
            Relevance score from 0.0 (unrelated) to 1.0 (perfect match)
        """
        score = 0.0
        weight_sum = 0.0
        
        # Keyword/tag matching
        if "keywords" in clip_metadata and "scene_keywords" in scene:
            clip_tags = set(clip_metadata.get("keywords", []))
            scene_tags = set(scene.get("scene_keywords", []))
            if clip_tags and scene_tags:
                overlap = len(clip_tags & scene_tags)
                total = len(clip_tags | scene_tags)
                tag_match = overlap / total if total > 0 else 0.0
                score += tag_match * 0.4  # 40% weight to tags
                weight_sum += 0.4
        
        # Thematic/conceptual relevance
        if "concept" in clip_metadata and "concept" in scene:
            concept_match = 1.0 if clip_metadata["concept"] == scene["concept"] else 0.5
            score += concept_match * 0.3  # 30% weight to concept
            weight_sum += 0.3
        
        # Temporal/contextual suitability
        if "suitable_for_phase" in clip_metadata and "phase" in scene:
            phases = clip_metadata.get("suitable_for_phase", [])
            if scene.get("phase") in phases:
                score += 0.3  # 30% weight to phase match
                weight_sum += 0.3
        
        return score / weight_sum if weight_sum > 0 else 0.5
    
    # =========================================================================
    # Face Detection Scoring
    # =========================================================================
    
    @staticmethod
    def score_faces(
        clip_metadata: Dict[str, Any],
        require_faces: bool = False,
    ) -> float:
        """Score face quality and presence in a clip.
        
        Args:
            clip_metadata: Dict with 'face_count', 'face_confidence'
            require_faces: If True, penalize clips with no faces
            
        Returns:
            Face score from 0.0 (no/bad faces) to 1.0 (excellent faces)
        """
        face_count = clip_metadata.get("face_count", 0)
        face_confidence = clip_metadata.get("face_confidence", 0.0)
        
        if face_count == 0:
            return 0.0 if require_faces else 0.5
        
        # Confidence is already normalized [0, 1]
        # Weight by count (prefer 1-2 clear faces over crowds)
        if face_count == 1:
            count_bonus = 1.0
        elif face_count == 2:
            count_bonus = 0.9
        else:
            count_bonus = max(0.5, 1.0 - (face_count - 2) * 0.1)
        
        return face_confidence * count_bonus
    
    # =========================================================================
    # Visual Novelty Scoring
    # =========================================================================
    
    @staticmethod
    def score_visual_novelty(
        clip_metadata: Dict[str, Any],
        recent_clips: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Score visual uniqueness compared to recent clips.
        
        Encourages variety and prevents monotony in the sequence.
        
        Args:
            clip_metadata: Current clip dict with visual features
            recent_clips: Previous 3-5 clips for comparison
            
        Returns:
            Novelty score from 0.0 (same as recent) to 1.0 (very different)
        """
        if not recent_clips:
            return 1.0  # No history = maximally novel
        
        # Compare color dominance
        color_diffs = []
        if "dominant_color" in clip_metadata:
            clip_color = clip_metadata["dominant_color"]
            for recent in recent_clips:
                if "dominant_color" in recent:
                    # Simple RGB distance (normalized to [0, 1])
                    dist = sum(abs(a - b) for a, b in zip(clip_color, recent["dominant_color"])) / (3 * 255)
                    color_diffs.append(dist)
        
        if color_diffs:
            avg_color_diff = sum(color_diffs) / len(color_diffs)
            return min(1.0, avg_color_diff)  # Return max novelty score
        
        # Fallback: compare metadata characteristics
        return 0.7  # Default moderate novelty
    
    # =========================================================================
    # Composite Scoring
    # =========================================================================
    
    def score_clip_comprehensive(
        self,
        clips: List[Dict[str, Any]],
        ordered: List[int],
        scene: Dict[str, Any],
        current_clip_idx: int,
        require_faces: bool = False,
    ) -> float:
        """Compute comprehensive clip score combining multiple factors.
        
        This is the primary interface for production use.
        
        Args:
            clips: All available clips
            ordered: Current sequence of selected clip indices
            scene: Scene requiring a clip
            current_clip_idx: Index of the candidate clip to score
            require_faces: Whether scene requires face presence
            
        Returns:
            Overall score from 0.0 to 1.0
        """
        clip_meta = clips[current_clip_idx]
        
        # Individual component scores
        variety_score = self.score_shot_variety(clips, ordered + [current_clip_idx])
        match_score = self.score_match_quality(scene, clip_meta)
        broll_score = self.score_broll_relevance(scene, clip_meta)
        face_score = self.score_faces(clip_meta, require_faces)
        
        # Get recent clips for novelty comparison
        recent = [clips[idx] for idx in ordered[-3:] if idx < len(clips)] if ordered else []
        novelty_score = self.score_visual_novelty(clip_meta, recent)
        
        # Weighted composite
        total = (
            variety_score * self.weights.shot_variety_weight
            + match_score * self.weights.match_quality_weight
            + broll_score * self.weights.broll_relevance_weight
            + face_score * self.weights.face_confidence_weight
            + novelty_score * self.weights.novelty_weight
        )
        
        return total / sum([
            self.weights.shot_variety_weight,
            self.weights.match_quality_weight,
            self.weights.broll_relevance_weight,
            self.weights.face_confidence_weight,
            self.weights.novelty_weight,
        ])


# ============================================================================
# Convenience Singletons
# ============================================================================

_default_scorer: Optional[ClipScorer] = None


def get_default_scorer() -> ClipScorer:
    """Get or create the default clip scorer instance.
    
    Returns:
        Shared ClipScorer instance
    """
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = ClipScorer()
    return _default_scorer


__all__ = [
    "ClipScorer",
    "ScoringWeights",
    "get_default_scorer",
]
