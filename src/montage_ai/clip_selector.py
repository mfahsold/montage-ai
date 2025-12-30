"""
Intelligent Clip Selection with LLM Reasoning

Enhances clip selection from heuristic scoring to context-aware LLM reasoning.

Architecture:
    Heuristic Filter → Top N Candidates → LLM Ranking → Best Clip

Benefits:
    - Context-aware decisions (style, previous clips, energy, position)
    - Explainable reasoning (shown in logs/UI)
    - Better continuity and flow
    - Preparation for advanced ML features

Version: 0.1.0
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .logger import logger

# Import Creative Director for LLM access
try:
    from .creative_director import CreativeDirector
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("CreativeDirector not available - LLM clip selection disabled")

VERSION = "0.1.0"

# Feature toggle - enabled by default for better clip selection
LLM_CLIP_SELECTION = os.environ.get("LLM_CLIP_SELECTION", "true").lower() == "true"


@dataclass
class ClipCandidate:
    """Represents a clip candidate with metadata."""
    path: str
    start_time: float
    duration: float
    heuristic_score: int
    metadata: Dict[str, Any]  # energy, action, shot_type, etc.


@dataclass
class ClipRanking:
    """LLM ranking result for a clip."""
    clip_index: int
    score: int  # 0-100
    reason: str


class IntelligentClipSelector:
    """
    LLM-powered clip selection with reasoning.

    Workflow:
        1. Get top N candidates by heuristic score (fast filter)
        2. Ask LLM to rank candidates considering context
        3. Return best clip with reasoning
        4. Fallback to heuristic if LLM fails
    """

    def __init__(self, style: str = "dynamic", use_llm: bool = None):
        """
        Initialize clip selector.

        Args:
            style: Editing style (hitchcock, mtv, etc.)
            use_llm: Force enable/disable LLM (None = auto from env)
        """
        self.style = style
        self.use_llm = LLM_CLIP_SELECTION if use_llm is None else use_llm

        # Initialize LLM if available and enabled
        self.llm = None
        if self.use_llm and LLM_AVAILABLE:
            try:
                self.llm = CreativeDirector()
                logger.info("Intelligent Clip Selector enabled (LLM-powered)")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.use_llm = False

        # Tracking
        self.selection_count = 0
        self.llm_success_count = 0

    def select_best_clip(
        self,
        candidates: List[ClipCandidate],
        context: Dict[str, Any],
        top_n: int = 3
    ) -> tuple[ClipCandidate, str]:
        """
        Select best clip from candidates using LLM reasoning.

        Args:
            candidates: List of clip candidates (already scored heuristically)
            context: Current editing context (energy, position, previous clips, etc.)
            top_n: Number of top candidates to send to LLM

        Returns:
            (best_clip, reasoning) tuple
        """
        self.selection_count += 1

        # If LLM disabled or not available, use heuristic
        if not self.use_llm or self.llm is None or len(candidates) <= 1:
            best = max(candidates, key=lambda c: c.heuristic_score)
            return best, "Heuristic selection (LLM disabled)"

        # Get top N by heuristic score (fast filter)
        candidates_sorted = sorted(candidates, key=lambda c: c.heuristic_score, reverse=True)
        top_candidates = candidates_sorted[:min(top_n, len(candidates))]

        # Try LLM ranking
        try:
            rankings = self._query_llm_ranking(top_candidates, context)

            if rankings:
                # Get best ranked clip
                best_ranking = max(rankings, key=lambda r: r.score)
                best_clip = top_candidates[best_ranking.clip_index]

                self.llm_success_count += 1

                return best_clip, f"LLM choice: {best_ranking.reason}"
            else:
                # LLM failed - fallback to heuristic
                best = top_candidates[0]  # Already sorted
                return best, "Heuristic fallback (LLM returned no ranking)"

        except Exception as e:
            logger.warning(f"LLM clip selection error: {e}")
            # Fallback to heuristic
            best = top_candidates[0]
            return best, f"Heuristic fallback (LLM error: {str(e)[:50]})"

    def _query_llm_ranking(
        self,
        candidates: List[ClipCandidate],
        context: Dict[str, Any]
    ) -> Optional[List[ClipRanking]]:
        """
        Query LLM to rank clip candidates.

        Args:
            candidates: Top N candidates to rank
            context: Editing context

        Returns:
            List of rankings or None if failed
        """
        # Build prompt
        prompt = self._build_ranking_prompt(candidates, context)

        # Query LLM (using same backend as Creative Director)
        response = self.llm._query_llm(prompt)

        if not response:
            return None

        # Parse response
        return self._parse_llm_ranking(response, len(candidates))

    def _build_ranking_prompt(
        self,
        candidates: List[ClipCandidate],
        context: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for clip ranking."""

        # Extract context
        style = self.style
        current_energy = context.get('current_energy', 0.5)
        position = context.get('position', 'middle')  # intro/build/climax/outro
        previous_clips = context.get('previous_clips', [])
        beat_position = context.get('beat_position', 0)

        # Describe previous clips (last 2)
        prev_desc = ""
        if previous_clips:
            last_clips = previous_clips[-2:]
            prev_desc = "\n".join([
                f"- {i+1}. {self._describe_clip(c)}"
                for i, c in enumerate(last_clips)
            ])
        else:
            prev_desc = "None (first clip)"

        # Describe candidates
        candidates_desc = "\n".join([
            f"{i+1}. {self._describe_clip_candidate(c)}"
            for i, c in enumerate(candidates)
        ])

        # Energy level description
        energy_desc = "high" if current_energy > 0.7 else "medium" if current_energy > 0.4 else "low"

        prompt = f"""You are a professional film editor working on a {style}-style montage.

CONTEXT:
Style: {style}
Current Energy: {energy_desc} ({current_energy:.2f})
Position in Timeline: {position}
Beat Position: {beat_position}

Previous Clips:
{prev_desc}

CANDIDATE CLIPS (rank these):
{candidates_desc}

TASK:
Rank these clips (1=best) considering:
1. Continuity with previous clips (avoid jarring transitions)
2. Energy/mood match with current position
3. Visual variety (avoid repetition)
4. {style}-specific aesthetics

Respond with ONLY valid JSON:
{{
  "rankings": [
    {{"clip": 1, "score": 95, "reason": "High-action close-up creates tension after wide shot"}},
    {{"clip": 2, "score": 78, "reason": "Good energy but too similar to previous"}},
    {{"clip": 3, "score": 45, "reason": "Low energy doesn't match climax position"}}
  ]
}}

CRITICAL: Respond with JSON only, no markdown, no explanations."""

        return prompt

    def _describe_clip(self, clip: Dict[str, Any]) -> str:
        """Describe a previous clip for context."""
        meta = clip.get('meta', {})
        action = meta.get('action', 'medium')
        shot = meta.get('shot', 'medium')
        energy = meta.get('energy', 0.5)

        return f"{shot.capitalize()} shot, {action} action, energy={energy:.2f}"

    def _describe_clip_candidate(self, candidate: ClipCandidate) -> str:
        """Describe a clip candidate for LLM."""
        meta = candidate.metadata
        action = meta.get('action', 'medium')
        shot = meta.get('shot', 'medium')
        energy = meta.get('energy', 0.5)

        return (f"{shot.capitalize()} shot, {action} action, "
                f"energy={energy:.2f}, duration={candidate.duration:.1f}s "
                f"(heuristic_score={candidate.heuristic_score})")

    def _parse_llm_ranking(
        self,
        llm_response: str,
        num_candidates: int
    ) -> Optional[List[ClipRanking]]:
        """
        Parse LLM ranking response.

        Args:
            llm_response: Raw JSON string from LLM
            num_candidates: Expected number of candidates

        Returns:
            List of ClipRanking or None if invalid
        """
        try:
            # Clean response (remove markdown if present)
            response = llm_response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            data = json.loads(response)
            rankings_data = data.get('rankings', [])

            if not rankings_data:
                logger.warning("LLM returned empty rankings")
                return None

            # Convert to ClipRanking objects
            rankings = []
            for item in rankings_data:
                clip_idx = item.get('clip', 1) - 1  # Convert 1-indexed to 0-indexed
                score = item.get('score', 50)
                reason = item.get('reason', 'No reason provided')

                # Validate index
                if 0 <= clip_idx < num_candidates:
                    rankings.append(ClipRanking(
                        clip_index=clip_idx,
                        score=score,
                        reason=reason
                    ))

            if not rankings:
                logger.warning("No valid rankings parsed")
                return None

            return rankings

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM ranking JSON: {e}")
            logger.debug(f"Response: {llm_response[:200]}...")
            return None
        except Exception as e:
            logger.warning(f"Error parsing LLM ranking: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        llm_success_rate = (self.llm_success_count / self.selection_count * 100) if self.selection_count > 0 else 0

        return {
            "total_selections": self.selection_count,
            "llm_selections": self.llm_success_count,
            "heuristic_selections": self.selection_count - self.llm_success_count,
            "llm_success_rate": f"{llm_success_rate:.1f}%",
            "llm_enabled": self.use_llm
        }


# Convenience function for direct usage
def select_clip_with_reasoning(
    candidates: List[Dict[str, Any]],
    context: Dict[str, Any],
    style: str = "dynamic"
) -> tuple[Dict[str, Any], str]:
    """
    Convenience function: Select best clip with LLM reasoning.

    Args:
        candidates: List of clip dicts with heuristic scores
        context: Editing context
        style: Editing style

    Returns:
        (best_clip, reasoning) tuple
    """
    # Convert to ClipCandidate objects
    clip_candidates = [
        ClipCandidate(
            path=c.get('path', ''),
            start_time=c.get('start', 0),
            duration=c.get('duration', 0),
            heuristic_score=c.get('score', 0),
            metadata=c.get('meta', {})
        )
        for c in candidates
    ]

    # Select
    selector = IntelligentClipSelector(style=style)
    best_candidate, reasoning = selector.select_best_clip(
        clip_candidates,
        context
    )

    # Convert back to dict
    best_clip = next((c for c in candidates if c.get('path') == best_candidate.path), candidates[0])

    return best_clip, reasoning


if __name__ == "__main__":
    # Test
    print(f"🧠 Intelligent Clip Selector v{VERSION}")
    print(f"   LLM Available: {LLM_AVAILABLE}")
    print(f"   LLM Enabled: {LLM_CLIP_SELECTION}")
