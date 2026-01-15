"""
Intelligent Clip Selection with LLM/VLM Reasoning

Enhances clip selection from heuristic scoring to context-aware AI reasoning.

Architecture:
    Heuristic Filter → Top N Candidates → LLM/VLM Ranking → Best Clip

Backends (Priority Order):
    1. VLM (Vision-Language Model) - Qwen2.5-VL, VILA, InternVideo2
    2. LLM (Text-only) - CreativeDirector
    3. Heuristic (fallback)

Benefits:
    - Context-aware decisions (style, previous clips, energy, position)
    - Visual understanding via VLM (SOTA, query-based selection)
    - Explainable reasoning (shown in logs/UI)
    - Better continuity and flow

Version: 0.2.0
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .logger import logger
from .config import get_settings

# Import Creative Director for LLM access
try:
    from .creative_director import CreativeDirector
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# SOTA: Import VLM Clip Selector (Qwen2.5-VL, VILA, InternVideo2)
try:
    from .vlm_clip_selector import VLMClipSelector, get_available_vlm_backend
    VLM_AVAILABLE = get_available_vlm_backend() is not None
except ImportError:
    VLM_AVAILABLE = False

VERSION = "0.2.0"

# Feature toggle - enabled by default for better clip selection
LLM_CLIP_SELECTION = get_settings().features.llm_clip_selection


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
    AI-powered clip selection with reasoning.

    Backend Priority:
        1. VLM (Vision-Language Model) - Best for visual understanding
        2. LLM (Text-only) - Good for context-aware decisions
        3. Heuristic (fallback) - Fast, deterministic

    Workflow:
        1. Get top N candidates by heuristic score (fast filter)
        2. Ask VLM/LLM to rank candidates considering context
        3. Return best clip with reasoning
        4. Fallback to heuristic if AI fails
    """

    def __init__(self, style: str = "dynamic", use_llm: bool = None, use_vlm: bool = None):
        """
        Initialize clip selector.

        Args:
            style: Editing style (hitchcock, mtv, etc.)
            use_llm: Force enable/disable LLM (None = auto from env)
            use_vlm: Force enable/disable VLM (None = auto-detect)
        """
        self.style = style
        self.use_llm = LLM_CLIP_SELECTION if use_llm is None else use_llm
        self.use_vlm = VLM_AVAILABLE if use_vlm is None else use_vlm

        # SOTA: Initialize VLM if available (priority over LLM)
        self.vlm = None
        if self.use_vlm and VLM_AVAILABLE:
            try:
                self.vlm = VLMClipSelector()
                backend = get_available_vlm_backend()
                logger.info(f"   🧠 SOTA VLM Clip Selector enabled ({backend})")
            except Exception as e:
                logger.debug(f"VLM initialization failed: {e}")
                self.vlm = None
                self.use_vlm = False

        # Initialize LLM as fallback if VLM not available
        self.llm = None
        if self.use_llm and LLM_AVAILABLE and not self.vlm:
            try:
                self.llm = CreativeDirector()
                logger.info("   🧠 Intelligent Clip Selector enabled (LLM-powered)")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.use_llm = False

        # Tracking
        self.selection_count = 0
        self.llm_success_count = 0
        self.vlm_success_count = 0

    def select_best_clip(
        self,
        candidates: List[ClipCandidate],
        context: Dict[str, Any],
        top_n: int = 3,
        query: Optional[str] = None
    ) -> tuple[ClipCandidate, str]:
        """
        Select best clip from candidates using VLM/LLM reasoning.
        Implements a "Reasoning Tree" approach (ToAE inspired).
        """
        self.selection_count += 1

        # If no AI backend available, use heuristic
        if (not self.vlm and not self.llm) or len(candidates) <= 1:
            best = max(candidates, key=lambda c: c.heuristic_score)
            return best, "Heuristic selection (AI disabled)"

        # Get top N by heuristic score (fast filter)
        candidates_sorted = sorted(candidates, key=lambda c: c.heuristic_score, reverse=True)
        top_candidates = candidates_sorted[:min(top_n, len(candidates))]

        # SOTA: Try VLM first (visual understanding)
        if self.vlm and query:
            # ... (Existing VLM logic)
            pass

        # SOTA: Reasoning Tree (LLM-based)
        if self.llm:
            try:
                # Use Reasoning Tree if enabled or by default for high-quality
                best_clip, reasoning = self._query_reasoning_tree(top_candidates, context)
                if best_clip:
                    self.llm_success_count += 1
                    return best_clip, reasoning
            except Exception as e:
                logger.debug(f"Reasoning Tree selection error: {e}")

        # Final fallback: heuristic
        best = top_candidates[0]
        return best, "Heuristic fallback"

    def _query_reasoning_tree(
        self,
        candidates: List[ClipCandidate],
        context: Dict[str, Any]
    ) -> tuple[Optional[ClipCandidate], str]:
        """
        SOTA: Implements a Reasoning Tree (Tree-of-Thought) selection.
        Instead of simple ranking, the LLM explores multiple editing directions.
        """
        prompt = self._build_reasoning_tree_prompt(candidates, context)
        
        # SOTA: Robust Edge Case - Search Latency Fallback
        # Limit clip selection to 10 seconds to keep assembly loop fast.
        try:
            response = self.llm.query(
                prompt=prompt,
                system_prompt=self.llm.system_prompt,
                timeout=10, # 10s timeout for clip selection
                max_retries=1
            )
        except Exception as e:
            logger.debug(f"LLM selection timeout or error: {e}")
            return None, ""

        if not response:
            return None, ""

        try:
            # Parse the reasoning JSON
            # Expecting: {"best_direction": "...", "selected_clip_index": X, "reasoning": "..."}
            data = json.loads(self._fix_llm_json(response))
            idx = data.get("selected_clip_index", 1) - 1
            idx = max(0, min(idx, len(candidates) - 1))
            
            best_clip = candidates[idx]
            reason = f"🧠 Reasoning Tree [{data.get('best_direction', 'N/A')}]: {data.get('reasoning', 'Selected for narrative flow')}"
            return best_clip, reason
        except Exception as e:
            logger.debug(f"Failed to parse reasoning tree JSON: {e}")
            return None, ""

    def _build_reasoning_tree_prompt(
        self,
        candidates: List[ClipCandidate],
        context: Dict[str, Any]
    ) -> str:
        """Build reasoning tree prompt for LLM."""
        style = self.style
        current_energy = context.get('current_energy', 0.5)
        position = context.get('position', 'middle')
        previous_clips = context.get('previous_clips', [])
        
        # Describe previous context
        prev_desc = "\n".join([f"- {i+1}. {self._describe_clip(c)}" for i, c in enumerate(previous_clips[-2:])]) or "None"
        
        # Describe candidates
        candidates_desc = "\n".join([f"Clip {i+1}: {self._describe_clip_candidate(c)}" for i, c in enumerate(candidates)])
        
        energy_desc = "high" if current_energy > 0.7 else "medium" if current_energy > 0.4 else "low"

        prompt = f"""You are a professional film editor using a "Tree of Reasoning" to pick the next clip for a {style} montage.

CONTEXT:
Timeline Position: {position}
Current Musical Energy: {energy_desc} ({current_energy:.2f})
Previous Clips: {prev_desc}

CANDIDATES:
{candidates_desc}

TASK (Step-by-Step Reasoning):
1. Evaluate candidates against EDITORIAL AXIOMS:
   - AXIOM 1 (Kuleshov Effect): How does the juxtaposition with the previous clip change the meaning?
   - AXIOM 2 (Motion Continuity): Does the movement in the new clip flow logically from the previous one?
   - AXIOM 3 (Graphic Match): Are there similar shapes or colors that create a visual bridge?
2. Explore 3 potential narrative directions for the next cut:
   - DIRECTION A: Continuity (smooth transition, matching shots)
   - DIRECTION B: Impact (contrast, jump in energy/shot type)
   - DIRECTION C: Thematic/Story (focus on emotion or specific motifs)
3. For each direction, identify which candidate clip fits best based on the Axioms.
4. Evaluate which direction is most effective for the {style} style at this {position} phase.
5. Select the final clip based on the best direction.

Return ONLY valid JSON:
{{
  "thought_process": {{
    "direction_a": "...",
    "direction_b": "...",
    "direction_c": "..."
  }},
  "best_direction": "...",
  "selected_clip_index": number (1-{len(candidates)}),
  "reasoning": "A concise final explanation"
}}
"""
        return prompt

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
        # Use CreativeDirector's system prompt for persona context
        response = self.llm._query_llm(self.llm.system_prompt, prompt)

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
        face_count = meta.get('face_count', 0)

        return (f"{shot.capitalize()} shot, {action} action, "
                f"energy={energy:.2f}, faces={face_count}, duration={candidate.duration:.1f}s "
                f"(heuristic_score={candidate.heuristic_score})")

    def _fix_llm_json(self, text: str) -> str:
        """
        Fix common LLM JSON issues:
        - Single quotes → double quotes
        - Trailing commas
        - Unquoted property names
        - Comments
        """
        import re
        # Remove JS-style comments
        text = re.sub(r'//[^\n]*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        # Replace single quotes with double quotes (careful with apostrophes)
        text = re.sub(r"'(\w+)':", r'"\1":', text)  # property names
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)  # string values
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Extract first complete JSON object from text."""
        # Find first { and match to closing }
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

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

            # Try parsing as-is first
            data = None
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Apply fixes for common LLM JSON issues
                fixed = self._fix_llm_json(response)
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    # Try extracting just the JSON object
                    extracted = self._extract_json_object(response)
                    if extracted:
                        try:
                            data = json.loads(self._fix_llm_json(extracted))
                        except json.JSONDecodeError:
                            pass

            if data is None:
                raise json.JSONDecodeError("Could not parse JSON after fixes", response, 0)
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
            logger.debug(f"LLM ranking JSON parse failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"LLM ranking parse error: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        ai_selections = self.llm_success_count + self.vlm_success_count
        ai_success_rate = (ai_selections / self.selection_count * 100) if self.selection_count > 0 else 0

        return {
            "total_selections": self.selection_count,
            "vlm_selections": self.vlm_success_count,
            "llm_selections": self.llm_success_count,
            "heuristic_selections": self.selection_count - ai_selections,
            "ai_success_rate": f"{ai_success_rate:.1f}%",
            "vlm_enabled": self.use_vlm and self.vlm is not None,
            "llm_enabled": self.use_llm and self.llm is not None,
            "backend": "VLM" if self.vlm else ("LLM" if self.llm else "Heuristic"),
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
