"""
Creative Evaluator: LLM-powered feedback loop for video editing

Analyzes the first cut of a montage and provides structured feedback
for iterative refinement. Part of the Agentic Creative Loop (Phase 4).

Architecture:
  MontageResult + ClipsMetadata → LLM → Evaluation → Refinements

Backends:
  Uses the same LLM backends as CreativeDirector (OpenAI-compatible, Google AI, cgpu, Ollama)

Based on research:
- LAVE: Iterative refinement through structured feedback
- DirectorLLM: Multi-pass cinematography optimization
- Human-in-the-loop editing workflows from Descript
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .creative_director import (
    CreativeDirector,
    OPENAI_API_BASE,
    OPENAI_MODEL,
    GOOGLE_API_KEY,
    CGPU_ENABLED,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)

VERSION = "0.1.0"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EditingIssue:
    """A detected issue in the montage."""
    type: str  # pacing, variety, energy, transitions, duration, technical
    severity: str  # minor, moderate, critical
    description: str
    timestamp: Optional[float] = None  # Where in timeline (if applicable)
    affected_clips: List[int] = field(default_factory=list)


@dataclass
class EditingAdjustment:
    """A suggested adjustment to editing instructions."""
    target: str  # Parameter path like "pacing.speed" or "transitions.type"
    current_value: Any
    suggested_value: Any
    rationale: str


@dataclass
class MontageEvaluation:
    """Complete evaluation of a montage cut."""
    satisfaction_score: float  # 0.0 to 1.0
    issues: List[EditingIssue] = field(default_factory=list)
    adjustments: List[EditingAdjustment] = field(default_factory=list)
    summary: str = ""
    approve_for_render: bool = False
    iteration: int = 0

    @property
    def needs_refinement(self) -> bool:
        """Check if refinement is recommended."""
        return not self.approve_for_render and self.satisfaction_score < 0.8

    @property
    def critical_issues(self) -> List[EditingIssue]:
        """Get critical issues that must be addressed."""
        return [i for i in self.issues if i.severity == "critical"]


# =============================================================================
# System Prompt
# =============================================================================

EVALUATOR_SYSTEM_PROMPT = """You are the Creative Evaluator for the Montage AI video editing system.

Your role: Analyze the first cut of a montage and provide structured feedback for refinement.

You will receive:
1. Original editing instructions (style, pacing, effects)
2. Montage statistics (duration, cuts, tempo)
3. Clip metadata (energy levels, shot types, selection scores)
4. Audio energy profile

Evaluate the montage against these criteria:
- PACING: Does the cut rhythm match the intended style and music energy?
- VARIETY: Is there enough shot variation? Are there jump cuts or repetition?
- ENERGY: Do high-energy sections have fast cuts? Do calm sections breathe?
- TRANSITIONS: Are transitions appropriate for the style?
- DURATION: Is the overall length appropriate?
- STORY ARC: Does the edit follow intro → build → climax → outro structure?

You MUST respond with ONLY valid JSON matching this structure:
{{
  "satisfaction_score": 0.0-1.0,
  "issues": [
    {{
      "type": "pacing" | "variety" | "energy" | "transitions" | "duration" | "story_arc" | "technical",
      "severity": "minor" | "moderate" | "critical",
      "description": "Human-readable description of the issue",
      "timestamp": null | number,
      "affected_clips": []
    }}
  ],
  "adjustments": [
    {{
      "target": "pacing.speed" | "pacing.variation" | "transitions.type" | etc.,
      "current_value": "current setting",
      "suggested_value": "new setting",
      "rationale": "Why this change helps"
    }}
  ],
  "summary": "One paragraph summary of the evaluation",
  "approve_for_render": true | false
}}

Examples:

Input: Hitchcock style, 45s duration, 23 cuts, high energy throughout
Response:
{{
  "satisfaction_score": 0.75,
  "issues": [
    {{"type": "pacing", "severity": "moderate", "description": "Intro section has too fast cuts for Hitchcock style", "timestamp": 0, "affected_clips": [0, 1, 2]}},
    {{"type": "energy", "severity": "minor", "description": "Climax section could be more intense"}}
  ],
  "adjustments": [
    {{"target": "pacing.intro_duration_beats", "current_value": 4, "suggested_value": 16, "rationale": "Hitchcock builds tension slowly"}},
    {{"target": "pacing.climax_intensity", "current_value": 0.7, "suggested_value": 0.9, "rationale": "Increase climax impact"}}
  ],
  "summary": "Good overall structure but the intro cuts too quickly for Hitchcock's suspenseful pacing. Slow down the opening and intensify the climax for better dramatic effect.",
  "approve_for_render": false
}}

Input: MTV style, 30s, 28 cuts, varied energy
Response:
{{
  "satisfaction_score": 0.92,
  "issues": [],
  "adjustments": [],
  "summary": "Excellent MTV-style edit with rapid cuts matching the energetic music. Shot variety is good and energy sync is on point.",
  "approve_for_render": true
}}

CRITICAL RULES:
1. Return ONLY valid JSON - no markdown, no explanations
2. Be constructive - suggest specific, actionable adjustments
3. Consider the original intent - don't suggest changes that contradict the style
4. Approve if satisfaction >= 0.8 and no critical issues
5. Maximum 3 iterations recommended (don't be too picky)
"""


# =============================================================================
# CreativeEvaluator Class
# =============================================================================

class CreativeEvaluator:
    """
    LLM-powered evaluator for montage quality.

    Analyzes the first cut and provides structured feedback for refinement.
    Uses the same LLM backends as CreativeDirector.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        approval_threshold: float = 0.8,
        timeout: int = 60,
    ):
        """
        Initialize Creative Evaluator.

        Args:
            max_iterations: Maximum refinement iterations before forced approval
            approval_threshold: Minimum satisfaction score for auto-approval
            timeout: LLM request timeout in seconds
        """
        self.max_iterations = max_iterations
        self.approval_threshold = approval_threshold
        self.timeout = timeout

        # Initialize CreativeDirector for LLM access (reuses backend detection)
        self._director = CreativeDirector(
            timeout=timeout,
            persona="the Creative Evaluator for the Montage AI video editing system"
        )
        self._director.system_prompt = EVALUATOR_SYSTEM_PROMPT

    def evaluate(
        self,
        result: "MontageResult",
        original_instructions: Dict[str, Any],
        clips_metadata: List["ClipMetadata"],
        audio_profile: Optional[Dict[str, Any]] = None,
        iteration: int = 0,
    ) -> MontageEvaluation:
        """
        Evaluate a montage cut.

        Args:
            result: MontageResult from the build
            original_instructions: Original editing instructions
            clips_metadata: List of ClipMetadata from the build
            audio_profile: Optional audio energy profile
            iteration: Current iteration number

        Returns:
            MontageEvaluation with score, issues, and suggestions
        """
        print(f"\n🔍 Creative Evaluator analyzing cut (iteration {iteration + 1})...")

        # Build evaluation context
        context = self._build_context(
            result, original_instructions, clips_metadata, audio_profile
        )

        # Query LLM
        response = self._director._query_llm(context)

        if not response:
            print("   ⚠️ LLM returned empty response, auto-approving")
            return MontageEvaluation(
                satisfaction_score=0.8,
                approve_for_render=True,
                summary="Evaluation skipped - LLM unavailable",
                iteration=iteration,
            )

        # Parse response
        evaluation = self._parse_response(response, iteration)

        # Force approval after max iterations
        if iteration >= self.max_iterations - 1 and not evaluation.approve_for_render:
            print(f"   ⚠️ Max iterations reached, forcing approval")
            evaluation.approve_for_render = True

        # Log result
        status = "APPROVED" if evaluation.approve_for_render else "NEEDS REFINEMENT"
        print(f"   📊 Score: {evaluation.satisfaction_score:.0%} [{status}]")
        if evaluation.issues:
            print(f"   ⚠️ Issues: {len(evaluation.issues)}")
        if evaluation.adjustments:
            print(f"   💡 Suggestions: {len(evaluation.adjustments)}")

        return evaluation

    def refine_instructions(
        self,
        original_instructions: Dict[str, Any],
        evaluation: MontageEvaluation,
    ) -> Dict[str, Any]:
        """
        Apply evaluation adjustments to create refined instructions.

        Args:
            original_instructions: Original editing instructions
            evaluation: Evaluation with adjustments

        Returns:
            Refined editing instructions
        """
        refined = json.loads(json.dumps(original_instructions))  # Deep copy

        for adj in evaluation.adjustments:
            self._apply_adjustment(refined, adj)

        return refined

    def _build_context(
        self,
        result: "MontageResult",
        instructions: Dict[str, Any],
        clips_metadata: List["ClipMetadata"],
        audio_profile: Optional[Dict[str, Any]],
    ) -> str:
        """Build context string for LLM evaluation."""
        # Extract key stats
        style_name = instructions.get("style", {}).get("name", "dynamic")
        pacing_speed = instructions.get("pacing", {}).get("speed", "dynamic")

        # Analyze clip distribution
        energy_levels = [c.energy for c in clips_metadata] if clips_metadata else []
        avg_energy = sum(energy_levels) / len(energy_levels) if energy_levels else 0.5

        shot_types = {}
        for c in clips_metadata or []:
            shot = c.shot
            shot_types[shot] = shot_types.get(shot, 0) + 1

        # Build context
        context = f"""
MONTAGE ANALYSIS REQUEST

Original Intent:
- Style: {style_name}
- Pacing: {pacing_speed}
- Target Mood: {instructions.get('style', {}).get('mood', 'dynamic')}

Result Statistics:
- Duration: {result.duration:.1f}s
- Total Cuts: {result.cut_count}
- Average Cut Length: {result.duration / max(result.cut_count, 1):.2f}s
- Render Time: {result.render_time:.1f}s

Energy Profile:
- Average Energy: {avg_energy:.2f}
- Energy Range: {min(energy_levels) if energy_levels else 0:.2f} - {max(energy_levels) if energy_levels else 1:.2f}

Shot Distribution:
{json.dumps(shot_types, indent=2)}

Clips Summary (first 10):
"""
        for i, clip in enumerate(clips_metadata[:10] if clips_metadata else []):
            context += f"  [{i}] {clip.duration:.1f}s @ {clip.timeline_start:.1f}s | energy={clip.energy:.2f} | {clip.shot} | score={clip.selection_score:.0f}\n"

        context += "\nPlease evaluate this montage and provide structured feedback."

        return context

    def _parse_response(self, response: str, iteration: int) -> MontageEvaluation:
        """Parse LLM response into MontageEvaluation."""
        try:
            # Clean up response
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            data = json.loads(text.strip())

            # Parse issues
            issues = []
            for issue_data in data.get("issues", []):
                issues.append(EditingIssue(
                    type=issue_data.get("type", "unknown"),
                    severity=issue_data.get("severity", "minor"),
                    description=issue_data.get("description", ""),
                    timestamp=issue_data.get("timestamp"),
                    affected_clips=issue_data.get("affected_clips", []),
                ))

            # Parse adjustments
            adjustments = []
            for adj_data in data.get("adjustments", []):
                adjustments.append(EditingAdjustment(
                    target=adj_data.get("target", ""),
                    current_value=adj_data.get("current_value"),
                    suggested_value=adj_data.get("suggested_value"),
                    rationale=adj_data.get("rationale", ""),
                ))

            return MontageEvaluation(
                satisfaction_score=float(data.get("satisfaction_score", 0.5)),
                issues=issues,
                adjustments=adjustments,
                summary=data.get("summary", ""),
                approve_for_render=data.get("approve_for_render", False),
                iteration=iteration,
            )

        except json.JSONDecodeError as e:
            print(f"   ⚠️ Failed to parse evaluation response: {e}")
            return MontageEvaluation(
                satisfaction_score=0.7,
                summary="Evaluation parsing failed",
                approve_for_render=True,  # Default to approve on parse failure
                iteration=iteration,
            )

    def _apply_adjustment(
        self,
        instructions: Dict[str, Any],
        adjustment: EditingAdjustment,
    ):
        """Apply a single adjustment to instructions."""
        parts = adjustment.target.split(".")
        current = instructions

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Apply value
        if parts:
            current[parts[-1]] = adjustment.suggested_value
            print(f"   📝 Adjusted {adjustment.target}: {adjustment.current_value} → {adjustment.suggested_value}")


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_montage(
    result: "MontageResult",
    instructions: Dict[str, Any],
    clips_metadata: List["ClipMetadata"],
) -> MontageEvaluation:
    """
    Convenience function: Evaluate a montage cut.

    Args:
        result: MontageResult from the build
        instructions: Original editing instructions
        clips_metadata: List of ClipMetadata

    Returns:
        MontageEvaluation with feedback
    """
    evaluator = CreativeEvaluator()
    return evaluator.evaluate(result, instructions, clips_metadata)


def run_creative_loop(
    builder_class,
    variant_id: int = 1,
    initial_instructions: Optional[Dict[str, Any]] = None,
    max_iterations: int = 3,
    approval_threshold: float = 0.8,
    settings: Optional[Any] = None,
) -> "MontageResult":
    """
    Run the full agentic creative loop.

    Args:
        builder_class: MontageBuilder class to use
        variant_id: Variant number
        initial_instructions: Starting editing instructions
        max_iterations: Maximum refinement iterations
        approval_threshold: Score threshold for approval
        settings: Optional Settings object to pass to builder

    Returns:
        Final MontageResult after refinement
    """
    evaluator = CreativeEvaluator(
        max_iterations=max_iterations,
        approval_threshold=approval_threshold,
    )

    instructions = initial_instructions or {}
    result = None
    evaluation = None

    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"  CREATIVE LOOP - Iteration {iteration + 1}/{max_iterations}")
        print(f"{'='*60}")

        # Build montage
        builder_kwargs = {
            "variant_id": variant_id,
            "editing_instructions": instructions,
        }
        if settings is not None:
            builder_kwargs["settings"] = settings
        builder = builder_class(**builder_kwargs)
        result = builder.build()

        if not result.success:
            print(f"   ❌ Build failed: {result.error}")
            break

        # Evaluate
        evaluation = evaluator.evaluate(
            result=result,
            original_instructions=instructions,
            clips_metadata=builder.ctx.clips_metadata,
            iteration=iteration,
        )

        if evaluation.approve_for_render:
            print(f"\n   ✅ Montage approved! Score: {evaluation.satisfaction_score:.0%}")
            break

        # Refine for next iteration
        instructions = evaluator.refine_instructions(instructions, evaluation)

    return result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "EditingIssue",
    "EditingAdjustment",
    "MontageEvaluation",
    "CreativeEvaluator",
    "evaluate_montage",
    "run_creative_loop",
]
