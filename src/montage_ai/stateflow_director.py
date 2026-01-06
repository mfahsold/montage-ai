"""
StateFlow Director - Deterministic Multi-State Pipeline for Creative Direction.

Based on StateFlow research (arxiv.org/html/2403.11322v1) and
Blueprint First, Model Second (arxiv.org/html/2508.02721v1).

The Creative Director is decomposed into explicit states with deterministic
transitions, enabling:
- Clear tracking of decision progress
- Backtracking on validation failure
- Bounded execution guarantees
- Separation of LLM calls from workflow logic
"""

import asyncio
import logging
import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from .prompts import (
    DirectorOutput,
    StyleConfig,
    StoryArcConfig,
    StoryArcType,
    PacingConfig,
    PacingSpeed,
    PacingVariation,
    CinematographyConfig,
    ShotVariationPriority,
    TransitionsConfig,
    TransitionType,
    EffectsConfig,
    ColorGrading,
    Mood,
    PROMPT_GUARDRAILS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# State Machine Definition
# =============================================================================

class DirectorState(Enum):
    """States in the Creative Director state machine."""
    PARSE_INTENT = "parse_intent"
    ANALYZE_FOOTAGE = "analyze_footage"
    RESOLVE_CONFLICTS = "resolve_conflicts"
    PLAN_STRUCTURE = "plan_structure"
    VALIDATE_CONSTRAINTS = "validate_constraints"
    EMIT_OUTPUT = "emit_output"

    @property
    def is_terminal(self) -> bool:
        return self == DirectorState.EMIT_OUTPUT


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IntentAnalysis:
    """Result of parsing user intent."""
    explicit: Dict[str, Any] = field(default_factory=dict)
    implicit: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)

    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0


@dataclass
class StateContext:
    """Context passed through states (Γ* in StateFlow formalization)."""
    user_request: str
    footage_info: Dict[str, Any]
    intent_analysis: Optional[Dict[str, Any]] = None
    resolved_constraints: Optional[Dict[str, Any]] = None
    structure_plan: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None


@dataclass
class StateTransitionRecord:
    """Record of a state transition."""
    from_state: DirectorState
    to_state: DirectorState
    result: Dict[str, Any]
    timestamp: float = 0.0


# =============================================================================
# State-Specific Prompts
# =============================================================================

PARSE_INTENT_PROMPT = """Analyze this video editing request and extract structured constraints.

User Request: {user_request}

Available Footage Summary:
- Total clips: {clip_count}
- Total duration: {total_duration:.1f}s
- Shot types: {shot_types}

Extract:
1. EXPLICIT constraints (directly stated): duration, style name, specific effects
2. IMPLICIT preferences (inferred from wording): mood, pacing hints
3. CONFLICTS: Any contradictions in the request
4. AMBIGUITIES: Things that need clarification

{guardrails}

Respond with ONLY valid JSON:
{{
    "explicit": {{"duration": number|null, "style": string|null, "effects": [...]}},
    "implicit": {{"mood": string|null, "pacing": string|null}},
    "conflicts": ["conflict description", ...],
    "ambiguities": ["ambiguity description", ...]
}}
"""

RESOLVE_CONFLICTS_PROMPT = """Resolve conflicts in video editing constraints.

Original Request: {user_request}
Conflicts Detected: {conflicts}
Available Footage: {total_duration:.1f}s total

Priority Order for Resolution:
1. SAFETY: min_clip_duration >= 0.5s, max reasonable duration
2. STYLE-DEFINING: Hitchcock=slow build, MTV=fast cuts (cannot violate)
3. USER PREFERENCES: Can be relaxed if conflicting with style

{guardrails}

Respond with ONLY valid JSON:
{{
    "resolved": {{
        "style": "style_name",
        "mood": "mood_value",
        "pacing_speed": "slow|medium|fast",
        "duration": number,
        "climax_position": 0.0-1.0,
        "relaxed_constraints": ["what was relaxed"]
    }}
}}
"""

PLAN_STRUCTURE_PROMPT = """Plan the story arc structure for this montage.

Style: {style}
Target Duration: {duration}s
Mood: {mood}
Available Footage Energy Distribution: {energy_distribution}

Design phases (INTRO, BUILD, CLIMAX, SUSTAIN, OUTRO) with:
- Duration for each phase
- Energy level target
- Preferred shot types

{guardrails}

Respond with ONLY valid JSON:
{{
    "phases": ["intro", "build", "climax", "sustain", "outro"],
    "phase_config": {{
        "intro": {{"duration": number, "target_energy": 0.0-1.0, "shots": ["wide", ...]}},
        ...
    }}
}}
"""


# =============================================================================
# StateFlowDirector
# =============================================================================

class StateFlowDirector:
    """
    Deterministic multi-state pipeline for creative direction.

    Implements the StateFlow pattern: ⟨S, s₀, F, δ, Γ, Ω⟩
    - S: Set of DirectorState
    - s₀: PARSE_INTENT (initial state)
    - F: EMIT_OUTPUT (final state)
    - δ: Transition function (_get_next_state)
    - Γ: Output alphabet (state results)
    - Ω: Output functions (_execute_state)
    """

    # State transition table
    TRANSITIONS = {
        DirectorState.PARSE_INTENT: {
            "success": DirectorState.ANALYZE_FOOTAGE,
            "failure": DirectorState.EMIT_OUTPUT,  # Fallback
        },
        DirectorState.ANALYZE_FOOTAGE: {
            "feasible_no_conflicts": DirectorState.PLAN_STRUCTURE,
            "feasible_with_conflicts": DirectorState.RESOLVE_CONFLICTS,
            "infeasible": DirectorState.EMIT_OUTPUT,  # Fallback with defaults
        },
        DirectorState.RESOLVE_CONFLICTS: {
            "resolved": DirectorState.PLAN_STRUCTURE,
            "unresolvable": DirectorState.EMIT_OUTPUT,
        },
        DirectorState.PLAN_STRUCTURE: {
            "planned": DirectorState.VALIDATE_CONSTRAINTS,
            "failure": DirectorState.EMIT_OUTPUT,
        },
        DirectorState.VALIDATE_CONSTRAINTS: {
            "valid": DirectorState.EMIT_OUTPUT,
            "invalid": DirectorState.RESOLVE_CONFLICTS,  # Backtrack
        },
    }

    def __init__(
        self,
        max_backtrack_attempts: int = 3,
        llm_timeout: int = 30,
    ):
        self.max_backtrack_attempts = max_backtrack_attempts
        self.llm_timeout = llm_timeout
        self.current_state = DirectorState.PARSE_INTENT
        self.state_history: List[StateTransitionRecord] = []
        self.backtrack_count = 0
        self._llm_client = None

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def run(
        self,
        user_request: str,
        footage_info: Dict[str, Any],
    ) -> DirectorOutput:
        """
        Execute the state machine and return DirectorOutput.

        Args:
            user_request: Natural language editing request
            footage_info: Dict with clips, total_duration, available_shots

        Returns:
            DirectorOutput: Structured editing instructions
        """
        self.current_state = DirectorState.PARSE_INTENT
        self.state_history = []
        self.backtrack_count = 0

        context = StateContext(
            user_request=user_request,
            footage_info=footage_info,
        )

        while not self.current_state.is_terminal:
            try:
                result = await self._execute_state(self.current_state, context)
                self._update_context(context, result)

                next_state = self._get_next_state(self.current_state, result)

                # Check for backtrack
                if self._is_backtrack(self.current_state, next_state):
                    self._record_backtrack(
                        self.current_state,
                        next_state,
                        result.get("reason", "validation failed"),
                    )

                    if self._max_backtracks_exceeded():
                        logger.warning("Max backtracks exceeded, forcing output")
                        next_state = DirectorState.EMIT_OUTPUT

                self._record_state_transition(self.current_state, next_state, result)
                self.current_state = next_state

            except Exception as e:
                logger.error(f"State {self.current_state} failed: {e}")
                # On error, try to emit output with what we have
                self.current_state = DirectorState.EMIT_OUTPUT

        return self._build_final_output(context)

    # =========================================================================
    # State Execution
    # =========================================================================

    async def _execute_state(
        self,
        state: DirectorState,
        context: StateContext,
    ) -> Dict[str, Any]:
        """Execute a single state and return result."""

        if state == DirectorState.PARSE_INTENT:
            return await self._execute_parse_intent(context)

        elif state == DirectorState.ANALYZE_FOOTAGE:
            return self._execute_analyze_footage(context)

        elif state == DirectorState.RESOLVE_CONFLICTS:
            return await self._execute_resolve_conflicts(context)

        elif state == DirectorState.PLAN_STRUCTURE:
            return await self._execute_plan_structure(context)

        elif state == DirectorState.VALIDATE_CONSTRAINTS:
            return self._execute_validate_constraints(context)

        return {"success": False}

    async def _execute_parse_intent(self, context: StateContext) -> Dict[str, Any]:
        """Parse user intent using LLM."""
        try:
            prompt = PARSE_INTENT_PROMPT.format(
                user_request=context.user_request,
                clip_count=len(context.footage_info.get("clips", [])),
                total_duration=context.footage_info.get("total_duration", 0),
                shot_types=", ".join(context.footage_info.get("available_shots", [])),
                guardrails=PROMPT_GUARDRAILS,
            )

            response = await self._llm_call(prompt)

            if response:
                parsed = self._parse_json_response(response)
                if parsed:
                    return {
                        "success": True,
                        "explicit": parsed.get("explicit", {}),
                        "implicit": parsed.get("implicit", {}),
                        "conflicts": parsed.get("conflicts", []),
                        "ambiguities": parsed.get("ambiguities", []),
                    }

            # Fallback to regex parsing
            return self._fallback_parse_intent(context.user_request)

        except Exception as e:
            logger.warning(f"LLM parse_intent failed: {e}, using fallback")
            return self._fallback_parse_intent(context.user_request)

    def _execute_analyze_footage(self, context: StateContext) -> Dict[str, Any]:
        """Analyze footage feasibility (deterministic, no LLM)."""
        intent = context.intent_analysis or {}
        explicit = intent.get("explicit", {})
        conflicts = intent.get("conflicts", [])

        target_duration = explicit.get("duration") or 60.0
        total_footage = context.footage_info.get("total_duration", 0)

        # Check duration feasibility (need at least target duration worth of footage)
        feasible = total_footage >= target_duration

        # Check phase coverage
        phase_coverage = self._validate_phase_coverage(
            context.footage_info,
            self._get_default_phase_requirements(target_duration),
        )

        has_conflicts = len(conflicts) > 0 or not phase_coverage.get("feasible", True)

        return {
            "feasible": feasible,
            "has_conflicts": has_conflicts,
            "coverage": phase_coverage.get("coverage", {}),
            "blocking_issues": phase_coverage.get("issues", []),
        }

    async def _execute_resolve_conflicts(self, context: StateContext) -> Dict[str, Any]:
        """Resolve conflicts using LLM or rules."""
        intent = context.intent_analysis or {}
        conflicts = intent.get("conflicts", [])

        # If no conflicts, just merge explicit/implicit
        if not conflicts:
            explicit = intent.get("explicit", {})
            implicit = intent.get("implicit", {})

            return {
                "resolved": True,
                "style": explicit.get("style", "dynamic"),
                "mood": implicit.get("mood", "energetic"),
                "pacing_speed": self._infer_pacing(explicit.get("style")),
                "duration": explicit.get("duration", 60.0),
                "climax_position": explicit.get("climax_position", 0.75),
            }

        # Use LLM to resolve conflicts
        try:
            prompt = RESOLVE_CONFLICTS_PROMPT.format(
                user_request=context.user_request,
                conflicts=json.dumps(conflicts),
                total_duration=context.footage_info.get("total_duration", 0),
                guardrails=PROMPT_GUARDRAILS,
            )

            response = await self._llm_call(prompt)
            if response:
                parsed = self._parse_json_response(response)
                if parsed and "resolved" in parsed:
                    return {"resolved": True, **parsed["resolved"]}

        except Exception as e:
            logger.warning(f"LLM resolve_conflicts failed: {e}")

        # Fallback: apply priority rules
        return self._fallback_resolve_conflicts(context)

    async def _execute_plan_structure(self, context: StateContext) -> Dict[str, Any]:
        """Plan story arc structure."""
        resolved = context.resolved_constraints or {}

        style = resolved.get("style", "dynamic")
        duration = resolved.get("duration", 60.0)
        mood = resolved.get("mood", "energetic")

        # Calculate phase durations based on style
        phase_ratios = self._get_phase_ratios(style)
        phase_config = {}

        for phase, ratio in phase_ratios.items():
            phase_config[phase] = {
                "duration": duration * ratio,
                "target_energy": self._get_phase_energy(phase, style),
                "shots": self._get_phase_shots(phase),
            }

        return {
            "planned": True,
            "phases": list(phase_ratios.keys()),
            "phase_config": phase_config,
            "structure": {
                "total_duration": duration,
                "climax_position": resolved.get("climax_position", 0.75),
            },
        }

    def _execute_validate_constraints(self, context: StateContext) -> Dict[str, Any]:
        """Validate all constraints are satisfiable (deterministic)."""
        structure = context.structure_plan or {}
        phase_config = structure.get("phase_config", {})

        issues = []

        # Check each phase can be filled
        clips = context.footage_info.get("clips", [])
        total_available = sum(c.get("duration", 0) for c in clips)
        total_required = sum(p.get("duration", 0) for p in phase_config.values())

        if total_available < total_required * 0.8:
            issues.append(f"Not enough footage: {total_available:.1f}s < {total_required:.1f}s required")

        # Check energy distribution
        high_energy_clips = [c for c in clips if c.get("energy", 0) >= 0.7]
        if not high_energy_clips and "climax" in phase_config:
            issues.append("No high-energy clips available for climax")

        feasible = len(issues) == 0

        return {
            "feasible": feasible,
            "blocking_issues": issues,
        }

    # =========================================================================
    # State Transitions
    # =========================================================================

    def _get_next_state(
        self,
        current: DirectorState,
        result: Dict[str, Any],
    ) -> DirectorState:
        """Determine next state based on current state and result."""

        if current == DirectorState.PARSE_INTENT:
            if result.get("success", False):
                return DirectorState.ANALYZE_FOOTAGE
            return DirectorState.EMIT_OUTPUT

        elif current == DirectorState.ANALYZE_FOOTAGE:
            if not result.get("feasible", False):
                return DirectorState.EMIT_OUTPUT
            if result.get("has_conflicts", False):
                return DirectorState.RESOLVE_CONFLICTS
            return DirectorState.PLAN_STRUCTURE

        elif current == DirectorState.RESOLVE_CONFLICTS:
            if result.get("resolved", False):
                return DirectorState.PLAN_STRUCTURE
            return DirectorState.EMIT_OUTPUT

        elif current == DirectorState.PLAN_STRUCTURE:
            if result.get("planned", False):
                return DirectorState.VALIDATE_CONSTRAINTS
            return DirectorState.EMIT_OUTPUT

        elif current == DirectorState.VALIDATE_CONSTRAINTS:
            if result.get("feasible", False):
                return DirectorState.EMIT_OUTPUT
            # Backtrack
            return DirectorState.RESOLVE_CONFLICTS

        return DirectorState.EMIT_OUTPUT

    def _is_backtrack(self, current: DirectorState, next_state: DirectorState) -> bool:
        """Check if transition is a backtrack."""
        state_order = list(DirectorState)
        current_idx = state_order.index(current)
        next_idx = state_order.index(next_state)
        return next_idx < current_idx

    # =========================================================================
    # Output Building
    # =========================================================================

    def _build_final_output(self, context: StateContext) -> DirectorOutput:
        """Build DirectorOutput from accumulated context."""
        resolved = context.resolved_constraints or {}
        structure = context.structure_plan or {}

        style_name = resolved.get("style", "dynamic")
        mood_str = resolved.get("mood", "energetic")

        # Map to enums with fallbacks
        try:
            mood = Mood(mood_str.lower())
        except ValueError:
            mood = Mood.ENERGETIC

        pacing_speed_str = resolved.get("pacing_speed", "medium")
        try:
            pacing_speed = PacingSpeed(pacing_speed_str.lower())
        except ValueError:
            pacing_speed = PacingSpeed.MEDIUM

        return DirectorOutput(
            director_commentary=f"StateFlow pipeline completed. Style: {style_name}, Mood: {mood_str}. "
                               f"States traversed: {len(self.state_history)}, Backtracks: {self.backtrack_count}",
            style=StyleConfig(
                name=style_name,
                mood=mood,
            ),
            story_arc=StoryArcConfig(
                type=StoryArcType.THREE_ACT,
                tension_target=self._get_tension_target(style_name),
                climax_position=resolved.get("climax_position", 0.75),
            ),
            pacing=PacingConfig(
                speed=pacing_speed,
                variation=PacingVariation.MODERATE,
                intro_duration_beats=8,
                climax_intensity=0.9,
            ),
            cinematography=CinematographyConfig(
                prefer_wide_shots=style_name in ["documentary", "minimalist"],
                prefer_high_action=style_name in ["action", "mtv"],
                match_cuts_enabled=True,
                invisible_cuts_enabled=style_name == "hitchcock",
                shot_variation_priority=ShotVariationPriority.HIGH,
            ),
            transitions=TransitionsConfig(
                type=self._get_transition_type(style_name),
                crossfade_duration_sec=0.5 if style_name != "mtv" else None,
            ),
            effects=EffectsConfig(
                color_grading=self._get_color_grading(style_name),
                stabilization=False,
                upscale=False,
                sharpness_boost=False,
            ),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _update_context(self, context: StateContext, result: Dict[str, Any]) -> None:
        """Update context with state result."""
        if self.current_state == DirectorState.PARSE_INTENT:
            context.intent_analysis = result
        elif self.current_state == DirectorState.RESOLVE_CONFLICTS:
            context.resolved_constraints = result
        elif self.current_state == DirectorState.PLAN_STRUCTURE:
            context.structure_plan = result
        elif self.current_state == DirectorState.VALIDATE_CONSTRAINTS:
            context.validation_result = result

    def _record_state_transition(
        self,
        from_state: DirectorState,
        to_state: DirectorState,
        result: Dict[str, Any],
    ) -> None:
        """Record a state transition in history."""
        import time
        self.state_history.append(StateTransitionRecord(
            from_state=from_state,
            to_state=to_state,
            result=result,
            timestamp=time.time(),
        ))

    def _record_backtrack(
        self,
        from_state: DirectorState,
        to_state: DirectorState,
        reason: str,
    ) -> None:
        """Record a backtrack event."""
        self.backtrack_count += 1
        logger.info(f"Backtrack #{self.backtrack_count}: {from_state} -> {to_state}, reason: {reason}")

    def _max_backtracks_exceeded(self) -> bool:
        """Check if max backtracks exceeded."""
        return self.backtrack_count >= self.max_backtrack_attempts

    async def _llm_call(self, prompt: str) -> Optional[str]:
        """Make LLM call (to be integrated with CreativeDirector's LLM)."""
        # This will be integrated with the existing LLM infrastructure
        # For now, return None to trigger fallback
        return None

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        try:
            # Strip markdown code fences
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def _fallback_parse_intent(self, user_request: str) -> Dict[str, Any]:
        """Fallback intent parsing using regex."""
        explicit = {}
        implicit = {}

        # Extract duration
        duration_match = re.search(r'(\d+)\s*(?:second|sec|s\b)', user_request, re.I)
        if duration_match:
            explicit["duration"] = float(duration_match.group(1))

        # Extract style
        styles = ["hitchcock", "mtv", "documentary", "action", "minimalist", "wes_anderson", "dynamic"]
        for style in styles:
            if style.lower() in user_request.lower():
                explicit["style"] = style
                break

        # Extract mood hints
        mood_keywords = {
            "suspenseful": ["suspense", "thriller", "tense", "hitchcock"],
            "energetic": ["energetic", "dynamic", "fast", "action", "mtv"],
            "calm": ["calm", "peaceful", "slow", "minimalist"],
            "playful": ["fun", "playful", "wes_anderson"],
        }

        for mood, keywords in mood_keywords.items():
            if any(kw in user_request.lower() for kw in keywords):
                implicit["mood"] = mood
                break

        # Extract climax position
        climax_match = re.search(r'climax.*?(\d+)\s*%', user_request, re.I)
        if climax_match:
            explicit["climax_position"] = float(climax_match.group(1)) / 100.0

        return {
            "success": True,
            "explicit": explicit,
            "implicit": implicit,
            "conflicts": [],
            "ambiguities": [],
        }

    def _fallback_resolve_conflicts(self, context: StateContext) -> Dict[str, Any]:
        """Fallback conflict resolution using priority rules."""
        intent = context.intent_analysis or {}
        explicit = intent.get("explicit", {})
        implicit = intent.get("implicit", {})

        style = explicit.get("style", "dynamic")

        return {
            "resolved": True,
            "style": style,
            "mood": implicit.get("mood", self._get_default_mood(style)),
            "pacing_speed": self._infer_pacing(style),
            "duration": min(
                explicit.get("duration", 60.0),
                context.footage_info.get("total_duration", 60.0),
            ),
            "climax_position": explicit.get("climax_position", 0.75),
        }

    def _validate_duration_constraint(
        self,
        footage_info: Dict[str, Any],
        target_duration: float,
    ) -> Dict[str, Any]:
        """Validate duration constraint."""
        total = footage_info.get("total_duration", 0)
        # Need at least as much footage as target duration
        feasible = total >= target_duration

        return {
            "feasible": feasible,
            "total_available": total,
            "target": target_duration,
        }

    def _validate_phase_coverage(
        self,
        footage_info: Dict[str, Any],
        phase_requirements: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Validate if footage can cover all phases."""
        clips = footage_info.get("clips", [])
        coverage = {}
        issues = []

        for phase, req in phase_requirements.items():
            min_dur = req.get("min_duration", 0)
            max_energy = req.get("max_energy", 1.0)
            min_energy = req.get("min_energy", 0.0)

            # Find clips that match phase requirements
            matching = [
                c for c in clips
                if min_energy <= c.get("energy", 0.5) <= max_energy
            ]

            available_dur = sum(c.get("duration", 0) for c in matching)
            coverage[phase] = {
                "required": min_dur,
                "available": available_dur,
                "clips": len(matching),
            }

            if available_dur < min_dur * 0.8:
                issues.append(f"{phase}: need {min_dur}s, have {available_dur}s")

        return {
            "feasible": len(issues) == 0,
            "coverage": coverage,
            "issues": issues,
        }

    def _get_default_phase_requirements(self, duration: float) -> Dict[str, Dict]:
        """Get default phase requirements based on duration."""
        return {
            "intro": {"min_duration": duration * 0.15, "max_energy": 0.5},
            "build": {"min_duration": duration * 0.25},
            "climax": {"min_duration": duration * 0.20, "min_energy": 0.6},
            "sustain": {"min_duration": duration * 0.25},
            "outro": {"min_duration": duration * 0.15, "max_energy": 0.5},
        }

    def _get_phase_ratios(self, style: str) -> Dict[str, float]:
        """Get phase duration ratios for style."""
        ratios = {
            "hitchcock": {"intro": 0.20, "build": 0.30, "climax": 0.15, "sustain": 0.20, "outro": 0.15},
            "mtv": {"intro": 0.10, "build": 0.20, "climax": 0.30, "sustain": 0.30, "outro": 0.10},
            "documentary": {"intro": 0.15, "build": 0.25, "climax": 0.20, "sustain": 0.25, "outro": 0.15},
        }
        return ratios.get(style, {"intro": 0.15, "build": 0.25, "climax": 0.20, "sustain": 0.25, "outro": 0.15})

    def _get_phase_energy(self, phase: str, style: str) -> float:
        """Get target energy for phase."""
        energies = {
            "intro": 0.3,
            "build": 0.5,
            "climax": 0.9,
            "sustain": 0.7,
            "outro": 0.3,
        }

        base = energies.get(phase, 0.5)

        # Style adjustments
        if style == "mtv":
            return min(base + 0.2, 1.0)
        elif style == "minimalist":
            return max(base - 0.2, 0.1)

        return base

    def _get_phase_shots(self, phase: str) -> List[str]:
        """Get preferred shot types for phase."""
        shots = {
            "intro": ["wide", "establishing"],
            "build": ["medium", "wide"],
            "climax": ["close", "medium"],
            "sustain": ["medium", "close"],
            "outro": ["wide", "establishing"],
        }
        return shots.get(phase, ["medium"])

    def _infer_pacing(self, style: Optional[str]) -> str:
        """Infer pacing speed from style."""
        pacing_map = {
            "hitchcock": "slow",
            "mtv": "very_fast",
            "action": "fast",
            "documentary": "medium",
            "minimalist": "very_slow",
            "wes_anderson": "medium",
        }
        return pacing_map.get(style or "", "medium")

    def _get_default_mood(self, style: str) -> str:
        """Get default mood for style."""
        mood_map = {
            "hitchcock": "suspenseful",
            "mtv": "energetic",
            "action": "energetic",
            "documentary": "calm",
            "minimalist": "calm",
            "wes_anderson": "playful",
        }
        return mood_map.get(style, "energetic")

    def _get_tension_target(self, style: str) -> float:
        """Get tension target for style."""
        tension_map = {
            "hitchcock": 0.9,
            "mtv": 0.7,
            "action": 0.8,
            "documentary": 0.5,
            "minimalist": 0.3,
        }
        return tension_map.get(style, 0.6)

    def _get_transition_type(self, style: str) -> TransitionType:
        """Get transition type for style."""
        type_map = {
            "hitchcock": TransitionType.MIXED,
            "mtv": TransitionType.HARD_CUTS,
            "action": TransitionType.HARD_CUTS,
            "documentary": TransitionType.CROSSFADE,
            "minimalist": TransitionType.CROSSFADE,
        }
        return type_map.get(style, TransitionType.MIXED)

    def _get_color_grading(self, style: str) -> ColorGrading:
        """Get color grading for style."""
        grading_map = {
            "hitchcock": ColorGrading.HIGH_CONTRAST,
            "mtv": ColorGrading.VIBRANT,
            "action": ColorGrading.HIGH_CONTRAST,
            "documentary": ColorGrading.NEUTRAL,
            "minimalist": ColorGrading.DESATURATED,
            "wes_anderson": ColorGrading.WARM,
        }
        return grading_map.get(style, ColorGrading.NEUTRAL)

    def _relax_constraints(
        self,
        constraints: Dict[str, Any],
        footage_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Relax constraints when validation fails."""
        relaxed = constraints.copy()

        total = footage_info.get("total_duration", 60.0)

        # Relax duration to what's available
        if constraints.get("duration", 0) > total:
            relaxed["duration"] = total * 0.9

        return relaxed


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_director_output(
    user_request: str,
    footage_info: Dict[str, Any],
    max_backtracks: int = 3,
) -> DirectorOutput:
    """Convenience function to run StateFlowDirector."""
    director = StateFlowDirector(max_backtrack_attempts=max_backtracks)
    return await director.run(user_request, footage_info)
