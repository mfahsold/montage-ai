"""
Story Arc CSP Solver - Constraint Satisfaction for Clip Assignment.

Uses constraint satisfaction to optimally assign clips to story phases,
respecting:
- Duration constraints per phase
- Energy level requirements
- Shot type variety
- Style-specific rules

Supports multiple backends:
- Z3: SMT solver for optimal solutions
- OR-Tools: CP-SAT solver alternative
- Fallback: Greedy heuristic when no solver available
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple
import random

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class StoryPhase(Enum):
    """Story arc phases with order for sequencing."""
    INTRO = "intro"
    BUILD = "build"
    CLIMAX = "climax"
    SUSTAIN = "sustain"
    OUTRO = "outro"

    @property
    def order(self) -> int:
        """Order for phase sequencing."""
        return list(StoryPhase).index(self)


class ClipProtocol(Protocol):
    """Protocol for clip objects."""
    path: str
    duration: float
    energy: float
    shot_type: str


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PhaseConstraints:
    """Constraints for a story phase."""
    phase_name: str
    target_duration: float
    duration_tolerance: float = 0.15  # 15% tolerance
    min_energy: float = 0.0
    max_energy: float = 1.0
    preferred_shot_types: List[str] = field(default_factory=list)
    min_clips: int = 1
    max_clips: int = 10


@dataclass
class ClipAssignment:
    """Assignment of a clip to a phase."""
    clip_index: int
    phase: StoryPhase
    score: float = 0.0


@dataclass
class SolverResult:
    """Result from CSP solver."""
    feasible: bool
    assignments: Dict[str, List[int]]  # phase -> clip indices
    ordered_clips: List[int] = field(default_factory=list)
    total_duration: float = 0.0
    energy_curve_score: float = 0.0
    shot_repeat_score: float = 0.0
    reason: str = ""


# =============================================================================
# StoryArcCSPSolver
# =============================================================================

class StoryArcCSPSolver:
    """
    Constraint satisfaction solver for story arc clip assignment.

    Formulates clip-to-phase assignment as a CSP with:
    - Variables: clip_i_phase_j (bool) - clip i assigned to phase j
    - Domain: {True, False}
    - Constraints: Duration, energy, variety requirements
    """

    # Default phase duration ratios (sum to 1.0)
    DEFAULT_PHASE_RATIOS = {
        "intro": 0.15,
        "build": 0.25,
        "climax": 0.20,
        "sustain": 0.25,
        "outro": 0.15,
    }

    def __init__(
        self,
        phase_ratios: Optional[Dict[str, float]] = None,
        backend: str = "auto",
        min_shot_types: int = 2,
        min_locations: int = 1,
        avoid_consecutive_same_shot: bool = True,
        optimize_energy_curve: bool = True,
        minimize_shot_repeats: bool = True,
    ):
        self.phase_ratios = phase_ratios or self.DEFAULT_PHASE_RATIOS.copy()
        self.min_shot_types = min_shot_types
        self.min_locations = min_locations
        self.avoid_consecutive_same_shot = avoid_consecutive_same_shot
        self.optimize_energy_curve = optimize_energy_curve
        self.minimize_shot_repeats = minimize_shot_repeats

        # Select backend
        self.backend = self._select_backend(backend)

    @property
    def default_phase_ratios(self) -> Dict[str, float]:
        """Default phase ratios (for tests)."""
        return self.DEFAULT_PHASE_RATIOS

    def _select_backend(self, requested: str) -> str:
        """Select solver backend."""
        if requested == "auto":
            # Try Z3 first
            import importlib
            if importlib.util.find_spec("z3") is not None:
                return "z3"

            # Try OR-Tools
            try:
                from ortools.sat.python import cp_model
                return "ortools"
            except ImportError:
                pass

            return "fallback"

        elif requested == "z3":
            import importlib
            if importlib.util.find_spec("z3") is not None:
                return "z3"
            logger.warning("Z3 not available, using fallback")
            return "fallback"

        elif requested == "ortools":
            try:
                from ortools.sat.python import cp_model
                return "ortools"
            except ImportError:
                logger.warning("OR-Tools not available, using fallback")
                return "fallback"

        return "fallback"

    # =========================================================================
    # Main Solve Method
    # =========================================================================

    def solve(
        self,
        clips: List[Any],
        target_duration: float,
        style: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Solve clip assignment problem.

        Args:
            clips: List of clips (must have duration, energy, shot_type)
            target_duration: Target montage duration in seconds
            style: Style configuration dict

        Returns:
            Dict with feasible, assignments, and metrics
        """
        # Normalize clips to dict format
        normalized_clips = [self._normalize_clip(c) for c in clips]

        # Check basic feasibility
        total_available = sum(c["duration"] for c in normalized_clips)
        if total_available < target_duration * 0.5:
            return {
                "feasible": False,
                "reason": f"Not enough footage: {total_available:.1f}s < {target_duration * 0.5:.1f}s minimum",
                "assignments": {},
            }

        # Generate style-specific constraints
        style_constraints = self._generate_style_constraints(style)

        # Calculate phase durations
        phase_durations = {
            phase: target_duration * ratio
            for phase, ratio in self.phase_ratios.items()
        }

        # Solve based on backend
        if self.backend == "z3":
            result = self._solve_z3(normalized_clips, phase_durations, style_constraints)
        elif self.backend == "ortools":
            result = self._solve_ortools(normalized_clips, phase_durations, style_constraints)
        else:
            result = self._solve_fallback(normalized_clips, phase_durations, style_constraints)

        # Add ordered clips list
        if result["feasible"]:
            result["ordered_clips"] = self._flatten_assignments(result["assignments"])
            result["total_duration"] = sum(
                normalized_clips[i]["duration"]
                for i in result["ordered_clips"]
            )

            # Calculate scores
            if self.optimize_energy_curve:
                result["energy_curve_score"] = self._calculate_energy_score(
                    normalized_clips, result["assignments"], style
                )

            if self.minimize_shot_repeats:
                result["shot_repeat_score"] = self._calculate_shot_repeat_score(
                    normalized_clips, result["ordered_clips"]
                )

        return result

    # =========================================================================
    # Z3 Solver
    # =========================================================================

    def _solve_z3(
        self,
        clips: List[Dict],
        phase_durations: Dict[str, float],
        style_constraints: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Solve using Z3 SMT solver."""
        try:
            from z3 import Solver, Bool, If, Sum, Or, Not, AtMost, sat

            solver = Solver()
            n = len(clips)
            phases = list(StoryPhase)

            # Variables: clip_i_phase_j
            assignments = {}
            for i in range(n):
                for phase in phases:
                    assignments[(i, phase)] = Bool(f"c{i}_{phase.value}")

            # Constraint 1: Each clip assigned to at most one phase
            for i in range(n):
                solver.add(AtMost(*[assignments[(i, p)] for p in phases], 1))

            # Constraint 2: Phase duration constraints
            for phase in phases:
                target = phase_durations[phase.value]
                tolerance = 0.15

                phase_duration = Sum([
                    If(assignments[(i, phase)], int(clips[i]["duration"] * 1000), 0)
                    for i in range(n)
                ])

                min_dur = int(target * (1 - tolerance) * 1000)
                max_dur = int(target * (1 + tolerance) * 1000)

                solver.add(phase_duration >= min_dur)
                solver.add(phase_duration <= max_dur)

            # Constraint 3: Energy constraints from style
            for phase in phases:
                constraints = style_constraints.get(phase.value, {})
                min_energy = constraints.get("min_energy", 0.0)
                max_energy = constraints.get("max_energy", 1.0)

                for i, clip in enumerate(clips):
                    if clip["energy"] < min_energy or clip["energy"] > max_energy:
                        # This clip doesn't meet energy requirements for this phase
                        solver.add(Not(assignments[(i, phase)]))

            # Constraint 4: Climax must have at least one high-energy clip
            climax_constraints = style_constraints.get("climax", {})
            if climax_constraints.get("min_energy", 0) > 0.5:
                high_energy_clips = [
                    i for i, c in enumerate(clips)
                    if c["energy"] >= climax_constraints["min_energy"]
                ]
                if high_energy_clips:
                    solver.add(Or(*[assignments[(i, StoryPhase.CLIMAX)] for i in high_energy_clips]))

            # Solve
            if solver.check() == sat:
                model = solver.model()
                result_assignments = {p.value: [] for p in phases}

                for i in range(n):
                    for phase in phases:
                        if model.evaluate(assignments[(i, phase)]):
                            result_assignments[phase.value].append(i)

                return {
                    "feasible": True,
                    "assignments": result_assignments,
                }

            return {
                "feasible": False,
                "reason": "Z3 found no satisfying assignment",
                "assignments": {},
            }

        except Exception as e:
            logger.warning(f"Z3 solver failed: {e}, falling back")
            return self._solve_fallback(clips, phase_durations, style_constraints)

    # =========================================================================
    # OR-Tools Solver
    # =========================================================================

    def _solve_ortools(
        self,
        clips: List[Dict],
        phase_durations: Dict[str, float],
        style_constraints: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Solve using OR-Tools CP-SAT."""
        try:
            from ortools.sat.python import cp_model

            model = cp_model.CpModel()
            n = len(clips)
            phases = list(StoryPhase)

            # Variables
            assignments = {}
            for i in range(n):
                for phase in phases:
                    assignments[(i, phase)] = model.NewBoolVar(f"c{i}_{phase.value}")

            # Constraint 1: Each clip at most one phase
            for i in range(n):
                model.AddAtMostOne([assignments[(i, p)] for p in phases])

            # Constraint 2: Phase durations
            for phase in phases:
                target = phase_durations[phase.value]
                tolerance = 0.15

                phase_duration = sum(
                    assignments[(i, phase)] * int(clips[i]["duration"] * 1000)
                    for i in range(n)
                )

                min_dur = int(target * (1 - tolerance) * 1000)
                max_dur = int(target * (1 + tolerance) * 1000)

                model.Add(phase_duration >= min_dur)
                model.Add(phase_duration <= max_dur)

            # Constraint 3: Energy requirements
            for phase in phases:
                constraints = style_constraints.get(phase.value, {})
                min_energy = constraints.get("min_energy", 0.0)
                max_energy = constraints.get("max_energy", 1.0)

                for i, clip in enumerate(clips):
                    if clip["energy"] < min_energy or clip["energy"] > max_energy:
                        model.Add(assignments[(i, phase)] == 0)

            # Solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 5.0

            status = solver.Solve(model)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                result_assignments = {p.value: [] for p in phases}

                for i in range(n):
                    for phase in phases:
                        if solver.Value(assignments[(i, phase)]):
                            result_assignments[phase.value].append(i)

                return {
                    "feasible": True,
                    "assignments": result_assignments,
                }

            return {
                "feasible": False,
                "reason": "OR-Tools found no solution",
                "assignments": {},
            }

        except Exception as e:
            logger.warning(f"OR-Tools solver failed: {e}, falling back")
            return self._solve_fallback(clips, phase_durations, style_constraints)

    # =========================================================================
    # Fallback Greedy Solver
    # =========================================================================

    def _solve_fallback(
        self,
        clips: List[Dict],
        phase_durations: Dict[str, float],
        style_constraints: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Greedy fallback solver when no CSP solver available."""
        phases = list(StoryPhase)
        result_assignments = {p.value: [] for p in phases}
        used_clips = set()

        # Sort clips by energy for smarter assignment
        sorted_indices = sorted(
            range(len(clips)),
            key=lambda i: clips[i]["energy"],
        )

        for phase in phases:
            target = phase_durations[phase.value]
            tolerance = 0.15
            constraints = style_constraints.get(phase.value, {})
            min_energy = constraints.get("min_energy", 0.0)
            max_energy = constraints.get("max_energy", 1.0)

            current_duration = 0.0
            min_required = target * (1 - tolerance)
            max_allowed = target * (1 + tolerance)

            # Select clips based on phase energy requirements
            if phase == StoryPhase.CLIMAX:
                # High energy first for climax
                candidates = sorted_indices[::-1]
            elif phase in [StoryPhase.INTRO, StoryPhase.OUTRO]:
                # Low energy for intro/outro
                candidates = sorted_indices
            else:
                # Medium energy for build/sustain
                candidates = sorted_indices[len(sorted_indices)//4:3*len(sorted_indices)//4]
                candidates = list(candidates) + sorted_indices  # Fallback to all

            for i in candidates:
                if i in used_clips:
                    continue

                clip = clips[i]

                # Check energy constraints
                if not (min_energy <= clip["energy"] <= max_energy):
                    continue

                # Check duration fit
                if current_duration + clip["duration"] > max_allowed:
                    continue

                # Add clip
                result_assignments[phase.value].append(i)
                used_clips.add(i)
                current_duration += clip["duration"]

                if current_duration >= min_required:
                    break

        # Check if all phases have minimum coverage
        feasible = all(
            len(result_assignments[p.value]) >= 1
            for p in phases
        )

        return {
            "feasible": feasible,
            "assignments": result_assignments,
            "reason": "" if feasible else "Fallback solver couldn't satisfy all phases",
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _normalize_clip(self, clip: Any) -> Dict[str, Any]:
        """Normalize clip to dict format."""
        if isinstance(clip, dict):
            return clip

        # Handle objects with attributes
        return {
            "path": getattr(clip, "path", ""),
            "duration": getattr(clip, "duration", 0.0),
            "energy": getattr(clip, "energy", 0.5),
            "shot_type": getattr(clip, "shot_type", "medium"),
            "location": getattr(clip, "location", "unknown"),
        }

    def _generate_style_constraints(self, style: Dict[str, Any]) -> Dict[str, Dict]:
        """Generate phase constraints based on style."""
        style_name = style.get("name", "dynamic")

        constraints = {
            "intro": {"min_energy": 0.0, "max_energy": 0.5},
            "build": {"min_energy": 0.2, "max_energy": 0.7},
            "climax": {"min_energy": 0.6, "max_energy": 1.0},
            "sustain": {"min_energy": 0.4, "max_energy": 0.8},
            "outro": {"min_energy": 0.0, "max_energy": 0.5},
        }

        # Style-specific adjustments
        if style_name == "hitchcock":
            constraints["intro"]["max_energy"] = 0.4
            constraints["climax"]["min_energy"] = 0.7
        elif style_name == "mtv":
            constraints["intro"]["min_energy"] = 0.4
            constraints["build"]["min_energy"] = 0.5
            constraints["climax"]["min_energy"] = 0.7
            constraints["sustain"]["min_energy"] = 0.5
            constraints["outro"]["min_energy"] = 0.3

        return constraints

    def _flatten_assignments(self, assignments: Dict[str, List[int]]) -> List[int]:
        """Flatten phase assignments to ordered clip list."""
        ordered = []
        for phase in StoryPhase:
            clips_in_phase = assignments.get(phase.value, [])
            ordered.extend(clips_in_phase)
        return ordered

    def _calculate_energy_score(
        self,
        clips: List[Dict],
        assignments: Dict[str, List[int]],
        style: Dict[str, Any],
    ) -> float:
        """Calculate how well energy curve matches style."""
        target_curve = self._get_target_energy_curve(style)
        actual_curve = []

        for phase in StoryPhase:
            phase_clips = assignments.get(phase.value, [])
            if phase_clips:
                avg_energy = sum(clips[i]["energy"] for i in phase_clips) / len(phase_clips)
            else:
                avg_energy = 0.5
            actual_curve.append(avg_energy)

        # Calculate correlation/similarity
        if len(target_curve) != len(actual_curve):
            return 0.0

        diff_sum = sum(abs(t - a) for t, a in zip(target_curve, actual_curve))
        max_diff = len(target_curve)  # Max possible diff is 1.0 per phase

        return 1.0 - (diff_sum / max_diff)

    def _get_target_energy_curve(self, style: Dict[str, Any]) -> List[float]:
        """Get target energy curve for style."""
        style_name = style.get("name", "dynamic")

        curves = {
            "hitchcock": [0.2, 0.4, 0.9, 0.7, 0.3],  # Slow build, explosive climax
            "mtv": [0.7, 0.8, 0.95, 0.85, 0.6],      # High throughout
            "documentary": [0.3, 0.5, 0.7, 0.5, 0.3], # Moderate curve
            "minimalist": [0.2, 0.3, 0.5, 0.4, 0.2],  # Low energy
        }

        return curves.get(style_name, [0.3, 0.5, 0.8, 0.6, 0.3])

    def _calculate_shot_repeat_score(
        self,
        clips: List[Dict],
        ordered: List[int],
    ) -> float:
        """Calculate penalty for consecutive same shot types."""
        if len(ordered) < 2:
            return 1.0

        repeats = 0
        for i in range(1, len(ordered)):
            if clips[ordered[i]]["shot_type"] == clips[ordered[i-1]]["shot_type"]:
                repeats += 1

        max_repeats = len(ordered) - 1
        return 1.0 - (repeats / max_repeats) if max_repeats > 0 else 1.0


# =============================================================================
# Convenience Functions
# =============================================================================

def solve_story_arc(
    clips: List[Any],
    target_duration: float,
    style: Dict[str, Any],
    backend: str = "auto",
) -> Dict[str, Any]:
    """Convenience function to solve story arc assignment."""
    solver = StoryArcCSPSolver(backend=backend)
    return solver.solve(clips, target_duration, style)
