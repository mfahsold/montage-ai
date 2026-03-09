"""
MoE Control Plane - Orchestrates experts with human-in-the-loop

Manages expert execution, conflict resolution, and delta composition.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

from ..contracts import EditingState, EditDelta, ParameterType, ImpactLevel
from ..experts import BaseExpert


@dataclass
class Conflict:
    """Represents a conflict between two deltas."""

    delta1: EditDelta
    delta2: EditDelta
    parameter: ParameterType
    severity: str  # "low", "medium", "high"
    description: str


@dataclass
class MoEConfig:
    """Configuration for the MoE Control Plane."""

    enable_human_review: bool = True
    auto_apply_low_impact: bool = True
    auto_apply_threshold: float = 0.8
    max_conflicts_before_human: int = 3
    execution_timeout: float = 30.0


class MoEControlPlane:
    """
    Orchestrates multiple editing experts.

    Flow:
    1. Execute all experts → collect deltas
    2. Detect conflicts
    3. Present conflicts to human (if enabled)
    4. Compose final edit state
    """

    def __init__(self, config: MoEConfig = None):
        self.config = config or MoEConfig()
        self.experts: List[BaseExpert] = []
        self.conflicts: List[Conflict] = []
        self.pending_human_review: List[EditDelta] = []
        self.execution_log: List[Dict[str, Any]] = []

    def register_expert(self, expert: BaseExpert) -> None:
        """Register an expert with the control plane."""
        self.experts.append(expert)

    def execute(
        self, state: EditingState, media_context: Dict[str, Any]
    ) -> Tuple[EditingState, List[Conflict]]:
        """
        Execute all experts and resolve conflicts.

        Returns:
            (new_state, conflicts_needing_human_review)
        """
        start_time = time.time()

        # Step 1: Execute all experts
        all_deltas = self._execute_experts(state, media_context)

        # Step 2: Detect conflicts
        conflicts = self._detect_conflicts(all_deltas)

        # Step 3: Auto-resolve or queue for human review
        auto_deltas, human_deltas = self._categorize_deltas(all_deltas, conflicts)

        # Step 4: Apply auto-approved deltas
        new_state = state
        for delta in auto_deltas:
            new_state = new_state.apply_delta(delta)

        # Step 5: Queue human review deltas
        self.pending_human_review = human_deltas

        # Log execution
        self.execution_log.append(
            {
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "experts_executed": len(self.experts),
                "deltas_proposed": len(all_deltas),
                "auto_applied": len(auto_deltas),
                "pending_human": len(human_deltas),
                "conflicts": len(conflicts),
            }
        )

        return new_state, conflicts

    def _execute_experts(
        self, state: EditingState, media_context: Dict[str, Any]
    ) -> List[EditDelta]:
        """Execute all registered experts and collect deltas."""
        all_deltas = []

        for expert in self.experts:
            try:
                deltas = expert.execute(state, media_context)
                all_deltas.extend(deltas)
            except Exception as e:
                # Log error but continue with other experts
                self.execution_log.append(
                    {
                        "expert": expert.expert_id,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                )

        # Sort by confidence (highest first)
        all_deltas.sort(key=lambda d: d.confidence, reverse=True)
        return all_deltas

    def _detect_conflicts(self, deltas: List[EditDelta]) -> List[Conflict]:
        """Detect conflicts between deltas targeting same parameter."""
        conflicts = []

        # Group by parameter
        by_param = defaultdict(list)
        for delta in deltas:
            by_param[delta.parameter].append(delta)

        # Check for conflicts within each parameter group
        for param, param_deltas in by_param.items():
            if len(param_deltas) < 2:
                continue

            # Check each pair
            for i, d1 in enumerate(param_deltas):
                for d2 in param_deltas[i + 1 :]:
                    if self._is_conflicting(d1, d2):
                        severity = self._assess_severity(d1, d2)
                        conflicts.append(
                            Conflict(
                                delta1=d1,
                                delta2=d2,
                                parameter=param,
                                severity=severity,
                                description=f"{d1.expert_id} vs {d2.expert_id} on {param.value}",
                            )
                        )

        return conflicts

    def _is_conflicting(self, d1: EditDelta, d2: EditDelta) -> bool:
        """Determine if two deltas conflict."""
        # Same parameter with different values
        if d1.parameter != d2.parameter:
            return False

        # Different values
        if d1.value == d2.value:
            return False

        # Both high impact = definite conflict
        if d1.impact in (ImpactLevel.HIGH, ImpactLevel.CRITICAL) and d2.impact in (
            ImpactLevel.HIGH,
            ImpactLevel.CRITICAL,
        ):
            return True

        # Different experts with medium+ impact
        if (
            d1.expert_id != d2.expert_id
            and d1.impact in (ImpactLevel.MEDIUM, ImpactLevel.HIGH)
            and d2.impact in (ImpactLevel.MEDIUM, ImpactLevel.HIGH)
        ):
            return True

        return False

    def _assess_severity(self, d1: EditDelta, d2: EditDelta) -> str:
        """Assess conflict severity."""
        if d1.impact == ImpactLevel.CRITICAL or d2.impact == ImpactLevel.CRITICAL:
            return "high"
        elif d1.impact == ImpactLevel.HIGH or d2.impact == ImpactLevel.HIGH:
            return "high"
        elif d1.impact == ImpactLevel.MEDIUM or d2.impact == ImpactLevel.MEDIUM:
            return "medium"
        else:
            return "low"

    def _categorize_deltas(
        self, deltas: List[EditDelta], conflicts: List[Conflict]
    ) -> Tuple[List[EditDelta], List[EditDelta]]:
        """
        Categorize deltas into auto-apply and human-review.

        Returns:
            (auto_deltas, human_deltas)
        """
        auto_deltas = []
        human_deltas = []

        # Find deltas involved in conflicts
        conflicting_delta_ids = set()
        for conflict in conflicts:
            conflicting_delta_ids.add(id(conflict.delta1))
            conflicting_delta_ids.add(id(conflict.delta2))

        for delta in deltas:
            # Check if involved in conflict
            if id(delta) in conflicting_delta_ids:
                human_deltas.append(delta)
                continue

            # Check auto-apply criteria
            if self._can_auto_apply(delta):
                auto_deltas.append(delta)
            else:
                human_deltas.append(delta)

        return auto_deltas, human_deltas

    def _can_auto_apply(self, delta: EditDelta) -> bool:
        """Determine if delta can be auto-applied."""
        if not self.config.auto_apply_low_impact:
            return False

        # High confidence + low impact
        if (
            delta.confidence >= self.config.auto_apply_threshold
            and delta.impact == ImpactLevel.LOW
        ):
            return True

        # Very high confidence + medium impact
        if delta.confidence >= 0.9 and delta.impact == ImpactLevel.MEDIUM:
            return True

        return False

    def apply_human_decision(
        self, state: EditingState, delta: EditDelta, approved: bool
    ) -> EditingState:
        """
        Apply human decision on a pending delta.

        Args:
            state: Current editing state
            delta: Delta to approve/reject
            approved: True to apply, False to reject

        Returns:
            Updated editing state
        """
        if delta in self.pending_human_review:
            self.pending_human_review.remove(delta)

        if approved:
            return state.apply_delta(delta)
        else:
            return state.reject_delta(delta)

    def get_status(self) -> Dict[str, Any]:
        """Get current control plane status."""
        return {
            "registered_experts": len(self.experts),
            "pending_human_review": len(self.pending_human_review),
            "total_conflicts": len(self.conflicts),
            "execution_count": len(self.execution_log),
            "last_execution": self.execution_log[-1] if self.execution_log else None,
        }
