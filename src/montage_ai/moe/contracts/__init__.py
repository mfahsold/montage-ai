"""
MoE Contracts - Core Delta Contract System

Defines the immutable state container and delta contract for MoE editing.
All experts communicate via EditDelta proposals that are validated and composed.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


class ImpactLevel(str, Enum):
    """Impact of a delta on the edit."""

    LOW = "low"  # Minor tweak, safe to auto-apply
    MEDIUM = "medium"  # Noticeable change, needs validation
    HIGH = "high"  # Major change, human approval recommended
    CRITICAL = "critical"  # Structural change, always require approval


class ParameterType(str, Enum):
    """Types of editable parameters."""

    # Timing
    CUT_TIME = "cut_time"
    TRANSITION_DURATION = "transition_duration"
    BEAT_SYNC_OFFSET = "beat_sync_offset"

    # Selection
    CLIP_SELECTION = "clip_selection"
    SHOT_TYPE_PREFERENCE = "shot_type_preference"

    # Audio
    AUDIO_DUCKING = "audio_ducking"
    NORMALIZATION_TARGET = "normalization_target"

    # Visual
    COLOR_GRADING = "color_grading"
    STABILIZATION_STRENGTH = "stabilization_strength"
    REFRAME_CROP = "reframe_crop"

    # Effects
    TRANSITION_TYPE = "transition_type"
    SPEED_RAMP = "speed_ramp"


@dataclass(frozen=True)
class EditDelta:
    """
    Immutable proposal for a single parameter change.

    Experts create deltas, Control Plane validates and composes them.
    """

    expert_id: str
    parameter: ParameterType
    value: Any
    confidence: float  # 0.0 - 1.0
    impact: ImpactLevel
    rationale: str
    revertible: bool = True

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "parameter": self.parameter.value,
            "value": self.value,
            "confidence": self.confidence,
            "impact": self.impact.value,
            "rationale": self.rationale,
            "revertible": self.revertible,
        }


@dataclass
class EditingState:
    """
    Immutable snapshot of the current editing state.

    All changes create new state instances (functional approach).
    """

    # Timeline
    clips: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)

    # Parameters
    parameters: Dict[ParameterType, Any] = field(default_factory=dict)

    # Metadata
    duration: float = 0.0
    style_template: str = "default"

    # History
    applied_deltas: List[EditDelta] = field(default_factory=list)
    pending_deltas: List[EditDelta] = field(default_factory=list)
    rejected_deltas: List[EditDelta] = field(default_factory=list)

    def apply_delta(self, delta: EditDelta) -> "EditingState":
        """Create new state with delta applied."""
        new_state = deepcopy(self)
        new_state.parameters[delta.parameter] = delta.value
        new_state.applied_deltas.append(delta)

        # Remove from pending if present
        if delta in new_state.pending_deltas:
            new_state.pending_deltas.remove(delta)

        return new_state

    def reject_delta(self, delta: EditDelta) -> "EditingState":
        """Create new state with delta rejected."""
        new_state = deepcopy(self)
        new_state.rejected_deltas.append(delta)

        if delta in new_state.pending_deltas:
            new_state.pending_deltas.remove(delta)

        return new_state

    def add_pending(self, delta: EditDelta) -> "EditingState":
        """Add delta to pending queue."""
        new_state = deepcopy(self)
        new_state.pending_deltas.append(delta)
        return new_state

    def get_parameter(self, param: ParameterType, default=None) -> Any:
        """Get current value for parameter."""
        return self.parameters.get(param, default)

    def get_expert_deltas(self, expert_id: str) -> List[EditDelta]:
        """Get all deltas from specific expert."""
        return [d for d in self.applied_deltas if d.expert_id == expert_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_count": len(self.clips),
            "duration": self.duration,
            "style": self.style_template,
            "applied_deltas": len(self.applied_deltas),
            "pending_deltas": len(self.pending_deltas),
            "rejected_deltas": len(self.rejected_deltas),
        }
