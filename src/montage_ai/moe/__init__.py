"""
MoE (Mixture of Experts) Editing System

Orchestrates multiple AI experts to collaboratively edit video.
Each expert specializes in a different aspect (rhythm, narrative, audio, etc.)
and proposes changes via immutable EditDelta contracts.

The Control Plane manages expert execution, detects conflicts,
and supports human-in-the-loop decision making.

Example:
    from montage_ai.moe import MoEControlPlane, RhythmExpert, NarrativeExpert

    # Setup
    moe = MoEControlPlane()
    moe.register_expert(RhythmExpert())
    moe.register_expert(NarrativeExpert())

    # Execute
    new_state, conflicts = moe.execute(current_state, media_analysis)

    # Review conflicts
    for conflict in conflicts:
        print(f"Conflict: {conflict.description}")
        # Human decides...
        new_state = moe.apply_human_decision(new_state, chosen_delta, approved=True)
"""

from .contracts import EditingState, EditDelta, ParameterType, ImpactLevel

from .experts import (
    BaseExpert,
    ExpertConfig,
    RhythmExpert,
    NarrativeExpert,
    AudioExpert,
)

from .control_plane import MoEControlPlane, MoEConfig, Conflict

__version__ = "0.1.0"

__all__ = [
    # Contracts
    "EditingState",
    "EditDelta",
    "ParameterType",
    "ImpactLevel",
    # Experts
    "BaseExpert",
    "ExpertConfig",
    "RhythmExpert",
    "NarrativeExpert",
    "AudioExpert",
    # Control Plane
    "MoEControlPlane",
    "MoEConfig",
    "Conflict",
]
