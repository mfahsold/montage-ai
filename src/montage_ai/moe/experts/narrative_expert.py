"""
Narrative Expert - Story arc and continuity specialist

Analyzes story structure and proposes edits for emotional flow and continuity.
"""

from typing import List, Dict, Any

from .base import BaseExpert, ExpertConfig
from ..contracts import EditingState, EditDelta, ParameterType, ImpactLevel


class NarrativeExpert(BaseExpert):
    """
    Expert for story structure and emotional flow.

    Specializes in:
    - Story arc construction (setup → build → climax → resolution)
    - Visual continuity between shots
    - Pacing based on narrative beats
    """

    def __init__(self, config: ExpertConfig = None):
        super().__init__(
            "narrative", config or ExpertConfig(weight=1.0, confidence_threshold=0.55)
        )

    def analyze(
        self, state: EditingState, media_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze scene content and story structure."""
        scenes = media_context.get("scenes", [])
        duration = state.duration

        if not scenes:
            return {"has_scenes": False}

        # Analyze scene types if available
        scene_types = [s.get("type", "unknown") for s in scenes]
        action_count = scene_types.count("action")
        dialogue_count = scene_types.count("dialogue")

        # Detect climax position (highest energy scene)
        energies = [s.get("energy", 0.5) for s in scenes]
        climax_idx = energies.index(max(energies)) if energies else 0
        climax_position = climax_idx / len(scenes) if scenes else 0.5

        return {
            "has_scenes": True,
            "scene_count": len(scenes),
            "scene_types": scene_types,
            "action_ratio": action_count / len(scenes) if scenes else 0,
            "dialogue_ratio": dialogue_count / len(scenes) if scenes else 0,
            "climax_idx": climax_idx,
            "climax_position": climax_position,
            "duration": duration,
        }

    def propose(self, state: EditingState, analysis: Dict[str, Any]) -> List[EditDelta]:
        """Propose narrative structure adjustments."""
        deltas = []

        if not analysis.get("has_scenes"):
            return deltas

        scene_count = analysis["scene_count"]
        duration = analysis["duration"]

        # Proposal 1: Climax position adjustment
        # Ideal climax at ~75% of timeline
        current_pos = analysis["climax_position"]
        ideal_pos = 0.75

        if abs(current_pos - ideal_pos) > 0.1:  # More than 10% off
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.CLIP_SELECTION,
                    value={
                        "action": "reorder",
                        "strategy": "climax_at_75",
                        "current_climax": current_pos,
                        "target_climax": ideal_pos,
                    },
                    confidence=0.70,
                    impact=ImpactLevel.HIGH,
                    rationale=f"Move climax from {current_pos:.0%} to ideal position at 75%",
                    revertible=True,
                )
            )

        # Proposal 2: Shot type preference based on content
        action_ratio = analysis["action_ratio"]
        if action_ratio > 0.6:
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.SHOT_TYPE_PREFERENCE,
                    value="high_action",
                    confidence=0.75,
                    impact=ImpactLevel.MEDIUM,
                    rationale="High action content: prefer dynamic shots",
                    revertible=True,
                )
            )
        elif analysis["dialogue_ratio"] > 0.5:
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.SHOT_TYPE_PREFERENCE,
                    value="dialogue_friendly",
                    confidence=0.72,
                    impact=ImpactLevel.MEDIUM,
                    rationale="Dialogue-heavy content: prefer stable shots",
                    revertible=True,
                )
            )

        # Proposal 3: Pacing based on story phase
        if duration > 60:  # Only for longer content
            # Propose different pacing for different story phases
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.CUT_TIME,
                    value={
                        "phased_pacing": True,
                        "intro": {"duration_pct": 0.15, "pace": "slow"},
                        "build": {"duration_pct": 0.45, "pace": "medium"},
                        "climax": {"duration_pct": 0.25, "pace": "fast"},
                        "outro": {"duration_pct": 0.15, "pace": "slow"},
                    },
                    confidence=0.80,
                    impact=ImpactLevel.HIGH,
                    rationale="Phased pacing: slow intro, medium build, fast climax, slow outro",
                    revertible=True,
                )
            )

        return deltas
