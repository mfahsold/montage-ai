"""
Rhythm Expert - Beat-sync and pacing specialist

Analyzes audio beats and proposes timing adjustments for cuts and transitions.
"""

from typing import List, Dict, Any
import numpy as np

from .base import BaseExpert, ExpertConfig
from ..contracts import EditingState, EditDelta, ParameterType, ImpactLevel


class RhythmExpert(BaseExpert):
    """
    Expert for beat-synchronized editing and pacing.

    Specializes in:
    - Cut timing alignment to beats
    - Transition duration based on music energy
    - Pacing consistency throughout timeline
    """

    def __init__(self, config: ExpertConfig = None):
        super().__init__(
            "rhythm",
            config
            or ExpertConfig(
                weight=1.2,  # High weight - timing is critical
                confidence_threshold=0.6,
            ),
        )

    def analyze(
        self, state: EditingState, media_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze beat times and energy profile."""
        beat_times = media_context.get("beat_times", [])
        energy_profile = media_context.get("energy_profile", [])

        if not beat_times:
            return {"has_beats": False}

        # Calculate beat intervals
        intervals = np.diff(beat_times)
        avg_interval = float(np.mean(intervals)) if len(intervals) > 0 else 1.0

        # Detect energy peaks
        high_energy_regions = []
        if energy_profile:
            threshold = np.percentile(energy_profile, 75)
            high_energy_regions = [
                i for i, e in enumerate(energy_profile) if e > threshold
            ]

        return {
            "has_beats": True,
            "beat_times": beat_times,
            "avg_beat_interval": avg_interval,
            "beat_count": len(beat_times),
            "high_energy_regions": high_energy_regions,
            "energy_profile": energy_profile,
        }

    def propose(self, state: EditingState, analysis: Dict[str, Any]) -> List[EditDelta]:
        """Propose beat-synced timing adjustments."""
        deltas = []

        if not analysis.get("has_beats"):
            return deltas

        beat_times = analysis["beat_times"]
        avg_interval = analysis["avg_beat_interval"]

        # Proposal 1: Beat-sync offset
        # Align first cut to nearest beat
        if state.clips:
            first_clip_start = state.clips[0].get("start", 0)
            nearest_beat = self._find_nearest_beat(first_clip_start, beat_times)
            offset = nearest_beat - first_clip_start

            if abs(offset) > 0.05:  # Only if significant (>50ms)
                deltas.append(
                    EditDelta(
                        expert_id=self.expert_id,
                        parameter=ParameterType.BEAT_SYNC_OFFSET,
                        value=float(offset),
                        confidence=0.85,
                        impact=ImpactLevel.MEDIUM,
                        rationale=f"Align first cut to beat at {nearest_beat:.2f}s "
                        f"(current offset: {offset * 1000:.0f}ms)",
                        revertible=True,
                    )
                )

        # Proposal 2: Transition duration based on energy
        if analysis.get("high_energy_regions"):
            # Shorter transitions during high energy
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.TRANSITION_DURATION,
                    value=0.3,  # 300ms
                    confidence=0.75,
                    impact=ImpactLevel.LOW,
                    rationale="Short transitions during high energy sections",
                    revertible=True,
                )
            )
        else:
            # Longer transitions for low energy
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.TRANSITION_DURATION,
                    value=0.5,  # 500ms
                    confidence=0.70,
                    impact=ImpactLevel.LOW,
                    rationale="Longer transitions for smooth flow in low energy",
                    revertible=True,
                )
            )

        # Proposal 3: Cut timing recommendations
        if len(state.clips) > 1 and len(beat_times) > 4:
            # Propose cuts on every 2nd or 4th beat depending on pace
            recommended_interval = avg_interval * 2  # Every 2 beats default

            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.CUT_TIME,
                    value={"interval": float(recommended_interval), "sync_to": "beats"},
                    confidence=0.80,
                    impact=ImpactLevel.HIGH,
                    rationale=f"Sync cuts to every 2nd beat (interval: {recommended_interval:.2f}s)",
                    revertible=True,
                )
            )

        return deltas

    def _find_nearest_beat(self, time: float, beat_times: List[float]) -> float:
        """Find nearest beat time to given timestamp."""
        if not beat_times:
            return time

        beat_array = np.array(beat_times)
        idx = np.argmin(np.abs(beat_array - time))
        return float(beat_times[idx])
