"""
Audio Expert - Audio processing and ducking specialist

Analyzes audio levels and proposes ducking, normalization, and mixing adjustments.
"""

from typing import List, Dict, Any

from .base import BaseExpert, ExpertConfig
from ..contracts import EditingState, EditDelta, ParameterType, ImpactLevel


class AudioExpert(BaseExpert):
    """
    Expert for audio processing and voice/dialogue optimization.

    Specializes in:
    - Voice/dialogue ducking under music
    - Audio normalization (LUFS targets)
    - Speech vs music balance
    """

    def __init__(self, config: ExpertConfig = None):
        super().__init__(
            "audio", config or ExpertConfig(weight=1.1, confidence_threshold=0.6)
        )

    def analyze(
        self, state: EditingState, media_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze audio characteristics."""
        audio_analysis = media_context.get("audio_analysis", {})

        has_voice = audio_analysis.get("has_voice", False)
        has_music = audio_analysis.get("has_music", False)

        # LUFS analysis
        current_lufs = audio_analysis.get("integrated_lufs", -23.0)
        target_lufs = -14.0  # YouTube/Streaming standard
        lufs_adjustment = target_lufs - current_lufs

        # Voice regions
        voice_regions = audio_analysis.get("voice_regions", [])
        music_regions = audio_analysis.get("music_regions", [])

        # Calculate overlap
        overlap_detected = len(voice_regions) > 0 and len(music_regions) > 0

        return {
            "has_voice": has_voice,
            "has_music": has_music,
            "current_lufs": current_lufs,
            "target_lufs": target_lufs,
            "lufs_adjustment": lufs_adjustment,
            "voice_regions": voice_regions,
            "music_regions": music_regions,
            "overlap_detected": overlap_detected,
            "needs_ducking": has_voice and has_music and overlap_detected,
        }

    def propose(self, state: EditingState, analysis: Dict[str, Any]) -> List[EditDelta]:
        """Propose audio processing adjustments."""
        deltas = []

        # Proposal 1: LUFS Normalization
        adjustment = analysis.get("lufs_adjustment", 0)
        if abs(adjustment) > 1.0:  # Significant deviation
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.NORMALIZATION_TARGET,
                    value=analysis["target_lufs"],
                    confidence=0.90,
                    impact=ImpactLevel.MEDIUM,
                    rationale=f"Normalize from {analysis['current_lufs']:.1f} LUFS to "
                    f"{analysis['target_lufs']:.1f} LUFS (adjustment: {adjustment:+.1f}dB)",
                    revertible=True,
                )
            )

        # Proposal 2: Voice Ducking
        if analysis.get("needs_ducking"):
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.AUDIO_DUCKING,
                    value={
                        "enabled": True,
                        "threshold_db": -30,
                        "reduction_db": -12,
                        "attack_ms": 50,
                        "release_ms": 300,
                        "apply_to": "music",
                    },
                    confidence=0.85,
                    impact=ImpactLevel.MEDIUM,
                    rationale="Ducking: Lower music -12dB when voice detected",
                    revertible=True,
                )
            )

        # Proposal 3: Voice Isolation recommendation (for heavy voice content)
        if analysis.get("has_voice") and len(analysis.get("voice_regions", [])) > 5:
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.AUDIO_DUCKING,
                    value={
                        "strategy": "voice_isolation",
                        "priority": "dialogue",
                        "sidechain_music": True,
                    },
                    confidence=0.75,
                    impact=ImpactLevel.HIGH,
                    rationale="Heavy dialogue content: isolate voice with sidechain compression",
                    revertible=True,
                )
            )

        # Proposal 4: Cross-fade recommendations for music transitions
        if analysis.get("has_music") and len(state.transitions) > 3:
            deltas.append(
                EditDelta(
                    expert_id=self.expert_id,
                    parameter=ParameterType.TRANSITION_TYPE,
                    value="audio_crossfade",
                    confidence=0.70,
                    impact=ImpactLevel.LOW,
                    rationale="Use audio crossfades for smoother music transitions",
                    revertible=True,
                )
            )

        return deltas
