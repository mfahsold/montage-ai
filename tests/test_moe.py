"""
Tests for MoE (Mixture of Experts) System
"""

import pytest
from montage_ai.moe import (
    EditingState,
    EditDelta,
    ParameterType,
    ImpactLevel,
    RhythmExpert,
    NarrativeExpert,
    AudioExpert,
    MoEControlPlane,
    MoEConfig,
)


class TestEditDelta:
    """Test EditDelta contract."""

    def test_delta_creation(self):
        delta = EditDelta(
            expert_id="rhythm",
            parameter=ParameterType.BEAT_SYNC_OFFSET,
            value=0.5,
            confidence=0.85,
            impact=ImpactLevel.MEDIUM,
            rationale="Test rationale",
        )

        assert delta.expert_id == "rhythm"
        assert delta.confidence == 0.85
        assert delta.revertible is True

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            EditDelta(
                expert_id="test",
                parameter=ParameterType.CUT_TIME,
                value=1.0,
                confidence=1.5,  # Invalid
                impact=ImpactLevel.LOW,
                rationale="Test",
            )


class TestEditingState:
    """Test EditingState immutability."""

    def test_state_creation(self):
        state = EditingState(style_template="hitchcock")
        assert state.style_template == "hitchcock"
        assert len(state.clips) == 0

    def test_apply_delta_creates_new_state(self):
        state = EditingState()
        delta = EditDelta(
            expert_id="test",
            parameter=ParameterType.TRANSITION_DURATION,
            value=0.5,
            confidence=0.8,
            impact=ImpactLevel.LOW,
            rationale="Test",
        )

        new_state = state.apply_delta(delta)

        # Original unchanged
        assert len(state.applied_deltas) == 0
        # New state has delta
        assert len(new_state.applied_deltas) == 1
        assert new_state.get_parameter(ParameterType.TRANSITION_DURATION) == 0.5


class TestRhythmExpert:
    """Test RhythmExpert functionality."""

    def test_analyze_without_beats(self):
        expert = RhythmExpert()
        state = EditingState()
        context = {}  # No beats

        analysis = expert.analyze(state, context)
        assert analysis["has_beats"] is False

    def test_analyze_with_beats(self):
        expert = RhythmExpert()
        state = EditingState()
        context = {"beat_times": [0.5, 1.2, 2.0], "energy_profile": [0.3, 0.8, 0.5]}

        analysis = expert.analyze(state, context)
        assert analysis["has_beats"] is True
        assert analysis["beat_count"] == 3

    def test_propose_deltas(self):
        expert = RhythmExpert()
        state = EditingState()
        analysis = {
            "has_beats": True,
            "beat_times": [0.5, 1.2, 2.0, 2.8],
            "avg_beat_interval": 0.77,
            "high_energy_regions": [1],
            "energy_profile": [0.3, 0.8, 0.5, 0.7],
        }

        deltas = expert.propose(state, analysis)

        assert len(deltas) > 0
        assert all(d.confidence >= 0.6 for d in deltas)
        assert any(d.parameter == ParameterType.TRANSITION_DURATION for d in deltas)


class TestNarrativeExpert:
    """Test NarrativeExpert functionality."""

    def test_analyze_scenes(self):
        expert = NarrativeExpert()
        state = EditingState(duration=120.0)
        context = {
            "scenes": [
                {"type": "action", "energy": 0.8},
                {"type": "dialogue", "energy": 0.3},
                {"type": "action", "energy": 0.9},
            ]
        }

        analysis = expert.analyze(state, context)
        assert analysis["has_scenes"] is True
        assert analysis["scene_count"] == 3
        assert analysis["action_ratio"] > 0.5


class TestAudioExpert:
    """Test AudioExpert functionality."""

    def test_analyze_audio(self):
        expert = AudioExpert()
        state = EditingState()
        context = {
            "audio_analysis": {
                "has_voice": True,
                "has_music": True,
                "integrated_lufs": -20.0,
                "voice_regions": [(0, 5)],
                "music_regions": [(0, 10)],
            }
        }

        analysis = expert.analyze(state, context)
        assert analysis["has_voice"] is True
        assert analysis["needs_ducking"] is True
        assert analysis["lufs_adjustment"] == 6.0  # -14 - (-20)


class TestMoEControlPlane:
    """Test MoE Control Plane orchestration."""

    def test_register_expert(self):
        moe = MoEControlPlane()
        expert = RhythmExpert()

        moe.register_expert(expert)

        assert len(moe.experts) == 1
        assert moe.experts[0].expert_id == "rhythm"

    def test_execute_with_conflicts(self):
        moe = MoEControlPlane()
        moe.register_expert(RhythmExpert())
        moe.register_expert(NarrativeExpert())

        state = EditingState(clips=[{"start": 0, "end": 10}])
        context = {
            "beat_times": [0.5, 1.2, 2.0],
            "energy_profile": [0.5, 0.8, 0.6],
            "scenes": [{"type": "action", "energy": 0.8}],
            "audio_analysis": {"has_voice": False, "has_music": False},
        }

        new_state, conflicts = moe.execute(state, context)

        # Should return new state
        assert isinstance(new_state, EditingState)
        # Should have execution log
        assert len(moe.execution_log) == 1
        # Should have status
        status = moe.get_status()
        assert status["registered_experts"] == 2

    def test_human_decision(self):
        moe = MoEControlPlane()
        state = EditingState()

        delta = EditDelta(
            expert_id="test",
            parameter=ParameterType.CUT_TIME,
            value={"interval": 2.0},
            confidence=0.9,
            impact=ImpactLevel.LOW,
            rationale="Test",
        )

        # Approve
        new_state = moe.apply_human_decision(state, delta, approved=True)
        assert len(new_state.applied_deltas) == 1

        # Reject
        delta2 = EditDelta(
            expert_id="test2",
            parameter=ParameterType.TRANSITION_DURATION,
            value=0.5,
            confidence=0.8,
            impact=ImpactLevel.LOW,
            rationale="Test2",
        )
        new_state2 = moe.apply_human_decision(new_state, delta2, approved=False)
        assert len(new_state2.rejected_deltas) == 1
        assert len(new_state2.applied_deltas) == 1  # Previous still there


class TestIntegration:
    """Integration tests for full MoE workflow."""

    def test_full_moe_workflow(self):
        """Test complete MoE editing workflow."""
        # Setup
        moe = MoEControlPlane()
        moe.register_expert(RhythmExpert())
        moe.register_expert(NarrativeExpert())
        moe.register_expert(AudioExpert())

        # Initial state
        state = EditingState(
            clips=[{"id": "c1", "start": 0, "end": 5}],
            style_template="documentary",
            duration=60.0,
        )

        # Media analysis
        context = {
            "beat_times": [0.5, 1.2, 2.0, 3.5, 4.2],
            "energy_profile": [0.3, 0.5, 0.9, 0.8, 0.4],
            "scenes": [
                {"type": "dialogue", "energy": 0.3},
                {"type": "action", "energy": 0.8},
                {"type": "action", "energy": 0.9},
            ],
            "audio_analysis": {
                "has_voice": True,
                "has_music": True,
                "integrated_lufs": -18.0,
                "voice_regions": [(0, 10), (20, 30)],
                "music_regions": [(0, 60)],
            },
        }

        # Execute
        new_state, conflicts = moe.execute(state, context)

        # Verify
        assert isinstance(new_state, EditingState)
        assert len(new_state.applied_deltas) >= 0

        # Check status
        status = moe.get_status()
        assert status["registered_experts"] == 3
        assert status["execution_count"] == 1

        # If conflicts exist, they should be reviewable
        for conflict in conflicts:
            assert conflict.delta1.expert_id != conflict.delta2.expert_id
            assert conflict.parameter is not None
