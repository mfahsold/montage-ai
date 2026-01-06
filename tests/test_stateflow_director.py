"""
Tests for StateFlow Director - Deterministic Multi-State Pipeline for Creative Direction.

TDD: Tests written before implementation.
"""

import pytest
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, MagicMock, AsyncMock


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_user_request():
    """Sample user editing request."""
    return "Create a 60-second Hitchcock-style montage with climax at 75%"


@pytest.fixture
def sample_footage_info():
    """Sample footage information."""
    return {
        "clips": [
            {"path": "/clip1.mp4", "duration": 15.0, "energy": 0.3, "shot_type": "wide"},
            {"path": "/clip2.mp4", "duration": 10.0, "energy": 0.5, "shot_type": "medium"},
            {"path": "/clip3.mp4", "duration": 8.0, "energy": 0.8, "shot_type": "close"},
            {"path": "/clip4.mp4", "duration": 12.0, "energy": 0.9, "shot_type": "close"},
            {"path": "/clip5.mp4", "duration": 20.0, "energy": 0.4, "shot_type": "wide"},
        ],
        "total_duration": 65.0,
        "available_shots": ["wide", "medium", "close"],
    }


@pytest.fixture
def sample_constraints():
    """Sample extracted constraints."""
    return {
        "explicit": {
            "duration": 60.0,
            "style": "hitchcock",
            "climax_position": 0.75,
        },
        "implicit": {
            "mood": "suspenseful",
            "pacing": "slow_build",
        },
        "conflicts": [],
    }


# =============================================================================
# DirectorState Enum Tests
# =============================================================================

class TestDirectorState:
    """Tests for DirectorState enum."""

    def test_state_enum_exists(self):
        """DirectorState enum should exist with required states."""
        from montage_ai.stateflow_director import DirectorState

        assert hasattr(DirectorState, 'PARSE_INTENT')
        assert hasattr(DirectorState, 'ANALYZE_FOOTAGE')
        assert hasattr(DirectorState, 'RESOLVE_CONFLICTS')
        assert hasattr(DirectorState, 'PLAN_STRUCTURE')
        assert hasattr(DirectorState, 'VALIDATE_CONSTRAINTS')
        assert hasattr(DirectorState, 'EMIT_OUTPUT')

    def test_state_values(self):
        """States should have string values."""
        from montage_ai.stateflow_director import DirectorState

        assert DirectorState.PARSE_INTENT.value == "parse_intent"
        assert DirectorState.EMIT_OUTPUT.value == "emit_output"


# =============================================================================
# StateContext Tests
# =============================================================================

class TestStateContext:
    """Tests for StateContext dataclass."""

    def test_context_creation(self, sample_user_request, sample_footage_info):
        """StateContext should store request, footage, and state data."""
        from montage_ai.stateflow_director import StateContext

        ctx = StateContext(
            user_request=sample_user_request,
            footage_info=sample_footage_info,
        )

        assert ctx.user_request == sample_user_request
        assert ctx.footage_info == sample_footage_info
        assert ctx.intent_analysis is None
        assert ctx.resolved_constraints is None
        assert ctx.structure_plan is None
        assert ctx.validation_result is None

    def test_context_update(self, sample_user_request, sample_footage_info, sample_constraints):
        """StateContext should allow updating fields."""
        from montage_ai.stateflow_director import StateContext

        ctx = StateContext(
            user_request=sample_user_request,
            footage_info=sample_footage_info,
        )

        ctx.intent_analysis = sample_constraints
        assert ctx.intent_analysis == sample_constraints


# =============================================================================
# IntentAnalysis Tests
# =============================================================================

class TestIntentAnalysis:
    """Tests for IntentAnalysis result model."""

    def test_intent_analysis_creation(self):
        """IntentAnalysis should capture explicit, implicit constraints and conflicts."""
        from montage_ai.stateflow_director import IntentAnalysis

        analysis = IntentAnalysis(
            explicit={"duration": 60.0, "style": "hitchcock"},
            implicit={"mood": "suspenseful"},
            conflicts=[],
            ambiguities=["climax_position not specified"],
        )

        assert analysis.explicit["duration"] == 60.0
        assert analysis.implicit["mood"] == "suspenseful"
        assert len(analysis.conflicts) == 0
        assert "climax_position" in analysis.ambiguities[0]

    def test_intent_analysis_with_conflicts(self):
        """IntentAnalysis should capture contradictions."""
        from montage_ai.stateflow_director import IntentAnalysis

        analysis = IntentAnalysis(
            explicit={"style": "hitchcock", "pacing": "fast"},
            implicit={},
            conflicts=["hitchcock style implies slow pacing, but fast was requested"],
            ambiguities=[],
        )

        assert len(analysis.conflicts) == 1
        assert "slow pacing" in analysis.conflicts[0]


# =============================================================================
# StateTransition Tests
# =============================================================================

class TestStateTransitions:
    """Tests for state transition logic."""

    def test_parse_intent_to_analyze_footage(self):
        """PARSE_INTENT should transition to ANALYZE_FOOTAGE on success."""
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector()

        result = {"success": True, "has_conflicts": False}
        next_state = director._get_next_state(DirectorState.PARSE_INTENT, result)

        assert next_state == DirectorState.ANALYZE_FOOTAGE

    def test_analyze_footage_to_resolve_conflicts(self):
        """ANALYZE_FOOTAGE should transition to RESOLVE_CONFLICTS if conflicts exist."""
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector()

        result = {"feasible": True, "has_conflicts": True}
        next_state = director._get_next_state(DirectorState.ANALYZE_FOOTAGE, result)

        assert next_state == DirectorState.RESOLVE_CONFLICTS

    def test_analyze_footage_to_plan_structure(self):
        """ANALYZE_FOOTAGE should transition to PLAN_STRUCTURE if no conflicts."""
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector()

        result = {"feasible": True, "has_conflicts": False}
        next_state = director._get_next_state(DirectorState.ANALYZE_FOOTAGE, result)

        assert next_state == DirectorState.PLAN_STRUCTURE

    def test_validate_failure_backtracks(self):
        """VALIDATE_CONSTRAINTS should backtrack to RESOLVE_CONFLICTS on failure."""
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector()

        result = {"feasible": False, "blocking_issues": ["not enough clips"]}
        next_state = director._get_next_state(DirectorState.VALIDATE_CONSTRAINTS, result)

        assert next_state == DirectorState.RESOLVE_CONFLICTS

    def test_validate_success_emits_output(self):
        """VALIDATE_CONSTRAINTS should transition to EMIT_OUTPUT on success."""
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector()

        result = {"feasible": True, "blocking_issues": []}
        next_state = director._get_next_state(DirectorState.VALIDATE_CONSTRAINTS, result)

        assert next_state == DirectorState.EMIT_OUTPUT


# =============================================================================
# StateFlowDirector Tests
# =============================================================================

class TestStateFlowDirectorInit:
    """Tests for StateFlowDirector initialization."""

    def test_default_init(self):
        """StateFlowDirector should initialize with default settings."""
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector()

        assert director.max_backtrack_attempts == 3
        assert director.current_state == DirectorState.PARSE_INTENT

    def test_custom_init(self):
        """StateFlowDirector should accept custom settings."""
        from montage_ai.stateflow_director import StateFlowDirector

        director = StateFlowDirector(max_backtrack_attempts=5)

        assert director.max_backtrack_attempts == 5


class TestStateFlowDirectorRun:
    """Tests for StateFlowDirector.run() method."""

    def test_run_full_pipeline_mocked(self, sample_user_request, sample_footage_info):
        """run() should execute all states and return DirectorOutput."""
        import asyncio
        from montage_ai.stateflow_director import StateFlowDirector
        from montage_ai.prompts import DirectorOutput

        director = StateFlowDirector()

        # Mock LLM calls
        async def mock_execute(state, context):
            from montage_ai.stateflow_director import DirectorState
            if state == DirectorState.PARSE_INTENT:
                return {"success": True, "explicit": {"duration": 60, "style": "hitchcock"}, "implicit": {}, "conflicts": []}
            elif state == DirectorState.ANALYZE_FOOTAGE:
                return {"feasible": True, "has_conflicts": False, "coverage": {}}
            elif state == DirectorState.PLAN_STRUCTURE:
                return {"planned": True, "phases": [], "phase_config": {}}
            elif state == DirectorState.VALIDATE_CONSTRAINTS:
                return {"feasible": True, "blocking_issues": []}
            return {}

        with patch.object(director, '_execute_state', side_effect=mock_execute):
            with patch.object(director, '_build_final_output') as mock_build:
                mock_build.return_value = Mock(spec=DirectorOutput)

                result = asyncio.run(director.run(sample_user_request, sample_footage_info))

                assert result is not None

    def test_run_with_backtrack(self, sample_user_request, sample_footage_info):
        """run() should handle backtracking on validation failure."""
        import asyncio
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector(max_backtrack_attempts=2)

        call_count = [0]

        async def mock_execute(state, context):
            call_count[0] += 1
            if state == DirectorState.PARSE_INTENT:
                return {"success": True, "explicit": {}, "implicit": {}, "conflicts": []}
            elif state == DirectorState.ANALYZE_FOOTAGE:
                return {"feasible": True, "has_conflicts": False}
            elif state == DirectorState.PLAN_STRUCTURE:
                return {"planned": True, "phases": [], "phase_config": {}}
            elif state == DirectorState.VALIDATE_CONSTRAINTS:
                # First time fail, second time succeed
                if call_count[0] <= 4:
                    return {"feasible": False, "blocking_issues": ["test"]}
                return {"feasible": True, "blocking_issues": []}
            elif state == DirectorState.RESOLVE_CONFLICTS:
                return {"resolved": True}
            return {}

        with patch.object(director, '_execute_state', side_effect=mock_execute):
            with patch.object(director, '_build_final_output') as mock_build:
                mock_build.return_value = Mock()

                result = asyncio.run(director.run(sample_user_request, sample_footage_info))

                # Should have backtracked at least once
                assert call_count[0] > 4


class TestStateFlowDirectorExecuteState:
    """Tests for individual state execution."""

    def test_execute_parse_intent(self, sample_user_request, sample_footage_info):
        """_execute_state(PARSE_INTENT) should extract constraints."""
        import asyncio
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState, StateContext

        director = StateFlowDirector()
        ctx = StateContext(user_request=sample_user_request, footage_info=sample_footage_info)

        # Mock _llm_call to return JSON string
        async def mock_llm(prompt):
            return '{"explicit": {"duration": 60, "style": "hitchcock"}, "implicit": {"mood": "suspenseful"}, "conflicts": [], "ambiguities": []}'

        with patch.object(director, '_llm_call', side_effect=mock_llm):
            result = asyncio.run(director._execute_state(DirectorState.PARSE_INTENT, ctx))

            assert "explicit" in result
            assert result["explicit"].get("style") == "hitchcock"

    def test_execute_analyze_footage(self, sample_user_request, sample_footage_info):
        """_execute_state(ANALYZE_FOOTAGE) should check footage feasibility."""
        import asyncio
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState, StateContext

        director = StateFlowDirector()
        ctx = StateContext(
            user_request=sample_user_request,
            footage_info=sample_footage_info,
            intent_analysis={"explicit": {"duration": 60}, "implicit": {}, "conflicts": []},
        )

        # This should be deterministic (no LLM), checking if footage can cover phases
        result = asyncio.run(director._execute_state(DirectorState.ANALYZE_FOOTAGE, ctx))

        assert "feasible" in result
        assert isinstance(result["feasible"], bool)


# =============================================================================
# Deterministic Validation Tests
# =============================================================================

class TestDeterministicValidation:
    """Tests for deterministic constraint validation (no LLM)."""

    def test_validate_duration_constraint(self, sample_footage_info):
        """Should validate if footage can cover target duration."""
        from montage_ai.stateflow_director import StateFlowDirector

        director = StateFlowDirector()

        # Total footage: 65s, target: 60s -> feasible
        result = director._validate_duration_constraint(
            footage_info=sample_footage_info,
            target_duration=60.0,
        )
        assert result["feasible"] == True

        # Target: 100s -> not feasible
        result = director._validate_duration_constraint(
            footage_info=sample_footage_info,
            target_duration=100.0,
        )
        assert result["feasible"] == False

    def test_validate_phase_coverage(self, sample_footage_info):
        """Should validate if footage can cover all story phases."""
        from montage_ai.stateflow_director import StateFlowDirector

        director = StateFlowDirector()

        phase_requirements = {
            "intro": {"min_duration": 9.0, "max_energy": 0.5},
            "build": {"min_duration": 15.0},
            "climax": {"min_duration": 12.0, "min_energy": 0.7},
            "sustain": {"min_duration": 15.0},
            "outro": {"min_duration": 9.0, "max_energy": 0.5},
        }

        result = director._validate_phase_coverage(
            footage_info=sample_footage_info,
            phase_requirements=phase_requirements,
        )

        assert "feasible" in result
        assert "coverage" in result


# =============================================================================
# State History Tests
# =============================================================================

class TestStateHistory:
    """Tests for state history tracking (Γ* context history)."""

    def test_history_tracking(self, sample_user_request, sample_footage_info):
        """StateFlowDirector should track state history."""
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector()

        director._record_state_transition(
            from_state=DirectorState.PARSE_INTENT,
            to_state=DirectorState.ANALYZE_FOOTAGE,
            result={"success": True},
        )

        assert len(director.state_history) == 1
        # StateTransitionRecord is a dataclass, access via attributes
        assert director.state_history[0].from_state == DirectorState.PARSE_INTENT
        assert director.state_history[0].to_state == DirectorState.ANALYZE_FOOTAGE

    def test_backtrack_count(self):
        """Should track backtrack attempts."""
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState

        director = StateFlowDirector(max_backtrack_attempts=3)

        # Simulate backtrack
        director._record_backtrack(
            from_state=DirectorState.VALIDATE_CONSTRAINTS,
            to_state=DirectorState.RESOLVE_CONFLICTS,
            reason="validation failed",
        )

        assert director.backtrack_count == 1

    def test_max_backtrack_exceeded(self):
        """Should detect when max backtracks exceeded."""
        from montage_ai.stateflow_director import StateFlowDirector

        director = StateFlowDirector(max_backtrack_attempts=2)

        director.backtrack_count = 2

        assert director._max_backtracks_exceeded() == True


# =============================================================================
# Integration with DirectorOutput Tests
# =============================================================================

class TestDirectorOutputIntegration:
    """Tests for integration with existing DirectorOutput schema."""

    def test_build_final_output(self, sample_footage_info):
        """_build_final_output should create valid DirectorOutput."""
        from montage_ai.stateflow_director import StateFlowDirector, StateContext
        from montage_ai.prompts import DirectorOutput

        director = StateFlowDirector()

        ctx = StateContext(
            user_request="60s hitchcock montage",
            footage_info=sample_footage_info,
            intent_analysis={
                "explicit": {"duration": 60, "style": "hitchcock"},
                "implicit": {"mood": "suspenseful"},
                "conflicts": [],
            },
            resolved_constraints={
                "style": "hitchcock",
                "mood": "suspenseful",
                "pacing_speed": "slow",
                "climax_position": 0.75,
            },
            structure_plan={
                "phases": ["intro", "build", "climax", "sustain", "outro"],
                "phase_durations": [9, 15, 12, 15, 9],
            },
            validation_result={"feasible": True},
        )

        output = director._build_final_output(ctx)

        assert isinstance(output, DirectorOutput)
        assert output.style.name == "hitchcock"
        assert output.style.mood.value == "suspenseful"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in StateFlowDirector."""

    def test_llm_failure_fallback(self, sample_user_request, sample_footage_info):
        """Should fall back to defaults on LLM failure."""
        import asyncio
        from montage_ai.stateflow_director import StateFlowDirector, DirectorState, StateContext

        director = StateFlowDirector()

        async def failing_llm(prompt):
            raise Exception("LLM timeout")

        with patch.object(director, '_llm_call', side_effect=failing_llm):
            with patch.object(director, '_fallback_parse_intent') as mock_fallback:
                mock_fallback.return_value = {
                    "success": True,
                    "explicit": {},
                    "implicit": {},
                    "conflicts": [],
                }

                ctx = StateContext(user_request=sample_user_request, footage_info=sample_footage_info)

                result = asyncio.run(director._execute_state(DirectorState.PARSE_INTENT, ctx))

                mock_fallback.assert_called_once()

    def test_infeasible_constraints_relaxation(self, sample_footage_info):
        """Should relax constraints when validation fails."""
        from montage_ai.stateflow_director import StateFlowDirector

        director = StateFlowDirector()

        # Impossible constraint: 120s from 65s footage
        constraints = {"duration": 120.0, "min_clip_duration": 2.0}

        relaxed = director._relax_constraints(constraints, sample_footage_info)

        # Should reduce duration to what's feasible
        assert relaxed["duration"] <= sample_footage_info["total_duration"]
