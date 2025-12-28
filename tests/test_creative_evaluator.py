"""
Tests for creative_evaluator.py - Agentic Creative Loop (Phase 4)

Tests the LLM-powered feedback loop for montage refinement.
"""

import pytest
import json
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock

from montage_ai.creative_evaluator import (
    EditingIssue,
    EditingAdjustment,
    MontageEvaluation,
    CreativeEvaluator,
    evaluate_montage,
    run_creative_loop,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockClipMetadata:
    """Mock ClipMetadata for testing."""
    source_path: str = "/path/to/clip.mp4"
    start_time: float = 0.0
    duration: float = 2.0
    timeline_start: float = 0.0
    energy: float = 0.5
    action: str = "medium"
    shot: str = "medium"
    beat_idx: int = 0
    beats_per_cut: float = 4.0
    selection_score: float = 50.0


@dataclass
class MockMontageResult:
    """Mock MontageResult for testing."""
    success: bool = True
    output_path: str = "/output/montage.mp4"
    duration: float = 45.0
    cut_count: int = 23
    render_time: float = 120.0
    file_size_mb: float = 50.0
    error: str = None


@pytest.fixture
def sample_instructions():
    """Sample editing instructions."""
    return {
        "style": {"name": "hitchcock", "mood": "suspenseful"},
        "pacing": {"speed": "dynamic", "variation": "high", "intro_duration_beats": 8},
        "transitions": {"type": "hard_cuts"},
        "effects": {"stabilization": False},
    }


@pytest.fixture
def sample_clips_metadata():
    """Sample clips metadata list."""
    return [
        MockClipMetadata(energy=0.3, shot="wide", duration=3.0, timeline_start=0.0),
        MockClipMetadata(energy=0.4, shot="medium", duration=2.0, timeline_start=3.0),
        MockClipMetadata(energy=0.6, shot="close", duration=1.5, timeline_start=5.0),
        MockClipMetadata(energy=0.8, shot="medium", duration=1.0, timeline_start=6.5),
        MockClipMetadata(energy=0.9, shot="close", duration=0.8, timeline_start=7.5),
    ]


@pytest.fixture
def sample_result():
    """Sample montage result."""
    return MockMontageResult()


# =============================================================================
# Data Classes Tests
# =============================================================================

class TestEditingIssue:
    """Tests for EditingIssue dataclass."""

    def test_basic_creation(self):
        issue = EditingIssue(
            type="pacing",
            severity="moderate",
            description="Intro is too fast",
        )
        assert issue.type == "pacing"
        assert issue.severity == "moderate"
        assert issue.description == "Intro is too fast"
        assert issue.timestamp is None
        assert issue.affected_clips == []

    def test_with_timestamp_and_clips(self):
        issue = EditingIssue(
            type="variety",
            severity="minor",
            description="Repetitive shots",
            timestamp=15.5,
            affected_clips=[3, 4, 5],
        )
        assert issue.timestamp == 15.5
        assert issue.affected_clips == [3, 4, 5]


class TestEditingAdjustment:
    """Tests for EditingAdjustment dataclass."""

    def test_basic_creation(self):
        adj = EditingAdjustment(
            target="pacing.speed",
            current_value="fast",
            suggested_value="medium",
            rationale="Better for suspense",
        )
        assert adj.target == "pacing.speed"
        assert adj.current_value == "fast"
        assert adj.suggested_value == "medium"
        assert adj.rationale == "Better for suspense"


class TestMontageEvaluation:
    """Tests for MontageEvaluation dataclass."""

    def test_basic_creation(self):
        evaluation = MontageEvaluation(
            satisfaction_score=0.75,
            summary="Good but needs work",
        )
        assert evaluation.satisfaction_score == 0.75
        assert evaluation.approve_for_render == False
        assert evaluation.issues == []
        assert evaluation.adjustments == []

    def test_needs_refinement(self):
        # Low score, not approved -> needs refinement
        eval1 = MontageEvaluation(satisfaction_score=0.6, approve_for_render=False)
        assert eval1.needs_refinement == True

        # High score, approved -> no refinement needed
        eval2 = MontageEvaluation(satisfaction_score=0.9, approve_for_render=True)
        assert eval2.needs_refinement == False

        # Low score but approved -> no refinement needed
        eval3 = MontageEvaluation(satisfaction_score=0.6, approve_for_render=True)
        assert eval3.needs_refinement == False

    def test_critical_issues(self):
        issues = [
            EditingIssue(type="pacing", severity="minor", description="Minor issue"),
            EditingIssue(type="energy", severity="critical", description="Critical issue"),
            EditingIssue(type="variety", severity="moderate", description="Moderate issue"),
        ]
        evaluation = MontageEvaluation(
            satisfaction_score=0.5,
            issues=issues,
        )
        critical = evaluation.critical_issues
        assert len(critical) == 1
        assert critical[0].type == "energy"
        assert critical[0].severity == "critical"


# =============================================================================
# CreativeEvaluator Tests
# =============================================================================

class TestCreativeEvaluatorInit:
    """Tests for CreativeEvaluator initialization."""

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_default_init(self, mock_director):
        evaluator = CreativeEvaluator()
        assert evaluator.max_iterations == 3
        assert evaluator.approval_threshold == 0.8
        assert evaluator.timeout == 60

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_custom_init(self, mock_director):
        evaluator = CreativeEvaluator(
            max_iterations=5,
            approval_threshold=0.9,
            timeout=120,
        )
        assert evaluator.max_iterations == 5
        assert evaluator.approval_threshold == 0.9
        assert evaluator.timeout == 120


class TestCreativeEvaluatorBuildContext:
    """Tests for context building."""

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_build_context(self, mock_director, sample_instructions, sample_clips_metadata, sample_result):
        evaluator = CreativeEvaluator()

        context = evaluator._build_context(
            result=sample_result,
            instructions=sample_instructions,
            clips_metadata=sample_clips_metadata,
            audio_profile=None,
        )

        # Check key elements are present
        assert "hitchcock" in context
        assert "45.0s" in context or "45" in context
        assert "23" in context  # cut count
        assert "wide" in context  # shot type
        assert "medium" in context  # shot type


class TestCreativeEvaluatorParseResponse:
    """Tests for response parsing."""

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_parse_valid_response(self, mock_director):
        evaluator = CreativeEvaluator()

        response = json.dumps({
            "satisfaction_score": 0.85,
            "issues": [
                {"type": "pacing", "severity": "minor", "description": "Slightly fast intro"}
            ],
            "adjustments": [
                {"target": "pacing.speed", "current_value": "fast", "suggested_value": "medium", "rationale": "Better flow"}
            ],
            "summary": "Good edit overall",
            "approve_for_render": True,
        })

        evaluation = evaluator._parse_response(response, iteration=0)

        assert evaluation.satisfaction_score == 0.85
        assert evaluation.approve_for_render == True
        assert len(evaluation.issues) == 1
        assert evaluation.issues[0].type == "pacing"
        assert len(evaluation.adjustments) == 1
        assert evaluation.adjustments[0].target == "pacing.speed"

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_parse_response_with_markdown(self, mock_director):
        evaluator = CreativeEvaluator()

        response = """```json
{
    "satisfaction_score": 0.9,
    "issues": [],
    "adjustments": [],
    "summary": "Excellent",
    "approve_for_render": true
}
```"""

        evaluation = evaluator._parse_response(response, iteration=0)
        assert evaluation.satisfaction_score == 0.9
        assert evaluation.approve_for_render == True

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_parse_invalid_response(self, mock_director):
        evaluator = CreativeEvaluator()

        response = "This is not valid JSON"

        evaluation = evaluator._parse_response(response, iteration=0)
        # Should return default with auto-approve on parse failure
        assert evaluation.satisfaction_score == 0.7
        assert evaluation.approve_for_render == True


class TestCreativeEvaluatorApplyAdjustment:
    """Tests for adjustment application."""

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_apply_simple_adjustment(self, mock_director, sample_instructions):
        evaluator = CreativeEvaluator()

        adjustment = EditingAdjustment(
            target="pacing.speed",
            current_value="dynamic",
            suggested_value="slow",
            rationale="Test",
        )

        evaluator._apply_adjustment(sample_instructions, adjustment)
        assert sample_instructions["pacing"]["speed"] == "slow"

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_apply_nested_adjustment(self, mock_director, sample_instructions):
        evaluator = CreativeEvaluator()

        adjustment = EditingAdjustment(
            target="style.mood",
            current_value="suspenseful",
            suggested_value="calm",
            rationale="Test",
        )

        evaluator._apply_adjustment(sample_instructions, adjustment)
        assert sample_instructions["style"]["mood"] == "calm"

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_apply_new_nested_path(self, mock_director, sample_instructions):
        evaluator = CreativeEvaluator()

        adjustment = EditingAdjustment(
            target="energy_mapping.sync_to_beats",
            current_value=None,
            suggested_value=True,
            rationale="Test",
        )

        evaluator._apply_adjustment(sample_instructions, adjustment)
        assert sample_instructions["energy_mapping"]["sync_to_beats"] == True


class TestCreativeEvaluatorRefineInstructions:
    """Tests for instruction refinement."""

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_refine_with_adjustments(self, mock_director, sample_instructions):
        evaluator = CreativeEvaluator()

        evaluation = MontageEvaluation(
            satisfaction_score=0.6,
            adjustments=[
                EditingAdjustment(
                    target="pacing.speed",
                    current_value="dynamic",
                    suggested_value="slow",
                    rationale="More suspense",
                ),
                EditingAdjustment(
                    target="pacing.intro_duration_beats",
                    current_value=8,
                    suggested_value=16,
                    rationale="Longer build",
                ),
            ],
        )

        refined = evaluator.refine_instructions(sample_instructions, evaluation)

        # Original should be unchanged
        assert sample_instructions["pacing"]["speed"] == "dynamic"
        assert sample_instructions["pacing"]["intro_duration_beats"] == 8

        # Refined should have changes
        assert refined["pacing"]["speed"] == "slow"
        assert refined["pacing"]["intro_duration_beats"] == 16


class TestCreativeEvaluatorEvaluate:
    """Tests for full evaluate method."""

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_evaluate_with_mocked_llm(
        self, mock_director_class,
        sample_instructions, sample_clips_metadata, sample_result
    ):
        # Setup mock
        mock_director = MagicMock()
        mock_director._query_llm.return_value = json.dumps({
            "satisfaction_score": 0.9,
            "issues": [],
            "adjustments": [],
            "summary": "Excellent montage",
            "approve_for_render": True,
        })
        mock_director_class.return_value = mock_director

        evaluator = CreativeEvaluator()
        evaluator._director = mock_director

        evaluation = evaluator.evaluate(
            result=sample_result,
            original_instructions=sample_instructions,
            clips_metadata=sample_clips_metadata,
        )

        assert evaluation.satisfaction_score == 0.9
        assert evaluation.approve_for_render == True

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_evaluate_empty_llm_response(
        self, mock_director_class,
        sample_instructions, sample_clips_metadata, sample_result
    ):
        # Setup mock to return None (LLM failure)
        mock_director = MagicMock()
        mock_director._query_llm.return_value = None
        mock_director_class.return_value = mock_director

        evaluator = CreativeEvaluator()
        evaluator._director = mock_director

        evaluation = evaluator.evaluate(
            result=sample_result,
            original_instructions=sample_instructions,
            clips_metadata=sample_clips_metadata,
        )

        # Should auto-approve on LLM failure
        assert evaluation.approve_for_render == True

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_evaluate_max_iterations_forces_approval(
        self, mock_director_class,
        sample_instructions, sample_clips_metadata, sample_result
    ):
        mock_director = MagicMock()
        mock_director._query_llm.return_value = json.dumps({
            "satisfaction_score": 0.5,
            "issues": [{"type": "pacing", "severity": "critical", "description": "Bad"}],
            "adjustments": [],
            "summary": "Needs work",
            "approve_for_render": False,
        })
        mock_director_class.return_value = mock_director

        evaluator = CreativeEvaluator(max_iterations=3)
        evaluator._director = mock_director

        # Iteration 2 (0-indexed) = 3rd iteration = max
        evaluation = evaluator.evaluate(
            result=sample_result,
            original_instructions=sample_instructions,
            clips_metadata=sample_clips_metadata,
            iteration=2,
        )

        # Should force approval at max iterations
        assert evaluation.approve_for_render == True


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch('montage_ai.creative_evaluator.CreativeEvaluator')
    def test_evaluate_montage(self, mock_evaluator_class, sample_instructions, sample_clips_metadata, sample_result):
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = MontageEvaluation(
            satisfaction_score=0.9,
            approve_for_render=True,
        )
        mock_evaluator_class.return_value = mock_evaluator

        evaluation = evaluate_montage(
            result=sample_result,
            instructions=sample_instructions,
            clips_metadata=sample_clips_metadata,
        )

        assert evaluation.satisfaction_score == 0.9
        mock_evaluator.evaluate.assert_called_once()


# =============================================================================
# Integration Tests (without LLM)
# =============================================================================

class TestIntegration:
    """Integration tests for the evaluator workflow."""

    @patch('montage_ai.creative_evaluator.CreativeDirector')
    def test_full_refinement_workflow(self, mock_director_class, sample_instructions):
        mock_director = MagicMock()
        mock_director_class.return_value = mock_director

        evaluator = CreativeEvaluator()
        evaluator._director = mock_director

        # First evaluation - needs refinement
        first_eval = MontageEvaluation(
            satisfaction_score=0.6,
            issues=[EditingIssue(type="pacing", severity="moderate", description="Too fast")],
            adjustments=[
                EditingAdjustment(
                    target="pacing.speed",
                    current_value="dynamic",
                    suggested_value="slow",
                    rationale="Slow down",
                )
            ],
            approve_for_render=False,
        )

        # Refine instructions
        refined = evaluator.refine_instructions(sample_instructions, first_eval)

        assert refined["pacing"]["speed"] == "slow"
        assert sample_instructions["pacing"]["speed"] == "dynamic"  # Original unchanged

        # Second evaluation - approved
        second_eval = MontageEvaluation(
            satisfaction_score=0.9,
            issues=[],
            adjustments=[],
            approve_for_render=True,
        )

        assert not first_eval.approve_for_render
        assert second_eval.approve_for_render
