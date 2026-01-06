"""
Tests for Story Arc CSP Solver - Constraint Satisfaction for Clip Assignment.

TDD: Tests written before implementation.
Uses Z3 or OR-Tools for constraint satisfaction.
"""

import pytest
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
from unittest.mock import Mock, patch


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockFootageClip:
    """Mock FootageClip for testing."""
    path: str
    duration: float
    energy: float
    shot_type: str = "medium"
    scene_type: str = "action"
    location: str = "location_a"

    @property
    def id(self) -> str:
        return self.path


@pytest.fixture
def sample_clips():
    """Sample clips for CSP testing."""
    return [
        MockFootageClip(path="/clip1.mp4", duration=5.0, energy=0.2, shot_type="wide", location="beach"),
        MockFootageClip(path="/clip2.mp4", duration=4.0, energy=0.3, shot_type="medium", location="beach"),
        MockFootageClip(path="/clip3.mp4", duration=6.0, energy=0.5, shot_type="close", location="city"),
        MockFootageClip(path="/clip4.mp4", duration=3.0, energy=0.7, shot_type="medium", location="city"),
        MockFootageClip(path="/clip5.mp4", duration=4.0, energy=0.9, shot_type="close", location="mountain"),
        MockFootageClip(path="/clip6.mp4", duration=5.0, energy=0.8, shot_type="wide", location="mountain"),
        MockFootageClip(path="/clip7.mp4", duration=3.0, energy=0.4, shot_type="medium", location="beach"),
        MockFootageClip(path="/clip8.mp4", duration=4.0, energy=0.3, shot_type="wide", location="city"),
    ]


@pytest.fixture
def hitchcock_style():
    """Hitchcock style configuration."""
    return {
        "name": "hitchcock",
        "pacing": "slow_build",
        "energy_curve": "exponential",
        "climax_position": 0.75,
    }


@pytest.fixture
def mtv_style():
    """MTV style configuration."""
    return {
        "name": "mtv",
        "pacing": "fast",
        "energy_curve": "constant_high",
        "climax_position": 0.5,
    }


# =============================================================================
# StoryPhase Enum Tests
# =============================================================================

class TestStoryPhase:
    """Tests for StoryPhase enum."""

    def test_phase_enum_exists(self):
        """StoryPhase enum should have all required phases."""
        from montage_ai.story_arc_csp import StoryPhase

        assert hasattr(StoryPhase, 'INTRO')
        assert hasattr(StoryPhase, 'BUILD')
        assert hasattr(StoryPhase, 'CLIMAX')
        assert hasattr(StoryPhase, 'SUSTAIN')
        assert hasattr(StoryPhase, 'OUTRO')

    def test_phase_order(self):
        """Phases should have correct order values."""
        from montage_ai.story_arc_csp import StoryPhase

        assert StoryPhase.INTRO.order < StoryPhase.BUILD.order
        assert StoryPhase.BUILD.order < StoryPhase.CLIMAX.order
        assert StoryPhase.CLIMAX.order < StoryPhase.SUSTAIN.order
        assert StoryPhase.SUSTAIN.order < StoryPhase.OUTRO.order


# =============================================================================
# PhaseConstraints Tests
# =============================================================================

class TestPhaseConstraints:
    """Tests for PhaseConstraints dataclass."""

    def test_phase_constraints_creation(self):
        """PhaseConstraints should specify requirements for a phase."""
        from montage_ai.story_arc_csp import PhaseConstraints

        constraints = PhaseConstraints(
            phase_name="climax",
            target_duration=12.0,
            duration_tolerance=0.1,
            min_energy=0.7,
            max_energy=1.0,
            preferred_shot_types=["close", "medium"],
            min_clips=2,
            max_clips=5,
        )

        assert constraints.target_duration == 12.0
        assert constraints.min_energy == 0.7
        assert "close" in constraints.preferred_shot_types

    def test_phase_constraints_defaults(self):
        """PhaseConstraints should have sensible defaults."""
        from montage_ai.story_arc_csp import PhaseConstraints

        constraints = PhaseConstraints(
            phase_name="intro",
            target_duration=9.0,
        )

        assert constraints.duration_tolerance == 0.15  # Default 15%
        assert constraints.min_energy == 0.0
        assert constraints.max_energy == 1.0


# =============================================================================
# StoryArcCSPSolver Tests
# =============================================================================

class TestStoryArcCSPSolverInit:
    """Tests for StoryArcCSPSolver initialization."""

    def test_default_init(self):
        """Solver should initialize with default phase ratios."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()

        assert solver.default_phase_ratios["intro"] == pytest.approx(0.15, rel=0.01)
        assert solver.default_phase_ratios["climax"] == pytest.approx(0.20, rel=0.01)

    def test_custom_phase_ratios(self):
        """Solver should accept custom phase ratios."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        custom_ratios = {
            "intro": 0.10,
            "build": 0.30,
            "climax": 0.25,
            "sustain": 0.20,
            "outro": 0.15,
        }

        solver = StoryArcCSPSolver(phase_ratios=custom_ratios)

        assert solver.phase_ratios["intro"] == 0.10
        assert solver.phase_ratios["build"] == 0.30


class TestStoryArcCSPSolverSolve:
    """Tests for StoryArcCSPSolver.solve() method."""

    def test_solve_basic(self, sample_clips, hitchcock_style):
        """solve() should return valid clip assignments."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver, StoryPhase

        solver = StoryArcCSPSolver()

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        assert result is not None
        assert "assignments" in result
        assert "feasible" in result
        assert result["feasible"] == True

        # Check all phases have clips
        for phase in StoryPhase:
            assert phase.value in result["assignments"]

    def test_solve_respects_duration(self, sample_clips, hitchcock_style):
        """Assigned clips should respect target duration."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()

        result = solver.solve(
            clips=sample_clips,
            target_duration=25.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            total_assigned_duration = sum(
                sample_clips[i].duration
                for phase_clips in result["assignments"].values()
                for i in phase_clips
            )

            # Within 15% tolerance
            assert total_assigned_duration >= 25.0 * 0.85
            assert total_assigned_duration <= 25.0 * 1.15

    def test_solve_no_double_assignment(self, sample_clips, hitchcock_style):
        """Each clip should be assigned to at most one phase."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            all_assigned = []
            for phase_clips in result["assignments"].values():
                all_assigned.extend(phase_clips)

            # No duplicates
            assert len(all_assigned) == len(set(all_assigned))

    def test_solve_infeasible_duration(self, sample_clips, hitchcock_style):
        """solve() should return infeasible when duration impossible."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()

        # Total footage: ~34s, requesting 100s is impossible
        result = solver.solve(
            clips=sample_clips,
            target_duration=100.0,
            style=hitchcock_style,
        )

        assert result["feasible"] == False
        assert "reason" in result


class TestEnergyConstraints:
    """Tests for energy-based constraints."""

    def test_hitchcock_climax_high_energy(self, sample_clips, hitchcock_style):
        """Hitchcock style should place high-energy clips in CLIMAX."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            climax_clips = result["assignments"].get("climax", [])
            climax_energies = [sample_clips[i].energy for i in climax_clips]

            # Climax should have at least one high-energy clip
            assert any(e >= 0.7 for e in climax_energies)

    def test_intro_low_energy(self, sample_clips, hitchcock_style):
        """INTRO phase should have lower energy clips."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            intro_clips = result["assignments"].get("intro", [])
            intro_energies = [sample_clips[i].energy for i in intro_clips]

            # Intro should not have the highest energy clips
            if intro_energies:
                assert max(intro_energies) < 0.8


class TestShotVarietyConstraints:
    """Tests for shot variety constraints."""

    def test_minimum_shot_types(self, sample_clips, hitchcock_style):
        """Solution should use multiple shot types when available."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver(min_shot_types=2)

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            all_assigned = []
            for phase_clips in result["assignments"].values():
                all_assigned.extend(phase_clips)

            shot_types_used = set(sample_clips[i].shot_type for i in all_assigned)

            assert len(shot_types_used) >= 2

    def test_no_consecutive_same_shot(self, sample_clips, hitchcock_style):
        """Should avoid same shot type in consecutive clips (soft constraint)."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver(avoid_consecutive_same_shot=True)

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        # This is a soft constraint, so we just check it doesn't break


class TestLocationConstraints:
    """Tests for location/variety constraints."""

    def test_minimum_locations(self, sample_clips, hitchcock_style):
        """Solution should use clips from multiple locations."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver(min_locations=2)

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            all_assigned = []
            for phase_clips in result["assignments"].values():
                all_assigned.extend(phase_clips)

            locations_used = set(sample_clips[i].location for i in all_assigned)

            assert len(locations_used) >= 2


# =============================================================================
# Style-Specific Constraint Tests
# =============================================================================

class TestStyleSpecificConstraints:
    """Tests for style-specific constraint generation."""

    def test_hitchcock_constraints(self, hitchcock_style):
        """Hitchcock style should generate slow-build constraints."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()
        constraints = solver._generate_style_constraints(hitchcock_style)

        # Hitchcock: slow intro, explosive climax
        assert constraints["intro"]["max_energy"] <= 0.5
        assert constraints["climax"]["min_energy"] >= 0.6

    def test_mtv_constraints(self, mtv_style):
        """MTV style should generate high-energy throughout."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()
        constraints = solver._generate_style_constraints(mtv_style)

        # MTV: high energy everywhere
        assert constraints["intro"]["min_energy"] >= 0.4
        assert constraints["climax"]["min_energy"] >= 0.7


# =============================================================================
# Solver Backend Tests
# =============================================================================

class TestSolverBackend:
    """Tests for different solver backends (Z3, OR-Tools, Fallback)."""

    def test_z3_backend_available(self):
        """Z3 backend should be available if z3-solver installed."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver(backend="z3")

        # Should not raise if z3 is available, or gracefully fallback
        assert solver.backend in ["z3", "fallback"]

    def test_fallback_backend(self, sample_clips, hitchcock_style):
        """Fallback backend should work without external solvers."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver(backend="fallback")

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        # Fallback uses greedy/heuristic approach
        assert "feasible" in result


# =============================================================================
# Optimization Tests
# =============================================================================

class TestOptimization:
    """Tests for optimization criteria."""

    def test_maximize_energy_curve_fit(self, sample_clips, hitchcock_style):
        """Solver should optimize for best energy curve fit."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver(optimize_energy_curve=True)

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            assert "energy_curve_score" in result
            assert result["energy_curve_score"] >= 0.0

    def test_minimize_shot_type_repeats(self, sample_clips, hitchcock_style):
        """Solver should minimize consecutive shot type repeats."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver(minimize_shot_repeats=True)

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            assert "shot_repeat_score" in result


# =============================================================================
# Integration with Footage Manager Tests
# =============================================================================

class TestFootageManagerIntegration:
    """Tests for integration with existing FootageManager."""

    def test_from_footage_manager_clips(self):
        """Should accept clips from FootageManager."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        # Mock FootageClip from footage_manager
        mock_fm_clip = Mock()
        mock_fm_clip.path = "/test.mp4"
        mock_fm_clip.duration = 5.0
        mock_fm_clip.energy = 0.5
        mock_fm_clip.shot_type = "medium"
        mock_fm_clip.scene_type = "action"

        solver = StoryArcCSPSolver()

        # Should handle FootageClip interface
        normalized = solver._normalize_clip(mock_fm_clip)

        assert normalized["duration"] == 5.0
        assert normalized["energy"] == 0.5

    def test_output_format_for_editor(self, sample_clips, hitchcock_style):
        """Output should be compatible with editor.py."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()

        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )

        if result["feasible"]:
            # Output should include ordered clip list for editor
            assert "ordered_clips" in result or "assignments" in result


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Tests for solver performance."""

    def test_solve_time_reasonable(self, sample_clips, hitchcock_style):
        """Solving should complete within reasonable time."""
        import time
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        solver = StoryArcCSPSolver()

        start = time.time()
        result = solver.solve(
            clips=sample_clips,
            target_duration=30.0,
            style=hitchcock_style,
        )
        elapsed = time.time() - start

        # Should complete within 5 seconds for small problem
        assert elapsed < 5.0

    def test_larger_clip_set(self, hitchcock_style):
        """Should handle larger clip sets."""
        from montage_ai.story_arc_csp import StoryArcCSPSolver

        # Generate 50 clips
        large_clips = [
            MockFootageClip(
                path=f"/clip{i}.mp4",
                duration=3.0 + (i % 5),
                energy=0.2 + (i % 8) * 0.1,
                shot_type=["wide", "medium", "close"][i % 3],
                location=f"loc_{i % 5}",
            )
            for i in range(50)
        ]

        solver = StoryArcCSPSolver()

        result = solver.solve(
            clips=large_clips,
            target_duration=120.0,
            style=hitchcock_style,
        )

        assert "feasible" in result
