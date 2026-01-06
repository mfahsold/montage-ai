"""
Tests for MontageBuilder module.

Tests the object-oriented pipeline for creating video montages.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import numpy as np

from src.montage_ai.core.montage_builder import MontageBuilder
from src.montage_ai.core.context import (
    MontageContext,
    MontageResult,
    AudioAnalysisResult,
    SceneInfo,
    OutputProfile,
    ClipMetadata,
)


class TestAudioAnalysisResult:
    """Tests for AudioAnalysisResult dataclass."""

    def test_beat_count(self):
        """beat_count returns correct number of beats."""
        result = AudioAnalysisResult(
            music_path="/test.mp3",
            beat_times=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            tempo=120.0,
            energy_times=np.array([0.0, 1.0]),
            energy_values=np.array([0.5, 0.7]),
            duration=30.0,
        )
        assert result.beat_count == 5

    def test_avg_energy(self):
        """avg_energy returns mean of energy values."""
        result = AudioAnalysisResult(
            music_path="/test.mp3",
            beat_times=np.array([]),
            tempo=120.0,
            energy_times=np.array([0.0, 1.0]),
            energy_values=np.array([0.4, 0.8]),
            duration=30.0,
        )
        assert result.avg_energy == pytest.approx(0.6)

    def test_energy_profile_high(self):
        """energy_profile returns 'high' for high energy."""
        result = AudioAnalysisResult(
            music_path="/test.mp3",
            beat_times=np.array([]),
            tempo=120.0,
            energy_times=np.array([0.0]),
            energy_values=np.array([0.8]),
            duration=30.0,
        )
        assert result.energy_profile == "high"

    def test_energy_profile_low(self):
        """energy_profile returns 'low' for low energy."""
        result = AudioAnalysisResult(
            music_path="/test.mp3",
            beat_times=np.array([]),
            tempo=120.0,
            energy_times=np.array([0.0]),
            energy_values=np.array([0.2]),
            duration=30.0,
        )
        assert result.energy_profile == "low"

    def test_energy_profile_mixed(self):
        """energy_profile returns 'mixed' for middle energy."""
        result = AudioAnalysisResult(
            music_path="/test.mp3",
            beat_times=np.array([]),
            tempo=120.0,
            energy_times=np.array([0.0]),
            energy_values=np.array([0.5]),
            duration=30.0,
        )
        assert result.energy_profile == "mixed"


class TestSceneInfo:
    """Tests for SceneInfo dataclass."""

    def test_midpoint(self):
        """midpoint returns center of scene."""
        scene = SceneInfo(
            path="/video.mp4",
            start=10.0,
            end=20.0,
            duration=10.0,
        )
        assert scene.midpoint == 15.0

    def test_to_dict(self):
        """to_dict returns legacy format."""
        scene = SceneInfo(
            path="/video.mp4",
            start=10.0,
            end=20.0,
            duration=10.0,
            meta={"action": "high"},
        )
        result = scene.to_dict()
        assert result["path"] == "/video.mp4"
        assert result["start"] == 10.0
        assert result["end"] == 20.0
        assert result["duration"] == 10.0
        assert result["meta"]["action"] == "high"


class TestOutputProfile:
    """Tests for OutputProfile dataclass."""

    def test_default_values(self):
        """OutputProfile has correct defaults."""
        profile = OutputProfile(
            width=1080,
            height=1920,
            fps=30.0,
            codec="libx264",
            pix_fmt="yuv420p",
        )
        assert profile.orientation == "vertical"
        assert profile.aspect_ratio == "9:16"
        assert profile.bitrate == 0


class TestClipMetadata:
    """Tests for ClipMetadata dataclass."""

    def test_enhancements_default(self):
        """enhancements defaults to empty dict."""
        clip = ClipMetadata(
            source_path="/video.mp4",
            start_time=10.0,
            duration=2.0,
            timeline_start=0.0,
            energy=0.5,
            action="medium",
            shot="medium",
            beat_idx=0,
            beats_per_cut=4.0,
            selection_score=50.0,
        )
        assert clip.enhancements == {}


class TestMontageContext:
    """Tests for MontageContext dataclass."""

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_reset_timeline_state(self, mock_settings):
        """reset_timeline_state clears timeline state."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")

        from src.montage_ai.core.montage_builder import MontagePaths

        paths = MontagePaths(
            input_dir=Path("/input"),
            music_dir=Path("/music"),
            assets_dir=Path("/assets"),
            output_dir=Path("/output"),
            temp_dir=Path("/tmp"),
        )

        ctx = MontageContext(
            job_id="test",
            variant_id=1,
            settings=mock_settings.return_value,
            paths=paths,
        )
        ctx.timeline.current_time = 30.0
        ctx.timeline.beat_idx = 10
        ctx.timeline.cut_number = 5

        ctx.reset_timeline_state()

        assert ctx.timeline.current_time == 0.0
        assert ctx.timeline.beat_idx == 0
        assert ctx.timeline.cut_number == 0
        assert ctx.timeline.clips_metadata == []

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_get_story_position(self, mock_settings):
        """get_story_position returns correct position."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")

        from src.montage_ai.core.montage_builder import MontagePaths

        paths = MontagePaths(
            input_dir=Path("/input"),
            music_dir=Path("/music"),
            assets_dir=Path("/assets"),
            output_dir=Path("/output"),
            temp_dir=Path("/tmp"),
        )
        
        ctx = MontageContext(
            job_id="test",
            variant_id=1,
            settings=mock_settings.return_value,
            paths=paths,
        )
        ctx.timeline.target_duration = 100.0
        ctx.timeline.current_time = 50.0

        assert ctx.get_story_position() == 0.5

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_get_story_phase_intro(self, mock_settings):
        """get_story_phase returns 'intro' for early position."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")

        from src.montage_ai.core.montage_builder import MontagePaths

        paths = MontagePaths(
            input_dir=Path("/input"),
            music_dir=Path("/music"),
            assets_dir=Path("/assets"),
            output_dir=Path("/output"),
            temp_dir=Path("/tmp"),
        )

        ctx = MontageContext(
            job_id="test",
            variant_id=1,
            settings=mock_settings.return_value,
            paths=paths,
        )
        ctx.timeline.target_duration = 100.0
        ctx.timeline.current_time = 5.0

        assert ctx.get_story_phase() == "intro"

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_get_story_phase_climax(self, mock_settings):
        """get_story_phase returns 'climax' for middle position."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")
        
        from src.montage_ai.core.montage_builder import MontagePaths

        paths = MontagePaths(
            input_dir=Path("/input"),
            music_dir=Path("/music"),
            assets_dir=Path("/assets"),
            output_dir=Path("/output"),
            temp_dir=Path("/tmp"),
        )
        
        ctx = MontageContext(
            job_id="test",
            variant_id=1,
            settings=mock_settings.return_value,
            paths=paths,
        )
        ctx.timeline.target_duration = 100.0
        ctx.timeline.current_time = 55.0

        assert ctx.get_story_phase() == "climax"


class TestMontageResult:
    """Tests for MontageResult dataclass."""

    def test_success_result(self):
        """Success result has correct attributes."""
        result = MontageResult(
            success=True,
            output_path="/output/video.mp4",
            duration=60.0,
            cut_count=15,
            render_time=45.0,
            file_size_mb=25.5,
        )
        assert result.success is True
        assert result.error is None
        assert result.output_path == "/output/video.mp4"

    def test_failure_result(self):
        """Failure result has error message."""
        result = MontageResult(
            success=False,
            output_path=None,
            duration=0.0,
            cut_count=0,
            render_time=0.0,
            error="Failed to render",
        )
        assert result.success is False
        assert result.error == "Failed to render"


class TestMontageBuilder:
    """Tests for MontageBuilder class."""

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_initialization(self, mock_settings):
        """MontageBuilder initializes with default settings."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.job_id = "test_job"
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")
        mock_settings.return_value.features.stabilize = False
        mock_settings.return_value.features.upscale = False
        mock_settings.return_value.features.enhance = False

        builder = MontageBuilder(variant_id=1)

        assert builder.variant_id == 1
        assert builder.ctx.job_id == "test_job"
        assert builder.ctx.variant_id == 1

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_initialization_with_editing_instructions(self, mock_settings):
        """MontageBuilder accepts editing instructions."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.job_id = "test_job"
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")
        mock_settings.return_value.features.stabilize = False
        mock_settings.return_value.features.upscale = False
        mock_settings.return_value.features.enhance = False

        instructions = {"style": {"name": "dynamic"}}
        builder = MontageBuilder(variant_id=1, editing_instructions=instructions)

        # Convert dict to model if needed by implementation, but test checked against dict?
        # The implementation converts dict to EditingInstructions model.
        # So builder.ctx.creative.editing_instructions is an object.
        # If instructions is a dict, we should expect equality or attribute match.
        inst = builder.ctx.creative.editing_instructions
        assert inst.style.name == "dynamic"

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_apply_creative_director_effects(self, mock_settings):
        """_apply_creative_director_effects applies effects from instructions."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.job_id = "test_job"
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")
        mock_settings.return_value.features.stabilize = False
        mock_settings.return_value.features.upscale = False
        mock_settings.return_value.features.enhance = False
        mock_settings.return_value.encoding.quality_profile = "standard"

        instructions = {
            "effects": {
                "stabilization": True,
                "upscale": True,
                "sharpness_boost": True,
            }
        }
        builder = MontageBuilder(variant_id=1, editing_instructions=instructions)
        builder._apply_creative_director_effects()

        assert builder.ctx.features.stabilize is True
        assert builder.ctx.features.upscale is True
        assert builder.ctx.features.enhance is True

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_env_overrides_template(self, mock_settings):
        """Environment settings override template effects."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.job_id = "test_job"
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")
        mock_settings.return_value.encoding.quality_profile = "standard"
        # ENV says stabilize=True
        mock_settings.return_value.features.stabilize = True
        mock_settings.return_value.features.upscale = False
        mock_settings.return_value.features.enhance = False

        instructions = {
            "effects": {
                "stabilization": False,  # Template says false
            }
        }
        builder = MontageBuilder(variant_id=1, editing_instructions=instructions)
        builder._apply_creative_director_effects()

        # ENV should win
        assert builder.ctx.features.stabilize is True


class TestMontageBuilderHelpers:
    """Tests for MontageBuilder helper methods."""

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_get_energy_at_time(self, mock_settings):
        """_get_energy_at_time returns correct energy."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.job_id = "test_job"
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")
        mock_settings.return_value.features.stabilize = False
        mock_settings.return_value.features.upscale = False
        mock_settings.return_value.features.enhance = False

        builder = MontageBuilder(variant_id=1)

        energy_times = np.array([0.0, 1.0, 2.0, 3.0])
        energy_values = np.array([0.2, 0.4, 0.6, 0.8])
        
        # Setup context for PacingEngine
        builder.ctx.media.audio_result = MagicMock()
        builder.ctx.media.audio_result.energy_times = energy_times
        builder.ctx.media.audio_result.energy_values = energy_values

        result = builder._pacing_engine.get_energy_at_time(1.5)
        # Should return energy at index 2 (closest to 1.5)
        assert result == 0.6

    @patch("src.montage_ai.core.montage_builder.get_settings")
    def test_get_energy_at_time_empty(self, mock_settings):
        """_get_energy_at_time returns 0.5 for empty arrays."""
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.job_id = "test_job"
        mock_settings.return_value.paths.input_dir = Path("/input")
        mock_settings.return_value.paths.music_dir = Path("/music")
        mock_settings.return_value.paths.assets_dir = Path("/assets")
        mock_settings.return_value.paths.output_dir = Path("/output")
        mock_settings.return_value.paths.temp_dir = Path("/tmp")
        mock_settings.return_value.features.stabilize = False
        mock_settings.return_value.features.upscale = False
        mock_settings.return_value.features.enhance = False

        builder = MontageBuilder(variant_id=1)
        
        # Setup empty contexts
        builder.ctx.media.audio_result = MagicMock()
        builder.ctx.media.audio_result.energy_times = np.array([])
        builder.ctx.media.audio_result.energy_values = np.array([])

        result = builder._pacing_engine.get_energy_at_time(1.0)
        assert result == 0.5
