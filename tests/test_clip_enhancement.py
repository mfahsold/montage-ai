"""
Tests for clip enhancement module.

Tests stabilization, upscaling, color grading, and enhancement operations.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, Mock

from src.montage_ai.clip_enhancement import (
    BrightnessAnalysis,
    EnhancementResult,
    ClipEnhancer,
    stabilize_clip,
    enhance_clip,
    upscale_clip,
    enhance_clips_parallel,
    color_match_clips,
    _check_vidstab_available,
)


class TestBrightnessAnalysis:
    """Tests for BrightnessAnalysis dataclass."""

    def test_neutral(self):
        """neutral() creates default neutral analysis."""
        analysis = BrightnessAnalysis.neutral()
        assert analysis.avg_brightness == 128
        assert analysis.is_dark is False
        assert analysis.is_bright is False
        assert analysis.suggested_brightness == 0

    def test_dark_analysis(self):
        """Dark clip analysis has correct flags."""
        analysis = BrightnessAnalysis(
            avg_brightness=50,
            is_dark=True,
            is_bright=False,
            suggested_brightness=0.10
        )
        assert analysis.is_dark is True
        assert analysis.is_bright is False
        assert analysis.suggested_brightness > 0

    def test_bright_analysis(self):
        """Bright clip analysis has correct flags."""
        analysis = BrightnessAnalysis(
            avg_brightness=200,
            is_dark=False,
            is_bright=True,
            suggested_brightness=-0.08
        )
        assert analysis.is_dark is False
        assert analysis.is_bright is True
        assert analysis.suggested_brightness < 0


class TestEnhancementResult:
    """Tests for EnhancementResult dataclass."""

    def test_success_result(self):
        """Success result has correct attributes."""
        result = EnhancementResult(
            input_path="/input.mp4",
            output_path="/output.mp4",
            success=True,
            method="enhanced"
        )
        assert result.success is True
        assert result.method == "enhanced"

    def test_failure_result(self):
        """Failure result returns original path."""
        result = EnhancementResult(
            input_path="/input.mp4",
            output_path="/input.mp4",
            success=False,
            method="original",
            details={"error": "ffmpeg failed"}
        )
        assert result.success is False
        assert result.input_path == result.output_path
        assert "error" in result.details


class TestClipEnhancer:
    """Tests for ClipEnhancer class."""

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_analyze_brightness_returns_analysis(self, mock_run, mock_get_config):
        """_analyze_brightness returns BrightnessAnalysis."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        # Mock ffprobe duration
        mock_run.side_effect = [
            Mock(stdout="10.0\n"),  # Duration
            Mock(stderr="YAVG=120\n"),  # Sample 1
            Mock(stderr="YAVG=125\n"),  # Sample 2
            Mock(stderr="YAVG=118\n"),  # Sample 3
        ]

        enhancer = ClipEnhancer()
        analysis = enhancer._analyze_brightness("/fake/video.mp4")

        assert isinstance(analysis, BrightnessAnalysis)
        assert analysis.is_dark is False
        assert analysis.is_bright is False

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_analyze_brightness_dark_clip(self, mock_run, mock_get_config):
        """Dark clip (avg < 70) is detected correctly."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_run.side_effect = [
            Mock(stdout="10.0\n"),
            Mock(stderr="YAVG=50\n"),
            Mock(stderr="YAVG=55\n"),
            Mock(stderr="YAVG=45\n"),
        ]

        enhancer = ClipEnhancer()
        analysis = enhancer._analyze_brightness("/fake/video.mp4")

        assert analysis.is_dark is True
        assert analysis.suggested_brightness > 0

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_analyze_brightness_fallback_on_error(self, mock_run, mock_get_config):
        """Returns neutral analysis on error."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_run.side_effect = Exception("ffprobe failed")

        enhancer = ClipEnhancer()
        analysis = enhancer._analyze_brightness("/fake/video.mp4")

        assert analysis.avg_brightness == 128
        assert analysis.is_dark is False
        assert analysis.is_bright is False

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_enhance_calls_ffmpeg(self, mock_run, mock_get_config):
        """enhance() calls ffmpeg with correct filters."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        # Mock brightness analysis
        mock_run.side_effect = [
            Mock(stdout="10.0\n"),  # Duration probe
            Mock(stderr="YAVG=120\n"),
            Mock(stderr="YAVG=120\n"),
            Mock(stderr="YAVG=120\n"),
            Mock(returncode=0),  # FFmpeg enhance
        ]

        enhancer = ClipEnhancer()
        result = enhancer.enhance("/input.mp4", "/output.mp4")

        assert result == "/output.mp4"
        # Verify FFmpeg was called
        ffmpeg_call = mock_run.call_args_list[-1]
        cmd = ffmpeg_call[0][0]
        assert "ffmpeg" in cmd
        assert "-vf" in cmd

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_enhance_returns_original_on_failure(self, mock_run, mock_get_config):
        """enhance() returns original path on FFmpeg failure."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_run.side_effect = [
            Mock(stdout="10.0\n"),
            Mock(stderr=""),
            Mock(stderr=""),
            Mock(stderr=""),
            Exception("FFmpeg failed"),
        ]

        enhancer = ClipEnhancer()
        result = enhancer.enhance("/input.mp4", "/output.mp4")

        assert result == "/input.mp4"


class TestStabilization:
    """Tests for stabilization functionality."""

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement._check_vidstab_available')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_stabilize_uses_vidstab_when_available(self, mock_run, mock_vidstab, mock_get_config):
        """Uses vidstab 2-pass when available."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_vidstab.return_value = True
        mock_run.return_value = Mock(returncode=0)

        with patch('os.path.exists', return_value=True):
            with patch('os.remove'):
                enhancer = ClipEnhancer()
                enhancer._stabilize_vidstab("/input.mp4", "/output.mp4")

        # Should have called ffmpeg twice (detect + transform)
        assert mock_run.call_count >= 2

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement._check_vidstab_available')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_stabilize_falls_back_to_deshake(self, mock_run, mock_vidstab, mock_get_config):
        """Falls back to deshake when vidstab unavailable."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_vidstab.return_value = False
        mock_run.return_value = Mock(returncode=0)

        enhancer = ClipEnhancer()
        result = enhancer._stabilize_deshake("/input.mp4", "/output.mp4")

        assert result == "/output.mp4"
        cmd = mock_run.call_args[0][0]
        assert "deshake" in str(cmd)

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_deshake_returns_original_on_error(self, mock_run, mock_get_config):
        """deshake returns original path on error."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_run.side_effect = Exception("FFmpeg error")

        enhancer = ClipEnhancer()
        result = enhancer._stabilize_deshake("/input.mp4", "/output.mp4")

        assert result == "/input.mp4"


class TestUpscaling:
    """Tests for upscaling functionality."""

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('subprocess.run')
    def test_check_realesrgan_available(self, mock_run, mock_get_config):
        """_check_realesrgan_available detects Vulkan GPU."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_run.side_effect = [
            Mock(stderr=""),  # realesrgan test
            Mock(returncode=0, stdout="nvidia geforce"),  # vulkaninfo
        ]

        enhancer = ClipEnhancer()
        result = enhancer._check_realesrgan_available()

        assert result is True

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('subprocess.run')
    def test_check_realesrgan_detects_software_renderer(self, mock_run, mock_get_config):
        """Detects and skips software Vulkan renderers."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_run.side_effect = [
            Mock(stderr=""),
            Mock(returncode=0, stdout="llvmpipe"),  # Software renderer
        ]

        enhancer = ClipEnhancer()
        result = enhancer._check_realesrgan_available()

        assert result is False

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    @patch('src.montage_ai.clip_enhancement.subprocess.check_output')
    def test_upscale_ffmpeg_calculates_dimensions(self, mock_check, mock_run, mock_get_config):
        """FFmpeg upscaling calculates 2x dimensions."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        # Mock ffprobe
        mock_check.return_value = b'{"streams": [{"width": 1920, "height": 1080}]}'
        mock_run.return_value = Mock(returncode=0)

        enhancer = ClipEnhancer()
        result = enhancer._upscale_ffmpeg("/input.mp4", "/output.mp4", scale=2)

        assert result == "/output.mp4"
        cmd = mock_run.call_args[0][0]
        assert "scale=3840:2160" in str(cmd)

    @patch('src.montage_ai.clip_enhancement.get_ffmpeg_config')
    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    @patch('src.montage_ai.clip_enhancement.subprocess.check_output')
    def test_upscale_ffmpeg_handles_rotation(self, mock_check, mock_run, mock_get_config):
        """FFmpeg upscaling handles rotated videos."""
        mock_get_config.return_value = MagicMock(threads=1, preset="medium", effective_codec="libx264", pix_fmt="yuv420p", profile="high", level="4.0")
        mock_check.return_value = b'{"streams": [{"width": 1080, "height": 1920, "side_data_list": [{"rotation": 90}]}]}'
        mock_run.return_value = Mock(returncode=0)

        enhancer = ClipEnhancer()
        result = enhancer._upscale_ffmpeg("/input.mp4", "/output.mp4", scale=2)

        assert result == "/output.mp4"
        # With 90 rotation, dimensions should be swapped before 2x
        cmd = mock_run.call_args[0][0]
        assert "scale=3840:2160" in str(cmd)


class TestColorMatching:
    """Tests for color matching functionality."""

    def test_color_match_returns_identity_without_library(self):
        """Returns identity mapping when color-matcher unavailable."""
        with patch('src.montage_ai.clip_enhancement._get_color_matcher', return_value=(None, None)):
            enhancer = ClipEnhancer()
            result = enhancer.color_match(["/a.mp4", "/b.mp4"])

        assert result == {"/a.mp4": "/a.mp4", "/b.mp4": "/b.mp4"}

    def test_color_match_single_clip(self):
        """Single clip returns identity mapping."""
        enhancer = ClipEnhancer()
        result = enhancer.color_match(["/single.mp4"])

        assert result == {"/single.mp4": "/single.mp4"}


class TestParallelEnhancement:
    """Tests for parallel enhancement."""

    @patch.object(ClipEnhancer, 'enhance')
    def test_enhance_batch_sequential(self, mock_enhance):
        """enhance_batch processes sequentially when disabled."""
        mock_enhance.return_value = "/output.mp4"

        enhancer = ClipEnhancer()
        enhancer.parallel_enhance = False

        jobs = [("/in1.mp4", "/out1.mp4"), ("/in2.mp4", "/out2.mp4")]
        result = enhancer.enhance_batch(jobs)

        assert len(result) == 2
        assert mock_enhance.call_count == 2

    @patch.object(ClipEnhancer, 'enhance')
    def test_enhance_batch_parallel(self, mock_enhance):
        """enhance_batch processes in parallel when enabled."""
        mock_enhance.return_value = "/output.mp4"

        enhancer = ClipEnhancer()
        enhancer.parallel_enhance = True
        enhancer.max_parallel_jobs = 2

        jobs = [("/in1.mp4", "/out1.mp4"), ("/in2.mp4", "/out2.mp4")]
        result = enhancer.enhance_batch(jobs)

        assert len(result) == 2


class TestVidstabAvailability:
    """Tests for vidstab availability checking."""

    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_vidstab_available(self, mock_run):
        """Detects vidstab when present in ffmpeg filters."""
        import src.montage_ai.clip_enhancement as ce
        ce._VIDSTAB_AVAILABLE = None  # Reset cache

        mock_run.return_value = Mock(stdout="... vidstabdetect ...")

        result = _check_vidstab_available()
        assert result is True

    @patch('src.montage_ai.clip_enhancement.subprocess.run')
    def test_vidstab_not_available(self, mock_run):
        """Detects vidstab absence."""
        import src.montage_ai.clip_enhancement as ce
        ce._VIDSTAB_AVAILABLE = None  # Reset cache

        mock_run.return_value = Mock(stdout="... deshake ...")

        result = _check_vidstab_available()
        assert result is False


class TestLegacyFunctions:
    """Tests for legacy compatibility functions."""

    @patch.object(ClipEnhancer, 'stabilize')
    def test_stabilize_clip_legacy(self, mock_method):
        """stabilize_clip calls ClipEnhancer.stabilize."""
        mock_method.return_value = "/output.mp4"

        result = stabilize_clip("/input.mp4", "/output.mp4")

        assert result == "/output.mp4"
        mock_method.assert_called_once()

    @patch.object(ClipEnhancer, 'enhance')
    def test_enhance_clip_legacy(self, mock_method):
        """enhance_clip calls ClipEnhancer.enhance."""
        mock_method.return_value = "/output.mp4"

        result = enhance_clip("/input.mp4", "/output.mp4")

        assert result == "/output.mp4"
        mock_method.assert_called_once()

    @patch.object(ClipEnhancer, 'upscale')
    def test_upscale_clip_legacy(self, mock_method):
        """upscale_clip calls ClipEnhancer.upscale."""
        mock_method.return_value = "/output.mp4"

        result = upscale_clip("/input.mp4", "/output.mp4")

        assert result == "/output.mp4"
        mock_method.assert_called_once()

    @patch.object(ClipEnhancer, 'enhance_batch')
    def test_enhance_clips_parallel_legacy(self, mock_method):
        """enhance_clips_parallel calls ClipEnhancer.enhance_batch."""
        mock_method.return_value = {"/in.mp4": "/out.mp4"}

        jobs = [("/in.mp4", "/out.mp4")]
        result = enhance_clips_parallel(jobs)

        assert result == {"/in.mp4": "/out.mp4"}
        mock_method.assert_called_once()

    @patch.object(ClipEnhancer, 'color_match')
    def test_color_match_clips_legacy(self, mock_method):
        """color_match_clips calls ClipEnhancer.color_match."""
        mock_method.return_value = {"/a.mp4": "/a.mp4"}

        result = color_match_clips(["/a.mp4"])

        assert result == {"/a.mp4": "/a.mp4"}
        mock_method.assert_called_once()
