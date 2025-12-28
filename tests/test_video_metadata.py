"""
Tests for video metadata module.

Tests metadata extraction, output profile determination, and FFmpeg parameter building.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import json

from src.montage_ai.video_metadata import (
    VideoMetadata,
    OutputProfile,
    MetadataProber,
    OutputProfileBuilder,
    probe_metadata,
    determine_output_profile,
    build_ffmpeg_params,
    ffprobe_video_metadata,
    _parse_frame_rate,
    _weighted_median,
    _even_int,
    _snap_aspect_ratio,
    _snap_resolution,
    _normalize_codec_name,
)


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    def test_aspect_ratio_horizontal(self):
        """aspect_ratio returns correct value for horizontal video."""
        meta = VideoMetadata(
            path="/test.mp4", width=1920, height=1080,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        assert meta.aspect_ratio == pytest.approx(16 / 9, rel=0.01)

    def test_aspect_ratio_vertical(self):
        """aspect_ratio returns correct value for vertical video."""
        meta = VideoMetadata(
            path="/test.mp4", width=1080, height=1920,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        assert meta.aspect_ratio == pytest.approx(9 / 16, rel=0.01)

    def test_orientation_horizontal(self):
        """orientation returns 'horizontal' for wide videos."""
        meta = VideoMetadata(
            path="/test.mp4", width=1920, height=1080,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        assert meta.orientation == "horizontal"

    def test_orientation_vertical(self):
        """orientation returns 'vertical' for tall videos."""
        meta = VideoMetadata(
            path="/test.mp4", width=1080, height=1920,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        assert meta.orientation == "vertical"

    def test_orientation_square(self):
        """orientation returns 'square' for equal dimensions."""
        meta = VideoMetadata(
            path="/test.mp4", width=1080, height=1080,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        assert meta.orientation == "square"

    def test_resolution(self):
        """resolution returns (width, height) tuple."""
        meta = VideoMetadata(
            path="/test.mp4", width=1920, height=1080,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        assert meta.resolution == (1920, 1080)

    def test_long_side(self):
        """long_side returns larger dimension."""
        meta = VideoMetadata(
            path="/test.mp4", width=1080, height=1920,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        assert meta.long_side == 1920

    def test_short_side(self):
        """short_side returns smaller dimension."""
        meta = VideoMetadata(
            path="/test.mp4", width=1080, height=1920,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        assert meta.short_side == 1080

    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        meta = VideoMetadata(
            path="/test.mp4", width=1920, height=1080,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )
        d = meta.to_dict()
        assert d["path"] == "/test.mp4"
        assert d["width"] == 1920
        assert d["height"] == 1080
        assert d["fps"] == 30.0
        assert d["codec"] == "h264"

    def test_from_dict(self):
        """from_dict creates VideoMetadata correctly."""
        data = {
            "path": "/video.mp4",
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "duration": 5.0,
            "codec": "hevc",
            "pix_fmt": "yuv420p",
            "bitrate": 3000000,
        }
        meta = VideoMetadata.from_dict(data)
        assert meta.path == "/video.mp4"
        assert meta.width == 1280
        assert meta.height == 720
        assert meta.fps == 24.0


class TestOutputProfile:
    """Tests for OutputProfile dataclass."""

    def test_default(self):
        """default() creates standard profile."""
        profile = OutputProfile.default()
        assert profile.width > 0
        assert profile.height > 0
        assert profile.fps > 0
        assert profile.reason == "defaults"

    def test_resolution(self):
        """resolution returns (width, height) tuple."""
        profile = OutputProfile(
            width=1080, height=1920, fps=30.0, pix_fmt="yuv420p",
            codec="libx264", profile="high", level="4.1"
        )
        assert profile.resolution == (1080, 1920)

    def test_is_4k(self):
        """is_4k returns True for 4K resolution."""
        profile = OutputProfile(
            width=3840, height=2160, fps=30.0, pix_fmt="yuv420p",
            codec="libx264", profile="high", level="5.1"
        )
        assert profile.is_4k is True

    def test_is_not_4k(self):
        """is_4k returns False for 1080p resolution."""
        profile = OutputProfile(
            width=1920, height=1080, fps=30.0, pix_fmt="yuv420p",
            codec="libx264", profile="high", level="4.1"
        )
        assert profile.is_4k is False

    def test_is_hd(self):
        """is_hd returns True for 1080p resolution."""
        profile = OutputProfile(
            width=1920, height=1080, fps=30.0, pix_fmt="yuv420p",
            codec="libx264", profile="high", level="4.1"
        )
        assert profile.is_hd is True

    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        profile = OutputProfile(
            width=1080, height=1920, fps=30.0, pix_fmt="yuv420p",
            codec="libx264", profile="high", level="4.1",
            orientation="vertical", aspect_ratio="9:16"
        )
        d = profile.to_dict()
        assert d["width"] == 1080
        assert d["height"] == 1920
        assert d["orientation"] == "vertical"
        assert d["aspect_ratio"] == "9:16"

    def test_from_dict(self):
        """from_dict creates OutputProfile correctly."""
        data = {
            "width": 1920,
            "height": 1080,
            "fps": 24.0,
            "pix_fmt": "yuv420p",
            "codec": "libx265",
            "profile": "main",
            "level": "4.0",
            "orientation": "horizontal",
        }
        profile = OutputProfile.from_dict(data)
        assert profile.width == 1920
        assert profile.height == 1080
        assert profile.codec == "libx265"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_frame_rate_fraction(self):
        """_parse_frame_rate parses fractional frame rate."""
        assert _parse_frame_rate("30000/1001") == pytest.approx(29.97, rel=0.01)

    def test_parse_frame_rate_integer(self):
        """_parse_frame_rate parses integer frame rate."""
        assert _parse_frame_rate("30") == 30.0

    def test_parse_frame_rate_float(self):
        """_parse_frame_rate parses float frame rate."""
        assert _parse_frame_rate("29.97") == pytest.approx(29.97, rel=0.01)

    def test_parse_frame_rate_empty(self):
        """_parse_frame_rate returns 0 for empty string."""
        assert _parse_frame_rate("") == 0.0

    def test_parse_frame_rate_invalid(self):
        """_parse_frame_rate returns 0 for invalid string."""
        assert _parse_frame_rate("invalid") == 0.0

    def test_weighted_median_simple(self):
        """_weighted_median computes correct median."""
        values = [1.0, 2.0, 3.0]
        weights = [1.0, 1.0, 1.0]
        assert _weighted_median(values, weights) == 2.0

    def test_weighted_median_weighted(self):
        """_weighted_median respects weights."""
        values = [1.0, 2.0, 3.0]
        weights = [0.1, 0.1, 10.0]  # Heavy weight on 3.0
        assert _weighted_median(values, weights) == 3.0

    def test_weighted_median_empty(self):
        """_weighted_median returns 0 for empty list."""
        assert _weighted_median([], []) == 0.0

    def test_even_int_odd(self):
        """_even_int rounds odd numbers up."""
        assert _even_int(5.0) == 6

    def test_even_int_even(self):
        """_even_int keeps even numbers."""
        assert _even_int(4.0) == 4

    def test_even_int_minimum(self):
        """_even_int returns at least 2."""
        assert _even_int(0.5) == 2

    def test_snap_aspect_ratio_16_9(self):
        """_snap_aspect_ratio snaps to 16:9."""
        name, ratio = _snap_aspect_ratio(1.77)  # Close to 16:9
        assert name == "16:9"
        assert ratio == pytest.approx(16 / 9, rel=0.01)

    def test_snap_aspect_ratio_9_16(self):
        """_snap_aspect_ratio snaps to 9:16."""
        name, ratio = _snap_aspect_ratio(0.5625)  # Exact 9:16
        assert name == "9:16"
        assert ratio == pytest.approx(9 / 16, rel=0.01)

    def test_snap_aspect_ratio_custom(self):
        """_snap_aspect_ratio returns custom for non-standard ratio."""
        name, ratio = _snap_aspect_ratio(2.5)  # Not close to any preset
        assert name == "custom"
        assert ratio == 2.5

    def test_snap_resolution_horizontal(self):
        """_snap_resolution snaps to 1080p for horizontal."""
        w, h = _snap_resolution((1900, 1070), "horizontal", 1920)
        assert (w, h) == (1920, 1080)

    def test_snap_resolution_vertical(self):
        """_snap_resolution snaps to 1080x1920 for vertical."""
        w, h = _snap_resolution((1070, 1900), "vertical", 1920)
        assert (w, h) == (1080, 1920)

    def test_normalize_codec_name_h264(self):
        """_normalize_codec_name normalizes h264 variants."""
        assert _normalize_codec_name("h264") == "h264"
        assert _normalize_codec_name("avc1") == "h264"
        assert _normalize_codec_name("H.264") == "h264"

    def test_normalize_codec_name_hevc(self):
        """_normalize_codec_name normalizes HEVC variants."""
        assert _normalize_codec_name("hevc") == "hevc"
        assert _normalize_codec_name("h265") == "hevc"
        assert _normalize_codec_name("H.265") == "hevc"


class TestProbeMetadata:
    """Tests for probe_metadata function."""

    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_returns_metadata(self, mock_run):
        """probe_metadata returns VideoMetadata object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "width": 1920,
                    "height": 1080,
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "r_frame_rate": "30/1",
                    "bit_rate": "5000000"
                }],
                "format": {"duration": "10.0", "bit_rate": "5000000"}
            })
        )

        result = probe_metadata("/test/video.mp4")

        assert result is not None
        assert isinstance(result, VideoMetadata)
        assert result.width == 1920
        assert result.height == 1080
        assert result.fps == 30.0
        assert result.codec == "h264"

    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_returns_none_on_failure(self, mock_run):
        """probe_metadata returns None when ffprobe fails."""
        mock_run.return_value = MagicMock(returncode=1)

        result = probe_metadata("/nonexistent.mp4")
        assert result is None


class TestMetadataProber:
    """Tests for MetadataProber class."""

    @patch('src.montage_ai.video_metadata.probe_metadata')
    def test_probe_calls_function(self, mock_probe):
        """probe() calls probe_metadata function."""
        mock_probe.return_value = VideoMetadata(
            path="/test.mp4", width=1920, height=1080,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )

        prober = MetadataProber()
        result = prober.probe("/test.mp4")

        assert result is not None
        mock_probe.assert_called_once_with("/test.mp4")

    @patch('src.montage_ai.video_metadata.probe_metadata')
    def test_probe_many_filters_none(self, mock_probe):
        """probe_many() filters out None results."""
        mock_probe.side_effect = [
            VideoMetadata(
                path="/v1.mp4", width=1920, height=1080,
                fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
            ),
            None,  # Failure
            VideoMetadata(
                path="/v3.mp4", width=1280, height=720,
                fps=24.0, duration=5.0, codec="h264", pix_fmt="yuv420p", bitrate=3000000
            ),
        ]

        prober = MetadataProber()
        results = prober.probe_many(["/v1.mp4", "/v2.mp4", "/v3.mp4"])

        assert len(results) == 2
        assert results[0].path == "/v1.mp4"
        assert results[1].path == "/v3.mp4"


class TestOutputProfileBuilder:
    """Tests for OutputProfileBuilder class."""

    @patch('src.montage_ai.video_metadata.probe_metadata')
    def test_build_empty_list(self, mock_probe):
        """build() with empty list returns default profile."""
        builder = OutputProfileBuilder()
        profile = builder.build([])

        assert profile.reason == "defaults"

    @patch('src.montage_ai.video_metadata.probe_metadata')
    def test_build_horizontal_dominant(self, mock_probe):
        """build() detects horizontal orientation."""
        mock_probe.return_value = VideoMetadata(
            path="/test.mp4", width=1920, height=1080,
            fps=30.0, duration=60.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )

        builder = OutputProfileBuilder()
        profile = builder.build(["/test.mp4"])

        assert profile.orientation == "horizontal"

    @patch('src.montage_ai.video_metadata.probe_metadata')
    def test_build_vertical_dominant(self, mock_probe):
        """build() detects vertical orientation."""
        mock_probe.return_value = VideoMetadata(
            path="/test.mp4", width=1080, height=1920,
            fps=30.0, duration=60.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )

        builder = OutputProfileBuilder()
        profile = builder.build(["/test.mp4"])

        assert profile.orientation == "vertical"


class TestDetermineOutputProfile:
    """Tests for determine_output_profile function."""

    @patch('src.montage_ai.video_metadata.probe_metadata')
    def test_returns_output_profile(self, mock_probe):
        """determine_output_profile returns OutputProfile object."""
        mock_probe.return_value = VideoMetadata(
            path="/test.mp4", width=1920, height=1080,
            fps=30.0, duration=60.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )

        result = determine_output_profile(["/test.mp4"])

        assert isinstance(result, OutputProfile)
        assert result.width > 0
        assert result.height > 0


class TestBuildFfmpegParams:
    """Tests for build_ffmpeg_params function."""

    def test_returns_list(self):
        """build_ffmpeg_params returns list of strings."""
        params = build_ffmpeg_params()
        assert isinstance(params, list)

    def test_with_crf(self):
        """build_ffmpeg_params accepts CRF parameter."""
        params = build_ffmpeg_params(crf=23)
        assert isinstance(params, list)


class TestLegacyFunctions:
    """Tests for legacy compatibility functions."""

    @patch('src.montage_ai.video_metadata.probe_metadata')
    def test_ffprobe_video_metadata_returns_dict(self, mock_probe):
        """ffprobe_video_metadata returns dictionary."""
        mock_probe.return_value = VideoMetadata(
            path="/test.mp4", width=1920, height=1080,
            fps=30.0, duration=10.0, codec="h264", pix_fmt="yuv420p", bitrate=5000000
        )

        result = ffprobe_video_metadata("/test.mp4")

        assert isinstance(result, dict)
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["codec"] == "h264"

    @patch('src.montage_ai.video_metadata.probe_metadata')
    def test_ffprobe_video_metadata_returns_none(self, mock_probe):
        """ffprobe_video_metadata returns None on failure."""
        mock_probe.return_value = None

        result = ffprobe_video_metadata("/nonexistent.mp4")
        assert result is None
