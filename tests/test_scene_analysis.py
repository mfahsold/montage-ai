"""
Tests for scene analysis module.

Tests scene detection, content analysis, and visual similarity calculation.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock

from src.montage_ai.scene_analysis import (
    ActionLevel,
    ShotType,
    Scene,
    SceneAnalysis,
    SceneDetector,
    SceneContentAnalyzer,
    detect_scenes,
    analyze_scene_content,
    calculate_visual_similarity,
    detect_motion_blur,
    find_best_start_point,
    _fallback_start_point,
)


class TestScene:
    """Tests for Scene dataclass."""

    def test_duration(self):
        """duration property calculates correctly."""
        scene = Scene(start=10.0, end=20.0, path="/video.mp4")
        assert scene.duration == 10.0

    def test_midpoint(self):
        """midpoint property calculates correctly."""
        scene = Scene(start=10.0, end=20.0, path="/video.mp4")
        assert scene.midpoint == 15.0

    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        scene = Scene(start=1.0, end=5.0, path="/test.mp4", meta={"action": "high"})
        d = scene.to_dict()
        assert d["start"] == 1.0
        assert d["end"] == 5.0
        assert d["path"] == "/test.mp4"
        assert d["meta"]["action"] == "high"


class TestSceneAnalysis:
    """Tests for SceneAnalysis dataclass."""

    def test_default(self):
        """default() creates medium-level analysis."""
        analysis = SceneAnalysis.default()
        assert analysis.quality == "YES"
        assert analysis.action == ActionLevel.MEDIUM
        assert analysis.shot == ShotType.MEDIUM

    def test_from_dict(self):
        """from_dict parses response correctly."""
        data = {
            "quality": "YES",
            "description": "Person walking",
            "action": "high",
            "shot": "wide"
        }
        analysis = SceneAnalysis.from_dict(data)
        assert analysis.quality == "YES"
        assert analysis.description == "Person walking"
        assert analysis.action == ActionLevel.HIGH
        assert analysis.shot == ShotType.WIDE

    def test_to_dict(self):
        """to_dict returns correct format."""
        analysis = SceneAnalysis(
            quality="NO",
            description="Blurry image",
            action=ActionLevel.LOW,
            shot=ShotType.CLOSE
        )
        d = analysis.to_dict()
        assert d["quality"] == "NO"
        assert d["action"] == "low"
        assert d["shot"] == "close"


class TestSceneDetector:
    """Tests for SceneDetector class."""

    @patch('src.montage_ai.scene_analysis.open_video')
    @patch('src.montage_ai.scene_analysis.SceneManager')
    def test_detect_returns_scenes(self, mock_manager_class, mock_open_video):
        """detect() returns list of Scene objects."""
        # Setup mocks
        mock_scene_manager = MagicMock()
        mock_manager_class.return_value = mock_scene_manager

        # Create mock scene list
        mock_scene1 = (MagicMock(), MagicMock())
        mock_scene1[0].get_seconds.return_value = 0.0
        mock_scene1[1].get_seconds.return_value = 5.0

        mock_scene2 = (MagicMock(), MagicMock())
        mock_scene2[0].get_seconds.return_value = 5.0
        mock_scene2[1].get_seconds.return_value = 10.0

        mock_scene_manager.get_scene_list.return_value = [mock_scene1, mock_scene2]

        detector = SceneDetector(threshold=30.0)
        scenes = detector.detect("/fake/video.mp4")

        assert len(scenes) == 2
        assert isinstance(scenes[0], Scene)
        assert scenes[0].start == 0.0
        assert scenes[0].end == 5.0
        assert scenes[1].start == 5.0
        assert scenes[1].end == 10.0

    @patch('src.montage_ai.scene_analysis.open_video')
    @patch('src.montage_ai.scene_analysis.SceneManager')
    @patch('src.montage_ai.scene_analysis.VideoFileClip')
    def test_detect_no_scenes_returns_full_video(self, mock_clip_class, mock_manager_class, mock_open_video):
        """When no scenes detected, return full video as single scene."""
        mock_scene_manager = MagicMock()
        mock_manager_class.return_value = mock_scene_manager
        mock_scene_manager.get_scene_list.return_value = []

        mock_clip = MagicMock()
        mock_clip.duration = 30.0
        mock_clip_class.return_value = mock_clip

        detector = SceneDetector()
        scenes = detector.detect("/fake/video.mp4")

        assert len(scenes) == 1
        assert scenes[0].start == 0.0
        assert scenes[0].end == 30.0


class TestSceneContentAnalyzer:
    """Tests for SceneContentAnalyzer class."""

    def test_disabled_ai_returns_default(self):
        """When AI disabled, returns default analysis."""
        analyzer = SceneContentAnalyzer(enable_ai=False)
        result = analyzer.analyze("/fake/video.mp4", 5.0)
        assert result.quality == "YES"
        assert result.action == ActionLevel.MEDIUM

    @patch('src.montage_ai.scene_analysis.cv2')
    def test_frame_extraction_failure(self, mock_cv2):
        """When frame extraction fails, returns NO quality."""
        mock_cv2.VideoCapture.return_value.read.return_value = (False, None)

        analyzer = SceneContentAnalyzer(enable_ai=True)
        result = analyzer.analyze("/fake/video.mp4", 5.0)

        assert result.quality == "NO"
        assert result.action == ActionLevel.LOW


class TestCalculateVisualSimilarity:
    """Tests for calculate_visual_similarity function."""

    @patch('src.montage_ai.scene_analysis.cv2')
    def test_returns_float(self, mock_cv2):
        """Returns a float between 0 and 1."""
        # Mock frame reads
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.calcHist.return_value = np.zeros((8, 8, 8))
        mock_cv2.compareHist.return_value = 0.75

        result = calculate_visual_similarity("/v1.mp4", 1.0, "/v2.mp4", 2.0)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch('src.montage_ai.scene_analysis.cv2')
    def test_frame_read_failure(self, mock_cv2):
        """Returns 0.0 when frame read fails."""
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap

        result = calculate_visual_similarity("/v1.mp4", 1.0, "/v2.mp4", 2.0)
        assert result == 0.0


class TestDetectMotionBlur:
    """Tests for detect_motion_blur function."""

    @patch('src.montage_ai.scene_analysis.cv2')
    def test_returns_float(self, mock_cv2):
        """Returns a float between 0 and 1."""
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)

        # Mock Laplacian returning low variance (blurry)
        mock_laplacian = MagicMock()
        mock_laplacian.var.return_value = 50.0  # Low variance = blurry
        mock_cv2.Laplacian.return_value = mock_laplacian

        result = detect_motion_blur("/video.mp4", 5.0)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch('src.montage_ai.scene_analysis.cv2')
    def test_sharp_image(self, mock_cv2):
        """Sharp image (high Laplacian variance) returns low blur score."""
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)

        mock_laplacian = MagicMock()
        mock_laplacian.var.return_value = 1000.0  # High variance = sharp
        mock_cv2.Laplacian.return_value = mock_laplacian

        result = detect_motion_blur("/video.mp4", 5.0)
        assert result < 0.5  # Should be low blur score


class TestFindBestStartPoint:
    """Tests for find_best_start_point function."""

    def test_fallback_start_point(self):
        """_fallback_start_point returns valid start."""
        start = _fallback_start_point(0.0, 10.0, 3.0)
        assert 0.0 <= start <= 7.0

    def test_fallback_when_no_room(self):
        """When target_duration >= scene duration, returns scene_start."""
        start = _fallback_start_point(5.0, 8.0, 5.0)
        assert start == 5.0

    @patch('src.montage_ai.scene_analysis.cv2')
    def test_returns_float(self, mock_cv2):
        """Returns a valid start point."""
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap

        result = find_best_start_point("/video.mp4", 0.0, 10.0, 2.0)
        assert isinstance(result, float)
        assert 0.0 <= result <= 8.0


class TestLegacyFunctions:
    """Tests for legacy compatibility functions."""

    @patch('src.montage_ai.scene_analysis.SceneDetector')
    def test_detect_scenes_returns_tuples(self, mock_detector_class):
        """detect_scenes returns list of tuples."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            Scene(start=0.0, end=5.0, path="/v.mp4"),
            Scene(start=5.0, end=10.0, path="/v.mp4"),
        ]
        mock_detector_class.return_value = mock_detector

        result = detect_scenes("/video.mp4")

        assert result == [(0.0, 5.0), (5.0, 10.0)]

    @patch('src.montage_ai.scene_analysis.SceneContentAnalyzer')
    def test_analyze_scene_content_returns_dict(self, mock_analyzer_class):
        """analyze_scene_content returns dictionary."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = SceneAnalysis(
            quality="YES",
            description="Test",
            action=ActionLevel.HIGH,
            shot=ShotType.WIDE
        )
        mock_analyzer_class.return_value = mock_analyzer

        result = analyze_scene_content("/video.mp4", 5.0)

        assert isinstance(result, dict)
        assert result["quality"] == "YES"
        assert result["action"] == "high"
        assert result["shot"] == "wide"
