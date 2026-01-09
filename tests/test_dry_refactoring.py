"""
Test suite for ConfigParser and ClipScorer modules.

Comprehensive tests for the DRY refactoring work.
"""

import os
import pytest
from typing import Dict, List, Any

from montage_ai.config_parser import ConfigParser
from montage_ai.clip_scoring import ClipScorer, ScoringWeights, get_default_scorer


# =============================================================================
# ConfigParser Tests
# =============================================================================

class TestConfigParser:
    """Test ConfigParser environment variable parsing."""
    
    def test_parse_int(self):
        """Test integer parsing from environment."""
        os.environ["TEST_INT_VAR"] = "42"
        assert ConfigParser.parse_int("TEST_INT_VAR", 0) == 42
        assert ConfigParser.parse_int("NONEXISTENT_INT", 99) == 99
        
        # Invalid value should return default
        os.environ["BAD_INT"] = "not_a_number"
        assert ConfigParser.parse_int("BAD_INT", 10) == 10
    
    def test_parse_float(self):
        """Test float parsing from environment."""
        os.environ["TEST_FLOAT_VAR"] = "3.14159"
        result = ConfigParser.parse_float("TEST_FLOAT_VAR", 0.0)
        assert abs(result - 3.14159) < 0.001
        assert ConfigParser.parse_float("NONEXISTENT_FLOAT", 2.71) == 2.71
    
    def test_parse_bool(self):
        """Test boolean parsing from environment."""
        # True values
        for val in ["true", "True", "TRUE", "1", "yes", "on"]:
            os.environ["TEST_BOOL"] = val
            assert ConfigParser.parse_bool("TEST_BOOL", False) is True
        
        # False values
        for val in ["false", "False", "FALSE", "0", "no", "off"]:
            os.environ["TEST_BOOL"] = val
            assert ConfigParser.parse_bool("TEST_BOOL", True) is False
        
        # Default when not set
        assert ConfigParser.parse_bool("MISSING_BOOL", True) is True
    
    def test_parse_str(self):
        """Test string parsing from environment."""
        os.environ["TEST_STR"] = "hello"
        assert ConfigParser.parse_str("TEST_STR", "") == "hello"
        assert ConfigParser.parse_str("MISSING_STR", "default") == "default"
    
    def test_make_int_parser(self):
        """Test factory method for int parsers."""
        os.environ["FACTORY_INT"] = "123"
        parser = ConfigParser.make_int_parser("FACTORY_INT", 0)
        assert callable(parser)
        assert parser() == 123
    
    def test_make_bool_parser(self):
        """Test factory method for bool parsers."""
        os.environ["FACTORY_BOOL"] = "true"
        parser = ConfigParser.make_bool_parser("FACTORY_BOOL", False)
        assert callable(parser)
        assert parser() is True
    
    def test_parse_dict(self):
        """Test JSON dict parsing from environment."""
        os.environ["CONFIG_JSON"] = '{"key": "value", "num": 42}'
        result = ConfigParser.parse_dict("CONFIG_JSON")
        assert result["key"] == "value"
        assert result["num"] == 42
        
        # Invalid JSON should return default
        os.environ["BAD_JSON"] = "{invalid json"
        assert ConfigParser.parse_dict("BAD_JSON", {}) == {}


# =============================================================================
# ClipScorer Tests
# =============================================================================

class TestClipScorer:
    """Test unified clip scoring functionality."""
    
    @staticmethod
    def create_test_clips() -> List[Dict[str, Any]]:
        """Create sample clips for testing."""
        return [
            {"id": "clip1", "shot_type": "wide", "energy_level": 0.3},
            {"id": "clip2", "shot_type": "close", "energy_level": 0.6},
            {"id": "clip3", "shot_type": "close", "energy_level": 0.7},
            {"id": "clip4", "shot_type": "wide", "energy_level": 0.4},
            {"id": "clip5", "shot_type": "medium", "energy_level": 0.5},
        ]
    
    def test_instantiation(self):
        """Test ClipScorer can be instantiated."""
        scorer = ClipScorer()
        assert scorer is not None
        assert scorer.weights is not None
    
    def test_custom_weights(self):
        """Test ClipScorer with custom weights."""
        custom_weights = ScoringWeights(
            shot_variety_weight=0.5,
            match_quality_weight=0.5,
        )
        scorer = ClipScorer(custom_weights)
        assert scorer.weights.shot_variety_weight == 0.5
    
    def test_score_shot_variety_all_same(self):
        """Test shot variety with all same types."""
        clips = [
            {"shot_type": "wide"},
            {"shot_type": "wide"},
            {"shot_type": "wide"},
        ]
        score = ClipScorer.score_shot_variety(clips, [0, 1, 2])
        assert score == 0.0  # All repeats
    
    def test_score_shot_variety_all_different(self):
        """Test shot variety with all different types."""
        clips = [
            {"shot_type": "wide"},
            {"shot_type": "close"},
            {"shot_type": "medium"},
        ]
        score = ClipScorer.score_shot_variety(clips, [0, 1, 2])
        assert score == 1.0  # No repeats
    
    def test_score_shot_variety_mixed(self):
        """Test shot variety with mixed repeats."""
        clips = self.create_test_clips()
        score = ClipScorer.score_shot_variety(clips, [0, 1, 2, 3, 4])
        # Repeats: close-close (position 2), all others different
        # 1 repeat out of 4 transitions = 0.75
        assert 0.65 < score < 0.85
    
    def test_score_shot_variety_single_clip(self):
        """Test shot variety with single clip."""
        clips = [{"shot_type": "wide"}]
        score = ClipScorer.score_shot_variety(clips, [0])
        assert score == 1.0  # No transitions possible
    
    def test_score_match_quality_with_pacing(self):
        """Test match quality considering pacing."""
        scene = {"pacing": 0.7}
        clip_meta = {"energy_level": 0.7}  # Perfect match
        score = ClipScorer.score_match_quality(scene, clip_meta)
        assert score > 0.8  # Should be high
    
    def test_score_match_quality_bad_pacing(self):
        """Test match quality with mismatched pacing."""
        scene = {"pacing": 0.1}  # Very slow
        clip_meta = {"energy_level": 0.9}  # Very fast
        score = ClipScorer.score_match_quality(scene, clip_meta)
        assert score < 0.3  # Should be low
    
    def test_score_broll_relevance_keyword_match(self):
        """Test B-roll relevance with keyword overlap."""
        scene = {"scene_keywords": ["nature", "forest", "birds"]}
        clip_meta = {"keywords": ["nature", "trees", "animals"]}
        score = ClipScorer.score_broll_relevance(scene, clip_meta)
        # Overlap: "nature" (1 match out of 5 total unique)
        assert 0.05 < score < 0.3
    
    def test_score_broll_relevance_no_keywords(self):
        """Test B-roll relevance with no metadata."""
        scene = {}
        clip_meta = {}
        score = ClipScorer.score_broll_relevance(scene, clip_meta)
        assert score == 0.5  # Default when no info
    
    def test_score_faces_single_confident(self):
        """Test face scoring with single confident face."""
        clip_meta = {"face_count": 1, "face_confidence": 0.95}
        score = ClipScorer.score_faces(clip_meta, require_faces=False)
        assert 0.9 < score <= 0.95
    
    def test_score_faces_multiple(self):
        """Test face scoring with multiple faces."""
        clip_meta = {"face_count": 3, "face_confidence": 0.8}
        score = ClipScorer.score_faces(clip_meta)
        # Multiple faces reduce bonus
        assert 0.5 < score < 0.8
    
    def test_score_faces_required_but_missing(self):
        """Test face scoring when faces required but absent."""
        clip_meta = {"face_count": 0}
        score = ClipScorer.score_faces(clip_meta, require_faces=True)
        assert score == 0.0
    
    def test_score_visual_novelty_no_history(self):
        """Test novelty scoring with no previous clips."""
        clip_meta = {"dominant_color": (255, 100, 50)}
        score = ClipScorer.score_visual_novelty(clip_meta, None)
        assert score == 1.0  # Maximally novel when no history
    
    def test_score_visual_novelty_with_history(self):
        """Test novelty scoring with similar recent clips."""
        clip_meta = {"dominant_color": (255, 100, 50)}
        recent = [
            {"dominant_color": (255, 100, 50)},  # Same
            {"dominant_color": (254, 101, 49)},  # Very similar
        ]
        score = ClipScorer.score_visual_novelty(clip_meta, recent)
        assert score < 0.1  # Low novelty
    
    def test_default_scorer_singleton(self):
        """Test that get_default_scorer returns same instance."""
        scorer1 = get_default_scorer()
        scorer2 = get_default_scorer()
        assert scorer1 is scorer2
    
    def test_comprehensive_scoring(self):
        """Test comprehensive clip scoring."""
        clips = self.create_test_clips()
        scene = {"pacing": 0.6, "description": "medium pace"}
        scorer = ClipScorer()
        
        # Score clip 2 given clips 0, 1 already used
        score = scorer.score_clip_comprehensive(
            clips=clips,
            ordered=[0, 1],
            scene=scene,
            current_clip_idx=2,
            require_faces=False,
        )
        
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
