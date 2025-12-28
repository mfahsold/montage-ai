"""
Tests for semantic matcher module.

Phase 2: Semantic Storytelling
Tests embedding-based semantic matching for clip selection.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.montage_ai.semantic_matcher import (
    SemanticScore,
    SemanticMatcher,
    get_semantic_matcher,
    reset_semantic_matcher,
    MOOD_KEYWORDS,
    SETTING_KEYWORDS,
)


class TestSemanticScore:
    """Tests for SemanticScore dataclass."""

    def test_create_score(self):
        """SemanticScore can be created with all fields."""
        score = SemanticScore(
            overall_score=0.75,
            tag_score=0.8,
            caption_score=0.6,
            mood_match=1.0,
            setting_match=0.0,
            matched_tags=["beach", "ocean"],
        )

        assert score.overall_score == 0.75
        assert score.tag_score == 0.8
        assert score.caption_score == 0.6
        assert score.mood_match == 1.0
        assert score.setting_match == 0.0
        assert score.matched_tags == ["beach", "ocean"]

    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        score = SemanticScore(
            overall_score=0.5,
            tag_score=0.4,
            caption_score=0.3,
            mood_match=0.0,
            setting_match=1.0,
            matched_tags=["city"],
        )

        d = score.to_dict()
        assert d["overall_score"] == 0.5
        assert d["tag_score"] == 0.4
        assert d["matched_tags"] == ["city"]


class TestMoodKeywords:
    """Tests for mood keyword mappings."""

    def test_energetic_keywords(self):
        """Energetic mood has expected keywords."""
        assert "action" in MOOD_KEYWORDS["energetic"]
        assert "fast" in MOOD_KEYWORDS["energetic"]
        assert "exciting" in MOOD_KEYWORDS["energetic"]

    def test_calm_keywords(self):
        """Calm mood has expected keywords."""
        assert "peaceful" in MOOD_KEYWORDS["calm"]
        assert "serene" in MOOD_KEYWORDS["calm"]

    def test_dramatic_keywords(self):
        """Dramatic mood has expected keywords."""
        assert "epic" in MOOD_KEYWORDS["dramatic"]
        assert "cinematic" in MOOD_KEYWORDS["dramatic"]


class TestSettingKeywords:
    """Tests for setting keyword mappings."""

    def test_beach_keywords(self):
        """Beach setting has expected keywords."""
        assert "ocean" in SETTING_KEYWORDS["beach"]
        assert "waves" in SETTING_KEYWORDS["beach"]
        assert "surf" in SETTING_KEYWORDS["beach"]

    def test_city_keywords(self):
        """City setting has expected keywords."""
        assert "urban" in SETTING_KEYWORDS["city"]
        assert "street" in SETTING_KEYWORDS["city"]


class TestSemanticMatcher:
    """Tests for SemanticMatcher class."""

    @pytest.fixture
    def matcher(self):
        """Create a matcher instance with mocked model."""
        reset_semantic_matcher()
        with patch.object(SemanticMatcher, '_init_model'):
            matcher = SemanticMatcher()
            # Mock the model
            SemanticMatcher._model = MagicMock()
            SemanticMatcher._model.encode.side_effect = lambda x: np.random.rand(384)
            return matcher

    def test_initialization_weights(self):
        """Matcher initializes with correct weights."""
        reset_semantic_matcher()
        with patch.object(SemanticMatcher, '_init_model'):
            matcher = SemanticMatcher(
                tag_weight=0.5,
                caption_weight=0.3,
                mood_weight=0.1,
                setting_weight=0.1,
            )

            assert matcher.tag_weight == 0.5
            assert matcher.caption_weight == 0.3
            assert matcher.mood_weight == 0.1
            assert matcher.setting_weight == 0.1

    def test_cosine_similarity_identical(self, matcher):
        """Cosine similarity is 1.0 for identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = matcher._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_cosine_similarity_orthogonal(self, matcher):
        """Cosine similarity is 0.0 for orthogonal vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        similarity = matcher._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_cosine_similarity_none_input(self, matcher):
        """Cosine similarity returns 0.0 for None input."""
        vec = np.array([1.0, 2.0])
        assert matcher._cosine_similarity(None, vec) == 0.0
        assert matcher._cosine_similarity(vec, None) == 0.0
        assert matcher._cosine_similarity(None, None) == 0.0

    def test_check_mood_match_energetic(self, matcher):
        """check_mood_match detects energetic keywords."""
        assert matcher._check_mood_match("action scene", "energetic") == 1.0
        assert matcher._check_mood_match("fast paced", "energetic") == 1.0
        assert matcher._check_mood_match("calm scene", "energetic") == 0.0

    def test_check_mood_match_calm(self, matcher):
        """check_mood_match detects calm keywords."""
        assert matcher._check_mood_match("peaceful nature", "calm") == 1.0
        assert matcher._check_mood_match("serene lake", "calm") == 1.0
        assert matcher._check_mood_match("action scene", "calm") == 0.0

    def test_check_mood_match_neutral(self, matcher):
        """check_mood_match returns 0.0 for neutral mood."""
        assert matcher._check_mood_match("any query", "neutral") == 0.0

    def test_check_setting_match_beach(self, matcher):
        """check_setting_match detects beach keywords."""
        assert matcher._check_setting_match("ocean waves", "beach") == 1.0
        assert matcher._check_setting_match("surfing video", "beach") == 1.0
        assert matcher._check_setting_match("city street", "beach") == 0.0

    def test_check_setting_match_city(self, matcher):
        """check_setting_match detects city keywords."""
        assert matcher._check_setting_match("urban exploration", "city") == 1.0
        assert matcher._check_setting_match("street photography", "city") == 1.0
        assert matcher._check_setting_match("nature walk", "city") == 0.0

    def test_check_setting_match_unknown(self, matcher):
        """check_setting_match returns 0.0 for unknown setting."""
        assert matcher._check_setting_match("any query", "unknown") == 0.0

    def test_find_matched_tags(self, matcher):
        """find_matched_tags returns matching tags."""
        tags = ["beach", "surfing", "waves", "sunset"]

        matched = matcher._find_matched_tags("beach waves", tags)
        assert "beach" in matched
        assert "waves" in matched
        assert "surfing" not in matched

    def test_find_matched_tags_empty(self, matcher):
        """find_matched_tags returns empty list for no matches."""
        tags = ["mountain", "hiking"]
        matched = matcher._find_matched_tags("beach waves", tags)
        assert matched == []

    def test_match_query_to_clip_basic(self, matcher):
        """match_query_to_clip returns SemanticScore."""
        clip = {
            "tags": ["beach", "surfing", "ocean"],
            "caption": "Person surfing on blue ocean waves",
            "mood": "energetic",
            "setting": "beach",
        }

        result = matcher.match_query_to_clip("beach action", clip)

        assert isinstance(result, SemanticScore)
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.tag_score <= 1.0
        assert 0.0 <= result.caption_score <= 1.0

    def test_match_query_to_clip_mood_match(self, matcher):
        """match_query_to_clip detects mood match."""
        clip = {
            "tags": ["scene"],
            "caption": "A scene",
            "mood": "energetic",
            "setting": "outdoor",
        }

        result = matcher.match_query_to_clip("action scene", clip)
        assert result.mood_match == 1.0

        result2 = matcher.match_query_to_clip("calm scene", clip)
        assert result2.mood_match == 0.0

    def test_match_query_to_clip_setting_match(self, matcher):
        """match_query_to_clip detects setting match."""
        clip = {
            "tags": ["scene"],
            "caption": "A scene",
            "mood": "neutral",
            "setting": "beach",
        }

        result = matcher.match_query_to_clip("ocean waves", clip)
        assert result.setting_match == 1.0

        result2 = matcher.match_query_to_clip("city street", clip)
        assert result2.setting_match == 0.0

    def test_match_query_to_clip_empty_analysis(self, matcher):
        """match_query_to_clip handles empty analysis gracefully."""
        clip = {}

        result = matcher.match_query_to_clip("beach", clip)

        assert isinstance(result, SemanticScore)
        assert result.overall_score == 0.0
        assert result.matched_tags == []

    def test_match_query_to_clips_sorted(self, matcher):
        """match_query_to_clips returns sorted results."""
        clips = [
            {"meta": {"tags": ["mountain"], "caption": "Mountain hike", "mood": "calm", "setting": "nature"}},
            {"meta": {"tags": ["beach", "ocean"], "caption": "Beach surfing", "mood": "energetic", "setting": "beach"}},
            {"meta": {"tags": ["city"], "caption": "City street", "mood": "neutral", "setting": "city"}},
        ]

        # Beach query should rank beach clip higher
        results = matcher.match_query_to_clips("ocean beach surfing", clips)

        assert len(results) == 3
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i][1].overall_score >= results[i + 1][1].overall_score

    def test_match_query_to_clips_min_score(self, matcher):
        """match_query_to_clips respects min_score threshold."""
        clips = [
            {"meta": {"tags": ["beach"], "caption": "Beach", "mood": "calm", "setting": "beach"}},
            {"meta": {"tags": ["city"], "caption": "City", "mood": "neutral", "setting": "city"}},
        ]

        results = matcher.match_query_to_clips("beach", clips, min_score=0.5)

        # Only clips with score >= 0.5 should be returned
        for clip, score in results:
            assert score.overall_score >= 0.5


class TestSemanticMatcherUnavailable:
    """Tests for SemanticMatcher when embeddings are unavailable."""

    def test_match_without_model(self):
        """Matcher works without model (returns zero scores)."""
        reset_semantic_matcher()

        with patch.object(SemanticMatcher, '_init_model'):
            matcher = SemanticMatcher()
            SemanticMatcher._model = None

            clip = {
                "tags": ["beach"],
                "caption": "Beach scene",
                "mood": "calm",
                "setting": "beach",
            }

            result = matcher.match_query_to_clip("beach waves", clip)

            # Without embeddings, tag/caption scores are 0
            assert result.tag_score == 0.0
            assert result.caption_score == 0.0
            # But keyword-based mood/setting still work
            assert result.setting_match == 1.0  # "waves" matches beach

    def test_is_available_false(self):
        """is_available returns False when model is None."""
        reset_semantic_matcher()

        with patch.object(SemanticMatcher, '_init_model'):
            matcher = SemanticMatcher()
            SemanticMatcher._model = None

            assert matcher.is_available is False


class TestGlobalMatcher:
    """Tests for global singleton accessor."""

    def test_get_semantic_matcher_singleton(self):
        """get_semantic_matcher returns same instance."""
        reset_semantic_matcher()

        with patch.object(SemanticMatcher, '_init_model'):
            matcher1 = get_semantic_matcher()
            matcher2 = get_semantic_matcher()

            assert matcher1 is matcher2

    def test_reset_semantic_matcher(self):
        """reset_semantic_matcher creates new instance."""
        reset_semantic_matcher()

        with patch.object(SemanticMatcher, '_init_model'):
            matcher1 = get_semantic_matcher()
            reset_semantic_matcher()
            matcher2 = get_semantic_matcher()

            assert matcher1 is not matcher2


class TestSemanticMatcherIntegration:
    """Integration tests for semantic matching."""

    def test_weighted_score_calculation(self):
        """Verify weighted score calculation is correct."""
        reset_semantic_matcher()

        with patch.object(SemanticMatcher, '_init_model'):
            matcher = SemanticMatcher(
                tag_weight=0.4,
                caption_weight=0.3,
                mood_weight=0.15,
                setting_weight=0.15,
            )

            # Mock embeddings to return fixed similarity
            def mock_cosine(a, b):
                if a is None or b is None:
                    return 0.0
                return 0.8  # Fixed similarity

            matcher._cosine_similarity = mock_cosine
            matcher._get_embedding = lambda x: np.array([1.0]) if x else None

            clip = {
                "tags": ["beach"],
                "caption": "Beach scene",
                "mood": "energetic",
                "setting": "beach",
            }

            result = matcher.match_query_to_clip("action beach waves", clip)

            # Expected: 0.4*0.8 + 0.3*0.8 + 0.15*1.0 + 0.15*1.0
            #         = 0.32 + 0.24 + 0.15 + 0.15 = 0.86
            expected = 0.4 * 0.8 + 0.3 * 0.8 + 0.15 * 1.0 + 0.15 * 1.0
            assert result.overall_score == pytest.approx(expected, rel=0.01)
