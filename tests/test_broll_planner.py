"""Tests for B-Roll Planner module."""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from montage_ai.broll_planner import split_script, format_plan, BRollSuggestion


class TestSplitScript(unittest.TestCase):
    """Test script splitting functionality."""

    def test_split_simple_sentences(self):
        """Split script on sentence boundaries."""
        script = "The athlete trains hard. Victory celebration follows."
        segments = split_script(script)
        self.assertEqual(len(segments), 2)
        self.assertIn("athlete trains hard", segments[0]["text"])

    def test_split_with_exclamation(self):
        """Handle exclamation marks as sentence boundaries."""
        script = "Action scene! Drama unfolds! Suspense builds!"
        segments = split_script(script)
        self.assertEqual(len(segments), 3)

    def test_split_with_question(self):
        """Handle question marks as sentence boundaries."""
        script = "What happens next? Nobody knows the answer."
        segments = split_script(script)
        self.assertEqual(len(segments), 2)

    def test_filter_short_segments(self):
        """Filter out segments shorter than 4 characters."""
        script = "Go. Run fast through the forest."
        segments = split_script(script)
        # "Go" is too short, should be filtered
        self.assertEqual(len(segments), 1)

    def test_empty_input(self):
        """Handle empty input gracefully."""
        segments = split_script("")
        self.assertEqual(segments, [])


class TestFormatPlan(unittest.TestCase):
    """Test plan formatting for terminal output."""

    def test_format_with_suggestions(self):
        """Format plan with clip suggestions."""
        results = [
            {
                "segment": "Athlete running",
                "suggestions": [
                    {"clip": "/data/run.mp4", "start": 0.0, "end": 5.0, "score": 0.85, "caption": "jogging"}
                ]
            }
        ]
        output = format_plan(results)
        self.assertIn("B-ROLL PLAN", output)
        self.assertIn("Athlete running", output)
        self.assertIn("run.mp4", output)

    def test_format_no_matches(self):
        """Format plan when no clips match."""
        results = [{"segment": "Alien invasion", "suggestions": []}]
        output = format_plan(results)
        self.assertIn("no matches", output)


class TestBRollSuggestion(unittest.TestCase):
    """Test BRollSuggestion dataclass."""

    def test_to_dict(self):
        """Convert suggestion to dictionary."""
        suggestion = BRollSuggestion(
            segment_text="test",
            keywords=["test"],
            clip_path="/data/clip.mp4",
            start_time=0.0,
            end_time=5.0,
            score=0.8
        )
        d = suggestion.to_dict()
        self.assertEqual(d["segment"], "test")
        self.assertEqual(d["keywords"], ["test"])
        self.assertEqual(d["clip"], "/data/clip.mp4")
        self.assertEqual(d["score"], 0.8)


class TestPlanBroll(unittest.TestCase):
    """Test full B-roll planning flow."""

    @patch('montage_ai.broll_planner.VIDEO_AGENT_AVAILABLE', False)
    def test_plan_without_video_agent(self):
        """Return error when video_agent unavailable."""
        # Re-import to pick up the patched value
        from montage_ai.broll_planner import plan_broll
        results = plan_broll("Test script")
        self.assertTrue(any("error" in r for r in results))


if __name__ == '__main__':
    unittest.main()
