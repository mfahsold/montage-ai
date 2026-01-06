import unittest
from unittest.mock import MagicMock
import numpy as np

from montage_ai.core.selection_engine import SelectionEngine
from montage_ai.core.context import MontageContext, MontageMedia, MontageCreative, MontageTimeline

class TestSelectionEngine(unittest.TestCase):
    def setUp(self):
        self.ctx = MagicMock(spec=MontageContext)
        self.ctx.media = MagicMock(spec=MontageMedia)
        self.ctx.creative = MagicMock(spec=MontageCreative)
        self.ctx.timeline = MagicMock(spec=MontageTimeline)
        self.ctx.settings = MagicMock()
        
        # Style params
        self.ctx.creative.style_params = {
            "weights": {"action": 0.5, "face_count": 0.2},
            "scoring_rules": {
                "fresh_clip_bonus": 50,
                "jump_cut_penalty": 50,
            }
        }
        self.ctx.creative.editing_instructions = {}
        self.ctx.creative.semantic_query = None
        self.ctx.creative.broll_plan = None
        
        # Audio
        self.ctx.media.audio_result = MagicMock()
        self.ctx.media.audio_result.sections = []
        self.ctx.media.audio_result.tempo = 120
        
        # Scenes
        # Note: In real code, available_footage wraps these or refers to them. 
        # For mock, we'll setup available_footage appropriately.
        self.scene1 = {'path': '/path/scene1.mp4', 'start': 0, 'duration': 5.0, 'meta': {'action': 'high', 'shot': 'close_up'}}
        self.scene2 = {'path': '/path/scene2.mp4', 'start': 0, 'duration': 5.0, 'meta': {'action': 'low', 'shot': 'wide'}}
        self.ctx.media.all_scenes_dicts = [self.scene1, self.scene2]
        
        # Timeline state
        self.ctx.timeline.last_used_path = None
        self.ctx.timeline.last_shot_type = None
        self.ctx.get_story_phase.return_value = "build"

        self.engine = SelectionEngine(self.ctx)

    def test_score_style_preferences(self):
        # scene1: action high. Weight 0.5 -> score += 15 * 0.5 = 7.5
        # Also face_count not present so 0.
        
        score = self.engine._score_style_preferences(self.scene1['meta'])
        self.assertEqual(score, 7.5) # 15.0 * 0.5

    def test_score_usage(self):
        # Mock footage clip object
        mock_clip = MagicMock()
        mock_clip.usage_count = 0
        
        score = self.engine._score_usage_and_story_phase(mock_clip, 0.5, 50)
        # Usage 0 -> +50
        # Phase build, energy 0.5 -> +15
        # Total = 65
        self.assertEqual(score, 65)

    def test_select_clip_probabilistic(self):
        # Mock available footage
        # We need mock objects that have a clip_id corresponding to id(scene)
        
        # Since id() is non-deterministic in tests if we create dicts inside test method compared to setUp?
        # No, id() works on object identity.
        
        class MockClip:
            def __init__(self, scene_dict):
                self.clip_id = id(scene_dict)
                self.usage_count = 0
                self.scene = scene_dict
                
        c1 = MockClip(self.scene1)
        c2 = MockClip(self.scene2)
        
        available = [c1, c2]
        
        # Run select
        selected, score = self.engine.select_clip(available, current_energy=0.8, unique_videos=2)
        
        self.assertIsNotNone(selected)
        self.assertIn(selected, [self.scene1, self.scene2])
        # Score should be non-zero (even with random noise) unless penalties are huge
        # Scene 1 (High action) should score well.
        
    def test_get_candidate_scenes(self):
        class MockClip:
            def __init__(self, scene_dict):
                self.clip_id = id(scene_dict)
                
        c1 = MockClip(self.scene1)
        candidates = self.engine._get_candidate_scenes([c1])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0], self.scene1)

if __name__ == '__main__':
    unittest.main()
