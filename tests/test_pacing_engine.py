import unittest
from unittest.mock import MagicMock
import numpy as np

from montage_ai.core.pacing_engine import PacingEngine
from montage_ai.core.context import MontageContext, MontageMedia, MontageCreative, MontageTimeline

class TestPacingEngine(unittest.TestCase):
    def setUp(self):
        # Mock Context
        self.ctx = MagicMock(spec=MontageContext)
        self.ctx.media = MagicMock(spec=MontageMedia)
        self.ctx.creative = MagicMock(spec=MontageCreative)
        self.ctx.timeline = MagicMock(spec=MontageTimeline)
        self.ctx.settings = MagicMock()
        
        # Timeline state
        self.ctx.timeline.current_time = 10.0
        self.ctx.timeline.beat_idx = 4.0
        self.ctx.timeline.target_duration = 60.0
        self.ctx.timeline.current_pattern = None
        self.ctx.timeline.pattern_idx = 0
        
        # Audio Result
        self.ctx.media.audio_result = MagicMock()
        self.ctx.media.audio_result.tempo = 120.0
        # Beat times every 0.5s (120 BPM)
        self.ctx.media.audio_result.beat_times = np.arange(0, 60, 0.5)
        self.ctx.media.audio_result.energy_times = np.arange(0, 60, 1.0)
        self.ctx.media.audio_result.energy_values = np.full(60, 0.5)
        self.ctx.media.audio_result.sections = []

        # Instructions
        self.ctx.creative.editing_instructions = {}
        self.ctx.creative.style_params = {}

        self.engine = PacingEngine(self.ctx)

    def test_get_energy_basic(self):
        val = self.engine.get_energy_at_time(10.5)
        self.assertEqual(val, 0.5)

    def test_cut_duration_calculation(self):
        # 4 beats @ 120BPM = 2.0 seconds
        # Engine defaults to standard pattern [4, 4, 4, 4] for dynamic if no history
        
        # We need to mock _calculate_beats_per_cut indirectly or test public API
        # Let's test get_next_cut_duration
        
        # Mock pattern matching to return specific beats
        # But _calculate_beats_per_cut is internal.
        # Let's verify standard behavior.
        
        duration = self.engine.get_next_cut_duration(0.5)
        
        # Expectation: 
        # Beat count: Likely 4 beats (from default pattern first element)
        # Duration: ~2.0s
        # Jitter: +/- 0.05
        
        # The engine logic: 
        # 1. _calculate_beats_per_cut -> checks constraints -> dynamic -> calculate_dynamic_cut_length
        # 2. calculate_dynamic_cut_length (from audio_analysis) is imported.
        #    If we don't mock it, it runs real logic.
        
        # It should be roughly 2.0s
        self.assertTrue(1.9 < duration < 2.1)
        
        # Check timeline update
        # beat_idx was 4.0 (time 2.0s). 4 beats added -> 8.0.
        # verify self.ctx.timeline.beat_idx was updated
        self.assertEqual(self.ctx.timeline.beat_idx, 8.0)

    def test_fast_pacing_instruction(self):
        self.ctx.creative.editing_instructions = {"pacing": {"speed": "fast"}}
        
        # Fast: 2 beats if tempo < 130, else 4. Here 120 -> 2 beats.
        # Duration: 1.0s
        duration = self.engine.get_next_cut_duration(0.5)
        
        self.assertTrue(0.9 < duration < 1.1)
        self.assertEqual(self.ctx.timeline.beat_idx, 6.0)

if __name__ == '__main__':
    unittest.main()
