import random
import pytest
from unittest.mock import MagicMock, patch
from montage_ai.core.montage_builder import MontageBuilder
from montage_ai.core.context import MontageContext
from montage_ai.config import Settings

@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.paths.input_dir = "/tmp/input"
    settings.paths.music_dir = "/tmp/music"
    settings.paths.output_dir = "/tmp/output"
    settings.paths.temp_dir = "/tmp/temp"
    settings.features.shorts_mode = False
    return settings

@pytest.fixture
def mock_broll_planner():
    with patch('montage_ai.broll_planner.plan_broll') as mock:
        yield mock

def test_broll_planning_integration(mock_settings, mock_broll_planner):
    """Test that MontageBuilder calls BrollPlanner when script is provided."""
    
    # Setup instructions with script
    instructions = {
        "script": "The athlete runs. The crowd cheers."
    }
    
    # Mock B-Roll plan result
    mock_plan = [
        {
            "segment": "The athlete runs",
            "start_time": 0.0,
            "end_time": 5.0,
            "suggestions": [
                {"clip": "/tmp/input/run.mp4", "score": 0.9}
            ]
        },
        {
            "segment": "The crowd cheers",
            "start_time": 5.0,
            "end_time": 10.0,
            "suggestions": [
                {"clip": "/tmp/input/cheer.mp4", "score": 0.9}
            ]
        }
    ]
    mock_broll_planner.return_value = mock_plan
    
    # Initialize builder
    # We patch initialization phases to prevent full setup
    with patch.object(MontageBuilder, 'setup_workspace'), \
         patch.object(MontageBuilder, 'analyze_assets'), \
         patch.object(MontageBuilder, '_run_assembly_loop'), \
         patch.object(MontageBuilder, 'render_output'), \
         patch('montage_ai.core.montage_builder.get_resource_manager') :
         
        builder = MontageBuilder(settings=mock_settings, editing_instructions=instructions)
        
        # Mock audio result to prevent errors if invoked
        mock_audio = MagicMock()
        mock_audio.duration = 60.0
        mock_audio.tempo = 120.0
        mock_audio.beat_times = [0.5, 1.0, 1.5]
        builder.ctx.media.audio_result = mock_audio
        builder.ctx.creative.broll_plan = None # Ensure it starts empty
        
        # Run build (which calls _plan_broll_sequence)
        # We assume build calls _plan_broll_sequence. 
        # Since analyze_assets is mocked (where it might be called), we simulate the call manually
        # mirroring how build() or analyze_assets() would do it.
        # However, checking MontageBuilder definition:
        # build() calls analyze_assets(), then _plan_broll_sequence() is inside build() or analyze_assets()?
        # We saw line 879: self._plan_broll_sequence() is in build().
        # So mocking analyze_assets is fine.
        
        builder.build()
        
        # Verify BrollPlanner was called
        mock_broll_planner.assert_called_once()
        args, _ = mock_broll_planner.call_args
        assert args[0] == "The athlete runs. The crowd cheers."
        
        # Verify plan is stored in context
        assert builder.ctx.creative.broll_plan == mock_plan

def test_score_broll_match_unit(mock_settings):
    """Test _score_broll_match logic in isolation."""
    builder = MontageBuilder(settings=mock_settings)
    
    # Setup Context with a plan
    builder.ctx.creative.broll_plan = [
        {
            "segment": "Target Segment",
            "start_time": 0.0,
            "end_time": 10.0,
            "suggestions": [
                {"clip": "/path/to/target.mp4", "score": 0.9}
            ]
        }
    ]
    
    # Case 1: Matching Time & Clip
    builder.ctx.timeline.current_time = 5.0
    scene = {"path": "/path/to/target.mp4"}
    meta = {}
    
    score = builder._selection_engine._score_broll_match(scene, meta)
    assert score == 100.0, "Should get max score for direct suggestion match"
    
    # Case 2: Matching Time, Non-Matching Clip
    scene_wrong = {"path": "/path/to/random.mp4"}
    score = builder._selection_engine._score_broll_match(scene_wrong, meta)
    assert score == 0.0, "Should get 0 for non-suggested clip"
    
    # Case 3: Wrong Time
    builder.ctx.timeline.current_time = 15.0
    score = builder._selection_engine._score_broll_match(scene, meta)
    assert score == 0.0, "Should get 0 if current time is outside segment"

def test_broll_selection_integration(mock_settings):
    """Test that select_clip prioritizes B-Roll plan suggestions."""
    # Seed RNG
    random.seed(42)
    
    builder = MontageBuilder(settings=mock_settings)
    
    # Setup candidate scenes
    scene_match = {
        "path": "/path/to/run.mp4",
        "start": 0.0,
        "end": 10.0,
        "duration": 10.0,
        "meta": {"tags": ["running"]}
    }
    scene_other = {
        "path": "/path/to/other.mp4",
        "start": 0.0,
        "end": 10.0,
        "duration": 10.0,
        "meta": {"tags": ["walking"]}
    }
    
    # Setup Context
    builder.ctx.creative.broll_plan = [
        {
            "segment": "The athlete runs",
            "start_time": 0.0,
            "end_time": 5.0,
            "suggestions": [
                {"clip": "/path/to/run.mp4", "score": 0.9}
            ]
        }
    ]
    builder.ctx.timeline.current_time = 2.0
    
    # This mock is needed for _get_candidate_scenes
    builder.ctx.media.all_scenes_dicts = [scene_match, scene_other]
    
    class MockClipUsage:
        def __init__(self, scene):
            self.clip_id = id(scene)
            self.usage_count = 0
            self.last_used_time = -1
    
    available_footage = [MockClipUsage(scene_match), MockClipUsage(scene_other)]
    
    # Use the SelectionEngine instance
    engine = builder._selection_engine
    
    # Mock internal scoring methods to return 0 except broll
    # In SelectionEngine, these are instance methods
    with patch.object(engine, '_score_usage_and_story_phase', return_value=0), \
         patch.object(engine, '_score_jump_cut', return_value=0), \
         patch.object(engine, '_score_action_energy', return_value=0), \
         patch.object(engine, '_score_style_preferences', return_value=0), \
         patch.object(engine, '_score_shot_variation', return_value=0), \
         patch.object(engine, '_score_match_cut', return_value=0), \
         patch.object(engine, '_score_semantic_match', return_value=0), \
         patch.object(engine, '_resolve_scoring_rules', return_value={
             "fresh_clip_bonus": 10,
             "jump_cut_penalty": -10,
             "shot_variation_bonus": 5,
             "shot_repetition_penalty": -5
         }):
        
        # Ensure probabilistic fallback
        builder._intelligent_selector = None 
        engine._intelligent_selector = None
        
        # Test selection
        selected, score = engine.select_clip(
            available_footage, 
            current_energy=0.5, 
            unique_videos=2
        )
        
        assert selected is not None
        assert selected["path"] == "/path/to/run.mp4"
        assert selected["_heuristic_score"] >= 85 # 100 + random(-15, 15)
