import pytest
from unittest.mock import MagicMock, patch
from montage_ai.core.montage_builder import MontageBuilder
from montage_ai.core.selection_engine import SelectionEngine

@patch("montage_ai.core.montage_builder.get_settings")
def test_apply_style_scoring(mock_get_settings):
    """Test the abstracted scoring logic."""
    # Use real MontageSettingsSpec instead of MagicMock
    from src.montage_ai.config import MontageSettingsSpec
    mock_settings = MontageSettingsSpec.create_default()
    mock_get_settings.return_value = mock_settings
    
    builder = MontageBuilder()
    engine = builder._selection_engine
    
    # Test Case 1: Vlog Style (Face Count + Close Shot)
    vlog_params = {
        "weights": {"face_count": 2.0},
        "preferred_shots": ["close"]
    }
    
    meta_face = {"face_count": 1, "shot": "close"}
    score = engine._apply_style_scoring(meta_face, vlog_params)
    
    # Expected: 
    # Face: 1 * 10.0 * 2.0 = 20.0
    # Shot: 20.0
    # Total: 40.0
    assert score == 40.0
    
    # Test Case 2: Sport Style (Action + Wide Shot)
    sport_params = {
        "weights": {"action": 2.5},
        "preferred_shots": ["wide"]
    }
    
    meta_action = {"action": "high", "shot": "wide"}
    score = engine._apply_style_scoring(meta_action, sport_params)
    
    # Expected:
    # Action High: 15.0 * 2.5 = 37.5
    # Shot: 20.0
    # Total: 57.5
    assert score == 57.5

    # Test Case 3: Generic Numeric Weight
    generic_params = {
        "weights": {"energy": 1.5}
    }
    meta_energy = {"energy": 0.8}
    score = builder._selection_engine._apply_style_scoring(meta_energy, generic_params)
    
    # Expected:
    # Energy: 0.8 * 10.0 * 1.5 = 12.0
    assert score == 12.0
