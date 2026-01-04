"""
Tests for Web UI Options Normalization and Queue Logic
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from src.montage_ai.web_ui.app import normalize_options, app

# =============================================================================
# normalize_options Tests
# =============================================================================

def test_normalize_options_defaults():
    """Test that defaults are applied correctly when input is empty."""
    data = {}
    opts = normalize_options(data)
    
    assert opts['target_duration'] == 0
    assert opts['music_start'] == 0
    assert opts['music_end'] is None
    assert opts['prompt'] == ''
    # Check boolean defaults (should match DEFAULT_OPTIONS in app.py)
    # Assuming defaults are mostly False except maybe enhance/llm_clip_selection depending on env
    # We check that they return booleans at least
    assert isinstance(opts['stabilize'], bool)
    assert isinstance(opts['upscale'], bool)

def test_normalize_options_flat_structure():
    """Test parsing flat dictionary structure."""
    data = {
        'target_duration': '60',
        'music_start': '10',
        'stabilize': 'true',
        'prompt': 'test prompt'
    }
    opts = normalize_options(data)
    
    assert opts['target_duration'] == 60.0
    assert opts['music_start'] == 10.0
    assert opts['stabilize'] is True
    assert opts['prompt'] == 'test prompt'

def test_normalize_options_nested_structure():
    """Test parsing nested 'options' dictionary."""
    data = {
        'options': {
            'target_duration': 30,
            'upscale': True
        }
    }
    opts = normalize_options(data)
    
    assert opts['target_duration'] == 30.0
    assert opts['upscale'] is True

def test_normalize_options_music_end_derivation():
    """Test that music_end is derived from target_duration if missing."""
    data = {
        'target_duration': 45,
        'music_start': 5
    }
    opts = normalize_options(data)
    
    # Should be start + duration = 5 + 45 = 50
    assert opts['music_end'] == 50.0

def test_normalize_options_music_end_explicit():
    """Test that explicit music_end overrides derivation."""
    data = {
        'target_duration': 45,
        'music_start': 5,
        'music_end': 100
    }
    opts = normalize_options(data)
    
    assert opts['music_end'] == 100.0

def test_normalize_options_clamping():
    """Test that values are clamped to sensible ranges."""
    data = {
        'target_duration': -10,  # Should be 0
        'music_start': -5,       # Should be 0
        'music_end': 2           # Should be > start (if start is 0, end > 0)
    }
    opts = normalize_options(data)
    
    assert opts['target_duration'] == 0
    assert opts['music_start'] == 0
    # music_end logic: max(music_start + 1, music_end) -> max(1, 2) = 2
    assert opts['music_end'] == 2.0

    # Test music_end clamping
    data2 = {
        'music_start': 10,
        'music_end': 5  # Should be clamped to start + 1 = 11
    }
    opts2 = normalize_options(data2)
    assert opts2['music_end'] == 11.0

# =============================================================================
# Queue Robustness Tests
# =============================================================================

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('src.montage_ai.web_ui.app.q')
@patch('src.montage_ai.web_ui.app.job_store')
def test_job_submission_queues_job(mock_job_store, mock_q, client):
    """Test that submitting a job adds it to the store and the queue."""
    # Setup
    mock_job_store.create_job.return_value = None
    mock_q.enqueue.return_value = None
    
    # Action
    res = client.post('/api/jobs', json={'style': 'test', 'options': {}})
    
    # Assert
    assert res.status_code == 200
    assert mock_job_store.create_job.called
    assert mock_q.enqueue.called
    
    # Verify arguments
    args, _ = mock_job_store.create_job.call_args
    job_id = args[0]
    job_data = args[1]
    assert job_data['status'] == 'queued'
    assert job_data['style'] == 'test'

@patch('src.montage_ai.web_ui.app.q')
@patch('src.montage_ai.web_ui.app.job_store')
def test_multiple_job_submission(mock_job_store, mock_q, client):
    """Test that multiple jobs can be submitted without blocking."""
    # Setup
    mock_job_store.create_job.return_value = None
    mock_q.enqueue.return_value = None
    
    # Action
    res1 = client.post('/api/jobs', json={'style': 'job1'})
    res2 = client.post('/api/jobs', json={'style': 'job2'})
    
    # Assert
    assert res1.status_code == 200
    assert res2.status_code == 200
    assert mock_job_store.create_job.call_count == 2
    assert mock_q.enqueue.call_count == 2
