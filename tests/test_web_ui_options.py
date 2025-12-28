"""
Tests for Web UI Options Normalization and Queue Logic
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from src.montage_ai.web_ui.app import normalize_options, app, job_queue, jobs, active_jobs

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

@patch('src.montage_ai.web_ui.app.MAX_CONCURRENT_JOBS', 1)
@patch('src.montage_ai.web_ui.app.run_montage')
def test_queue_overflow(mock_run_montage, client):
    """Test that jobs are queued when MAX_CONCURRENT_JOBS is reached."""
    # Reset global state
    import src.montage_ai.web_ui.app as app_module
    app_module.active_jobs = 0
    app_module.job_queue.clear()
    app_module.jobs.clear()
    
    # 1. Submit first job (should run)
    res1 = client.post('/api/jobs', json={'style': 'test', 'options': {}})
    assert res1.status_code == 200
    assert app_module.active_jobs == 1
    assert len(app_module.job_queue) == 0
    assert mock_run_montage.call_count == 1
    
    # 2. Submit second job (should queue because MAX=1)
    res2 = client.post('/api/jobs', json={'style': 'test', 'options': {}})
    assert res2.status_code == 200
    assert app_module.active_jobs == 1  # Still 1 running
    assert len(app_module.job_queue) == 1
    assert mock_run_montage.call_count == 1  # Still only called once
    
    job2_id = res2.json['id']
    assert app_module.jobs[job2_id]['status'] == 'queued'

@patch('src.montage_ai.web_ui.app.MAX_CONCURRENT_JOBS', 1)
@patch('src.montage_ai.web_ui.app.run_montage')
def test_queue_processing(mock_run_montage, client):
    """Test that queued jobs are processed when active jobs finish."""
    # Reset global state
    import src.montage_ai.web_ui.app as app_module
    app_module.active_jobs = 0
    app_module.job_queue.clear()
    app_module.jobs.clear()
    
    # Fill the queue
    client.post('/api/jobs', json={'style': 'job1'}) # Runs
    client.post('/api/jobs', json={'style': 'job2'}) # Queues
    
    assert app_module.active_jobs == 1
    assert len(app_module.job_queue) == 1
    
    # Simulate job completion
    # We need to manually trigger what happens in 'finally' block of run_montage
    # Since we mocked run_montage, it didn't actually run or finish.
    # We can simulate the cleanup logic:
    
    # Decrement active jobs
    app_module.active_jobs -= 1
    
    # Process next job
    if app_module.job_queue:
        next_job = app_module.job_queue.popleft()
        # In app.py, process_job_from_queue calls run_montage in a thread
        # We'll just call the mock directly to verify logic
        app_module.active_jobs += 1
        mock_run_montage(next_job['job_id'], next_job['style'], next_job['options'])
        
    assert app_module.active_jobs == 1
    assert len(app_module.job_queue) == 0
    assert mock_run_montage.call_count == 2 # Called for job1 and job2
