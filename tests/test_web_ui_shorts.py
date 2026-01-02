import pytest
from unittest.mock import patch, MagicMock
from src.montage_ai.web_ui.app import normalize_options, app, jobs

def test_normalize_options_shorts_mode():
    """Test that shorts_mode is correctly normalized."""
    data = {'shorts_mode': 'true'}
    opts = normalize_options(data)
    assert opts['shorts_mode'] is True
    
    data = {'shorts_mode': False}
    opts = normalize_options(data)
    assert opts['shorts_mode'] is False

def test_normalize_options_export_resolution():
    """Test that export resolution is correctly normalized."""
    data = {'export_width': '640', 'export_height': 360}
    opts = normalize_options(data)
    assert opts['export_width'] == 640
    assert opts['export_height'] == 360

@patch('src.montage_ai.web_ui.app.run_montage')
def test_api_create_job_preview_resolution(mock_run_montage):
    """Test that preview preset sets 360p resolution."""
    client = app.test_client()
    
    # Test 1: Normal Preview (Horizontal)
    data = {
        'style': 'dynamic',
        'preset': 'fast',
        'shorts_mode': False
    }
    resp = client.post('/api/jobs', json=data)
    assert resp.status_code == 200
    job_id = resp.json['id']
    
    # Check options in the job store
    options = jobs[job_id]['options']
    
    assert options['export_width'] == 640
    assert options['export_height'] == 360
    assert options['shorts_mode'] is False

@patch('src.montage_ai.web_ui.app.run_montage')
def test_api_create_job_preview_shorts_resolution(mock_run_montage):
    """Test that preview preset sets 360p vertical resolution for shorts."""
    client = app.test_client()
    
    # Test 2: Shorts Preview (Vertical)
    data = {
        'style': 'dynamic',
        'preset': 'fast',
        'shorts_mode': True
    }
    resp = client.post('/api/jobs', json=data)
    assert resp.status_code == 200
    job_id = resp.json['id']
    
    options = jobs[job_id]['options']
    
    assert options['export_width'] == 360
    assert options['export_height'] == 640
    assert options['shorts_mode'] is True

@patch('src.montage_ai.web_ui.app.run_montage')
def test_api_create_job_preview_shorts_nested(mock_run_montage):
    """Test that preview preset sets 360p vertical resolution for shorts (nested options)."""
    client = app.test_client()
    
    # Test 3: Shorts Preview with nested options
    data = {
        'style': 'dynamic',
        'preset': 'fast',
        'options': {
            'shorts_mode': True
        }
    }
    resp = client.post('/api/jobs', json=data)
    assert resp.status_code == 200
    job_id = resp.json['id']
    
    options = jobs[job_id]['options']
    
    assert options['export_width'] == 360
    assert options['export_height'] == 640
    assert options['shorts_mode'] is True
