from src.montage_ai.config import get_settings
from src.montage_ai.web_ui.app import DEFAULT_OPTIONS
from src.montage_ai.web_ui.job_options import normalize_options


SETTINGS = get_settings()

def test_normalize_options_shorts_mode():
    """Test that shorts_mode is correctly normalized."""
    data = {'shorts_mode': 'true'}
    opts = normalize_options(data, DEFAULT_OPTIONS, SETTINGS)
    assert opts['shorts_mode'] is True
    
    data = {'shorts_mode': False}
    opts = normalize_options(data, DEFAULT_OPTIONS, SETTINGS)
    assert opts['shorts_mode'] is False

def test_normalize_options_export_resolution():
    """Test that export resolution is correctly normalized."""
    data = {'export_width': '640', 'export_height': 360}
    opts = normalize_options(data, DEFAULT_OPTIONS, SETTINGS)
    assert opts['export_width'] == 640
    assert opts['export_height'] == 360

def test_api_create_job_preview_resolution(client, mock_redis_and_rq):
    """Test that preview preset sets 360p resolution."""
    mock_job_store = mock_redis_and_rq['job_store']
    mock_q = mock_redis_and_rq['q']
    
    # Test 1: Normal Preview (Horizontal)
    data = {
        'style': 'dynamic',
        'preset': 'fast',
        'shorts_mode': False
    }
    resp = client.post('/api/jobs', json=data)
    assert resp.status_code == 200
    
    # Verify options passed to job_store
    args, _ = mock_job_store.create_job.call_args
    job_data = args[1]
    options = job_data['options']
    
    assert options['export_width'] == 640
    assert options['export_height'] == 360
    assert options['shorts_mode'] is False

def test_api_create_job_preview_shorts_resolution(client, mock_redis_and_rq):
    """Test that preview preset sets 360p vertical resolution for shorts."""
    mock_job_store = mock_redis_and_rq['job_store']
    mock_q = mock_redis_and_rq['q']
    
    # Test 2: Shorts Preview (Vertical)
    data = {
        'style': 'dynamic',
        'preset': 'fast',
        'shorts_mode': True
    }
    resp = client.post('/api/jobs', json=data)
    assert resp.status_code == 200
    
    # Verify options passed to job_store
    args, _ = mock_job_store.create_job.call_args
    job_data = args[1]
    options = job_data['options']
    
    assert options['export_width'] == 360
    assert options['export_height'] == 640
    assert options['shorts_mode'] is True

def test_api_create_job_preview_shorts_nested(client, mock_redis_and_rq):
    """Test that preview preset sets 360p vertical resolution for shorts (nested options)."""
    mock_job_store = mock_redis_and_rq['job_store']
    mock_q = mock_redis_and_rq['q']
    
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
    
    # Verify options passed to job_store
    args, _ = mock_job_store.create_job.call_args
    job_data = args[1]
    options = job_data['options']
    
    assert options['export_width'] == 360
    assert options['export_height'] == 640
    assert options['shorts_mode'] is True
