"""
Tests for Web UI

Simple tests following KISS principle.
"""

import pytest
import json
from pathlib import Path
from src.montage_ai.web_ui.app import app





def test_index_page(client):
    """Test main page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Montage AI' in response.data


def test_api_status(client):
    """Test API health check."""
    response = client.get('/api/status')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'ok'
    assert 'version' in data


def test_api_list_files(client):
    """Test file listing."""
    response = client.get('/api/files')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'videos' in data
    assert 'music' in data
    assert isinstance(data['videos'], list)
    assert isinstance(data['music'], list)


def test_api_list_styles(client):
    """Test styles listing."""
    response = client.get('/api/styles')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'styles' in data
    assert len(data['styles']) > 0

    # Check first style has required fields
    style = data['styles'][0]
    assert 'id' in style
    assert 'name' in style
    assert 'description' in style


def test_api_create_job(client):
    """Test job creation."""
    job_data = {
        'style': 'dynamic',
        'prompt': 'test prompt',
        'stabilize': False,
        'upscale': False
    }

    response = client.post(
        '/api/jobs',
        data=json.dumps(job_data),
        content_type='application/json'
    )

    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'id' in data
    assert data['style'] == 'dynamic'
    assert data['status'] in ['queued', 'running']


def test_api_create_job_missing_style(client):
    """Test job creation fails without style."""
    response = client.post(
        '/api/jobs',
        data=json.dumps({}),
        content_type='application/json'
    )

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_api_list_jobs(client):
    """Test jobs listing."""
    response = client.get('/api/jobs')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'jobs' in data
    assert isinstance(data['jobs'], list)


def test_api_get_job_not_found(client):
    """Test getting non-existent job."""
    response = client.get('/api/jobs/nonexistent123')
    assert response.status_code == 404


def test_file_upload_validation(client):
    """Test file upload validation."""
    # Test without file
    response = client.post('/api/upload')
    assert response.status_code == 400

    # Test with invalid file type
    data = {
        'file': (Path(__file__).open('rb'), 'test.txt'),
        'type': 'video'
    }
    response = client.post('/api/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
