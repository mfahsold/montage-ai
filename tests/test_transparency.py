
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.montage_ai.web_ui.app import app, OUTPUT_DIR

def test_api_get_job_decisions_success(tmp_path):
    """Test fetching job decisions when file exists."""
    job_id = "test_job_123"
    decisions_data = {
        "director_commentary": "Test commentary",
        "style": {"name": "test_style"}
    }
    
    # Mock OUTPUT_DIR to point to tmp_path
    with patch('src.montage_ai.web_ui.app.OUTPUT_DIR', tmp_path):
        # Create the decisions file
        decisions_file = tmp_path / f"decisions_{job_id}.json"
        with open(decisions_file, 'w') as f:
            json.dump(decisions_data, f)
            
        client = app.test_client()
        resp = client.get(f'/api/jobs/{job_id}/decisions')
        
        assert resp.status_code == 200
        assert resp.json['director_commentary'] == "Test commentary"

def test_api_get_job_decisions_not_found(tmp_path):
    """Test fetching job decisions when file is missing."""
    job_id = "missing_job"
    
    with patch('src.montage_ai.web_ui.app.OUTPUT_DIR', tmp_path):
        client = app.test_client()
        resp = client.get(f'/api/jobs/{job_id}/decisions')
        
        assert resp.status_code == 200
        assert resp.json['available'] is False
        assert "message" in resp.json
