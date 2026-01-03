import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock cv2 globally
mock_cv2 = MagicMock()
sys.modules['cv2'] = mock_cv2

from montage_ai.web_ui.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('montage_ai.smart_reframing.SmartReframer')
def test_shorts_visualize(mock_reframer, client, tmp_path):
    # Setup mock cv2
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [1920, 1080]
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

    # Mock reframer
    mock_instance = mock_reframer.return_value
    from dataclasses import dataclass
    @dataclass
    class MockCrop:
        time: float
        x: int
        y: int
        width: int
        height: int
        score: float
    
    mock_instance.analyze.return_value = [
        MockCrop(0.0, 100, 0, 608, 1080, 0.9),
        MockCrop(1.0, 110, 0, 608, 1080, 0.95)
    ]

    # Create dummy video file
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    response = client.post('/api/shorts/visualize', json={
        'video_path': str(video_path)
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert len(data['crops']) == 2
    assert data['original_width'] == 1920
    assert data['original_height'] == 1080

@patch('montage_ai.cgpu_jobs.VoiceIsolationJob')
@patch('montage_ai.web_ui.app.is_cgpu_available')
@patch('subprocess.run')
@patch('montage_ai.smart_reframing.SmartReframer')
def test_shorts_render_audio_polish(mock_reframer, mock_run, mock_is_cgpu, mock_job, client, tmp_path):
    # Mock cgpu availability
    mock_is_cgpu.return_value = True

    # Mock VoiceIsolationJob
    mock_job_instance = mock_job.return_value
    mock_job_instance.execute.return_value.success = True
    mock_job_instance.execute.return_value.metadata = {
        "stems": {"vocals": "/tmp/vocals.wav"}
    }

    # Create dummy video file
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    response = client.post('/api/shorts/render', json={
        'video_path': str(video_path),
        'audioPolish': True,
        'reframeMode': 'center'
    })

    # Check if VoiceIsolationJob was called
    mock_job.assert_called_once()
    
    # Check if ffmpeg was called
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "-i" in args
    assert "/tmp/vocals.wav" in args
