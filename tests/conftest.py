import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.montage_ai.web_ui.app import app


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def mock_redis_and_rq():
    """Mock Redis and RQ for all tests."""
    with patch('src.montage_ai.web_ui.app.redis_conn') as mock_redis, \
         patch('src.montage_ai.web_ui.app.q') as mock_q, \
         patch('src.montage_ai.web_ui.app.job_store') as mock_job_store:
        
        # Setup mock returns
        mock_redis.ping.return_value = True
        mock_q.enqueue = MagicMock(return_value=None)
        mock_q.started_job_registry = []
        mock_q.__len__ = MagicMock(return_value=0)
        
        # Mock job_store
        mock_job_store.get_job = MagicMock(return_value=None)
        mock_job_store.create_job = MagicMock(return_value=None)
        mock_job_store.update_job = MagicMock(return_value=None)
        mock_job_store.list_jobs = MagicMock(return_value={})
        
        yield {
            'redis': mock_redis,
            'q': mock_q,
            'job_store': mock_job_store
        }


@pytest.fixture
def client():
    """Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory (generated via `make test-fixtures`)."""
    return PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture
def test_video(fixtures_dir):
    """Path to synthetic test video (1080p, 5s).

    Generate with: make test-fixtures
    Skip test if not available.
    """
    video_path = fixtures_dir / "video" / "test_1080p_5s.mp4"
    if not video_path.exists():
        pytest.skip("Test fixtures not generated. Run: make test-fixtures")
    return video_path


@pytest.fixture
def test_audio(fixtures_dir):
    """Path to synthetic test audio (120 BPM, 10s).

    Generate with: make test-fixtures
    Skip test if not available.
    """
    audio_path = fixtures_dir / "audio" / "test_120bpm_10s.wav"
    if not audio_path.exists():
        pytest.skip("Test fixtures not generated. Run: make test-fixtures")
    return audio_path


@pytest.fixture
def bunny_trailer():
    """Path to minimal Big Buck Bunny trailer (143KB, included in repo)."""
    trailer_path = PROJECT_ROOT / "data" / "input_test" / "bunny_trailer.mp4"
    if not trailer_path.exists():
        pytest.skip("bunny_trailer.mp4 not found in data/input_test/")
    return trailer_path


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (require real media files)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
