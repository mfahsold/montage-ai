import pytest
from src.montage_ai.web_ui.app import app

@pytest.fixture
def client():
    """Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
