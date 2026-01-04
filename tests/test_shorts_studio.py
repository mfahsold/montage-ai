import pytest
import json
from unittest.mock import MagicMock, patch
from montage_ai.auto_reframe import AutoReframeEngine, CropWindow
from montage_ai.web_ui.app import app

# --- Tracking Logic Tests ---

class MockDetection:
    def __init__(self, x, y, w, h, score=0.9):
        self.score = [score]
        self.location_data = MagicMock()
        self.location_data.relative_bounding_box.xmin = x
        self.location_data.relative_bounding_box.ymin = y
        self.location_data.relative_bounding_box.width = w
        self.location_data.relative_bounding_box.height = h

class MockResults:
    def __init__(self, detections):
        self.detections = detections

@pytest.fixture
def mock_cv2():
    with patch('montage_ai.auto_reframe.cv2') as mock:
        # Mock VideoCapture
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            3: 1920, # WIDTH
            4: 1080, # HEIGHT
            5: 30,   # FPS
            7: 100   # FRAME_COUNT
        }.get(prop, 0)
        
        # Mock read() to return True, dummy_frame for 3 frames, then False
        cap.read.side_effect = [(True, MagicMock()), (True, MagicMock()), (True, MagicMock()), (False, None)]
        
        mock.VideoCapture.return_value = cap
        mock.CAP_PROP_FRAME_WIDTH = 3
        mock.CAP_PROP_FRAME_HEIGHT = 4
        mock.CAP_PROP_FPS = 5
        mock.CAP_PROP_FRAME_COUNT = 7
        yield mock

@pytest.fixture
def mock_mp():
    with patch('montage_ai.auto_reframe.mp') as mock:
        yield mock

def test_tracking_logic(mock_cv2, mock_mp):
    """
    Test that the tracker sticks to the original subject even if a larger face appears.
    """
    reframer = AutoReframeEngine()
    
    # Mock the face detector
    detector = reframer.face_detector
    
    # Frame 1: Subject A (Center-ish)
    det_a1 = MockDetection(x=0.4, y=0.2, w=0.2, h=0.2)
    
    # Frame 2: Subject A (Moved slightly) AND Subject B (Larger, far right)
    det_a2 = MockDetection(x=0.42, y=0.2, w=0.2, h=0.2)
    det_b2 = MockDetection(x=0.8, y=0.2, w=0.3, h=0.3) # Larger!
    
    # Frame 3: Subject A only
    det_a3 = MockDetection(x=0.44, y=0.2, w=0.2, h=0.2)
    
    # Setup process return values
    detector.process.side_effect = [
        MockResults([det_a1]),          # Frame 1
        MockResults([det_a2, det_b2]),  # Frame 2
        MockResults([det_a3])           # Frame 3
    ]
    
    crops = reframer.analyze("dummy.mp4")
    
    # Check Frame 2 crop center
    # Target width for 9:16 from 1920x1080 is 607 pixels
    # Center of A2 is (0.42 + 0.1) * 1920 = 0.52 * 1920 = 998.4
    # Center of B2 is (0.8 + 0.15) * 1920 = 0.95 * 1920 = 1824
    
    # The crop.x should be centered around A2, not B2.
    # Crop X = Center - TargetWidth/2
    # Expected X approx = 998 - 303 = 695
    
    frame_2_crop = crops[1]
    
    # Allow some smoothing deviation, but it should be much closer to A than B
    assert abs(frame_2_crop.x - 695) < 200, f"Tracker jumped to distractor! X={frame_2_crop.x}"


# --- API Tests ---

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('montage_ai.web_ui.app.AutoReframeEngine')
@patch('montage_ai.web_ui.app.os.path.exists')
def test_analyze_crops_api(mock_exists, mock_reframer_cls, client):
    # Mock file existence
    mock_exists.return_value = True
    
    # Mock AutoReframeEngine instance and analyze method
    mock_reframer = MagicMock()
    mock_reframer_cls.return_value = mock_reframer
    
    # Mock CropWindow objects
    mock_crop = MagicMock()
    mock_crop.time = 0.0
    mock_crop.x = 100
    mock_crop.y = 0
    mock_crop.width = 607
    mock_crop.height = 1080
    mock_crop.score = 0.9
    
    mock_reframer.analyze.return_value = [mock_crop]
    
    # Call API
    response = client.post('/api/analyze-crops', json={'filename': 'test.mp4'})
    
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]['x'] == 100
    assert data[0]['width'] == 607
