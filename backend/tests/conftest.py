import io
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
from PIL import Image

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.feedback_service import FeedbackService
import backend.app.api.v1.endpoints.feedback as feedback_endpoint


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Returns valid encoded JPEG bytes representing a 48x48 synthetic face/image."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Draw simple facial feature shapes
    cv2.circle(img, (35, 40), 10, (255, 255, 255), -1)
    cv2.circle(img, (65, 40), 10, (255, 255, 255), -1)
    cv2.rectangle(img, (30, 70), (70, 80), (255, 255, 255), -1)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


@pytest.fixture
def sample_crop_bytes():
    """Returns valid 48x48 grayscale-compatible JPEG bytes."""
    img = np.full((48, 48, 3), 200, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


@pytest.fixture
def isolated_feedback_service(tmp_path):
    """Creates an isolated FeedbackService in a temporary directory for testing."""
    fb_service = FeedbackService(feedback_dir=tmp_path)
    old_service = feedback_endpoint.feedback_service
    feedback_endpoint.feedback_service = fb_service
    yield fb_service
    feedback_endpoint.feedback_service = old_service
