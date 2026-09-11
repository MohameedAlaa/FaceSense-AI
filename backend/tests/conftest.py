import io
import tempfile
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.db.session import get_db
from backend.app.models.feedback import FeedbackRecord
from backend.app.services.feedback_service import FeedbackService
import backend.app.api.v1.endpoints.feedback as feedback_endpoint


@pytest.fixture
def client():
    """Default client with no DB dependency overrides."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Returns valid encoded JPEG bytes representing a synthetic face/image."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
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


@pytest.fixture
def test_db_engine():
    """Creates an in-memory SQLite database engine with all tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def test_db_session(test_db_engine) -> Generator[Session, None, None]:
    """Yields a database session bound to the in-memory test database."""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_db_engine,
        expire_on_commit=False,
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client_with_db(test_db_session):
    """Test client with get_db dependency overridden to in-memory SQLite session."""
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)
