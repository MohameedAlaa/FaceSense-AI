"""
Regression and Integration Tests for Face Crop Feedback Storage
Verifies:
1. Real face crop base64 is decoded, saved to disk, and recorded in feedback store.
2. When image_base64 is omitted, image_path is None and NO fake image is written to disk.
3. Bounding box coordinates [x, y, w, h] and face_index are persisted accurately.
4. Multi-face independent crops map accurately to each distinct face.
5. Invalid image payloads are safely rejected.
"""

import base64
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings


@pytest.fixture
def test_client():
    return TestClient(app)


def _create_synthetic_face_jpeg_base64(width=100, height=100, color=(120, 140, 180)):
    """Generates a valid test face crop image encoded as Base64 JPEG."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Add simple facial landmark features so it's not a solid flat color
    cv2.circle(img, (width // 3, height // 3), width // 8, (40, 50, 60), -1)
    cv2.circle(img, (2 * width // 3, height // 3), width // 8, (40, 50, 60), -1)
    cv2.rectangle(img, (width // 3, 2 * height // 3), (2 * width // 3, 3 * height // 4), (50, 60, 200), -1)
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}", img


def test_feedback_submission_with_real_crop(client_with_db, tmp_path):
    """Submitting feedback with valid image_base64 stores the real image crop and its metadata."""
    b64_crop, orig_img = _create_synthetic_face_jpeg_base64(120, 120)

    payload = {
        "feedback_type": "correct",
        "predicted_emotion": "happy",
        "confidence": 0.94,
        "face_index": 0,
        "bounding_box": [50, 60, 120, 120],
        "image_base64": b64_crop,
        "notes": "Webcam face crop regression test",
    }

    response = client_with_db.post(f"{settings.API_V1_STR}/feedback", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    record_id = data["record_id"]

    # Verify JSONL record in the isolated test feedback directory
    log_file = tmp_path / settings.FEEDBACK_LOG_FILE
    assert log_file.exists(), "Feedback log file should exist in isolated storage"

    import json
    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    matched = [r for r in records if r.get("feedback_id") == record_id]
    assert len(matched) == 1
    rec = matched[0]

    assert rec["state"] == "correct"
    assert rec["predicted_emotion"] == "happy"
    assert rec["bounding_box"] == [50, 60, 120, 120]
    assert rec["metadata"].get("face_index") == 0
    assert rec["image_path"] is not None

    # Verify physical file existence and dimensions
    saved_file = Path(rec["image_path"])
    assert saved_file.exists(), f"Crop image file must exist: {saved_file}"
    assert saved_file.stat().st_size > 1000, "Crop file must be a real image with non-trivial size"

    decoded = cv2.imread(str(saved_file))
    assert decoded is not None
    assert decoded.shape == (120, 120, 3)
    # Ensure it's NOT a flat black image
    assert decoded.mean() > 10.0


def test_feedback_submission_without_crop_does_not_create_fake_image(client_with_db, tmp_path):
    """When image_base64 is omitted, image_path must be None and NO fake image should be written."""
    payload = {
        "feedback_type": "uncertain",
        "predicted_emotion": "neutral",
        "confidence": 0.45,
        "face_index": 1,
        "bounding_box": [200, 150, 80, 80],
        "image_base64": None,
        "notes": "Feedback without image crop",
    }

    response = client_with_db.post(f"{settings.API_V1_STR}/feedback", json=payload)
    assert response.status_code == 201
    record_id = response.json()["record_id"]

    import json
    log_file = tmp_path / settings.FEEDBACK_LOG_FILE
    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    matched = [r for r in records if r.get("feedback_id") == record_id]
    assert len(matched) == 1
    rec = matched[0]

    assert rec["image_path"] is None, "When image_base64 is omitted, image_path must be None"
    assert rec["bounding_box"] == [200, 150, 80, 80]
    assert rec["metadata"].get("face_index") == 1

    # Check images directory in tmp_path to ensure no dummy file was created
    images_dir = tmp_path / "images"
    created_files = list(images_dir.glob("*.jpg")) if images_dir.exists() else []
    assert len(created_files) == 0, "No dummy 48x48 image should be created when image_base64 is None"


def test_multi_face_independent_crops(client_with_db, tmp_path):
    """Submitting feedback for Face 01 and Face 02 produces two distinct records with their respective crops."""
    b64_face1, _ = _create_synthetic_face_jpeg_base64(80, 80, color=(100, 150, 200))
    b64_face2, _ = _create_synthetic_face_jpeg_base64(95, 95, color=(200, 100, 150))

    # Submit Face 01
    res1 = client_with_db.post(
        f"{settings.API_V1_STR}/feedback",
        json={
            "feedback_type": "correct",
            "predicted_emotion": "happy",
            "confidence": 0.92,
            "face_index": 0,
            "bounding_box": [50, 50, 80, 80],
            "image_base64": b64_face1,
        },
    )
    assert res1.status_code == 201
    id1 = res1.json()["record_id"]

    # Submit Face 02
    res2 = client_with_db.post(
        f"{settings.API_V1_STR}/feedback",
        json={
            "feedback_type": "incorrect",
            "predicted_emotion": "sad",
            "corrected_emotion": "neutral",
            "confidence": 0.65,
            "face_index": 1,
            "bounding_box": [300, 150, 95, 95],
            "image_base64": b64_face2,
        },
    )
    assert res2.status_code == 201
    id2 = res2.json()["record_id"]

    import json
    log_file = tmp_path / settings.FEEDBACK_LOG_FILE
    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    r1 = next(r for r in records if r.get("feedback_id") == id1)
    r2 = next(r for r in records if r.get("feedback_id") == id2)

    assert r1["bounding_box"] == [50, 50, 80, 80]
    assert r1["metadata"].get("face_index") == 0
    assert r2["bounding_box"] == [300, 150, 95, 95]
    assert r2["metadata"].get("face_index") == 1
    assert r1["image_path"] != r2["image_path"]

    # Verify physical file dimensions match each face
    img1 = cv2.imread(r1["image_path"])
    img2 = cv2.imread(r2["image_path"])
    assert img1.shape == (80, 80, 3)
    assert img2.shape == (95, 95, 3)


def test_invalid_image_base64_rejected(client_with_db):
    """Submitting corrupt or invalid base64 image data is rejected with 422."""
    payload = {
        "feedback_type": "correct",
        "predicted_emotion": "happy",
        "confidence": 0.90,
        "image_base64": "not-valid-base64!!!",
    }
    response = client_with_db.post(f"{settings.API_V1_STR}/feedback", json=payload)
    assert response.status_code == 422
