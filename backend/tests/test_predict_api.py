import io
from pathlib import Path
from backend.app.core.config import WORKSPACE_ROOT


def test_predict_image_with_fallback(client, sample_crop_bytes):
    response = client.post(
        "/api/v1/predict/image?allow_fallback=true",
        files={"file": ("face.jpg", sample_crop_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["faces_detected"] >= 1
    assert len(data["predictions"]) >= 1
    pred = data["predictions"][0]
    assert "emotion" in pred
    assert "confidence" in pred
    assert "all_probabilities" in pred
    assert len(pred["all_probabilities"]) == 7
    assert data["processing_time_ms"] > 0


def test_predict_image_no_face_detected_error(client):
    # Blank 20x20 black square will not match any Haar cascade face
    import numpy as np
    import cv2

    blank = np.zeros((20, 20, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)

    response = client.post(
        "/api/v1/predict/image?allow_fallback=false",
        files={"file": ("blank.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 422
    data = response.json()
    assert "NoFaceDetectedError" in data.get("error", "")


def test_predict_invalid_image_corrupt_bytes(client):
    corrupt_bytes = b"not_a_real_image_data_stream"
    response = client.post(
        "/api/v1/predict/image",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Invalid image content" in data.get("detail", "") or "ImageDecodeError" in data.get("error", "")


def test_predict_with_real_fer2013_sample(client):
    real_sample_path = WORKSPACE_ROOT / "Data(FER2013)" / "test" / "happy" / "PrivateTest_10077120.jpg"
    if real_sample_path.exists():
        with open(real_sample_path, "rb") as f:
            img_bytes = f.read()

        response = client.post(
            "/api/v1/predict/image?allow_fallback=true",
            files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["faces_detected"] == 1
        assert data["predictions"][0]["emotion"] in [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "uncertain"
        ]
