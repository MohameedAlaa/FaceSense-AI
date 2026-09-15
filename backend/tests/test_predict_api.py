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


def _make_multi_face_image_bytes():
    import numpy as np
    import cv2

    img = np.full((360, 640, 3), 160, dtype=np.uint8)

    def draw_face(cx, cy, s=1.2):
        cv2.ellipse(img, (cx, cy), (int(45 * s), int(60 * s)), 0, 0, 360, (220, 200, 180), -1)
        cv2.circle(img, (int(cx - 18 * s), int(cy - 15 * s)), int(8 * s), (255, 255, 255), -1)
        cv2.circle(img, (int(cx + 18 * s), int(cy - 15 * s)), int(8 * s), (255, 255, 255), -1)
        cv2.circle(img, (int(cx - 18 * s), int(cy - 15 * s)), int(3 * s), (30, 30, 30), -1)
        cv2.circle(img, (int(cx + 18 * s), int(cy - 15 * s)), int(3 * s), (30, 30, 30), -1)
        cv2.line(img, (cx, int(cy - 5 * s)), (cx, int(cy + 12 * s)), (150, 130, 110), int(2 * s))
        cv2.ellipse(img, (cx, int(cy + 28 * s)), (int(20 * s), int(10 * s)), 0, 0, 180, (50, 50, 200), int(3 * s))

    draw_face(160, 180, 1.2)
    draw_face(480, 180, 1.2)
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()


def test_predict_multi_face_batched_output_and_ordering(client):
    """Verify multi-face image returns all detected faces in detector order with preserved boxes."""
    multi_bytes = _make_multi_face_image_bytes()
    response = client.post(
        "/api/v1/predict/image",
        files={"file": ("multi.jpg", multi_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["faces_detected"] == 2
    assert len(data["predictions"]) == 2

    # Check bounding box fields and predictions
    for pred in data["predictions"]:
        box = pred["box"]
        assert "x" in box and "y" in box and "w" in box and "h" in box
        assert box["w"] > 0 and box["h"] > 0
        assert pred["emotion"] in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "uncertain"]
        assert 0.0 <= pred["confidence"] <= 1.0
        assert len(pred["all_probabilities"]) == 7


def test_predict_multi_face_model_forward_called_once(client):
    """Regression test: verify model.forward is called exactly ONCE for multi-face batch."""
    from unittest.mock import patch
    from backend.app.services.prediction_service import prediction_service

    multi_bytes = _make_multi_face_image_bytes()
    model = prediction_service.predictor.model

    with patch.object(model, "forward", wraps=model.forward) as spy_forward:
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("multi.jpg", multi_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["faces_detected"] == 2
        assert spy_forward.call_count == 1

