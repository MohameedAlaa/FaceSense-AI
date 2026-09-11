import base64


def test_submit_correct_feedback(client, isolated_feedback_service, sample_crop_bytes):
    b64_str = base64.b64encode(sample_crop_bytes).decode("utf-8")
    payload = {
        "feedback_type": "correct",
        "predicted_emotion": "happy",
        "confidence": 0.88,
        "image_base64": b64_str,
        "notes": "Verified happy expression",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["record_id"].startswith("fb_")


def test_submit_incorrect_feedback(client, isolated_feedback_service):
    payload = {
        "feedback_type": "incorrect",
        "predicted_emotion": "neutral",
        "confidence": 0.55,
        "corrected_emotion": "sad",
        "notes": "Subject was actually sad",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert "fb_" in data["record_id"]


def test_submit_invalid_feedback_type_rejected(client, isolated_feedback_service):
    payload = {
        "feedback_type": "invalid_type",
        "predicted_emotion": "happy",
        "confidence": 0.9,
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422


def test_get_feedback_stats(client, isolated_feedback_service):
    # Post one correct, one incorrect
    client.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "correct",
            "predicted_emotion": "happy",
            "confidence": 0.95,
        },
    )
    client.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "incorrect",
            "predicted_emotion": "angry",
            "confidence": 0.60,
            "corrected_emotion": "disgust",
        },
    )

    stats_response = client.get("/api/v1/feedback/stats")
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["total_records"] == 2
    assert stats["correct"] == 1
    assert stats["incorrect"] == 1
    assert stats["uncertain"] == 0
    assert stats["accuracy_rate"] == 0.5
    assert "happy" in stats["emotions"]
