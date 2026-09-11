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


from backend.app.models.user import User
from backend.app.core.security import create_access_token

def test_get_feedback_stats_unauthorized(client, isolated_feedback_service):
    response = client.get("/api/v1/feedback/stats")
    assert response.status_code == 401

def test_get_feedback_stats_forbidden(client_with_db, test_db_session, isolated_feedback_service):
    user = User(email="user@example.com", password_hash="pw", role="user", is_active=True)
    test_db_session.add(user)
    test_db_session.commit()
    
    token = create_access_token(user.id, user.role)
    response = client_with_db.get("/api/v1/feedback/stats", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403

def test_get_feedback_stats_admin(client_with_db, test_db_session, isolated_feedback_service):
    # Post one correct, one incorrect
    client_with_db.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "correct",
            "predicted_emotion": "happy",
            "confidence": 0.95,
        },
    )
    client_with_db.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "incorrect",
            "predicted_emotion": "angry",
            "confidence": 0.60,
            "corrected_emotion": "disgust",
        },
    )

    admin = User(email="admin@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    
    token = create_access_token(admin.id, admin.role)
    stats_response = client_with_db.get("/api/v1/feedback/stats", headers={"Authorization": f"Bearer {token}"})
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["total_records"] == 2
    assert stats["correct"] == 1
    assert stats["incorrect"] == 1
    assert stats["uncertain"] == 0
    assert stats["accuracy_rate"] == 0.5
    assert "happy" in stats["emotions"]
