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


# ==============================================================================
# Human-in-the-Loop Feedback Review Tests
# ==============================================================================

def test_feedback_review_overview_unauthorized(client, isolated_feedback_service):
    """Anonymous user cannot access feedback review overview."""
    res = client.get("/api/v1/feedback/review/overview")
    assert res.status_code == 401


def test_feedback_review_overview_forbidden(client_with_db, test_db_session, isolated_feedback_service):
    """Normal user cannot access feedback review overview."""
    user = User(email="normal_user@example.com", password_hash="pw", role="user", is_active=True)
    test_db_session.add(user)
    test_db_session.commit()
    token = create_access_token(user.id, user.role)

    res = client_with_db.get(
        "/api/v1/feedback/review/overview",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403


def test_feedback_review_overview_admin(client_with_db, test_db_session, isolated_feedback_service):
    """Admin can fetch overview with counts across all 7 emotions."""
    admin = User(email="admin_review@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    token = create_access_token(admin.id, admin.role)

    # Submit feedback for happy and fear
    client_with_db.post("/api/v1/feedback", json={"feedback_type": "correct", "predicted_emotion": "happy", "confidence": 0.90})
    client_with_db.post("/api/v1/feedback", json={"feedback_type": "incorrect", "predicted_emotion": "happy", "confidence": 0.60, "corrected_emotion": "sad"})
    client_with_db.post("/api/v1/feedback", json={"feedback_type": "correct", "predicted_emotion": "fear", "confidence": 0.75})

    res = client_with_db.get(
        "/api/v1/feedback/review/overview",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "emotions" in data
    assert data["emotions"]["happy"]["pending"] == 2
    assert data["emotions"]["happy"]["total"] == 2
    assert data["emotions"]["fear"]["pending"] == 1
    assert data["emotions"]["fear"]["total"] == 1
    assert data["emotions"]["angry"]["total"] == 0
    assert data["total_records"] == 3


def test_feedback_review_by_emotion_admin(client_with_db, test_db_session, isolated_feedback_service):
    """Admin can fetch feedback records filtered by emotion and status."""
    admin = User(email="admin_list@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    token = create_access_token(admin.id, admin.role)

    # Create 2 happy items
    fb_res = client_with_db.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "incorrect",
            "predicted_emotion": "happy",
            "confidence": 0.72,
            "corrected_emotion": "sad",
            "notes": "Looked sad",
        },
    )
    assert fb_res.status_code == 201

    # Fetch happy feedback
    res = client_with_db.get(
        "/api/v1/feedback/review/emotion/happy",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["emotion"] == "happy"
    assert data["total"] >= 1
    item = data["items"][0]
    assert item["predicted_emotion"] == "happy"
    assert item["corrected_emotion"] == "sad"
    assert item["review_status"] == "pending"
    assert item["notes"] == "Looked sad"


def test_feedback_review_by_invalid_emotion(client_with_db, test_db_session, isolated_feedback_service):
    """Invalid emotion parameter is rejected with 422."""
    admin = User(email="admin_err@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    token = create_access_token(admin.id, admin.role)

    res = client_with_db.get(
        "/api/v1/feedback/review/emotion/invalid_emotion",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 422


def test_feedback_review_decision_approve(client_with_db, test_db_session, isolated_feedback_service):
    """Admin approves feedback: review_status becomes approved, final_label is set authoritatively."""
    admin = User(email="admin_decide@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    token = create_access_token(admin.id, admin.role)

    # Submit feedback: Model = happy, User correction = sad
    create_res = client_with_db.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "incorrect",
            "predicted_emotion": "happy",
            "confidence": 0.65,
            "corrected_emotion": "sad",
        },
    )
    record_id = create_res.json()["record_id"]

    # Admin approves with final_label='sad'
    decision_res = client_with_db.patch(
        f"/api/v1/feedback/review/{record_id}/decision",
        json={"decision": "approve", "final_label": "sad"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert decision_res.status_code == 200
    dec_data = decision_res.json()
    assert dec_data["status"] == "success"
    assert dec_data["review_status"] == "approved"
    assert dec_data["final_label"] == "sad"

    # Verify item status updated in listing
    list_res = client_with_db.get(
        "/api/v1/feedback/review/emotion/happy?status=approved",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 1
    assert list_res.json()["items"][0]["review_status"] == "approved"
    assert list_res.json()["items"][0]["final_label"] == "sad"


def test_feedback_review_decision_reject(client_with_db, test_db_session, isolated_feedback_service):
    """Admin rejects feedback: review_status becomes rejected, excluded from training."""
    admin = User(email="admin_reject@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    token = create_access_token(admin.id, admin.role)

    create_res = client_with_db.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "incorrect",
            "predicted_emotion": "surprise",
            "confidence": 0.50,
            "corrected_emotion": "angry",
        },
    )
    record_id = create_res.json()["record_id"]

    decision_res = client_with_db.patch(
        f"/api/v1/feedback/review/{record_id}/decision",
        json={"decision": "reject"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert decision_res.status_code == 200
    dec_data = decision_res.json()
    assert dec_data["review_status"] == "rejected"
    assert dec_data["final_label"] is None


def test_feedback_review_decision_forbidden_for_user(client_with_db, test_db_session, isolated_feedback_service):
    """Normal non-admin user cannot submit review decisions."""
    user = User(email="non_admin@example.com", password_hash="pw", role="user", is_active=True)
    test_db_session.add(user)
    test_db_session.commit()
    user_token = create_access_token(user.id, user.role)

    res = client_with_db.patch(
        "/api/v1/feedback/review/fb_dummy/decision",
        json={"decision": "approve"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert res.status_code == 403


def test_feedback_review_decision_invalid_values(client_with_db, test_db_session, isolated_feedback_service):
    """Invalid decision or invalid final_label rejected by schema validator."""
    admin = User(email="admin_val@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    token = create_access_token(admin.id, admin.role)

    # Invalid decision
    res1 = client_with_db.patch(
        "/api/v1/feedback/review/fb_dummy/decision",
        json={"decision": "maybe"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res1.status_code == 422

    # Invalid final_label emotion
    res2 = client_with_db.patch(
        "/api/v1/feedback/review/fb_dummy/decision",
        json={"decision": "approve", "final_label": "ecstatic"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res2.status_code == 422


# ==============================================================================
# Feedback Image Endpoint Authorization & Security Tests
# ==============================================================================

def test_feedback_image_unauthorized(client):
    """Unauthenticated request to retrieve feedback image is rejected with 401."""
    res = client.get("/api/v1/feedback/images/test_crop.jpg")
    assert res.status_code == 401


def test_feedback_image_forbidden_for_normal_user(client_with_db, test_db_session):
    """Normal user cannot access feedback images (403 Forbidden)."""
    user = User(email="img_user@example.com", password_hash="pw", role="user", is_active=True)
    test_db_session.add(user)
    test_db_session.commit()
    user_token = create_access_token(user.id, user.role)

    res = client_with_db.get(
        "/api/v1/feedback/images/test_crop.jpg",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert res.status_code == 403


def test_feedback_image_admin_success(client_with_db, test_db_session, tmp_path, monkeypatch):
    """Admin user can successfully fetch a stored face crop image."""
    from backend.app.core.config import settings

    # Create dummy image in actual feedback dir
    img_dir = settings.FEEDBACK_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    dummy_img = img_dir / "test_crop_admin_success.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")

    try:
        admin = User(email="img_admin@example.com", password_hash="pw", role="admin", is_active=True)
        test_db_session.add(admin)
        test_db_session.commit()
        admin_token = create_access_token(admin.id, admin.role)

        res = client_with_db.get(
            "/api/v1/feedback/images/test_crop_admin_success.jpg",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 200
        assert "image/jpeg" in res.headers.get("content-type", "")
        assert len(res.content) > 0
    finally:
        if dummy_img.exists():
            dummy_img.unlink()


def test_feedback_image_missing_returns_404(client_with_db, test_db_session):
    """Admin requesting nonexistent image receives 404."""
    admin = User(email="img_admin_404@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    admin_token = create_access_token(admin.id, admin.role)

    res = client_with_db.get(
        "/api/v1/feedback/images/nonexistent_crop_999999.jpg",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 404


def test_feedback_image_path_traversal_rejected(client_with_db, test_db_session):
    """Path traversal sequences in image filename are rejected with 400."""
    admin = User(email="img_admin_sec@example.com", password_hash="pw", role="admin", is_active=True)
    test_db_session.add(admin)
    test_db_session.commit()
    admin_token = create_access_token(admin.id, admin.role)

    res = client_with_db.get(
        "/api/v1/feedback/images/..%2f..%2fsecret.txt",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code in (400, 404)


