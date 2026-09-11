import pytest
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.db.session import check_db_connection
from backend.app.models.feedback import FeedbackRecord
from backend.app.schemas.feedback import FeedbackCreateRequest
from backend.app.services.feedback_service import FeedbackService
from ml.feedback.collector import FeedbackState


def test_feedback_record_model(test_db_session: Session):
    """Verify FeedbackRecord ORM model insertion, indexing, and to_dict serialization."""
    record = FeedbackRecord(
        feedback_id="test_fb_001",
        state="correct",
        predicted_emotion="happy",
        confidence=0.95,
        corrected_emotion=None,
        image_path="outputs/feedback/images/test.jpg",
        bounding_box=[10, 20, 30, 40],
        notes="Model was accurate",
        model_version="test_model_v1",
        extra_metadata={"client": "test", "latency_ms": 42},
    )
    test_db_session.add(record)
    test_db_session.commit()
    test_db_session.refresh(record)

    assert record.id is not None
    assert record.feedback_id == "test_fb_001"
    assert record.state == "correct"
    assert record.confidence == 0.95
    assert record.created_at is not None

    record_dict = record.to_dict()
    assert record_dict["feedback_id"] == "test_fb_001"
    assert record_dict["bounding_box"] == [10, 20, 30, 40]
    assert record_dict["metadata"]["client"] == "test"


def test_feedback_record_unique_constraint(test_db_session: Session):
    """Verify that feedback_id must be unique."""
    record1 = FeedbackRecord(
        feedback_id="duplicate_id",
        state="correct",
        predicted_emotion="happy",
        confidence=0.88,
    )
    test_db_session.add(record1)
    test_db_session.commit()

    record2 = FeedbackRecord(
        feedback_id="duplicate_id",
        state="incorrect",
        predicted_emotion="sad",
        confidence=0.55,
    )
    test_db_session.add(record2)
    with pytest.raises(IntegrityError):
        test_db_session.commit()
    test_db_session.rollback()


def test_dual_write_service(tmp_path, test_db_session: Session):
    """Verify FeedbackService persists to both JSONL and DB when session is supplied."""
    service = FeedbackService(feedback_dir=tmp_path)
    req = FeedbackCreateRequest(
        feedback_type="incorrect",
        predicted_emotion="neutral",
        confidence=0.65,
        corrected_emotion="sad",
        notes="Actually sad",
    )

    resp = service.create_feedback(req, db=test_db_session)
    assert resp.status == "success"
    assert resp.record_id.startswith("fb_")

    # Verify written to DB
    stmt = select(FeedbackRecord).where(FeedbackRecord.feedback_id == resp.record_id)
    db_rec = test_db_session.scalar(stmt)
    assert db_rec is not None
    assert db_rec.state == "incorrect"
    assert db_rec.predicted_emotion == "neutral"
    assert db_rec.corrected_emotion == "sad"
    assert db_rec.notes == "Actually sad"

    # Verify written to JSONL
    jsonl_records = service.collector.load_feedback_records()
    assert len(jsonl_records) == 1
    assert jsonl_records[0]["feedback_id"] == resp.record_id


def test_dual_write_db_failure_fallback(tmp_path, monkeypatch):
    """Verify that if DB commit raises an exception, the JSONL write still succeeds."""
    service = FeedbackService(feedback_dir=tmp_path)
    req = FeedbackCreateRequest(
        feedback_type="correct",
        predicted_emotion="happy",
        confidence=0.99,
        notes="Great prediction",
    )

    class BrokenSession:
        def add(self, obj):
            pass
        def commit(self):
            raise RuntimeError("Database connection suddenly dropped!")
        def rollback(self):
            pass

    broken_db = BrokenSession()
    # Should not raise exception
    resp = service.create_feedback(req, db=broken_db)
    assert resp.status == "success"
    assert resp.record_id.startswith("fb_")

    # JSONL must still contain the record
    jsonl_records = service.collector.load_feedback_records()
    assert len(jsonl_records) == 1
    assert jsonl_records[0]["feedback_id"] == resp.record_id


def test_get_stats_from_db(tmp_path, test_db_session: Session):
    """Verify get_stats aggregates counts correctly from the database."""
    service = FeedbackService(feedback_dir=tmp_path)

    # Insert a set of records directly into DB
    records = [
        FeedbackRecord(feedback_id="f1", state="correct", predicted_emotion="happy", confidence=0.9),
        FeedbackRecord(feedback_id="f2", state="correct", predicted_emotion="happy", confidence=0.8),
        FeedbackRecord(feedback_id="f3", state="incorrect", predicted_emotion="sad", confidence=0.7, corrected_emotion="angry"),
        FeedbackRecord(feedback_id="f4", state="uncertain", predicted_emotion="fear", confidence=0.4),
    ]
    test_db_session.add_all(records)
    test_db_session.commit()

    stats = service.get_stats(db=test_db_session)
    assert stats.total_records == 4
    assert stats.correct == 2
    assert stats.incorrect == 1
    assert stats.uncertain == 1
    assert stats.emotions["happy"] == 2
    assert stats.emotions["sad"] == 1
    assert stats.emotions["fear"] == 1
    # Accuracy rate: 2 / (2 + 1) = 0.6667
    assert stats.accuracy_rate == 0.6667


def test_check_db_connection(test_db_session: Session):
    """Verify check_db_connection succeeds with active session."""
    assert check_db_connection(test_db_session) is True


def test_api_feedback_endpoint_with_db(client_with_db, isolated_feedback_service, test_db_session: Session):
    """Test API POST /api/v1/feedback persists to DB and returns 201."""
    payload = {
        "feedback_type": "correct",
        "predicted_emotion": "happy",
        "confidence": 0.91,
        "notes": "Integration test note",
    }
    response = client_with_db.post("/api/v1/feedback", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    rec_id = data["record_id"]

    # Verify directly in test DB
    stmt = select(FeedbackRecord).where(FeedbackRecord.feedback_id == rec_id)
    rec = test_db_session.scalar(stmt)
    assert rec is not None
    assert rec.state == "correct"
    assert rec.predicted_emotion == "happy"


def test_api_feedback_stats_with_db(client_with_db, test_db_session: Session):
    """Test API GET /api/v1/feedback/stats reads from database."""
    record = FeedbackRecord(
        feedback_id="stat_test_01",
        state="correct",
        predicted_emotion="surprise",
        confidence=0.85,
    )
    test_db_session.add(record)
    test_db_session.commit()

    response = client_with_db.get("/api/v1/feedback/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] >= 1
    assert data["correct"] >= 1
