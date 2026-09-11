import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.feedback import FeedbackRecord
from backend.app.schemas.feedback import (
    FeedbackCreateRequest,
    FeedbackCreateResponse,
    FeedbackStatsResponse,
)
from ml.feedback.collector import FeedbackCollector, FeedbackState

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Feedback service providing dual-write support:
    - Primary write: structured JSONL via FeedbackCollector
    - Secondary write: PostgreSQL database via SQLAlchemy FeedbackRecord
    Gracefully handles database unavailability without interrupting service.
    """

    def __init__(self, feedback_dir: Optional[Path] = None):
        self._feedback_dir = feedback_dir or settings.FEEDBACK_DIR
        self._collector: Optional[FeedbackCollector] = None

    @property
    def collector(self) -> FeedbackCollector:
        if self._collector is None:
            self._collector = FeedbackCollector(
                feedback_dir=str(self._feedback_dir),
                log_filename=settings.FEEDBACK_LOG_FILE,
            )
        return self._collector

    def create_feedback(
        self,
        request: FeedbackCreateRequest,
        db: Optional[Session] = None,
    ) -> FeedbackCreateResponse:
        """
        Processes feedback request:
        1. Decodes and saves face crop image if provided.
        2. Primary write: Appends record to JSONL via FeedbackCollector.
        3. Secondary write: Inserts record into database if session is provided.
        """
        metadata: Dict[str, Any] = {}
        if request.notes:
            metadata["notes"] = request.notes
        metadata["source"] = "api_v1"

        state_val = request.feedback_type.lower()
        if state_val == "correct":
            fb_state = FeedbackState.CORRECT
        elif state_val == "incorrect":
            fb_state = FeedbackState.INCORRECT
        else:
            fb_state = FeedbackState.UNCERTAIN

        if request.image_base64:
            try:
                raw_b64 = request.image_base64
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(raw_b64)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                crop_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if crop_img is None:
                    raise ValueError("Failed to decode base64 image data into valid image")
            except Exception as e:
                logger.warning("Base64 image decoding failed, falling back to dummy image: %s", e)
                crop_img = np.zeros((48, 48, 3), dtype=np.uint8)
        else:
            # Create a 48x48 neutral placeholder if no image was provided
            crop_img = np.zeros((48, 48, 3), dtype=np.uint8)

        # 1. Primary storage: JSONL via FeedbackCollector
        record = self.collector.save_face_crop_and_add_feedback(
            frame_or_face=crop_img,
            predicted_emotion=request.predicted_emotion,
            confidence=request.confidence,
            corrected_emotion=request.corrected_emotion,
            state=fb_state,
            metadata=metadata,
        )
        record_id = record["feedback_id"]

        # 2. Secondary storage: PostgreSQL database (additive, resilient)
        if db is not None:
            try:
                db_record = FeedbackRecord(
                    feedback_id=record_id,
                    state=record.get("state", fb_state.value),
                    predicted_emotion=record.get("predicted_emotion", request.predicted_emotion),
                    confidence=float(record.get("confidence", request.confidence)),
                    corrected_emotion=record.get("corrected_emotion"),
                    image_path=record.get("image_path"),
                    bounding_box=record.get("bounding_box"),
                    notes=request.notes,
                    model_version=record.get("model_version"),
                    extra_metadata=record.get("metadata"),
                )
                db.add(db_record)
                db.commit()
                db.refresh(db_record)
                logger.info("Feedback persisted to database: %s (id=%s)", record_id, db_record.id)
            except Exception as e:
                db.rollback()
                logger.warning(
                    "Failed to persist feedback to database (JSONL write succeeded): %s", e
                )

        return FeedbackCreateResponse(
            status="success",
            record_id=record_id,
            message="Feedback recorded successfully.",
        )

    def get_stats(self, db: Optional[Session] = None) -> FeedbackStatsResponse:
        """
        Retrieves aggregate statistics from the database if available;
        gracefully falls back to JSONL summary statistics if DB is not provided or fails.
        """
        if db is not None:
            try:
                # Query total records
                total = db.scalar(select(func.count(FeedbackRecord.id))) or 0

                # Query state counts
                state_stmt = (
                    select(FeedbackRecord.state, func.count(FeedbackRecord.id))
                    .group_by(FeedbackRecord.state)
                )
                state_rows = db.execute(state_stmt).all()
                state_counts = {row[0]: row[1] for row in state_rows}

                correct_count = state_counts.get("correct", 0)
                incorrect_count = state_counts.get("incorrect", 0)
                uncertain_count = state_counts.get("uncertain", 0)

                # Query emotion counts
                emotion_stmt = (
                    select(FeedbackRecord.predicted_emotion, func.count(FeedbackRecord.id))
                    .group_by(FeedbackRecord.predicted_emotion)
                )
                emotion_rows = db.execute(emotion_stmt).all()
                emotions_breakdown = {row[0]: row[1] for row in emotion_rows}

                evaluated = correct_count + incorrect_count
                accuracy = (
                    round(float(correct_count) / float(evaluated), 4) if evaluated > 0 else None
                )

                return FeedbackStatsResponse(
                    total_records=total,
                    correct=correct_count,
                    incorrect=incorrect_count,
                    uncertain=uncertain_count,
                    emotions=emotions_breakdown,
                    accuracy_rate=accuracy,
                )
            except Exception as e:
                logger.warning("Database stats query failed, falling back to JSONL: %s", e)

        # Fallback to JSONL
        raw_stats = self.collector.get_summary_statistics()
        by_state = raw_stats.get("by_state", {})
        correct_count = by_state.get(FeedbackState.CORRECT.value, 0)
        incorrect_count = by_state.get(FeedbackState.INCORRECT.value, 0)
        uncertain_count = by_state.get(FeedbackState.UNCERTAIN.value, 0)
        total = raw_stats.get("total_records", 0)

        evaluated = correct_count + incorrect_count
        accuracy = round(float(correct_count) / float(evaluated), 4) if evaluated > 0 else None
        emotions_breakdown = raw_stats.get("by_predicted_emotion", {})

        return FeedbackStatsResponse(
            total_records=total,
            correct=correct_count,
            incorrect=incorrect_count,
            uncertain=uncertain_count,
            emotions=emotions_breakdown,
            accuracy_rate=accuracy,
        )

    def get_by_feedback_id(
        self,
        feedback_id: str,
        db: Optional[Session] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a feedback record by feedback_id from database or JSONL."""
        if db is not None:
            try:
                stmt = select(FeedbackRecord).where(FeedbackRecord.feedback_id == feedback_id)
                record = db.scalar(stmt)
                if record is not None:
                    return record.to_dict()
            except Exception as e:
                logger.warning("Database query by feedback_id failed, falling back: %s", e)

        # Fallback to collector
        all_records = self.collector.load_feedback_records()
        for r in all_records:
            if r.get("feedback_id") == feedback_id:
                return r
        return None

    def list_feedback(
        self,
        limit: int = 50,
        offset: int = 0,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """List feedback records with pagination."""
        if db is not None:
            try:
                stmt = (
                    select(FeedbackRecord)
                    .order_by(FeedbackRecord.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                records = db.scalars(stmt).all()
                return [r.to_dict() for r in records]
            except Exception as e:
                logger.warning("Database listing failed, falling back: %s", e)

        all_records = self.collector.load_feedback_records()
        return all_records[offset : offset + limit]


feedback_service = FeedbackService()
