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
        user_id: Optional[int] = None,
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
        if user_id is not None:
            metadata["user_id"] = user_id
        if request.face_index is not None:
            metadata["face_index"] = request.face_index

        state_val = request.feedback_type.lower()
        if state_val == "correct":
            fb_state = FeedbackState.CORRECT
        elif state_val == "incorrect":
            fb_state = FeedbackState.INCORRECT
        else:
            fb_state = FeedbackState.UNCERTAIN

        if request.image_base64:
            raw_b64 = request.image_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(raw_b64.strip())
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image data: invalid encoding") from e

            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            crop_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if crop_img is None or crop_img.size == 0:
                raise ValueError("Failed to decode base64 image data into valid image: corrupted or invalid image format")

            # 1. Primary storage: JSONL via FeedbackCollector with real face crop
            record = self.collector.save_face_crop_and_add_feedback(
                frame_or_face=crop_img,
                predicted_emotion=request.predicted_emotion,
                confidence=request.confidence,
                corrected_emotion=request.corrected_emotion,
                bounding_box=request.bounding_box,
                state=fb_state,
                metadata=metadata,
            )
        else:
            # If no real source image is provided, do NOT fabricate an image. Store image_path = None.
            record = self.collector.add_feedback(
                image_path=None,
                predicted_emotion=request.predicted_emotion,
                confidence=request.confidence,
                corrected_emotion=request.corrected_emotion,
                bounding_box=request.bounding_box,
                state=fb_state,
                metadata=metadata,
            )
        record_id = record["feedback_id"]

        # 2. Secondary storage: PostgreSQL database (additive, resilient)
        if db is not None:
            try:
                db_record = FeedbackRecord(
                    feedback_id=record_id,
                    user_id=user_id,
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


    def get_review_overview(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Retrieves review status counts (pending, approved, rejected, total) grouped
        across all 7 FER2013 emotion categories.
        """
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        overview: Dict[str, Dict[str, int]] = {
            emo: {"pending": 0, "approved": 0, "rejected": 0, "total": 0}
            for emo in emotions
        }
        total_records = 0

        if db is not None:
            try:
                stmt = (
                    select(
                        FeedbackRecord.predicted_emotion,
                        FeedbackRecord.review_status,
                        func.count(FeedbackRecord.id),
                    )
                    .group_by(FeedbackRecord.predicted_emotion, FeedbackRecord.review_status)
                )
                rows = db.execute(stmt).all()
                for emo, status_val, count in rows:
                    emo_clean = str(emo).lower()
                    st_clean = str(status_val).lower()
                    if emo_clean in overview:
                        if st_clean in overview[emo_clean]:
                            overview[emo_clean][st_clean] = count
                        overview[emo_clean]["total"] += count
                    total_records += count

                return {
                    "emotions": overview,
                    "total_records": total_records,
                }
            except Exception as e:
                logger.warning("Database review overview query failed, falling back to JSONL: %s", e)

        # JSONL Fallback
        all_records = self.collector.load_feedback_records()
        for r in all_records:
            total_records += 1
            pred = str(r.get("predicted_emotion", "")).lower()
            rev_st = str(r.get("review_status", "pending")).lower()
            if pred in overview:
                if rev_st in overview[pred]:
                    overview[pred][rev_st] += 1
                overview[pred]["total"] += 1

        return {
            "emotions": overview,
            "total_records": total_records,
        }

    def list_review_feedback(
        self,
        emotion: str,
        status_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Lists feedback records for a specific emotion, optionally filtered by review_status.
        """
        emo_clean = emotion.strip().lower()
        st_clean = status_filter.strip().lower() if status_filter else None

        if db is not None:
            try:
                query = select(FeedbackRecord).where(func.lower(FeedbackRecord.predicted_emotion) == emo_clean)
                if st_clean and st_clean != "all":
                    query = query.where(func.lower(FeedbackRecord.review_status) == st_clean)

                total_stmt = select(func.count()).select_from(query.subquery())
                total = db.scalar(total_stmt) or 0

                records = db.scalars(
                    query.order_by(FeedbackRecord.id.desc()).limit(limit).offset(offset)
                ).all()

                items = [r.to_dict() for r in records]
                return {
                    "emotion": emo_clean,
                    "status_filter": st_clean,
                    "total": total,
                    "items": items,
                }
            except Exception as e:
                logger.warning("Database review listing failed, falling back to JSONL: %s", e)

        # JSONL Fallback
        all_records = self.collector.load_feedback_records()
        matched = []
        for r in all_records:
            if str(r.get("predicted_emotion", "")).lower() == emo_clean:
                rec_st = str(r.get("review_status", "pending")).lower()
                if not st_clean or st_clean == "all" or rec_st == st_clean:
                    rec_copy = dict(r)
                    if "review_status" not in rec_copy:
                        rec_copy["review_status"] = "pending"
                    matched.append(rec_copy)

        total = len(matched)
        sliced = matched[offset : offset + limit]
        return {
            "emotion": emo_clean,
            "status_filter": st_clean,
            "total": total,
            "items": sliced,
        }

    def update_review_decision(
        self,
        feedback_id: str,
        decision: str,
        final_label: Optional[str] = None,
        reviewer_id: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Updates review_status (approved | rejected) and final_label authoritative decision.
        """
        from datetime import datetime, timezone
        import json

        decision_clean = decision.strip().lower()
        target_status = "approved" if decision_clean == "approve" else "rejected"
        reviewed_at = datetime.now(timezone.utc)
        resolved_label: Optional[str] = None

        if target_status == "approved":
            if final_label:
                resolved_label = final_label.strip().lower()
            else:
                # Default: corrected_emotion if given, else predicted_emotion
                record_dict = self.get_by_feedback_id(feedback_id, db=db)
                if record_dict:
                    corr = record_dict.get("corrected_emotion")
                    pred = record_dict.get("predicted_emotion")
                    resolved_label = (corr or pred or "neutral").strip().lower()
                else:
                    resolved_label = "neutral"

        # 1. Update DB if available
        db_updated = False
        if db is not None:
            try:
                stmt = select(FeedbackRecord).where(FeedbackRecord.feedback_id == feedback_id)
                db_record = db.scalar(stmt)
                if db_record is not None:
                    db_record.review_status = target_status
                    db_record.final_label = resolved_label
                    db_record.reviewed_at = reviewed_at
                    db_record.reviewed_by_id = reviewer_id
                    db.commit()
                    db.refresh(db_record)
                    db_updated = True
            except Exception as e:
                db.rollback()
                logger.warning("Database review decision update failed, continuing with JSONL: %s", e)

        # 2. Update JSONL store for redundancy
        try:
            log_path = self.collector.log_path
            if log_path.exists():
                lines = []
                found = False
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            item = json.loads(line_str)
                            if item.get("feedback_id") == feedback_id:
                                item["review_status"] = target_status
                                item["final_label"] = resolved_label
                                item["reviewed_at"] = reviewed_at.isoformat()
                                item["reviewed_by_id"] = reviewer_id
                                found = True
                            lines.append(json.dumps(item, ensure_ascii=False))
                        except Exception:
                            lines.append(line_str)
                if found:
                    with open(log_path, "w", encoding="utf-8") as f:
                        for l in lines:
                            f.write(l + "\n")
        except Exception as e:
            logger.warning("Failed to update JSONL file with review decision: %s", e)

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "review_status": target_status,
            "final_label": resolved_label,
            "message": f"Feedback record successfully marked as {target_status}.",
        }


feedback_service = FeedbackService()


