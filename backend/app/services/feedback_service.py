import base64
import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image
import io

from backend.app.core.config import settings
from backend.app.schemas.feedback import (
    FeedbackCreateRequest,
    FeedbackCreateResponse,
    FeedbackStatsResponse,
)
from ml.feedback.collector import FeedbackCollector, FeedbackState

logger = logging.getLogger(__name__)


class FeedbackService:
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

    def create_feedback(self, request: FeedbackCreateRequest) -> FeedbackCreateResponse:
        """
        Processes feedback request, saves face crop if base64 provided,
        and logs the record using the existing FeedbackCollector.
        """
        metadata = {}
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
                # Remove header if data URL format (e.g. 'data:image/jpeg;base64,...')
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

        record = self.collector.save_face_crop_and_add_feedback(
            frame_or_face=crop_img,
            predicted_emotion=request.predicted_emotion,
            confidence=request.confidence,
            corrected_emotion=request.corrected_emotion,
            state=fb_state,
            metadata=metadata,
        )

        return FeedbackCreateResponse(
            status="success",
            record_id=record["feedback_id"],
            message="Feedback recorded successfully.",
        )

    def get_stats(self) -> FeedbackStatsResponse:
        """
        Retrieves aggregate statistics from the feedback records.
        """
        raw_stats = self.collector.get_summary_statistics()
        by_state = raw_stats.get("by_state", {})
        correct_count = by_state.get(FeedbackState.CORRECT.value, 0)
        incorrect_count = by_state.get(FeedbackState.INCORRECT.value, 0)
        uncertain_count = by_state.get(FeedbackState.UNCERTAIN.value, 0)
        total = raw_stats.get("total_records", 0)

        evaluated = correct_count + incorrect_count
        accuracy = round(float(correct_count) / float(evaluated), 4) if evaluated > 0 else None

        # Combine emotion counts
        emotions_breakdown = raw_stats.get("by_predicted_emotion", {})

        return FeedbackStatsResponse(
            total_records=total,
            correct=correct_count,
            incorrect=incorrect_count,
            uncertain=uncertain_count,
            emotions=emotions_breakdown,
            accuracy_rate=accuracy,
        )


feedback_service = FeedbackService()
