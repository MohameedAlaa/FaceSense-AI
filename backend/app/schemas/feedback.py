import base64
from typing import Optional, Dict
from pydantic import BaseModel, Field, field_validator

from backend.app.core.upload_security import validate_image_bytes

MAX_NOTES_LENGTH = 500
MAX_FEEDBACK_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MB max decoded image size
MAX_FEEDBACK_BASE64_LEN = 3_000_000  # ~2.8MB encoded payload limit + header margin


class FeedbackCreateRequest(BaseModel):
    feedback_type: str = Field(..., description="'correct', 'incorrect', or 'uncertain'")
    predicted_emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    corrected_emotion: Optional[str] = Field(None, description="Corrected emotion if feedback_type is 'incorrect'")
    image_base64: Optional[str] = Field(
        None,
        max_length=MAX_FEEDBACK_BASE64_LEN,
        description="Optional Base64-encoded face crop image (max ~2MB decoded)",
    )
    notes: Optional[str] = Field(
        None,
        max_length=MAX_NOTES_LENGTH,
        description="Optional user notes or remarks (max 500 characters)",
    )

    @field_validator("feedback_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = {"correct", "incorrect", "uncertain"}
        v_clean = v.strip().lower()
        if v_clean not in valid_types:
            raise ValueError(f"feedback_type must be one of {sorted(valid_types)}")
        return v_clean

    @field_validator("image_base64")
    @classmethod
    def validate_image_base64(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v_clean = v.strip()
        if not v_clean:
            return None

        # Pre-filter string length before attempting decode
        if len(v_clean) > MAX_FEEDBACK_BASE64_LEN:
            raise ValueError(
                f"image_base64 payload exceeds maximum length of {MAX_FEEDBACK_BASE64_LEN} characters."
            )

        raw_b64 = v_clean
        if raw_b64.startswith("data:"):
            if "," not in raw_b64:
                raise ValueError("Invalid data URI format: missing comma separator.")
            header, _, raw_b64 = raw_b64.partition(",")
            if not header.startswith("data:image/"):
                raise ValueError("Invalid data URI format: MIME type must be an image format (e.g. data:image/jpeg;base64,...).")
            raw_b64 = raw_b64.strip()
        elif "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1].strip()

        try:
            img_bytes = base64.b64decode(raw_b64, validate=True)
        except Exception:
            raise ValueError("Invalid Base64 format: payload could not be decoded.")

        if not img_bytes:
            raise ValueError("Decoded image payload is empty.")

        if len(img_bytes) > MAX_FEEDBACK_IMAGE_BYTES:
            raise ValueError(
                f"Decoded image size ({len(img_bytes)} bytes) exceeds maximum limit of {MAX_FEEDBACK_IMAGE_BYTES} bytes."
            )

        # Validate magic bytes and verify image decoding integrity via OpenCV
        validate_image_bytes(img_bytes, max_bytes=MAX_FEEDBACK_IMAGE_BYTES)

        return v_clean


class FeedbackCreateResponse(BaseModel):
    status: str = "success"
    record_id: str
    message: str


class FeedbackStatsResponse(BaseModel):
    total_records: int
    correct: int
    incorrect: int
    uncertain: int
    emotions: Dict[str, int]
    accuracy_rate: Optional[float] = None
