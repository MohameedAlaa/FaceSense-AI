import base64
from typing import Optional, Dict, Any, List
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
    bounding_box: Optional[List[int]] = Field(
        None,
        description="Optional [x, y, w, h] face bounding box in image pixel coordinates",
    )
    face_index: Optional[int] = Field(
        None,
        ge=0,
        description="Optional 0-based index of the detected face",
    )

    @field_validator("bounding_box")
    @classmethod
    def validate_bounding_box(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is None:
            return None
        if len(v) != 4:
            raise ValueError("bounding_box must contain exactly 4 integers [x, y, w, h]")
        x, y, w, h = v
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            raise ValueError("bounding_box coordinates must be non-negative with positive width/height")
        return v

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


class FeedbackReviewItem(BaseModel):
    id: Optional[int] = None
    feedback_id: str
    user_id: Optional[int] = None
    state: str
    predicted_emotion: str
    confidence: float
    corrected_emotion: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    bounding_box: Optional[Any] = None
    notes: Optional[str] = None
    model_version: Optional[str] = None
    review_status: str = "pending"
    final_label: Optional[str] = None
    reviewed_at: Optional[str] = None
    reviewed_by_id: Optional[int] = None
    created_at: Optional[str] = None


class FeedbackReviewStatusCounts(BaseModel):
    pending: int = 0
    approved: int = 0
    rejected: int = 0
    total: int = 0


class FeedbackEmotionReviewSummary(BaseModel):
    emotion: str
    counts: FeedbackReviewStatusCounts


class FeedbackReviewOverviewResponse(BaseModel):
    emotions: Dict[str, FeedbackReviewStatusCounts]
    total_records: int


class FeedbackReviewListResponse(BaseModel):
    emotion: str
    status_filter: Optional[str] = None
    total: int
    items: List[FeedbackReviewItem]


class FeedbackReviewDecisionRequest(BaseModel):
    decision: str = Field(..., description="'approve' or 'reject'")
    final_label: Optional[str] = Field(
        None,
        description="Authoritative final emotion label (if approved). If omitted on approve, defaults to corrected_emotion (if present) or predicted_emotion.",
    )

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if v_clean not in {"approve", "reject"}:
            raise ValueError("decision must be either 'approve' or 'reject'")
        return v_clean

    @field_validator("final_label")
    @classmethod
    def validate_final_label(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v_clean = v.strip().lower()
        valid_emotions = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}
        if v_clean not in valid_emotions:
            raise ValueError(f"final_label must be one of {sorted(valid_emotions)}")
        return v_clean


class FeedbackReviewDecisionResponse(BaseModel):
    status: str = "success"
    feedback_id: str
    review_status: str
    final_label: Optional[str] = None
    message: str

